"""LanceDB read/write operations and ChunkRecord schema."""

import json
import logging
import re
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from pydantic import BaseModel

from specagent.config import settings
from specagent.retrieval.exceptions import StoreError

logger = logging.getLogger(__name__)


class ChunkRecord(BaseModel):
    """One row in the LanceDB documents table — a single embedded chunk."""

    id: str
    doc_id: str
    library: str
    source: str
    content_hash: str
    title: str
    content: str
    embedding: list[float]
    chunk_index: int
    created_at: str
    metadata: str  # JSON-serialised dict
    file_type: str  # e.g. "pdf", "docx", "html", "url"; "unknown" if undetectable
    last_modified: str  # ISO 8601 from file mtime or HTTP Last-Modified; "" if unknown
    page: int  # 1-indexed page number; 0 = unknown or not applicable


def _lance_schema() -> pa.Schema:
    """Return the canonical Arrow schema for the documents table.

    Uses explicit types so the schema is never inferred from Python literals,
    which prevents dimension-mismatch bugs when environment variables override
    settings between runs.
    """
    dim = settings.embedding_dimension
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("library", pa.string()),
            pa.field("source", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
            pa.field("chunk_index", pa.int64()),
            pa.field("created_at", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("last_modified", pa.string()),
            pa.field("page", pa.int64()),
        ]
    )


def _open_table(uri: str, table_name: str) -> lancedb.table.Table:
    """Open (or create) the LanceDB table.

    Args:
        uri: LanceDB URI (local path or s3:// URI).
        table_name: Name of the table within the database.

    Returns:
        An open LanceDB Table object.

    Raises:
        StoreError: If the connection or table open fails, or if the existing
            table's embedding dimension does not match settings.embedding_dimension.
    """
    try:
        # Expand ~ for local paths
        resolved = str(Path(uri).expanduser()) if not uri.startswith("s3://") else uri
        db = lancedb.connect(resolved)
        table_list = db.list_tables()
        existing = (
            table_list.tables if hasattr(table_list, "tables") else list(table_list)
        )
        if table_name in existing:
            table = db.open_table(table_name)
            _validate_embedding_dimension(table)
            _migrate_table(table)
        else:
            # Create table with an explicit PyArrow schema — no dummy record needed.
            table = db.create_table(table_name, schema=_lance_schema())
        return table
    except StoreError:
        raise
    except Exception as e:
        raise StoreError(
            f"Failed to open LanceDB table {table_name!r} at {uri!r}"
        ) from e


def _ensure_scalar_indexes(table: lancedb.table.Table) -> None:
    """Create scalar indexes on commonly filtered columns (idempotent).

    Called once per Store instance after the table is first opened. Scalar indexes
    require at least one row to be created; failures are logged as warnings rather
    than silently ignored.

    Args:
        table: Open LanceDB table to index.
    """
    for col in ("library", "doc_id", "source"):
        try:
            table.create_scalar_index(col, replace=True)
        except Exception as e:
            logger.warning(
                "Scalar index on %r not created (table may be empty — will retry on next write): %s",
                col,
                e,
            )


def _validate_embedding_dimension(table: lancedb.table.Table) -> None:
    """Raise StoreError if the table's embedding column dimension doesn't match settings.

    Args:
        table: Open LanceDB table to validate.

    Raises:
        StoreError: If the dimension in the stored schema differs from
            settings.embedding_dimension.
    """
    try:
        field = table.schema.field("embedding")
        stored_dim = field.type.list_size
    except Exception:
        return  # can't determine — skip validation

    expected = settings.embedding_dimension
    if stored_dim != expected:
        raise StoreError(
            f"Embedding dimension mismatch: the existing index stores {stored_dim}d "
            f"vectors but EMBEDDING_DIMENSION={expected}. "
            "Either restore the original EMBEDDING_DIMENSION value or delete the index "
            "and re-ingest all documents."
        )


def _migrate_table(table: lancedb.table.Table) -> None:
    """Add columns introduced in later schema versions to an existing table.

    Args:
        table: Open LanceDB table to migrate in place.
    """
    existing = {field.name for field in table.schema}
    to_add: dict[str, str] = {}
    if "file_type" not in existing:
        to_add["file_type"] = "''"
    if "last_modified" not in existing:
        to_add["last_modified"] = "''"
    if "page" not in existing:
        to_add["page"] = "CAST(0 AS INT)"
    if not to_add:
        return
    try:
        table.add_columns(to_add)
        logger.info("Migrated table schema: added columns %s", list(to_add))
    except Exception as e:
        logger.warning("Schema migration failed — old data may lack new fields: %s", e)


_SAFE_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _build_where_clause(
    library: str | None,
    filter: dict | None,  # noqa: A002
) -> str | None:
    """Build a SQL WHERE clause from a library restriction and field equality filters.

    Args:
        library: Restrict results to this library name if provided.
        filter: Optional dict mapping column names to equality values.
            String values are single-quoted; int values are unquoted.
            Keys must match ``[a-zA-Z_][a-zA-Z0-9_]*``.

    Returns:
        SQL WHERE clause string, or None if there are no conditions.

    Raises:
        StoreError: If a filter key contains invalid characters.
    """
    conditions: list[str] = []
    if library is not None:
        safe_lib = library.replace("'", "''")
        conditions.append(f"library = '{safe_lib}'")
    if filter:
        for key, value in filter.items():
            if not _SAFE_KEY.match(key):
                raise StoreError(f"Invalid filter key: {key!r}")
            if isinstance(value, int):
                conditions.append(f"{key} = {value}")
            else:
                safe_val = str(value).replace("'", "''")
                conditions.append(f"{key} = '{safe_val}'")
    return " AND ".join(conditions) if conditions else None


class Store:
    """Provides read/write access to the LanceDB documents table.

    Each method opens a fresh connection — LanceDB is embedded and cheap to open.
    """

    def __init__(
        self,
        uri: str | None = None,
        table_name: str | None = None,
    ) -> None:
        """Initialise the store with connection parameters.

        Args:
            uri: LanceDB URI. Defaults to settings.lancedb_uri.
            table_name: Table name. Defaults to settings.lancedb_table_name.
        """
        self._uri: str = uri if uri is not None else str(settings.lancedb_uri)
        self._table_name = table_name or settings.lancedb_table_name
        self._indexes_created = False

    def _table(self) -> lancedb.table.Table:
        """Open and return the LanceDB table, creating scalar indexes on first call."""
        table = _open_table(self._uri, self._table_name)
        if not self._indexes_created:
            _ensure_scalar_indexes(table)
            self._indexes_created = True
        return table

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Write a list of chunk records to the store.

        Args:
            chunks: Chunk records to insert.

        Raises:
            StoreError: If the write fails.
        """
        if not chunks:
            return
        try:
            table = self._table()
            rows = [c.model_dump() for c in chunks]
            # Convert embeddings to float32 numpy arrays to satisfy the
            # FixedSizeList<float32> Arrow schema — Python lists are typed as
            # ListType and only cast correctly when the sizes match, while numpy
            # arrays are always unambiguous.
            for row in rows:
                row["embedding"] = np.array(row["embedding"], dtype=np.float32)
            table.add(rows)
            logger.info("Upserted %d chunks (doc_id=%s)", len(chunks), chunks[0].doc_id)
            try:
                table.create_fts_index("content", replace=True)
                logger.debug("FTS index rebuilt on 'content'")
            except Exception as fts_err:
                logger.warning(
                    "FTS index rebuild failed (hybrid search degraded): %s", fts_err
                )
        except Exception as e:
            raise StoreError(f"Failed to upsert {len(chunks)} chunks") from e

    def find_existing(self, source: str, library: str) -> tuple[str | None, str | None]:
        """Look up an existing document by (source, library) dedup key.

        Args:
            source: File path or URL string.
            library: Library name.

        Returns:
            Tuple of (doc_id, content_hash) if found, else (None, None).

        Raises:
            StoreError: If the query fails.
        """
        try:
            table = self._table()
            # Escape single quotes in source to prevent injection
            safe_source = source.replace("'", "''")
            safe_library = library.replace("'", "''")
            results = (
                table.search()
                .where(f"source = '{safe_source}' AND library = '{safe_library}'")
                .limit(1)
                .to_list()
            )
            if results:
                row = results[0]
                return row["doc_id"], row["content_hash"]
            return None, None
        except Exception as e:
            raise StoreError(f"find_existing failed for source={source!r}") from e

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: The document UUID to delete.

        Returns:
            Number of rows deleted.

        Raises:
            StoreError: If the delete fails.
        """
        try:
            table = self._table()
            safe_id = doc_id.replace("'", "''")
            before = table.count_rows()
            table.delete(f"doc_id = '{safe_id}'")
            after = table.count_rows()
            deleted = before - after
            logger.info("Deleted %d chunks for doc_id=%s", deleted, doc_id)
            try:
                table.create_fts_index("content", replace=True)
                logger.debug("FTS index rebuilt after delete of doc_id=%s", doc_id)
            except Exception as fts_err:
                logger.warning(
                    "FTS index rebuild failed after delete (hybrid search may return stale results): %s",
                    fts_err,
                )
            return deleted
        except Exception as e:
            raise StoreError(f"Failed to delete document {doc_id!r}") from e

    def _vector_search(
        self,
        table: lancedb.table.Table,
        embedding: list[float],
        where: str | None,
        top_k: int,
    ) -> list[dict]:
        """Run a pure vector (ANN) search against an open table.

        Args:
            table: Open LanceDB table.
            embedding: Query vector.
            where: Optional SQL WHERE clause string.
            top_k: Maximum number of results to return.

        Returns:
            Raw row dicts from LanceDB.
        """
        q = table.search(np.array(embedding, dtype=np.float32))
        if where is not None:
            q = q.where(where)
        return q.refine_factor(settings.search_refine_factor).limit(top_k).to_list()

    def search(
        self,
        embedding: list[float],
        query_text: str,
        top_k: int,
        library: str | None,
        filter: dict | None,  # noqa: A002
    ) -> list[tuple["ChunkRecord", float]]:
        """Hybrid (BM25 + vector) search over stored chunks.

        Attempts hybrid search when hybrid_search_enabled is True; falls back to
        vector-only search if the FTS index is absent or the hybrid query fails.

        Args:
            embedding: Query vector of shape (embedding_dimension,).
            query_text: Raw query string for the BM25 leg of hybrid search.
            top_k: Maximum number of results to return.
            library: Restrict search to this library if provided.
            filter: Optional equality filters, e.g. ``{"file_type": "pdf"}``.
                Keys must be valid column names; string and int values supported.

        Returns:
            List of (ChunkRecord, similarity_score) tuples sorted by relevance
            descending. similarity_score is in [0.0, 1.0].

        Raises:
            StoreError: If the search fails or a filter key is invalid.
        """
        try:
            table = self._table()
            where = _build_where_clause(library, filter)

            rows: list[dict] = []
            if settings.hybrid_search_enabled:
                try:
                    query = table.search(query_text, query_type="hybrid").vector(
                        np.array(embedding, dtype=np.float32)
                    )
                    if where is not None:
                        query = query.where(where)
                    rows = (
                        query.refine_factor(settings.search_refine_factor)
                        .limit(top_k)
                        .to_list()
                    )
                except Exception as hybrid_err:
                    logger.warning(
                        "Hybrid search fell back to vector-only: %s", hybrid_err
                    )
                    rows = self._vector_search(table, embedding, where, top_k)
            else:
                rows = self._vector_search(table, embedding, where, top_k)

            results = []
            for row in rows:
                distance = float(row.get("_distance") or 0.0)
                similarity = max(0.0, 1.0 - distance)
                record = ChunkRecord(**{k: v for k, v in row.items() if k != "_distance"})
                results.append((record, similarity))
            return results
        except StoreError:
            raise
        except Exception as e:
            raise StoreError("Search failed") from e

    def get_document(self, doc_id: str) -> list[ChunkRecord]:
        """Return all chunks for a document, ordered by chunk_index.

        Args:
            doc_id: The document UUID.

        Returns:
            List of ChunkRecord objects sorted by chunk_index.

        Raises:
            StoreError: If the query fails.
        """
        try:
            table = self._table()
            safe_id = doc_id.replace("'", "''")
            rows = table.search().where(f"doc_id = '{safe_id}'").to_list()
            records = [
                ChunkRecord(**{k: v for k, v in row.items() if k != "_distance"})
                for row in rows
            ]
            records.sort(key=lambda r: r.chunk_index)
            return records
        except Exception as e:
            raise StoreError(f"get_document failed for doc_id={doc_id!r}") from e

    def list_documents(
        self,
        library: str | None,
        limit: int,
        offset: int,
    ) -> list[dict]:
        """List indexed documents with metadata, one entry per doc_id.

        Args:
            library: Filter by library name if provided.
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.

        Returns:
            List of dicts with doc-level metadata.

        Raises:
            StoreError: If the query fails.
        """
        try:
            table = self._table()
            q = table.search()
            if library is not None:
                safe_lib = library.replace("'", "''")
                q = q.where(f"library = '{safe_lib}'")
            rows = q.to_list()

            # Group by doc_id — keep first occurrence for metadata
            seen: dict[str, dict] = {}
            for row in rows:
                did = row["doc_id"]
                if did not in seen:
                    seen[did] = {
                        "doc_id": did,
                        "source": row["source"],
                        "title": row["title"],
                        "library": row["library"],
                        "content_hash": row["content_hash"],
                        "created_at": row["created_at"],
                        "metadata": json.loads(row["metadata"]),
                        "chunk_count": 0,
                    }
                seen[did]["chunk_count"] += 1

            docs = list(seen.values())
            docs.sort(key=lambda d: d["created_at"], reverse=True)
            return docs[offset : offset + limit]
        except Exception as e:
            raise StoreError("list_documents failed") from e

    def list_libraries(self) -> list[dict]:
        """List all libraries with document and chunk counts.

        Returns:
            List of dicts with keys: library, document_count, chunk_count.

        Raises:
            StoreError: If the query fails.
        """
        try:
            table = self._table()
            rows = table.search().to_list()

            libs: dict[str, dict] = {}
            for row in rows:
                lib = row["library"]
                if lib not in libs:
                    libs[lib] = {
                        "library": lib,
                        "document_count": 0,
                        "chunk_count": 0,
                        "_docs": set(),
                    }
                libs[lib]["chunk_count"] += 1
                libs[lib]["_docs"].add(row["doc_id"])

            return sorted(
                [
                    {
                        "library": lib_data["library"],
                        "document_count": len(lib_data["_docs"]),
                        "chunk_count": lib_data["chunk_count"],
                    }
                    for lib_data in libs.values()
                ],
                key=lambda d: d["library"],
            )
        except Exception as e:
            raise StoreError("list_libraries failed") from e
