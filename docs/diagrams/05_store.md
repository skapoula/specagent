# LanceDB Store State Machine

The `Store` class (`store.py`) wraps LanceDB read/write operations. It uses double-checked locking on `_table_lock` for the one-time table open (open-or-create with schema validation and optional migration), and `_write_lock` to serialise concurrent `upsert_chunks()` calls from the `asyncio.to_thread` pool. The FTS index is rebuilt inside `upsert_chunks()` by default but can be deferred during bulk ingest. Search returns early if the table is empty (`_is_empty` flag).

```mermaid
stateDiagram-v2
    [*] --> Unconnected

    Unconnected --> OpeningTable : first call to _table()

    state OpeningTable {
        [*] --> CheckCachedTable
        CheckCachedTable --> ReturnCached : _cached_table is not None
        CheckCachedTable --> AcquireTableLock : _cached_table is None
        AcquireTableLock --> DoubleCheckTable
        DoubleCheckTable --> ReturnCached : set by another thread
        DoubleCheckTable --> ConnectingLanceDB : still None — open/create table

        state ConnectingLanceDB {
            [*] --> ResolveURI
            ResolveURI --> ConnectDB
            ConnectDB --> TableExists : table found in db.list_tables()
            ConnectDB --> CreateTable : table not found

            TableExists --> ValidateDimension
            ValidateDimension --> DimensionMismatch : stored_dim != settings.embedding_dimension
            ValidateDimension --> MigrateTable : dimensions match

            DimensionMismatch --> [*] : raises StoreError

            MigrateTable --> EnsureScalarIndexes : adds missing columns (file_type, last_modified, page, release)

            CreateTable --> EnsureScalarIndexes : created with explicit PyArrow schema
            EnsureScalarIndexes --> [*]
        }

        ConnectingLanceDB --> ReturnCached : _cached_table written last
        ReturnCached --> [*]
    }

    OpeningTable --> TableReady : _cached_table set

    state TableReady {
        [*] --> Idle

        Idle --> Upserting : upsert_chunks(chunks) called
        Idle --> Searching : search() called
        Idle --> FindingExisting : find_existing() called
        Idle --> DeletingDocument : delete_document(doc_id) called
        Idle --> DeletingLibrary : delete_library(library) called
        Idle --> RebuildingFTS : rebuild_fts_index() called

        state Upserting {
            [*] --> CheckChunksEmpty
            CheckChunksEmpty --> [*] : chunks=[] → return immediately
            CheckChunksEmpty --> AcquireWriteLock

            AcquireWriteLock --> BuildRows : serialize ChunkRecord → dicts + np arrays
            BuildRows --> DeleteExisting : delete by doc_id for each unique doc_id in batch
            DeleteExisting --> AddRows : table.add(rows)
            AddRows --> UpdateIsEmpty : _is_empty = False
            UpdateIsEmpty --> CreateScalarIndexes : was_empty=True → create indexes
            UpdateIsEmpty --> RebuildFTSConditional : was_empty=False

            CreateScalarIndexes --> RebuildFTSConditional
            RebuildFTSConditional --> FTSRebuilt : rebuild_fts=True → table.create_fts_index()
            RebuildFTSConditional --> FTSSkipped : rebuild_fts=False (bulk ingest deferral)

            FTSRebuilt --> [*]
            FTSSkipped --> [*]

            BuildRows --> UpsertFailed : serialisation raises
            AddRows --> UpsertFailed : table.add() raises
            UpsertFailed --> [*] : raises StoreError
        }

        state Searching {
            [*] --> CheckIsEmpty
            CheckIsEmpty --> ReturnEmpty : _is_empty == True
            CheckIsEmpty --> BuildWhereClause : table has rows

            ReturnEmpty --> [*] : returns []

            BuildWhereClause --> InvalidFilterKey : key fails _SAFE_KEY regex
            BuildWhereClause --> RunHybridSearch : WHERE clause built (or None)

            InvalidFilterKey --> [*] : raises StoreError

            RunHybridSearch --> SearchDone : hybrid BM25+vector query executed
            RunHybridSearch --> SearchFailed : Exception

            SearchDone --> [*] : returns list[(ChunkRecord, similarity_score)]
            SearchFailed --> [*] : raises StoreError
        }

        Upserting --> Idle
        Searching --> Idle
        FindingExisting --> Idle
        DeletingDocument --> Idle
        DeletingLibrary --> Idle
        RebuildingFTS --> Idle
    }
```
