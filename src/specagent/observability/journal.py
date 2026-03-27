"""Thread-safe rotating JSONL journal for RAG pipeline observability events."""

import logging
import threading
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from specagent.observability.models import LLMCallRecord, QueryEvent, RetrievalRecord

logger = logging.getLogger(__name__)


class QueryJournal:
    """Appends structured observability records to a rotating daily JSONL file."""

    def __init__(self, journal_dir: Path) -> None:
        """Create the journal directory and initialise the thread lock."""
        self._dir = journal_dir
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _current_path(self) -> Path:
        """Return the path for today's UTC journal file."""
        return self._dir / f"journal-{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"

    def write(self, record: LLMCallRecord | RetrievalRecord | QueryEvent) -> None:
        """Append a JSON record to today's journal file. Never raises."""
        with self._lock:
            try:
                path = self._current_path()
                with path.open("a") as f:
                    f.write(record.model_dump_json() + "\n")
            except OSError as exc:
                logger.warning("Failed to write journal record: %s", exc)


@lru_cache(maxsize=1)
def get_journal() -> QueryJournal:
    """Return the singleton QueryJournal instance configured from settings."""
    from specagent.config import settings  # noqa: PLC0415

    return QueryJournal(journal_dir=settings.journal_dir)
