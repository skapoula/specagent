"""Disk-backed cache for Groq vision API results, keyed by image content hash."""

import hashlib
import json
import logging
from pathlib import Path

from specagent.retrieval.groq_vision_client import ImageAnalysisResult

logger = logging.getLogger(__name__)


class VisionCache:
    """JSON file cache mapping image SHA-256 → ImageAnalysisResult.

    Thread-safety: single-process only (asyncio-safe via to_thread callers).
    Corrupted or missing cache files are silently reset.
    """

    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Vision cache corrupted at %s; starting fresh", self._path)
                self._data = {}

    def get(self, image_bytes: bytes) -> ImageAnalysisResult | None:
        """Return a cached result for *image_bytes*, or ``None`` on cache miss."""
        key = hashlib.sha256(image_bytes).hexdigest()
        raw = self._data.get(key)
        return ImageAnalysisResult.model_validate(raw) if raw is not None else None

    def put(self, image_bytes: bytes, result: ImageAnalysisResult) -> None:
        """Store *result* under the SHA-256 hash of *image_bytes* and flush to disk."""
        key = hashlib.sha256(image_bytes).hexdigest()
        self._data[key] = result.model_dump()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write vision cache: %s", exc)
