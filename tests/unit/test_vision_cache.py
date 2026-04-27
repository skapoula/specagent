"""Unit tests for VisionCache — Issue 8 (TDD)."""

from __future__ import annotations

import pytest


def _make_result(placeholder: str = "image0.png") -> ImageAnalysisResult:
    from specagent.retrieval.groq_vision_client import ImageAnalysisResult

    return ImageAnalysisResult(
        placeholder_name=placeholder,
        markdown_content="```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ok\n```",
        image_type="call_flow",
        prose_fallback="A call flow.",
    )


@pytest.mark.unit
class TestVisionCache:
    def test_get_returns_none_for_unseen_image(self, tmp_path) -> None:
        """get() returns None when the image hash has not been cached."""
        from specagent.retrieval.vision_cache import VisionCache

        cache = VisionCache(tmp_path / "cache.json")
        assert cache.get(b"unknown image bytes") is None

    def test_put_then_get_round_trips_result(self, tmp_path) -> None:
        """put() stores a result; get() returns it for the same bytes."""
        from specagent.retrieval.vision_cache import VisionCache

        cache = VisionCache(tmp_path / "cache.json")
        image_bytes = b"fake image content"
        result = _make_result()
        cache.put(image_bytes, result)

        retrieved = cache.get(image_bytes)
        assert retrieved is not None
        assert retrieved.placeholder_name == result.placeholder_name
        assert retrieved.image_type == result.image_type
        assert retrieved.markdown_content == result.markdown_content

    def test_cache_persists_across_instances(self, tmp_path) -> None:
        """A new VisionCache instance reads back entries written by a prior instance."""
        from specagent.retrieval.vision_cache import VisionCache

        cache_path = tmp_path / "cache.json"
        image_bytes = b"persistent image"
        result = _make_result("image1.png")

        VisionCache(cache_path).put(image_bytes, result)

        loaded = VisionCache(cache_path).get(image_bytes)
        assert loaded is not None
        assert loaded.placeholder_name == "image1.png"

    def test_different_image_bytes_get_different_entries(self, tmp_path) -> None:
        """Two different byte strings map to independent cache entries."""
        from specagent.retrieval.vision_cache import VisionCache

        cache = VisionCache(tmp_path / "cache.json")
        cache.put(b"image_a", _make_result("a.png"))
        cache.put(b"image_b", _make_result("b.png"))

        assert cache.get(b"image_a").placeholder_name == "a.png"
        assert cache.get(b"image_b").placeholder_name == "b.png"

    def test_corrupted_cache_file_starts_fresh(self, tmp_path) -> None:
        """A corrupted cache JSON file is silently discarded — get() returns None."""
        from specagent.retrieval.vision_cache import VisionCache

        cache_path = tmp_path / "cache.json"
        cache_path.write_text("NOT VALID JSON {{{{")

        cache = VisionCache(cache_path)
        assert cache.get(b"anything") is None

    def test_put_writes_cache_to_disk(self, tmp_path) -> None:
        """put() persists the cache file to disk immediately."""
        from specagent.retrieval.vision_cache import VisionCache

        cache_path = tmp_path / "cache.json"
        cache = VisionCache(cache_path)
        cache.put(b"bytes", _make_result())

        assert cache_path.exists()
