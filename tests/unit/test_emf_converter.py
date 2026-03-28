"""Tests for the EMF/WMF to JPEG converter."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from specagent.retrieval.emf_converter import EMF_MIME_TYPES, convert_emf_to_jpeg
from specagent.retrieval.exceptions import IngestionError

_FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # minimal JPEG header


@pytest.mark.unit
class TestEmfMimeTypes:
    def test_contains_emf_variants(self):
        assert "image/x-emf" in EMF_MIME_TYPES
        assert "image/emf" in EMF_MIME_TYPES

    def test_contains_wmf_variants(self):
        assert "image/x-wmf" in EMF_MIME_TYPES
        assert "image/wmf" in EMF_MIME_TYPES

    def test_is_frozenset(self):
        assert isinstance(EMF_MIME_TYPES, frozenset)


@pytest.mark.unit
class TestConvertEmfToJpeg:
    def _make_fitz_mock(self, jpeg_bytes: bytes = _FAKE_JPEG) -> MagicMock:
        """Build a fitz module mock that returns jpeg_bytes on tobytes()."""
        pixmap = MagicMock()
        pixmap.tobytes.return_value = jpeg_bytes

        page = MagicMock()
        page.get_pixmap.return_value = pixmap

        doc = MagicMock()
        doc.__getitem__ = MagicMock(return_value=page)

        fitz_mock = MagicMock()
        fitz_mock.open.return_value = doc
        return fitz_mock

    def test_returns_jpeg_bytes(self):
        """Successful conversion returns non-empty bytes."""
        fitz_mock = self._make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": fitz_mock}):
            result = convert_emf_to_jpeg(b"fake-emf-content")
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result == _FAKE_JPEG

    def test_jpeg_header_present(self):
        """Output starts with the JPEG SOI marker \\xff\\xd8."""
        fitz_mock = self._make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": fitz_mock}):
            result = convert_emf_to_jpeg(b"fake-emf-content")
        assert result[:2] == b"\xff\xd8"

    def test_default_filetype_is_emf(self):
        """fitz.open is called with filetype='emf' by default."""
        fitz_mock = self._make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": fitz_mock}):
            convert_emf_to_jpeg(b"fake-emf-content")
        fitz_mock.open.assert_called_once_with(stream=b"fake-emf-content", filetype="emf")

    def test_wmf_filetype_passed_through(self):
        """fitz.open is called with filetype='wmf' when specified."""
        fitz_mock = self._make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": fitz_mock}):
            convert_emf_to_jpeg(b"fake-wmf-content", filetype="wmf")
        fitz_mock.open.assert_called_once_with(stream=b"fake-wmf-content", filetype="wmf")

    def test_dpi_forwarded_to_get_pixmap(self):
        """The dpi parameter is forwarded to page.get_pixmap()."""
        fitz_mock = self._make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": fitz_mock}):
            convert_emf_to_jpeg(b"fake-emf-content", dpi=300)
        doc_mock = fitz_mock.open.return_value
        page_mock = doc_mock[0]
        page_mock.get_pixmap.assert_called_once_with(dpi=300)

    def test_fitz_error_raises_ingestion_error(self):
        """A fitz exception is wrapped as IngestionError."""
        fitz_mock = MagicMock()
        fitz_mock.open.side_effect = RuntimeError("bad EMF data")
        with (
            patch.dict("sys.modules", {"fitz": fitz_mock}),
            pytest.raises(IngestionError, match="EMF rasterization failed"),
        ):
            convert_emf_to_jpeg(b"garbage")

    def test_import_error_raises_ingestion_error(self):
        """Missing PyMuPDF installation raises IngestionError."""
        original = sys.modules.pop("fitz", None)
        try:
            with (
                patch.dict("sys.modules", {"fitz": None}),
                pytest.raises(IngestionError, match="PyMuPDF is not installed"),
            ):
                convert_emf_to_jpeg(b"fake-emf-content")
        finally:
            if original is not None:
                sys.modules["fitz"] = original
