"""Tests for the EMF/WMF to JPEG converter."""

import io
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from specagent.retrieval.emf_converter import EMF_MIME_TYPES, convert_emf_to_jpeg
from specagent.retrieval.exceptions import IngestionError


def _make_png_bytes() -> bytes:
    """Return minimal 1×1 white PNG — used as the fake Inkscape output."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


_FAKE_PNG = _make_png_bytes()


def _make_completed_process(returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr="")


def _inkscape_side_effect(png_bytes: bytes = _FAKE_PNG, returncode: int = 0):
    """Return a side_effect function that writes png_bytes to the output path."""

    def fake_run(cmd, **kwargs):
        out_arg = next(a for a in cmd if a.startswith("--export-filename="))
        out_path = Path(out_arg.split("=", 1)[1])
        out_path.write_bytes(png_bytes)
        return _make_completed_process(returncode=returncode)

    return fake_run


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
    def test_returns_jpeg_bytes(self):
        """Successful conversion returns non-empty JPEG bytes."""
        with patch(
            "specagent.retrieval.emf_converter.subprocess.run",
            side_effect=_inkscape_side_effect(),
        ):
            result = convert_emf_to_jpeg(b"fake-emf-content")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_jpeg_header_present(self):
        """Output starts with the JPEG SOI marker \\xff\\xd8."""
        with patch(
            "specagent.retrieval.emf_converter.subprocess.run",
            side_effect=_inkscape_side_effect(),
        ):
            result = convert_emf_to_jpeg(b"fake-emf-content")
        assert result[:2] == b"\xff\xd8"

    def test_default_filetype_is_emf(self):
        """Inkscape is invoked with an .emf input file by default."""
        with patch(
            "specagent.retrieval.emf_converter.subprocess.run",
            side_effect=_inkscape_side_effect(),
        ) as mock_run:
            convert_emf_to_jpeg(b"fake-emf-content")
        cmd = mock_run.call_args[0][0]
        assert any("input.emf" in str(a) for a in cmd)

    def test_wmf_filetype_passed_through(self):
        """Inkscape is invoked with a .wmf input file when filetype='wmf'."""
        with patch(
            "specagent.retrieval.emf_converter.subprocess.run",
            side_effect=_inkscape_side_effect(),
        ) as mock_run:
            convert_emf_to_jpeg(b"fake-wmf-content", filetype="wmf")
        cmd = mock_run.call_args[0][0]
        assert any("input.wmf" in str(a) for a in cmd)

    def test_dpi_forwarded_to_inkscape(self):
        """The dpi parameter is forwarded as --export-dpi."""
        with patch(
            "specagent.retrieval.emf_converter.subprocess.run",
            side_effect=_inkscape_side_effect(),
        ) as mock_run:
            convert_emf_to_jpeg(b"fake-emf-content", dpi=300)
        cmd = mock_run.call_args[0][0]
        assert "--export-dpi=300" in cmd

    def test_inkscape_nonzero_exit_raises_ingestion_error(self):
        """A non-zero Inkscape exit code raises IngestionError."""
        with (
            patch(
                "specagent.retrieval.emf_converter.subprocess.run",
                return_value=_make_completed_process(returncode=1),
            ),
            pytest.raises(IngestionError, match="Inkscape exited 1"),
        ):
            convert_emf_to_jpeg(b"garbage")

    def test_inkscape_not_found_raises_ingestion_error(self):
        """Missing Inkscape binary raises IngestionError."""
        with (
            patch(
                "specagent.retrieval.emf_converter.subprocess.run",
                side_effect=FileNotFoundError("inkscape not found"),
            ),
            pytest.raises(IngestionError, match="not installed"),
        ):
            convert_emf_to_jpeg(b"fake-emf-content")

    def test_inkscape_timeout_raises_ingestion_error(self):
        """Inkscape hanging past the timeout raises IngestionError."""
        with (
            patch(
                "specagent.retrieval.emf_converter.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="inkscape", timeout=60),
            ),
            pytest.raises(IngestionError, match="timed out"),
        ):
            convert_emf_to_jpeg(b"fake-emf-content")
