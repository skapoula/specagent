"""
Pytest configuration and shared fixtures.

Provides common fixtures for:
    - Configuration with test values
    - Mock LLM API responses
    - Sample document chunks
    - Temporary directories for indexes
    - Docx ZIP helpers for structural/error-path tests only
"""

import io
import json
import struct
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from specagent.retrieval.resources import clear_resource_cache

# ---------------------------------------------------------------------------
# Real test data — actual 3GPP .docx files from the project data directory
# ---------------------------------------------------------------------------

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# The three real .docx files available for tests.
# 38108-i40.docx is the smallest (857 KB) and is the default for most tests.
DOCX_SMALL = RAW_DATA_DIR / "38108-i40.docx"  # 857 KB, 98 images (87 wmf, 8 emf, 3 png)
DOCX_MEDIUM = RAW_DATA_DIR / "38104-ic0.docx"  # 3.3 MB, 164 images
DOCX_LARGE = RAW_DATA_DIR / "23502-j70.docx"  # 16.9 MB, 288 images

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Provide test settings without requiring .env file.

    Clears the get_settings lru_cache before and after the test so that any
    code calling get_settings() during the test receives a fresh instance.
    Settings are constructed via kwargs (shell env vars are excluded from the
    settings source chain, so patch.dict(os.environ) has no effect here).
    """
    from specagent.config import Settings, get_settings

    get_settings.cache_clear()
    new_settings = Settings(
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        enable_tracing=False,
    )
    # Patch the module-level singletons imported at load time in each module
    with (
        patch("specagent.config.settings", new_settings),
        patch("specagent.graph.workflow.settings", new_settings),
    ):
        yield new_settings
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def reset_resource_cache():
    """
    Automatically clear resource cache before each test.

    Ensures tests don't interfere with each other by
    sharing cached index/embedder instances.
    """
    clear_resource_cache()
    yield
    clear_resource_cache()


@pytest.fixture(autouse=True)
def reset_llm_rate_limiter_singleton():
    """Reset the Groq LLM rate limiter singleton before each test.

    Prevents test cross-contamination when settings are patched during singleton
    initialisation (e.g. patch("specagent.config.settings") in factory tests).
    """
    import specagent.llm.groq_rate_limiter as _mod

    _mod._llm_rate_limiter = None
    yield
    _mod._llm_rate_limiter = None


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_chunks():
    """Sample ChunkRecord objects for testing."""
    import json
    import uuid as _uuid

    from specagent.retrieval.store import ChunkRecord

    data = [
        (
            "TS38.321.docx",
            "5.4 HARQ Entity",
            "The maximum number of HARQ processes for NR is 16 for both FDD and TDD.",
        ),
        (
            "TS38.101-1.docx",
            "5.5A Carrier Aggregation",
            "The UE shall support a maximum of 16 component carriers for CA.",
        ),
        (
            "TS38.331.docx",
            "5.3.7 RRC Connection Re-establishment",
            "RRC connection re-establishment is initiated when T311 expires.",
        ),
        (
            "TS38.211.docx",
            "7.3 Physical Downlink Control Channel",
            "The PDCCH carries downlink control information (DCI).",
        ),
        (
            "TS38.401.docx",
            "6.1 F1 Interface",
            "The gNB-DU and gNB-CU are connected via the F1 interface.",
        ),
    ]
    return [
        ChunkRecord(
            id=str(_uuid.uuid4()),
            doc_id=str(_uuid.uuid4()),
            library="3gpp-specs",
            source=src,
            content_hash=f"hash{i}",
            title=src.replace(".docx", ""),
            content=content,
            embedding=[0.0] * 768,
            chunk_index=0,
            created_at="2026-03-01T00:00:00Z",
            metadata=json.dumps({"section_header": section}),
            file_type="docx",
            last_modified="2026-03-01T00:00:00Z",
            page=0,
        )
        for i, (src, section, content) in enumerate(data)
    ]


@pytest.fixture
def sample_embeddings():
    """Sample 768d embeddings for testing."""
    rng = np.random.default_rng(42)
    embeddings = rng.random((5, 768)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_question():
    """Provide a sample 3GPP-related question."""
    return "What is the maximum number of HARQ processes in NR?"


@pytest.fixture
def sample_off_topic_question():
    """Provide a sample off-topic question."""
    return "What is the best recipe for chocolate cake?"


# =============================================================================
# Mock API Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_response():
    """Mock embedding API response."""

    def _mock_response(texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(hash(tuple(texts)) % 2**32)
        embeddings = rng.random((len(texts), 768)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        return normalized.tolist()

    return _mock_response


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""

    def _mock_response(prompt: str) -> str:
        # Return structured responses based on prompt content
        if "router" in prompt.lower():
            return '{"route": "retrieve", "reasoning": "This is a 3GPP question"}'
        elif "grader" in prompt.lower():
            return '{"relevant": "yes", "confidence": 0.85}'
        elif "rewriter" in prompt.lower():
            return "What is the maximum number of HARQ processes in 5G NR Release 18?"
        elif "hallucination" in prompt.lower():
            return '{"grounded": "yes", "ungrounded_claims": []}'
        else:
            return "The maximum number of HARQ processes in NR is 16. [TS 38.321 §5.4]"

    return _mock_response


# =============================================================================
# Directory Fixtures
# =============================================================================


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for vector index artifacts."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def tmp_data_dir(tmp_path: Path, sample_markdown_files) -> Path:
    """Provide temporary directory with sample markdown files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for filename, content in sample_markdown_files.items():
        (data_dir / filename).write_text(content)

    return data_dir


@pytest.fixture
def sample_markdown_files():
    """Provide sample 3GPP markdown content."""
    return {
        "TS38.321.md": """# TS 38.321 - Medium Access Control (MAC)

## 5.4 HARQ Entity

### 5.4.1 HARQ Processes

The UE shall support a maximum of 16 HARQ processes per cell for FDD and TDD.

Each HARQ process handles one transport block at a time.
""",
        "TS38.331.md": """# TS 38.331 - Radio Resource Control (RRC)

## 5.3 RRC Connection Control

### 5.3.7 RRC Connection Re-establishment

The RRC connection re-establishment procedure is used to re-establish RRC 
connection after radio link failure.

Timer T311 is started upon detection of radio link failure.
""",
    }


# =============================================================================
# Graph State Fixtures
# =============================================================================


@pytest.fixture
def initial_graph_state(sample_question):
    """Provide initial graph state for testing."""
    from specagent.graph.state import create_initial_state

    return create_initial_state(sample_question)


@pytest.fixture
def state_after_retrieval(initial_graph_state):
    """Graph state after retrieval, using the unified RetrievedChunk schema."""
    from specagent.graph.state import RetrievedChunk

    state = initial_graph_state.copy()
    state["route_decision"] = "retrieve"
    state["retrieved_chunks"] = [
        RetrievedChunk(
            content="The maximum number of HARQ processes for NR is 16.",
            chunk_id="TS38.321.docx:0",
            doc_id="doc-uuid-1",
            source="TS38.321.docx",
            title="TS 38.321 MAC Protocol",
            chunk_index=0,
            file_type="docx",
            spec_id="TS38.321",
            section="5.4 HARQ Entity",
            similarity_score=0.85,
        ),
        RetrievedChunk(
            content="The UE shall support a maximum of 16 component carriers for CA.",
            chunk_id="TS38.101-1.docx:0",
            doc_id="doc-uuid-2",
            source="TS38.101-1.docx",
            title="TS 38.101-1",
            chunk_index=0,
            file_type="docx",
            spec_id="TS38.101",
            section="5.5A Carrier Aggregation",
            similarity_score=0.75,
        ),
        RetrievedChunk(
            content="RRC re-establishment is initiated when T311 expires.",
            chunk_id="TS38.331.docx:0",
            doc_id="doc-uuid-3",
            source="TS38.331.docx",
            title="TS 38.331 RRC",
            chunk_index=0,
            file_type="docx",
            spec_id="TS38.331",
            section="5.3.7 RRC Connection Re-establishment",
            similarity_score=0.65,
        ),
    ]
    return state


# =============================================================================
# Benchmark Fixtures
# =============================================================================


@pytest.fixture
def sample_benchmark_questions():
    """Provide sample benchmark questions for testing."""
    return [
        {
            "id": "q1",
            "question": "What is the maximum number of HARQ processes in NR?",
            "answer": "16",
            "difficulty": "easy",
        },
        {
            "id": "q2",
            "question": "What is the maximum number of component carriers for CA?",
            "answer": "16",
            "difficulty": "easy",
        },
        {
            "id": "q3",
            "question": "What timer is started upon radio link failure detection?",
            "answer": "T311",
            "difficulty": "medium",
        },
    ]


@pytest.fixture
def benchmark_file(tmp_path: Path, sample_benchmark_questions) -> Path:
    """Create temporary benchmark file."""
    benchmark_path = tmp_path / "benchmark.json"
    with benchmark_path.open("w") as f:
        json.dump(sample_benchmark_questions, f)
    return benchmark_path


# =============================================================================
# Docx / OCR fixtures
# =============================================================================


def _make_png_bytes(n_bytes: int | None = None) -> bytes:
    """Return a valid minimal PNG (1×1 red pixel).

    If n_bytes is given the PNG is padded with a tEXt chunk so that
    ``len(result) >= n_bytes``.  Used to simulate large/small images.
    """

    def chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00\xff\x00\x00"  # filter + R G B
    compressed = zlib.compress(raw_row)
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")
    base = signature + ihdr + idat + iend

    if n_bytes is not None and len(base) < n_bytes:
        padding = b"x" * (n_bytes - len(base))
        text_chunk = chunk(b"tEXt", b"Comment\x00" + padding)
        base = signature + ihdr + idat + text_chunk + iend

    return base


def make_docx_zip(images: list[tuple[str, bytes]] | None = None) -> bytes:
    """Build a minimal in-memory .docx ZIP.

    Args:
        images: List of (media_filename, image_bytes) pairs in relationship order.

    Returns:
        Raw bytes of a valid .docx ZIP archive.
    """
    _IMAGE_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p/></w:body></w:document>",
        )
        if images:
            rels = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
            ]
            for idx, (media_filename, img_bytes) in enumerate(images, start=1):
                rels.append(
                    f'  <Relationship Id="rId{idx}" Type="{_IMAGE_NS}"'
                    f' Target="media/{media_filename}"/>'
                )
                zf.writestr(f"word/media/{media_filename}", img_bytes)
            rels.append("</Relationships>")
            zf.writestr("word/_rels/document.xml.rels", "\n".join(rels))
        else:
            zf.writestr(
                "word/_rels/document.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>',
            )
    return buf.getvalue()


def make_docx_zip_with_caption(
    image_filename: str,
    image_bytes: bytes,
    caption_text: str,
) -> bytes:
    """Build a minimal .docx ZIP with one image followed by a Caption paragraph.

    word/document.xml contains a paragraph with a DrawingML a:blip (r:embed="rId1")
    followed immediately by a Caption-style paragraph containing caption_text.

    Args:
        image_filename: Filename for the image in word/media/.
        image_bytes: Raw bytes of the image file.
        caption_text: Text content of the Caption-style paragraph.

    Returns:
        Raw bytes of a valid .docx ZIP archive.
    """
    _IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    _PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
    _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    _R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    _A = "http://schemas.openxmlformats.org/drawingml/2006/main"

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{_W}" xmlns:r="{_R}">'
        "<w:body>"
        "<w:p>"
        "<w:r>"
        f'<w:drawing><a:blip xmlns:a="{_A}" r:embed="rId1"/></w:drawing>'
        "</w:r>"
        "</w:p>"
        "<w:p>"
        "<w:pPr>"
        '<w:pStyle w:val="Caption"/>'
        "</w:pPr>"
        "<w:r>"
        f"<w:t>{caption_text}</w:t>"
        "</w:r>"
        "</w:p>"
        "</w:body>"
        "</w:document>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_PKG_REL_NS}">'
        f'<Relationship Id="rId1" Type="{_IMAGE_REL_TYPE}"'
        f' Target="media/{image_filename}"/>'
        "</Relationships>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", rels_xml)
        zf.writestr(f"word/media/{image_filename}", image_bytes)
    return buf.getvalue()


@pytest.fixture
def small_png() -> bytes:
    """PNG well below the 10 KB default min threshold (~100 bytes).

    Still synthetic — used only for size-threshold edge-case tests where
    a real sub-threshold PNG is required as bytes (not a file path).
    """
    return _make_png_bytes()


@pytest.fixture
def large_png() -> bytes:
    """Real PNG bytes from the smallest 3GPP docx (image4.png, 33 KB).

    Above the 10 KB vision_min_image_bytes threshold.
    """
    import zipfile

    with zipfile.ZipFile(DOCX_SMALL) as zf:
        return zf.read("word/media/image4.png")


@pytest.fixture
def docx_no_images(tmp_path: Path) -> Path:
    """A .docx file containing no embedded images (synthetic minimal ZIP).

    Used only for structural/error-path tests that require a docx with
    zero images. Real 3GPP files all contain images, so this remains
    synthetic for those specific tests.
    """
    p = tmp_path / "no_images.docx"
    p.write_bytes(make_docx_zip())
    return p


@pytest.fixture
def docx_one_image() -> Path:
    """Real 3GPP .docx file (38108-i40.docx) — contains 98 embedded images."""
    return DOCX_SMALL


@pytest.fixture
def docx_three_images() -> Path:
    """Real 3GPP .docx file (38108-i40.docx) — contains 98 embedded images.

    Tests using this fixture verify that multiple images are processed;
    assertions should use >= 3 rather than exactly 3.
    """
    return DOCX_SMALL
