"""Unit tests for the retriever node (LanceDB-backed)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import RetrievedChunk, create_initial_state


def _mock_record(source="TS38.321.docx", section="5.4 HARQ Entity", content="HARQ content."):
    """Build a minimal mock ChunkRecord."""
    r = MagicMock()
    r.id = "chunk-1"
    r.doc_id = "doc-1"
    r.source = source
    r.title = "TS 38.321 MAC"
    r.content = content
    r.chunk_index = 0
    r.file_type = "docx"
    r.metadata = json.dumps({"section_header": section})
    return r


@pytest.mark.unit
def test_retriever_populates_retrieved_chunks():
    from specagent.nodes.retriever import retriever_node

    state = create_initial_state("What is HARQ in NR?")
    record = _mock_record()

    with (
        patch("specagent.nodes.retriever.get_store") as mock_store_fn,
        patch("specagent.nodes.retriever.get_embedder") as mock_emb_fn,
    ):
        mock_store_fn.return_value.search.return_value = [(record, 0.9)]
        mock_emb = MagicMock()
        mock_emb.embed.return_value = iter([[0.1] * 768])
        mock_emb_fn.return_value = mock_emb

        result = retriever_node(state)

    chunks = result["retrieved_chunks"]
    assert len(chunks) == 1
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].content == "HARQ content."
    assert chunks[0].spec_id == "TS38.321"
    assert chunks[0].section == "5.4 HARQ Entity"
    assert chunks[0].source == "TS38.321.docx"


@pytest.mark.unit
def test_retriever_uses_rewritten_question():
    from specagent.nodes.retriever import retriever_node

    state = create_initial_state("HARQ?")
    state["rewritten_question"] = "What is the HARQ process in 5G NR?"

    with (
        patch("specagent.nodes.retriever.get_store") as mock_store_fn,
        patch("specagent.nodes.retriever.get_embedder") as mock_emb_fn,
    ):
        mock_store_fn.return_value.search.return_value = []
        mock_emb = MagicMock()
        mock_emb.embed.return_value = iter([[0.1] * 768])
        mock_emb_fn.return_value = mock_emb

        retriever_node(state)

    call_text = list(mock_emb.embed.call_args[0][0])
    assert "5G NR" in call_text[0]


@pytest.mark.unit
def test_retriever_handles_store_exception_gracefully():
    from specagent.nodes.retriever import retriever_node

    state = create_initial_state("test")

    with (
        patch("specagent.nodes.retriever.get_store") as mock_store_fn,
        patch("specagent.nodes.retriever.get_embedder") as mock_emb_fn,
    ):
        mock_store_fn.side_effect = Exception("LanceDB not found")
        mock_emb_fn.return_value = MagicMock()

        result = retriever_node(state)

    assert result.get("error") is not None
    assert result.get("retrieved_chunks", []) == []


@pytest.mark.unit
def test_retriever_adds_query_prefix_to_embed():
    from specagent.nodes.retriever import _QUERY_PREFIX, retriever_node

    state = create_initial_state("test query")

    with (
        patch("specagent.nodes.retriever.get_store") as mock_store_fn,
        patch("specagent.nodes.retriever.get_embedder") as mock_emb_fn,
    ):
        mock_store_fn.return_value.search.return_value = []
        mock_emb = MagicMock()
        mock_emb.embed.return_value = iter([[0.1] * 768])
        mock_emb_fn.return_value = mock_emb

        retriever_node(state)

    call_text = list(mock_emb.embed.call_args[0][0])
    assert call_text[0].startswith(_QUERY_PREFIX)


@pytest.mark.unit
def test_retriever_empty_query_sets_error():
    from specagent.nodes.retriever import retriever_node

    state = {"question": "", "rewritten_question": None}
    result = retriever_node(state)
    assert "error" in result
    assert result["retrieved_chunks"] == []


@pytest.mark.unit
def test_retriever_bad_json_metadata_uses_empty_section():
    from specagent.nodes.retriever import retriever_node

    rec = MagicMock()
    rec.id = "c1"
    rec.doc_id = "d1"
    rec.source = "TS38.321.docx"
    rec.title = "T"
    rec.content = "content"
    rec.chunk_index = 0
    rec.file_type = "docx"
    rec.metadata = "{invalid"
    emb = MagicMock()
    emb.embed.return_value = iter([[0.1] * 768])
    store = MagicMock()
    store.search.return_value = [(rec, 0.8)]
    with (
        patch("specagent.nodes.retriever.get_embedder", return_value=emb),
        patch("specagent.nodes.retriever.get_store", return_value=store),
    ):
        result = retriever_node({"question": "test?", "rewritten_question": None})
    assert result["retrieved_chunks"][0].section == ""
