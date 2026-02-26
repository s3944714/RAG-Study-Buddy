import pytest
from unittest.mock import AsyncMock, MagicMock

from app.retrieval.retriever import RetrievedChunk, retrieve
from app.services.embeddings_client import DummyEmbeddingsClient
from app.services.vector_store import (
    InMemoryVectorStoreClient,
    QueryResult,
    VectorDocument,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> list[float]:
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


async def _seed(
    store: InMemoryVectorStoreClient,
    workspace_id: int,
    docs: list[dict],
) -> None:
    """Upsert a list of dicts into the in-memory store.

    Each dict must have: embedding, content, document_id, page, heading, filename.
    """
    collection = f"workspace_{workspace_id}"
    vector_docs = [
        VectorDocument(
            embedding=d["embedding"],
            content=d["content"],
            metadata={
                "document_id": d["document_id"],
                "page": d["page"],
                "heading": d.get("heading", ""),
                "filename": d.get("filename", ""),
                "chunk_index": d.get("chunk_index", 0),
            },
        )
        for d in docs
    ]
    await store.upsert(collection, vector_docs)


WORKSPACE = 1
DIM = 8


def _embeddings() -> DummyEmbeddingsClient:
    return DummyEmbeddingsClient(dimensions=DIM)


def _store() -> InMemoryVectorStoreClient:
    return InMemoryVectorStoreClient()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_returns_list_of_retrieved_chunks() -> None:
    emb = _embeddings()
    store = _store()
    vec = await emb.embed_one("neural networks")
    await _seed(store, WORKSPACE, [
        {"embedding": vec, "content": "Neural networks are ...", "document_id": 1,
         "page": 1, "heading": "Intro", "filename": "ai.pdf"},
    ])

    results = await retrieve(
        WORKSPACE, "neural networks",
        embeddings_client=emb, vector_store=store,
    )
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)


@pytest.mark.asyncio
async def test_retrieve_chunk_fields_are_populated() -> None:
    emb = _embeddings()
    store = _store()
    vec = await emb.embed_one("backpropagation")
    await _seed(store, WORKSPACE, [
        {"embedding": vec, "content": "Backprop computes gradients",
         "document_id": 7, "page": 3, "heading": "Training", "filename": "dl.pdf"},
    ])

    results = await retrieve(
        WORKSPACE, "backpropagation",
        embeddings_client=emb, vector_store=store,
    )
    chunk = results[0]
    assert chunk.text == "Backprop computes gradients"
    assert chunk.doc_id == 7
    assert chunk.page == 3
    assert chunk.heading == "Training"
    assert chunk.filename == "dl.pdf"
    assert isinstance(chunk.score, float)
    assert isinstance(chunk.embedding_id, str)


# ---------------------------------------------------------------------------
# Sorting + top_k
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_results_sorted_descending_by_score() -> None:
    """
    Use hand-crafted unit vectors so cosine similarities are predictable
    and guaranteed positive, making the sort assertion reliable regardless
    of the DummyEmbeddingsClient's hash outputs.

    query  = [1, 0, 0, 0, 0, 0, 0, 0]
    close  = [0.9, 0.1, ...] → high cosine with query
    far    = [0.1, 0.9, ...] → lower cosine with query

    Both scores are positive so neither is filtered by the default
    score_threshold=0.0.
    """
    emb = _embeddings()
    store = _store()

    query_vec  = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    close_vec  = _unit([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # more similar
    far_vec    = _unit([0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # less similar

    await _seed(store, WORKSPACE, [
        {"embedding": far_vec,   "content": "Far text",
         "document_id": 1, "page": 1, "heading": "", "filename": "far.pdf"},
        {"embedding": close_vec, "content": "Close text",
         "document_id": 2, "page": 1, "heading": "", "filename": "close.pdf"},
    ])

    # Bypass the embed_one call — inject the query vector directly via a
    # mock embeddings client so the exact vector is controlled.
    mock_emb = MagicMock()
    mock_emb.embed_one = AsyncMock(return_value=query_vec)

    results = await retrieve(
        WORKSPACE, "attention mechanism",
        embeddings_client=mock_emb, vector_store=store, top_k=5,
    )

    assert len(results) == 2, (
        f"Expected 2 results, got {len(results)}. "
        f"Scores: {[r.score for r in results]}"
    )
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by descending score"
    assert results[0].text == "Close text"
    assert results[1].text == "Far text"


@pytest.mark.asyncio
async def test_top_k_limits_results() -> None:
    emb = _embeddings()
    store = _store()

    await _seed(store, WORKSPACE, [
        {"embedding": await emb.embed_one(f"doc {i}"), "content": f"Content {i}",
         "document_id": i, "page": 1, "heading": "", "filename": f"{i}.pdf"}
        for i in range(10)
    ])

    results = await retrieve(
        WORKSPACE, "topic",
        embeddings_client=emb, vector_store=store, top_k=3,
    )
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_top_k_one_returns_single_best() -> None:
    """
    Use explicit unit vectors so the 'best' result is unambiguous.
    query = [1, 0, ...], best = [1, 0, ...] (identical), other = [0, 1, ...]
    """
    emb = _embeddings()
    store = _store()

    query_vec = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_vec  = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    other_vec = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    await _seed(store, WORKSPACE, [
        {"embedding": best_vec,  "content": "Best content",
         "document_id": 1, "page": 1, "heading": "", "filename": "best.pdf"},
        {"embedding": other_vec, "content": "Other content",
         "document_id": 2, "page": 1, "heading": "", "filename": "other.pdf"},
    ])

    mock_emb = MagicMock()
    mock_emb.embed_one = AsyncMock(return_value=query_vec)

    results = await retrieve(
        WORKSPACE, "deep learning",
        embeddings_client=mock_emb, vector_store=store, top_k=1,
    )
    assert len(results) == 1
    assert results[0].text == "Best content"


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_empty_collection_returns_empty() -> None:
    results = await retrieve(
        WORKSPACE, "anything",
        embeddings_client=_embeddings(), vector_store=_store(),
    )
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty() -> None:
    results = await retrieve(
        WORKSPACE, "   ",
        embeddings_client=_embeddings(), vector_store=_store(),
    )
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_different_workspaces_are_isolated() -> None:
    emb = _embeddings()
    store = _store()
    vec = await emb.embed_one("shared topic")

    await _seed(store, workspace_id=1, docs=[
        {"embedding": vec, "content": "WS1 content",
         "document_id": 1, "page": 1, "heading": "", "filename": "ws1.pdf"},
    ])
    # workspace 2 is empty
    results = await retrieve(2, "shared topic", embeddings_client=emb, vector_store=store)
    assert results == []


# ---------------------------------------------------------------------------
# Score threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_threshold_filters_low_scoring_results() -> None:
    """
    Seed a vector that is orthogonal to the query (score = 0.0) and one
    that is identical (score = 1.0). With threshold=0.5 only the identical
    one should survive.
    """
    emb = _embeddings()
    store = _store()

    query_vec = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ortho_vec = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # score = 0.0
    ident_vec = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # score = 1.0

    await _seed(store, WORKSPACE, [
        {"embedding": ortho_vec, "content": "Orthogonal content",
         "document_id": 1, "page": 1, "heading": "", "filename": "orth.pdf"},
        {"embedding": ident_vec, "content": "Identical content",
         "document_id": 2, "page": 1, "heading": "", "filename": "ident.pdf"},
    ])

    mock_emb = MagicMock()
    mock_emb.embed_one = AsyncMock(return_value=query_vec)

    # Without threshold: both returned (orthogonal score == 0.0, which is not < 0.0)
    all_results = await retrieve(
        WORKSPACE, "query",
        embeddings_client=mock_emb, vector_store=store,
        score_threshold=0.0,
    )
    assert len(all_results) >= 1

    # With threshold 0.5: only the identical vector should pass
    filtered = await retrieve(
        WORKSPACE, "query",
        embeddings_client=mock_emb, vector_store=store,
        score_threshold=0.5,
    )
    assert all(r.score >= 0.5 for r in filtered)
    contents = [r.text for r in filtered]
    assert "Identical content" in contents


# ---------------------------------------------------------------------------
# doc_ids allow-list filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_doc_ids_single_filters_correctly() -> None:
    emb = _embeddings()
    store = _store()
    vec = await emb.embed_one("machine learning")

    await _seed(store, WORKSPACE, [
        {"embedding": vec, "content": "Doc 1 content",
         "document_id": 1, "page": 1, "heading": "", "filename": "d1.pdf"},
        {"embedding": vec, "content": "Doc 2 content",
         "document_id": 2, "page": 1, "heading": "", "filename": "d2.pdf"},
    ])

    results = await retrieve(
        WORKSPACE, "machine learning",
        embeddings_client=emb, vector_store=store,
        doc_ids=[1],
    )
    assert all(r.doc_id == 1 for r in results)


@pytest.mark.asyncio
async def test_doc_ids_multi_filters_correctly() -> None:
    emb = _embeddings()
    store = _store()

    await _seed(store, WORKSPACE, [
        {"embedding": await emb.embed_one(f"doc {i}"), "content": f"Doc {i} content",
         "document_id": i, "page": 1, "heading": "", "filename": f"d{i}.pdf"}
        for i in range(1, 5)  # docs 1–4
    ])

    results = await retrieve(
        WORKSPACE, "topic",
        embeddings_client=emb, vector_store=store,
        doc_ids=[1, 3], top_k=10,
    )
    returned_ids = {r.doc_id for r in results}
    assert returned_ids.issubset({1, 3})


@pytest.mark.asyncio
async def test_doc_ids_empty_list_returns_empty() -> None:
    emb = _embeddings()
    store = _store()
    vec = await emb.embed_one("topic")
    await _seed(store, WORKSPACE, [
        {"embedding": vec, "content": "Something",
         "document_id": 1, "page": 1, "heading": "", "filename": "x.pdf"},
    ])

    results = await retrieve(
        WORKSPACE, "topic",
        embeddings_client=emb, vector_store=store,
        doc_ids=[],
    )
    assert results == []


# ---------------------------------------------------------------------------
# Malformed metadata is skipped gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_metadata_skipped_gracefully() -> None:
    """A vector store result missing 'document_id' must be skipped, not crash."""
    emb = _embeddings()

    bad_result = QueryResult(
        id="bad-id",
        content="some content",
        metadata={},   # missing document_id, page, etc.
        score=0.9,
    )
    good_result = QueryResult(
        id="good-id",
        content="good content",
        metadata={
            "document_id": 42, "page": 1,
            "heading": "Intro", "filename": "ok.pdf",
        },
        score=0.8,
    )

    mock_store = MagicMock()
    mock_store.query = AsyncMock(return_value=[bad_result, good_result])

    results = await retrieve(
        WORKSPACE, "some query",
        embeddings_client=emb, vector_store=mock_store,
    )
    # Only the well-formed result should survive
    assert len(results) == 1
    assert results[0].doc_id == 42