import os
from unittest.mock import MagicMock, patch

import pytest

from app.services.vector_store import (
    InMemoryVectorStoreClient,
    VectorDocument,
    get_vector_store_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, embedding: list[float], metadata: dict | None = None) -> VectorDocument:
    return VectorDocument(embedding=embedding, content=content, metadata=metadata or {})


def _unit(direction: list[float]) -> list[float]:
    """Normalise a vector to unit length."""
    mag = sum(x * x for x in direction) ** 0.5
    return [x / mag for x in direction]


# ---------------------------------------------------------------------------
# VectorDocument
# ---------------------------------------------------------------------------

def test_vector_document_auto_assigns_id() -> None:
    doc = _doc("hello", [0.1, 0.2])
    assert isinstance(doc.id, str)
    assert len(doc.id) > 0


def test_vector_document_preserves_explicit_id() -> None:
    doc = VectorDocument(embedding=[0.1], content="x", id="my-id-123")
    assert doc.id == "my-id-123"


# ---------------------------------------------------------------------------
# InMemoryVectorStoreClient — upsert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_returns_ids() -> None:
    store = InMemoryVectorStoreClient()
    docs = [_doc("a", [1.0, 0.0]), _doc("b", [0.0, 1.0])]
    ids = await store.upsert("col1", docs)
    assert len(ids) == 2
    assert set(ids) == {docs[0].id, docs[1].id}


@pytest.mark.asyncio
async def test_upsert_is_idempotent() -> None:
    store = InMemoryVectorStoreClient()
    doc = _doc("hello", [1.0, 0.0])
    await store.upsert("col", [doc])
    doc.content = "updated"
    await store.upsert("col", [doc])

    results = await store.query("col", [1.0, 0.0], top_k=1)
    assert results[0].content == "updated"


# ---------------------------------------------------------------------------
# InMemoryVectorStoreClient — query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_returns_most_similar_first() -> None:
    store = InMemoryVectorStoreClient()
    # Three docs with clearly separated directions
    await store.upsert("col", [
        _doc("north", _unit([0.0, 1.0])),
        _doc("east",  _unit([1.0, 0.0])),
        _doc("south", _unit([0.0, -1.0])),
    ])

    # Query pointing north → "north" should be first
    results = await store.query("col", _unit([0.01, 1.0]), top_k=3)
    assert results[0].content == "north"
    assert len(results) == 3


@pytest.mark.asyncio
async def test_query_top_k_limits_results() -> None:
    store = InMemoryVectorStoreClient()
    docs = [_doc(f"doc{i}", [float(i), 0.0]) for i in range(1, 6)]
    await store.upsert("col", docs)

    results = await store.query("col", [1.0, 0.0], top_k=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_query_empty_collection_returns_empty() -> None:
    store = InMemoryVectorStoreClient()
    results = await store.query("empty-col", [1.0, 0.0], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_query_scores_are_between_minus_one_and_one() -> None:
    store = InMemoryVectorStoreClient()
    await store.upsert("col", [_doc("x", _unit([1.0, 0.5]))])
    results = await store.query("col", _unit([1.0, 0.5]), top_k=1)
    assert -1.0 <= results[0].score <= 1.0


@pytest.mark.asyncio
async def test_query_metadata_filter() -> None:
    store = InMemoryVectorStoreClient()
    await store.upsert("col", [
        _doc("doc A", _unit([1.0, 0.0]), {"doc_id": 1}),
        _doc("doc B", _unit([1.0, 0.1]), {"doc_id": 2}),
    ])

    results = await store.query("col", _unit([1.0, 0.0]), top_k=5, where={"doc_id": 1})
    assert len(results) == 1
    assert results[0].content == "doc A"


# ---------------------------------------------------------------------------
# InMemoryVectorStoreClient — delete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_removes_document() -> None:
    store = InMemoryVectorStoreClient()
    doc = _doc("to delete", [1.0, 0.0])
    await store.upsert("col", [doc])
    await store.delete("col", [doc.id])

    results = await store.query("col", [1.0, 0.0], top_k=5)
    assert all(r.id != doc.id for r in results)


@pytest.mark.asyncio
async def test_delete_nonexistent_id_is_noop() -> None:
    store = InMemoryVectorStoreClient()
    await store.delete("col", ["does-not-exist"])  # must not raise


@pytest.mark.asyncio
async def test_delete_collection() -> None:
    store = InMemoryVectorStoreClient()
    await store.upsert("col-to-drop", [_doc("x", [1.0, 0.0])])
    await store.delete_collection("col-to-drop")

    results = await store.query("col-to-drop", [1.0, 0.0], top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Collections are isolated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collections_are_isolated() -> None:
    store = InMemoryVectorStoreClient()
    await store.upsert("col-a", [_doc("only in A", [1.0, 0.0])])
    await store.upsert("col-b", [_doc("only in B", [1.0, 0.0])])

    results_a = await store.query("col-a", [1.0, 0.0], top_k=5)
    assert all(r.content == "only in A" for r in results_a)

    results_b = await store.query("col-b", [1.0, 0.0], top_k=5)
    assert all(r.content == "only in B" for r in results_b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_in_memory_by_default() -> None:
    with patch.dict(os.environ, {"VECTOR_STORE_PROVIDER": "memory"}, clear=False):
        client = get_vector_store_client()
    assert isinstance(client, InMemoryVectorStoreClient)


def test_factory_returns_in_memory_when_env_unset() -> None:
    env = {k: v for k, v in os.environ.items() if k != "VECTOR_STORE_PROVIDER"}
    with patch.dict(os.environ, env, clear=True):
        client = get_vector_store_client()
    assert isinstance(client, InMemoryVectorStoreClient)


def test_factory_returns_chroma_when_provider_set() -> None:
    """Chroma is lazy-imported so we stub chromadb in sys.modules."""
    import sys
    from types import ModuleType

    fake_chroma = ModuleType("chromadb")
    fake_chroma.HttpClient = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    with patch.dict(os.environ, {"VECTOR_STORE_PROVIDER": "chroma"}, clear=False):
        with patch.dict(sys.modules, {"chromadb": fake_chroma}):
            from app.services.vector_store import ChromaVectorStoreClient
            client = get_vector_store_client()
    assert isinstance(client, ChromaVectorStoreClient)