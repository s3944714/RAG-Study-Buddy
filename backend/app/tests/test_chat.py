import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401
from app.db.base import Base
from app.db.session import get_session
from app.main import app as fastapi_app
from app.api.routes.chat import _get_embeddings, _get_llm, _get_vector_store
from app.api.schemas import ChatRequest, ChatResponse, Citation
from app.retrieval.retriever import RetrievedChunk
from app.services.chat_service import _build_citations, _inject_citation_numbers, run_chat
from app.services.embeddings_client import DummyEmbeddingsClient
from app.services.llm_client import DummyLLMClient
from app.services.vector_store import InMemoryVectorStoreClient


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

def _chunk(
    doc_id: int = 1,
    text: str = "Photosynthesis converts sunlight into glucose.",
    page: int = 3,
    heading: str = "Energy",
    score: float = 0.9,
    filename: str = "bio.pdf",
) -> RetrievedChunk:
    return RetrievedChunk(
        embedding_id=f"emb-{doc_id}",
        text=text,
        doc_id=doc_id,
        page=page,
        heading=heading,
        score=score,
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Unit tests — chat_service internals
# ---------------------------------------------------------------------------

def test_build_citations_converts_chunks() -> None:
    chunks = [_chunk(doc_id=1), _chunk(doc_id=2)]
    citations = _build_citations(chunks)
    assert len(citations) == 2
    assert all(isinstance(c, Citation) for c in citations)


def test_build_citations_deduplicates() -> None:
    # Same doc_id + page + embedding_id → should collapse to one
    same = _chunk(doc_id=1, page=1)
    citations = _build_citations([same, same])
    assert len(citations) == 1


def test_build_citations_preserves_order() -> None:
    chunks = [_chunk(doc_id=3), _chunk(doc_id=1), _chunk(doc_id=2)]
    citations = _build_citations(chunks)
    assert [c.document_id for c in citations] == [3, 1, 2]


def test_build_citations_excerpt_max_200_chars() -> None:
    long_text = "word " * 100
    citations = _build_citations([_chunk(text=long_text)])
    assert len(citations[0].excerpt) <= 200


def test_build_citations_page_none_when_zero() -> None:
    chunk = RetrievedChunk(
        embedding_id="x", text="t", doc_id=1, page=0,
        heading="", score=0.5, filename="f.pdf",
    )
    citations = _build_citations([chunk])
    assert citations[0].page is None


def test_build_citations_empty_chunks_returns_empty() -> None:
    assert _build_citations([]) == []


def test_inject_citation_numbers_appends_references() -> None:
    citations = _build_citations([_chunk(doc_id=1, page=5, filename="bio.pdf")])
    result = _inject_citation_numbers("Some answer.", citations)
    assert "**References**" in result
    assert "[1]" in result
    assert "bio.pdf" in result


def test_inject_citation_numbers_no_citations_unchanged() -> None:
    answer = "Here is my answer."
    result = _inject_citation_numbers(answer, [])
    assert result == answer


def test_inject_citation_numbers_page_in_reference() -> None:
    citations = _build_citations([_chunk(page=7)])
    result = _inject_citation_numbers("Answer.", citations)
    assert "p.7" in result


def test_inject_citation_numbers_no_page_omits_page_part() -> None:
    chunk = RetrievedChunk(
        embedding_id="x", text="t", doc_id=1, page=0,
        heading="", score=0.5, filename="notes.pdf",
    )
    citations = _build_citations([chunk])
    result = _inject_citation_numbers("Answer.", citations)
    assert "p." not in result
    assert "notes.pdf" in result


# ---------------------------------------------------------------------------
# Unit tests — run_chat pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chat_returns_chat_response() -> None:
    emb   = DummyEmbeddingsClient(dimensions=8)
    vs    = InMemoryVectorStoreClient()
    llm   = DummyLLMClient()

    response = await run_chat(
        workspace_id=1,
        question="What is photosynthesis?",
        embeddings_client=emb,
        vector_store=vs,
        llm_client=llm,
    )
    assert isinstance(response, ChatResponse)
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0


@pytest.mark.asyncio
async def test_run_chat_empty_workspace_still_responds() -> None:
    """No indexed chunks → DummyLLM still returns an answer string."""
    response = await run_chat(
        workspace_id=99,
        question="Tell me about entropy.",
        embeddings_client=DummyEmbeddingsClient(dimensions=8),
        vector_store=InMemoryVectorStoreClient(),
        llm_client=DummyLLMClient(),
    )
    assert isinstance(response.answer, str)
    assert response.citations == []


@pytest.mark.asyncio
async def test_run_chat_citations_populated_when_chunks_retrieved() -> None:
    emb   = DummyEmbeddingsClient(dimensions=8)
    vs    = InMemoryVectorStoreClient()
    llm   = DummyLLMClient()

    # Seed the vector store with a chunk
    from app.services.vector_store import VectorDocument
    vec = await emb.embed_one("photosynthesis")
    await vs.upsert("workspace_1", [
        VectorDocument(
            embedding=vec,
            content="Photosynthesis uses chlorophyll.",
            metadata={
                "document_id": 5,
                "page": 2,
                "heading": "Plants",
                "filename": "bio.pdf",
                "chunk_index": 0,
            },
        )
    ])

    response = await run_chat(
        workspace_id=1,
        question="photosynthesis",
        embeddings_client=emb,
        vector_store=vs,
        llm_client=llm,
    )
    assert len(response.citations) >= 1
    assert response.citations[0].document_id == 5


@pytest.mark.asyncio
async def test_run_chat_model_name_in_response() -> None:
    response = await run_chat(
        workspace_id=1,
        question="Test question",
        embeddings_client=DummyEmbeddingsClient(dimensions=8),
        vector_store=InMemoryVectorStoreClient(),
        llm_client=DummyLLMClient(),
    )
    assert response.model == "dummy-echo-v1"


@pytest.mark.asyncio
async def test_run_chat_answer_contains_references_block_when_chunks() -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    from app.services.vector_store import VectorDocument
    vec = await emb.embed_one("topic")
    await vs.upsert("workspace_1", [
        VectorDocument(
            embedding=vec,
            content="Some relevant content.",
            metadata={
                "document_id": 1, "page": 1,
                "heading": "", "filename": "doc.pdf", "chunk_index": 0,
            },
        )
    ])
    response = await run_chat(
        workspace_id=1, question="topic",
        embeddings_client=emb, vector_store=vs, llm_client=DummyLLMClient(),
    )
    assert "**References**" in response.answer


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(session_factory):
    emb   = DummyEmbeddingsClient(dimensions=8)
    vs    = InMemoryVectorStoreClient()
    llm   = DummyLLMClient()

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    fastapi_app.dependency_overrides[get_session]      = override_session
    fastapi_app.dependency_overrides[_get_embeddings]  = lambda: emb
    fastapi_app.dependency_overrides[_get_vector_store] = lambda: vs
    fastapi_app.dependency_overrides[_get_llm]         = lambda: llm

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_endpoint_returns_200(client: AsyncClient) -> None:
    payload = {"workspace_id": 1, "question": "What is ATP?"}
    resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_endpoint_response_schema(client: AsyncClient) -> None:
    payload = {"workspace_id": 1, "question": "What is ATP?"}
    resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert "model" in data
    assert isinstance(data["citations"], list)


@pytest.mark.asyncio
async def test_chat_endpoint_workspace_id_mismatch_returns_422(
    client: AsyncClient,
) -> None:
    payload = {"workspace_id": 99, "question": "anything"}
    resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    assert resp.status_code == 422
    assert "workspace_id" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_chat_endpoint_empty_question_returns_422(client: AsyncClient) -> None:
    payload = {"workspace_id": 1, "question": ""}
    resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_endpoint_question_too_long_returns_422(client: AsyncClient) -> None:
    payload = {"workspace_id": 1, "question": "x" * 2001}
    resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_endpoint_pipeline_error_returns_500(client: AsyncClient) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.api.routes.chat.run_chat",
            AsyncMock(side_effect=RuntimeError("LLM unavailable")),
        )
        payload = {"workspace_id": 1, "question": "What is DNA?"}
        resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    assert resp.status_code == 500
    assert "Chat pipeline error" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chat_endpoint_answer_is_non_empty_string(client: AsyncClient) -> None:
    payload = {"workspace_id": 1, "question": "Explain mitosis."}
    resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    assert isinstance(resp.json()["answer"], str)
    assert len(resp.json()["answer"]) > 0


@pytest.mark.asyncio
async def test_chat_endpoint_model_field_populated(client: AsyncClient) -> None:
    payload = {"workspace_id": 1, "question": "Explain meiosis."}
    resp = await client.post("/api/v1/workspaces/1/chat", json=payload)
    assert resp.json()["model"] == "dummy-echo-v1"