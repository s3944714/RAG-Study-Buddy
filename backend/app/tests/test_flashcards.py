from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401
from app.api.routes.flashcards import _get_embeddings, _get_llm, _get_vector_store
from app.api.schemas import FlashcardSetResponse
from app.db.base import Base
from app.db.session import get_session
from app.main import app as fastapi_app
from app.retrieval.retriever import RetrievedChunk
from app.services.embeddings_client import DummyEmbeddingsClient
from app.services.flashcards_service import (
    FLASHCARD_SYSTEM_PROMPT,
    _build_context,
    _cards_from_chunks,
    _chunk_citation,
    _parse_llm_json,
    _validate_card,
    generate_flashcards,
)
from app.services.llm_client import DummyLLMClient
from app.services.vector_store import InMemoryVectorStoreClient, VectorDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    doc_id: int = 1,
    text: str = "The nucleus contains DNA.",
    page: int = 2,
    heading: str = "Cell Nucleus",
    score: float = 0.88,
    filename: str = "bio.pdf",
) -> RetrievedChunk:
    return RetrievedChunk(
        embedding_id=f"emb-{doc_id}-{page}",
        text=text,
        doc_id=doc_id,
        page=page,
        heading=heading,
        score=score,
        filename=filename,
    )


async def _seed_store(
    store: InMemoryVectorStoreClient,
    emb: DummyEmbeddingsClient,
    workspace_id: int = 1,
    texts: list[str] | None = None,
) -> None:
    texts = texts or ["The nucleus stores genetic information."]
    docs = []
    for i, text in enumerate(texts):
        vec = await emb.embed_one(text)
        docs.append(
            VectorDocument(
                embedding=vec,
                content=text,
                metadata={
                    "document_id": i + 1,
                    "page": i + 1,
                    "heading": "Section",
                    "filename": f"doc{i+1}.pdf",
                    "chunk_index": 0,
                },
            )
        )
    await store.upsert(f"workspace_{workspace_id}", docs)


# ---------------------------------------------------------------------------
# _parse_llm_json
# ---------------------------------------------------------------------------

def test_parse_llm_json_clean_array() -> None:
    raw = json.dumps([{"front": "Q?", "back": "A."}])
    result = _parse_llm_json(raw)
    assert len(result) == 1
    assert result[0]["front"] == "Q?"


def test_parse_llm_json_embedded_in_prose() -> None:
    raw = 'Here are your cards:\n[{"front": "What?", "back": "This."}]\nDone.'
    result = _parse_llm_json(raw)
    assert len(result) == 1


def test_parse_llm_json_multiple_cards() -> None:
    cards = [{"front": f"Q{i}?", "back": f"A{i}."} for i in range(5)]
    result = _parse_llm_json(json.dumps(cards))
    assert len(result) == 5


def test_parse_llm_json_invalid_returns_empty() -> None:
    assert _parse_llm_json("not json at all") == []


def test_parse_llm_json_empty_string_returns_empty() -> None:
    assert _parse_llm_json("") == []


def test_parse_llm_json_object_not_array_returns_empty() -> None:
    # Top-level object rather than array
    assert _parse_llm_json('{"front": "Q", "back": "A"}') == []


def test_parse_llm_json_strips_markdown_fences() -> None:
    raw = '```json\n[{"front": "Q?", "back": "A."}]\n```'
    # The regex strategy should still find the [...] block
    result = _parse_llm_json(raw)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _validate_card
# ---------------------------------------------------------------------------

def test_validate_card_valid() -> None:
    assert _validate_card({"front": "Q?", "back": "A."}) == {"front": "Q?", "back": "A."}


def test_validate_card_missing_back_returns_none() -> None:
    assert _validate_card({"front": "Q?"}) is None


def test_validate_card_missing_front_returns_none() -> None:
    assert _validate_card({"back": "A."}) is None


def test_validate_card_empty_front_returns_none() -> None:
    assert _validate_card({"front": "  ", "back": "A."}) is None


def test_validate_card_empty_back_returns_none() -> None:
    assert _validate_card({"front": "Q?", "back": ""}) is None


def test_validate_card_not_dict_returns_none() -> None:
    assert _validate_card("not a dict") is None


def test_validate_card_strips_whitespace() -> None:
    result = _validate_card({"front": "  Q?  ", "back": "  A.  "})
    assert result == {"front": "Q?", "back": "A."}


# ---------------------------------------------------------------------------
# _cards_from_chunks (fallback)
# ---------------------------------------------------------------------------

def test_cards_from_chunks_uses_heading_as_front() -> None:
    chunk = _chunk(heading="Photosynthesis")
    cards = _cards_from_chunks([chunk], n=1)
    assert len(cards) == 1
    assert "Photosynthesis" in cards[0]["front"]


def test_cards_from_chunks_no_heading_generic_front() -> None:
    chunk = _chunk(heading="")
    cards = _cards_from_chunks([chunk], n=1)
    assert len(cards) == 1
    assert isinstance(cards[0]["front"], str)


def test_cards_from_chunks_back_is_chunk_text() -> None:
    chunk = _chunk(text="Mitochondria produce ATP.")
    cards = _cards_from_chunks([chunk], n=1)
    assert "Mitochondria produce ATP." in cards[0]["back"]


def test_cards_from_chunks_caps_at_n() -> None:
    chunks = [_chunk(doc_id=i) for i in range(10)]
    cards = _cards_from_chunks(chunks, n=3)
    assert len(cards) == 3


def test_cards_from_chunks_empty_chunks_returns_empty() -> None:
    assert _cards_from_chunks([], n=5) == []


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

def test_build_context_includes_doc_id() -> None:
    chunk = _chunk(doc_id=7)
    ctx = _build_context([chunk])
    assert "7" in ctx


def test_build_context_includes_page() -> None:
    chunk = _chunk(page=4)
    ctx = _build_context([chunk])
    assert "4" in ctx


def test_build_context_includes_chunk_text() -> None:
    chunk = _chunk(text="Specific passage content.")
    ctx = _build_context([chunk])
    assert "Specific passage content." in ctx


def test_build_context_numbers_passages() -> None:
    chunks = [_chunk(doc_id=i) for i in range(1, 4)]
    ctx = _build_context(chunks)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "[3]" in ctx


# ---------------------------------------------------------------------------
# _chunk_citation
# ---------------------------------------------------------------------------

def test_chunk_citation_fields() -> None:
    c = _chunk(doc_id=5, page=3, filename="test.pdf", heading="Intro")
    cit = _chunk_citation(c)
    assert cit.document_id == 5
    assert cit.page == 3
    assert cit.filename == "test.pdf"
    assert cit.heading == "Intro"


def test_chunk_citation_page_none_when_zero() -> None:
    chunk = RetrievedChunk(
        embedding_id="x", text="t", doc_id=1, page=0,
        heading="", score=0.5, filename="f.pdf",
    )
    cit = _chunk_citation(chunk)
    assert cit.page is None


# ---------------------------------------------------------------------------
# generate_flashcards — service integration with DummyLLM (echo fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_flashcards_returns_response_type() -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed_store(vs, emb)

    result = await generate_flashcards(
        workspace_id=1, topic="nucleus",
        embeddings_client=emb, vector_store=vs, llm_client=DummyLLMClient(),
    )
    assert isinstance(result, FlashcardSetResponse)


@pytest.mark.asyncio
async def test_generate_flashcards_returns_cards() -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed_store(vs, emb, texts=[
        "Ribosomes build proteins.",
        "The Golgi apparatus packages proteins.",
        "Lysosomes digest waste materials.",
    ])

    result = await generate_flashcards(
        workspace_id=1, topic="organelles", number_of_cards=3,
        embeddings_client=emb, vector_store=vs, llm_client=DummyLLMClient(),
    )
    # DummyLLM echoes so JSON parse fails → fallback cards from chunks
    assert len(result.flashcards) >= 1


@pytest.mark.asyncio
async def test_generate_flashcards_with_json_llm() -> None:
    """Simulate a real LLM returning valid JSON."""
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed_store(vs, emb)

    json_response = json.dumps([
        {"front": "What stores DNA?", "back": "The nucleus."},
        {"front": "What is ATP?",     "back": "Energy currency of the cell."},
    ])
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=json_response)
    mock_llm.model_name = "mock-gpt"

    result = await generate_flashcards(
        workspace_id=1, topic="cell biology", number_of_cards=2,
        embeddings_client=emb, vector_store=vs, llm_client=mock_llm,
    )
    assert len(result.flashcards) == 2
    assert result.flashcards[0].front == "What stores DNA?"
    assert result.flashcards[1].back == "Energy currency of the cell."


@pytest.mark.asyncio
async def test_generate_flashcards_caps_at_requested_number() -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed_store(vs, emb, texts=[f"Fact number {i}." for i in range(10)])

    json_response = json.dumps([
        {"front": f"Q{i}?", "back": f"A{i}."} for i in range(20)
    ])
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=json_response)
    mock_llm.model_name = "mock"

    result = await generate_flashcards(
        workspace_id=1, topic="facts", number_of_cards=5,
        embeddings_client=emb, vector_store=vs, llm_client=mock_llm,
    )
    assert len(result.flashcards) <= 5


@pytest.mark.asyncio
async def test_generate_flashcards_model_name_in_response() -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed_store(vs, emb)

    result = await generate_flashcards(
        workspace_id=1, topic="test",
        embeddings_client=emb, vector_store=vs, llm_client=DummyLLMClient(),
    )
    assert result.model == "dummy-echo-v1"


@pytest.mark.asyncio
async def test_generate_flashcards_empty_topic_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        await generate_flashcards(
            workspace_id=1, topic="  ",
            embeddings_client=DummyEmbeddingsClient(dimensions=8),
            vector_store=InMemoryVectorStoreClient(),
            llm_client=DummyLLMClient(),
        )


@pytest.mark.asyncio
async def test_generate_flashcards_empty_workspace_returns_response() -> None:
    """No indexed chunks → fallback cards list is empty, response is still valid."""
    result = await generate_flashcards(
        workspace_id=99, topic="anything",
        embeddings_client=DummyEmbeddingsClient(dimensions=8),
        vector_store=InMemoryVectorStoreClient(),
        llm_client=DummyLLMClient(),
    )
    assert isinstance(result, FlashcardSetResponse)
    assert result.flashcards == []


@pytest.mark.asyncio
async def test_generate_flashcards_document_ids_populated() -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed_store(vs, emb, texts=["Content A", "Content B"])

    result = await generate_flashcards(
        workspace_id=1, topic="content",
        embeddings_client=emb, vector_store=vs, llm_client=DummyLLMClient(),
    )
    assert isinstance(result.document_ids, list)


@pytest.mark.asyncio
async def test_generate_flashcards_system_prompt_used() -> None:
    """Verify the system prompt constant is passed to the LLM."""
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed_store(vs, emb)

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="[]")
    mock_llm.model_name = "mock"

    await generate_flashcards(
        workspace_id=1, topic="test",
        embeddings_client=emb, vector_store=vs, llm_client=mock_llm,
    )
    call_kwargs = mock_llm.generate.call_args.kwargs
    assert call_kwargs.get("system_prompt") == FLASHCARD_SYSTEM_PROMPT


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
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    llm = DummyLLMClient()

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    fastapi_app.dependency_overrides[get_session]       = override_session
    fastapi_app.dependency_overrides[_get_embeddings]   = lambda: emb
    fastapi_app.dependency_overrides[_get_vector_store] = lambda: vs
    fastapi_app.dependency_overrides[_get_llm]          = lambda: llm

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_flashcards_endpoint_returns_200(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/workspaces/1/flashcards",
        json={"topic": "photosynthesis", "number_of_cards": 3},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_flashcards_endpoint_response_schema(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/workspaces/1/flashcards",
        json={"topic": "mitosis"},
    )
    data = resp.json()
    assert "workspace_id" in data
    assert "flashcards"   in data
    assert "model"        in data
    assert isinstance(data["flashcards"], list)


@pytest.mark.asyncio
async def test_flashcards_endpoint_workspace_id_in_response(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/workspaces/7/flashcards",
        json={"topic": "enzymes"},
    )
    assert resp.json()["workspace_id"] == 7


@pytest.mark.asyncio
async def test_flashcards_endpoint_empty_topic_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/workspaces/1/flashcards",
        json={"topic": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_flashcards_endpoint_number_too_high_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/workspaces/1/flashcards",
        json={"topic": "DNA", "number_of_cards": 999},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_flashcards_endpoint_number_zero_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/workspaces/1/flashcards",
        json={"topic": "DNA", "number_of_cards": 0},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_flashcards_endpoint_pipeline_error_returns_500(
    client: AsyncClient,
) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.api.routes.flashcards.generate_flashcards",
            AsyncMock(side_effect=RuntimeError("vector store down")),
        )
        resp = await client.post(
            "/api/v1/workspaces/1/flashcards",
            json={"topic": "enzymes"},
        )
    assert resp.status_code == 500
    assert "Flashcard generation error" in resp.json()["detail"]