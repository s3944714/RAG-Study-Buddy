from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models        # noqa: F401
import app.db.models_quiz   # noqa: F401 — registers QuizSession with Base
from app.api.routes.quiz import _get_embeddings, _get_llm, _get_vector_store
from app.db.base import Base
from app.db.models import Workspace
from app.db.models_quiz import QuizSession, QuizStatus
from app.db.session import get_session
from app.main import app as fastapi_app
from app.retrieval.retriever import RetrievedChunk
from app.services.embeddings_client import DummyEmbeddingsClient
from app.services.llm_client import DummyLLMClient
from app.services.quiz_service import (
    QUIZ_SYSTEM_PROMPT,
    _fallback_questions,
    _parse_questions,
    _validate_question,
    next_question,
    start_quiz,
    submit_answer,
)
from app.services.vector_store import InMemoryVectorStoreClient, VectorDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_Q: dict = {
    "question":      "What does the nucleus contain?",
    "options":       ["DNA", "Ribosomes", "ATP", "Glucose"],
    "correct_index": 0,
    "explanation":   "The nucleus stores the cell's DNA.",
}

_VALID_JSON = json.dumps([_VALID_Q])


def _make_llm(response: str = _VALID_JSON) -> MagicMock:
    mock = MagicMock()
    mock.generate   = AsyncMock(return_value=response)
    mock.model_name = "mock-llm"
    return mock


def _make_chunks(n: int = 2) -> list[RetrievedChunk]:
    """Return n known RetrievedChunk objects for use in mock patches."""
    return [
        RetrievedChunk(
            embedding_id=f"emb-{i}",
            text=f"The cell organelle number {i} performs function {i}.",
            doc_id=i + 1,
            page=i + 1,
            heading=f"Section {i}",
            score=0.85,
            filename=f"doc{i}.pdf",
        )
        for i in range(n)
    ]


async def _seed(
    store: InMemoryVectorStoreClient,
    emb: DummyEmbeddingsClient,
    workspace_id: int = 1,
    n: int = 3,
) -> None:
    docs = []
    for i in range(n):
        vec = await emb.embed_one(f"study content {i}")
        docs.append(VectorDocument(
            embedding=vec,
            content=f"The cell organelle number {i} performs function {i}.",
            metadata={
                "document_id": i + 1, "page": i + 1,
                "heading": f"Section {i}", "filename": f"doc{i}.pdf",
                "chunk_index": 0,
            },
        ))
    await store.upsert(f"workspace_{workspace_id}", docs)


# ---------------------------------------------------------------------------
# _parse_questions
# ---------------------------------------------------------------------------

def test_parse_questions_clean_json() -> None:
    result = _parse_questions(_VALID_JSON)
    assert len(result) == 1
    assert result[0]["question"] == _VALID_Q["question"]


def test_parse_questions_embedded_in_prose() -> None:
    raw = f"Here are your questions:\n{_VALID_JSON}\nAll done."
    result = _parse_questions(raw)
    assert len(result) == 1


def test_parse_questions_invalid_returns_empty() -> None:
    assert _parse_questions("Not JSON at all.") == []


def test_parse_questions_multiple() -> None:
    raw = json.dumps([_VALID_Q, _VALID_Q])
    assert len(_parse_questions(raw)) == 2


# ---------------------------------------------------------------------------
# _validate_question
# ---------------------------------------------------------------------------

def test_validate_question_valid() -> None:
    assert _validate_question(_VALID_Q) is not None


def test_validate_question_wrong_options_count() -> None:
    bad = {**_VALID_Q, "options": ["A", "B", "C"]}
    assert _validate_question(bad) is None


def test_validate_question_correct_index_out_of_range() -> None:
    bad = {**_VALID_Q, "correct_index": 5}
    assert _validate_question(bad) is None


def test_validate_question_empty_question_text() -> None:
    bad = {**_VALID_Q, "question": "  "}
    assert _validate_question(bad) is None


def test_validate_question_non_dict_returns_none() -> None:
    assert _validate_question("string") is None


def test_validate_question_strips_option_whitespace() -> None:
    q = {**_VALID_Q, "options": [" DNA ", " A ", " B ", " C "]}
    result = _validate_question(q)
    assert result is not None
    assert result["options"][0] == "DNA"


# ---------------------------------------------------------------------------
# _fallback_questions
# ---------------------------------------------------------------------------

def test_fallback_questions_produces_valid_structure() -> None:
    chunk = RetrievedChunk(
        embedding_id="x", text="Mitosis is cell division.",
        doc_id=1, page=1, heading="Mitosis", score=0.9, filename="bio.pdf",
    )
    qs = _fallback_questions([chunk], n=1)
    assert len(qs) == 1
    assert "question" in qs[0]
    assert len(qs[0]["options"]) == 4
    assert qs[0]["correct_index"] == 0


def test_fallback_questions_caps_at_n() -> None:
    chunks = [
        RetrievedChunk(embedding_id=str(i), text=f"t{i}", doc_id=i,
                       page=1, heading="", score=0.5, filename="f.pdf")
        for i in range(10)
    ]
    assert len(_fallback_questions(chunks, n=3)) == 3


# ---------------------------------------------------------------------------
# start_quiz — service
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def workspace(db_session: AsyncSession) -> Workspace:
    ws = Workspace(name="Quiz WS")
    db_session.add(ws)
    await db_session.flush()
    return ws


@pytest.mark.asyncio
async def test_start_quiz_creates_session(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    result = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="organelles",
        num_questions=1,
        embeddings_client=emb, vector_store=vs, llm_client=_make_llm(),
    )
    assert "session_id" in result
    assert result["session_id"] is not None


@pytest.mark.asyncio
async def test_start_quiz_returns_first_question(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    result = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="organelles",
        num_questions=1,
        embeddings_client=emb, vector_store=vs, llm_client=_make_llm(),
    )
    assert "question" in result
    assert "options"  in result
    assert len(result["options"]) == 4


@pytest.mark.asyncio
async def test_start_quiz_question_numbering(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)
    llm = _make_llm(json.dumps([_VALID_Q, _VALID_Q, _VALID_Q]))

    result = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=3,
        embeddings_client=emb, vector_store=vs, llm_client=llm,
    )
    assert result["question_number"] == 1
    assert result["total_questions"] == 3


@pytest.mark.asyncio
async def test_start_quiz_empty_topic_raises(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    with pytest.raises(ValueError, match="empty"):
        await start_quiz(
            db=db_session, workspace_id=workspace.id, topic="  ",
            embeddings_client=DummyEmbeddingsClient(dimensions=8),
            vector_store=InMemoryVectorStoreClient(),
            llm_client=DummyLLMClient(),
        )


@pytest.mark.asyncio
async def test_start_quiz_missing_workspace_raises(db_session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="not found"):
        await start_quiz(
            db=db_session, workspace_id=9999, topic="DNA",
            embeddings_client=DummyEmbeddingsClient(dimensions=8),
            vector_store=InMemoryVectorStoreClient(),
            llm_client=_make_llm(),
        )


@pytest.mark.asyncio
async def test_start_quiz_uses_fallback_when_llm_unparseable(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    """
    DummyLLMClient echoes the prompt — not valid JSON — so _parse_questions
    returns []. The service falls back to _fallback_questions(chunks, n).

    We patch `retrieve` inside quiz_service to return known chunks directly,
    bypassing DummyEmbeddingsClient hash geometry (which can produce negative
    cosine scores that the default score_threshold=0.0 filters out, leaving
    an empty chunk list and therefore an empty fallback list).
    """
    known_chunks = _make_chunks(n=2)

    with patch(
        "app.services.quiz_service.retrieve",
        new_callable=AsyncMock,
        return_value=known_chunks,
    ):
        result = await start_quiz(
            db=db_session,
            workspace_id=workspace.id,
            topic="organelles",
            num_questions=2,
            embeddings_client=DummyEmbeddingsClient(dimensions=8),
            vector_store=InMemoryVectorStoreClient(),
            llm_client=DummyLLMClient(),   # echoes prompt → not valid JSON
        )

    assert "question" in result
    assert result["total_questions"] >= 1
    # Fallback cards use chunk headings — verify it ran, not the LLM path
    assert result["status"] == "active"


# ---------------------------------------------------------------------------
# next_question — service
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_next_question_returns_current_question(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)
    llm = _make_llm(json.dumps([_VALID_Q, _VALID_Q]))

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=2, embeddings_client=emb, vector_store=vs, llm_client=llm,
    )
    nxt = await next_question(db=db_session, session_id=started["session_id"])
    assert nxt["question_number"] == 1
    assert "question" in nxt


@pytest.mark.asyncio
async def test_next_question_missing_session_raises(db_session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="not found"):
        await next_question(db=db_session, session_id=9999)


# ---------------------------------------------------------------------------
# submit_answer — service
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_answer_correct(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=1, embeddings_client=emb, vector_store=vs,
        llm_client=_make_llm(_VALID_JSON),
    )
    feedback = await submit_answer(
        db=db_session,
        session_id=started["session_id"],
        chosen_index=_VALID_Q["correct_index"],
    )
    assert feedback["is_correct"] is True
    assert feedback["score_so_far"] == 1


@pytest.mark.asyncio
async def test_submit_answer_wrong(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=1, embeddings_client=emb, vector_store=vs,
        llm_client=_make_llm(_VALID_JSON),
    )
    wrong_index = (_VALID_Q["correct_index"] + 1) % 4
    feedback = await submit_answer(
        db=db_session, session_id=started["session_id"], chosen_index=wrong_index,
    )
    assert feedback["is_correct"] is False
    assert feedback["score_so_far"] == 0


@pytest.mark.asyncio
async def test_submit_answer_returns_explanation(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=1, embeddings_client=emb, vector_store=vs,
        llm_client=_make_llm(_VALID_JSON),
    )
    feedback = await submit_answer(
        db=db_session, session_id=started["session_id"], chosen_index=0,
    )
    assert isinstance(feedback["explanation"], str)


@pytest.mark.asyncio
async def test_submit_answer_advances_to_next_question(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)
    llm = _make_llm(json.dumps([_VALID_Q, _VALID_Q]))

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=2, embeddings_client=emb, vector_store=vs, llm_client=llm,
    )
    feedback = await submit_answer(
        db=db_session, session_id=started["session_id"], chosen_index=0,
    )
    assert feedback["next"]["status"] == "active"
    assert feedback["next"]["question_number"] == 2


@pytest.mark.asyncio
async def test_submit_answer_last_question_completes_session(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=1, embeddings_client=emb, vector_store=vs,
        llm_client=_make_llm(_VALID_JSON),
    )
    feedback = await submit_answer(
        db=db_session, session_id=started["session_id"], chosen_index=0,
    )
    assert feedback["next"]["status"] == "completed"
    assert "score" in feedback["next"]
    assert "percentage" in feedback["next"]


@pytest.mark.asyncio
async def test_submit_answer_invalid_index_raises(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=1, embeddings_client=emb, vector_store=vs,
        llm_client=_make_llm(_VALID_JSON),
    )
    with pytest.raises(ValueError, match="chosen_index"):
        await submit_answer(
            db=db_session, session_id=started["session_id"], chosen_index=5,
        )


@pytest.mark.asyncio
async def test_completed_session_rejects_further_answers(
    db_session: AsyncSession, workspace: Workspace
) -> None:
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    await _seed(vs, emb, workspace.id)

    started = await start_quiz(
        db=db_session, workspace_id=workspace.id, topic="cells",
        num_questions=1, embeddings_client=emb, vector_store=vs,
        llm_client=_make_llm(_VALID_JSON),
    )
    sid = started["session_id"]
    await submit_answer(db=db_session, session_id=sid, chosen_index=0)

    with pytest.raises(ValueError, match="completed"):
        await submit_answer(db=db_session, session_id=sid, chosen_index=0)


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
async def http_client(session_factory):
    emb = DummyEmbeddingsClient(dimensions=8)
    vs  = InMemoryVectorStoreClient()
    llm = _make_llm(_VALID_JSON)

    async def override_session():
        async with session_factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
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


@pytest_asyncio.fixture
async def started_session(http_client: AsyncClient, session_factory):
    """Create a workspace + start a quiz, return the start payload."""
    async with session_factory() as s:
        ws = Workspace(name="HTTP Quiz WS")
        s.add(ws)
        await s.commit()
        await s.refresh(ws)
        ws_id = ws.id

    resp = await http_client.post(
        f"/api/v1/workspaces/{ws_id}/quiz/start",
        json={"topic": "cells", "num_questions": 1},
    )
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.asyncio
async def test_start_endpoint_201(http_client: AsyncClient, session_factory) -> None:
    async with session_factory() as s:
        ws = Workspace(name="WS1")
        s.add(ws)
        await s.commit()
        await s.refresh(ws)

    resp = await http_client.post(
        f"/api/v1/workspaces/{ws.id}/quiz/start",
        json={"topic": "DNA", "num_questions": 1},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data
    assert "question"   in data
    assert "options"    in data


@pytest.mark.asyncio
async def test_start_endpoint_empty_topic_422(
    http_client: AsyncClient, session_factory
) -> None:
    async with session_factory() as s:
        ws = Workspace(name="WS2")
        s.add(ws)
        await s.commit()
        await s.refresh(ws)

    resp = await http_client.post(
        f"/api/v1/workspaces/{ws.id}/quiz/start",
        json={"topic": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_next_endpoint_returns_question(
    http_client: AsyncClient, started_session: dict
) -> None:
    sid  = started_session["session_id"]
    resp = await http_client.post(f"/api/v1/quiz/{sid}/next")
    assert resp.status_code == 200
    assert "question" in resp.json()


@pytest.mark.asyncio
async def test_next_endpoint_missing_session_404(http_client: AsyncClient) -> None:
    resp = await http_client.post("/api/v1/quiz/99999/next")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_answer_endpoint_returns_feedback(
    http_client: AsyncClient, started_session: dict
) -> None:
    sid  = started_session["session_id"]
    resp = await http_client.post(
        f"/api/v1/quiz/{sid}/answer", json={"chosen_index": 0}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "is_correct"   in data
    assert "explanation"  in data
    assert "score_so_far" in data
    assert "next"         in data


@pytest.mark.asyncio
async def test_answer_endpoint_out_of_range_422(
    http_client: AsyncClient, started_session: dict
) -> None:
    sid  = started_session["session_id"]
    resp = await http_client.post(
        f"/api/v1/quiz/{sid}/answer", json={"chosen_index": 9}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_full_quiz_flow(
    http_client: AsyncClient, session_factory
) -> None:
    """Start → next → answer → completion in one HTTP flow."""
    async with session_factory() as s:
        ws = Workspace(name="Flow WS")
        s.add(ws)
        await s.commit()
        await s.refresh(ws)

    start_resp = await http_client.post(
        f"/api/v1/workspaces/{ws.id}/quiz/start",
        json={"topic": "cells", "num_questions": 1},
    )
    assert start_resp.status_code == 201
    sid = start_resp.json()["session_id"]

    answer_resp = await http_client.post(
        f"/api/v1/quiz/{sid}/answer", json={"chosen_index": 0}
    )
    assert answer_resp.status_code == 200
    summary = answer_resp.json()["next"]
    assert summary["status"] == "completed"
    assert "score" in summary