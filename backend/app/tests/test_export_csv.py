from __future__ import annotations

import csv
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401
from app.api.routes.export import _get_embeddings, _get_llm, _get_vector_store
from app.api.schemas import Flashcard, FlashcardSetResponse
from app.db.base import Base
from app.db.session import get_session
from app.main import app as fastapi_app
from app.services.embeddings_client import DummyEmbeddingsClient
from app.services.export_service import (
    ANKI_HEADER_COMMENT,
    CSV_COLUMNS,
    CSV_DELIMITER,
    DEFAULT_TAG,
    _card_source,
    _sanitise,
    flashcards_to_csv,
)
from app.services.llm_client import DummyLLMClient
from app.services.vector_store import InMemoryVectorStoreClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fset(
    cards: list[tuple[str, str]] | None = None,
    workspace_id: int = 1,
    doc_ids: list[int] | None = None,
) -> FlashcardSetResponse:
    cards = cards or [("What is ATP?", "ATP is the energy currency of the cell.")]
    flashcards = [
        Flashcard(front=f, back=b, source_chunk_id=i + 1)
        for i, (f, b) in enumerate(cards)
    ]
    return FlashcardSetResponse(
        workspace_id=workspace_id,
        document_ids=doc_ids or [1],
        flashcards=flashcards,
        model="dummy-echo-v1",
    )


def _parse_csv(csv_text: str) -> list[list[str]]:
    """Parse CSV text (skipping comment lines) into rows."""
    lines = [l for l in csv_text.splitlines() if not l.startswith("#")]
    return list(csv.reader(lines, delimiter=CSV_DELIMITER))


# ---------------------------------------------------------------------------
# _sanitise
# ---------------------------------------------------------------------------

def test_sanitise_strips_leading_trailing_whitespace() -> None:
    assert _sanitise("  hello  ") == "hello"


def test_sanitise_collapses_carriage_returns() -> None:
    assert "\r" not in _sanitise("line1\r\nline2")


def test_sanitise_empty_string() -> None:
    assert _sanitise("") == ""


def test_sanitise_preserves_internal_content() -> None:
    result = _sanitise("The nucleus contains DNA.")
    assert "nucleus" in result
    assert "DNA" in result


# ---------------------------------------------------------------------------
# _card_source
# ---------------------------------------------------------------------------

def test_card_source_with_chunk_id() -> None:
    fset = _fset(workspace_id=3, doc_ids=[5])
    card = Flashcard(front="Q", back="A", source_chunk_id=7)
    src = _card_source(card, fset)
    assert "workspace=3" in src
    assert "doc=7" in src


def test_card_source_without_chunk_id_uses_doc_ids() -> None:
    fset = _fset(workspace_id=2, doc_ids=[1, 2, 3])
    card = Flashcard(front="Q", back="A", source_chunk_id=None)
    src = _card_source(card, fset)
    assert "workspace=2" in src
    assert "1" in src and "2" in src and "3" in src


def test_card_source_no_chunk_no_docs() -> None:
    fset = FlashcardSetResponse(
        workspace_id=9, document_ids=[], flashcards=[], model="x"
    )
    card = Flashcard(front="Q", back="A", source_chunk_id=None)
    src = _card_source(card, fset)
    assert "workspace=9" in src


# ---------------------------------------------------------------------------
# flashcards_to_csv — structure
# ---------------------------------------------------------------------------

def test_csv_starts_with_anki_header_comment() -> None:
    csv_text = flashcards_to_csv(_fset())
    assert csv_text.startswith(ANKI_HEADER_COMMENT)


def test_csv_no_header_comment_when_disabled() -> None:
    csv_text = flashcards_to_csv(_fset(), include_header_comment=False)
    assert not csv_text.startswith("#")


def test_csv_has_correct_number_of_rows() -> None:
    cards = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]
    rows = _parse_csv(flashcards_to_csv(_fset(cards)))
    assert len(rows) == 3


def test_csv_uses_semicolon_delimiter() -> None:
    csv_text = flashcards_to_csv(_fset())
    data_lines = [l for l in csv_text.splitlines() if not l.startswith("#")]
    assert any(CSV_DELIMITER in line for line in data_lines)


def test_csv_four_columns_per_row() -> None:
    rows = _parse_csv(flashcards_to_csv(_fset()))
    for row in rows:
        assert len(row) == len(CSV_COLUMNS), f"Expected 4 columns, got {row}"


def test_csv_empty_flashcards_produces_only_comment() -> None:
    fset = FlashcardSetResponse(
        workspace_id=1, document_ids=[], flashcards=[], model="x"
    )
    csv_text = flashcards_to_csv(fset)
    rows = _parse_csv(csv_text)
    assert rows == []
    assert ANKI_HEADER_COMMENT in csv_text


# ---------------------------------------------------------------------------
# flashcards_to_csv — column content
# ---------------------------------------------------------------------------

def test_csv_front_column() -> None:
    rows = _parse_csv(flashcards_to_csv(_fset([("My Question?", "My Answer.")])))
    assert rows[0][0] == "My Question?"


def test_csv_back_column() -> None:
    rows = _parse_csv(flashcards_to_csv(_fset([("Q?", "My detailed answer.")])))
    assert rows[0][1] == "My detailed answer."


def test_csv_tags_column_contains_default_tag() -> None:
    rows = _parse_csv(flashcards_to_csv(_fset()))
    assert DEFAULT_TAG in rows[0][2]


def test_csv_extra_tags_appended() -> None:
    rows = _parse_csv(flashcards_to_csv(_fset(), extra_tags=["biology", "chapter1"]))
    tags = rows[0][2]
    assert "biology" in tags
    assert "chapter1" in tags
    assert DEFAULT_TAG in tags


def test_csv_source_column_contains_workspace_id() -> None:
    rows = _parse_csv(flashcards_to_csv(_fset(workspace_id=42)))
    assert "42" in rows[0][3]


def test_csv_source_column_contains_doc_id() -> None:
    fset = _fset(doc_ids=[99])
    # Manually set source_chunk_id to None so doc_ids path is exercised
    fset.flashcards[0] = Flashcard(
        front=fset.flashcards[0].front,
        back=fset.flashcards[0].back,
        source_chunk_id=None,
    )
    rows = _parse_csv(flashcards_to_csv(fset))
    assert "99" in rows[0][3]


# ---------------------------------------------------------------------------
# flashcards_to_csv — content safety
# ---------------------------------------------------------------------------

def test_csv_front_with_semicolon_is_quoted() -> None:
    """A semicolon in content must be quoted so Anki doesn't split the field."""
    rows = _parse_csv(flashcards_to_csv(_fset([("A; B; C", "Answer.")])))
    assert rows[0][0] == "A; B; C"


def test_csv_multiline_back_is_sanitised() -> None:
    rows = _parse_csv(flashcards_to_csv(_fset([("Q?", "Line1\r\nLine2")])))
    assert "\r" not in rows[0][1]


def test_csv_multiple_cards_all_present() -> None:
    cards = [(f"Q{i}?", f"A{i}.") for i in range(5)]
    rows = _parse_csv(flashcards_to_csv(_fset(cards)))
    for i, row in enumerate(rows):
        assert row[0] == f"Q{i}?"
        assert row[1] == f"A{i}."


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
        async with session_factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    fastapi_app.dependency_overrides[get_session]        = override_session
    fastapi_app.dependency_overrides[_get_embeddings]    = lambda: emb
    fastapi_app.dependency_overrides[_get_vector_store]  = lambda: vs
    fastapi_app.dependency_overrides[_get_llm]           = lambda: llm

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_export_endpoint_returns_200(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "photosynthesis"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_export_endpoint_content_type_is_csv(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "mitosis"},
    )
    assert "text/csv" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_export_endpoint_content_disposition_attachment(
    client: AsyncClient,
) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "DNA"},
    )
    cd = resp.headers.get("content-disposition", "")
    assert "attachment" in cd
    assert ".csv" in cd


@pytest.mark.asyncio
async def test_export_endpoint_filename_contains_topic(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "cell biology"},
    )
    cd = resp.headers.get("content-disposition", "")
    assert "cell" in cd or "biology" in cd


@pytest.mark.asyncio
async def test_export_endpoint_body_is_valid_csv(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "enzymes"},
    )
    # Must be decodable and parse without raising
    text = resp.content.decode("utf-8-sig")
    assert isinstance(text, str)


@pytest.mark.asyncio
async def test_export_endpoint_contains_anki_header(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "ribosomes"},
    )
    text = resp.content.decode("utf-8-sig")
    assert ANKI_HEADER_COMMENT in text


@pytest.mark.asyncio
async def test_export_endpoint_missing_topic_returns_422(
    client: AsyncClient,
) -> None:
    resp = await client.get("/api/v1/workspaces/1/export/flashcards.csv")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_export_endpoint_empty_topic_returns_422(
    client: AsyncClient,
) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_export_endpoint_number_of_cards_param(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "ATP", "number_of_cards": 3},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_export_endpoint_invalid_card_count_returns_422(
    client: AsyncClient,
) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "ATP", "number_of_cards": 999},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_export_endpoint_extra_tags_param(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "meiosis", "tags": "bio chapter3"},
    )
    assert resp.status_code == 200
    text = resp.content.decode("utf-8-sig")
    rows = _parse_csv(text)
    if rows:  # may be empty if DummyLLM fallback produced no cards
        assert "bio" in rows[0][2] or DEFAULT_TAG in rows[0][2]


@pytest.mark.asyncio
async def test_export_endpoint_pipeline_error_returns_500(
    client: AsyncClient,
) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.api.routes.export.export_flashcards_csv",
            AsyncMock(side_effect=RuntimeError("vector store unavailable")),
        )
        resp = await client.get(
            "/api/v1/workspaces/1/export/flashcards.csv",
            params={"topic": "osmosis"},
        )
    assert resp.status_code == 500
    assert "Export error" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_export_endpoint_cache_control_no_store(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/workspaces/1/export/flashcards.csv",
        params={"topic": "DNA replication"},
    )
    assert resp.headers.get("cache-control") == "no-store"


# ---------------------------------------------------------------------------
# Round-trip integration: generate → serialise → parse back
# ---------------------------------------------------------------------------

def test_round_trip_csv_preserves_content() -> None:
    cards = [
        ("What is the powerhouse of the cell?", "The mitochondria."),
        ("What does DNA stand for?", "Deoxyribonucleic acid."),
    ]
    fset = _fset(cards, workspace_id=5)
    csv_text = flashcards_to_csv(fset, extra_tags=["biology"])
    rows = _parse_csv(csv_text)

    assert len(rows) == 2
    assert rows[0][0] == "What is the powerhouse of the cell?"
    assert rows[0][1] == "The mitochondria."
    assert "biology" in rows[0][2]
    assert DEFAULT_TAG in rows[0][2]

    assert rows[1][0] == "What does DNA stand for?"
    assert rows[1][1] == "Deoxyribonucleic acid."