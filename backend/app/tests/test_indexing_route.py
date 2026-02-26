import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401
from app.db.base import Base
from app.db.session import get_session
from app.main import app as fastapi_app
from app.services.embeddings_client import DummyEmbeddingsClient
from app.services.vector_store import InMemoryVectorStoreClient
from app.api.routes.indexing import _get_embeddings_client, _get_vector_store


# ---------------------------------------------------------------------------
# Minimal valid PDF helper (same pattern used throughout test suite)
# ---------------------------------------------------------------------------

def _make_pdf(content: str = "INTRODUCTION\nThis is test content for indexing.") -> bytes:
    stream = f"BT\n/F1 12 Tf\n72 720 Td\n({content}) Tj\nET".encode()
    stream_obj = (
        b"4 0 obj\n<< /Length " + str(len(stream)).encode() + b" >>\n"
        b"stream\n" + stream + b"\nendstream\nendobj\n"
    )
    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R\n"
        b"   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >> >> >> >>\n"
        b">>\nendobj\n"
    )
    body = header + obj1 + obj2 + obj3 + stream_obj

    def _off(full: bytes, chunk: bytes) -> bytes:
        return f"{full.find(chunk):010d} 00000 n \n".encode()

    xref_pos = len(body)
    xref = (
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        + _off(body, obj1)
        + _off(body, obj2)
        + _off(body, obj3)
        + _off(body, stream_obj)
    )
    trailer = (
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return body + xref + trailer


# ---------------------------------------------------------------------------
# Fixtures
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
async def client(session_factory, tmp_path: Path):
    """HTTP client with in-memory DB, dummy embeddings, and in-memory vector store."""
    embeddings = DummyEmbeddingsClient(dimensions=8)
    vector_store = InMemoryVectorStoreClient()

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    fastapi_app.dependency_overrides[get_session] = override_session
    fastapi_app.dependency_overrides[_get_embeddings_client] = lambda: embeddings
    fastapi_app.dependency_overrides[_get_vector_store] = lambda: vector_store

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def uploaded_doc(client: AsyncClient, tmp_path: Path) -> dict:
    """Create a workspace + upload a real PDF, return the document JSON."""
    # Create workspace
    ws_resp = await client.post("/api/v1/workspaces/", json={"name": "Index WS"})
    assert ws_resp.status_code == 201
    workspace_id = ws_resp.json()["id"]

    # Patch STORAGE_ROOT so the file lands in tmp_path
    import app.services.documents_service as svc
    original = svc.STORAGE_ROOT
    svc.STORAGE_ROOT = tmp_path

    upload_resp = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/upload",
        files={"file": ("lecture.pdf", io.BytesIO(_make_pdf()), "application/pdf")},
    )
    svc.STORAGE_ROOT = original

    assert upload_resp.status_code == 201
    return upload_resp.json()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_endpoint_returns_200(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    doc_id = uploaded_doc["document_id"]
    resp = await client.post(f"/api/v1/documents/{doc_id}/index")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_index_endpoint_response_schema(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    doc_id = uploaded_doc["document_id"]
    resp = await client.post(f"/api/v1/documents/{doc_id}/index")
    data = resp.json()

    assert data["document_id"] == doc_id
    assert data["status"] == "indexed"
    assert data["chunks_indexed"] >= 1
    assert "message" in data


@pytest.mark.asyncio
async def test_index_endpoint_chunk_count_positive(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    doc_id = uploaded_doc["document_id"]
    resp = await client.post(f"/api/v1/documents/{doc_id}/index")
    assert resp.json()["chunks_indexed"] >= 1


@pytest.mark.asyncio
async def test_index_endpoint_is_idempotent(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    """Calling index twice must not raise and must return consistent chunk count."""
    doc_id = uploaded_doc["document_id"]

    first = await client.post(f"/api/v1/documents/{doc_id}/index")
    second = await client.post(f"/api/v1/documents/{doc_id}/index")

    assert first.status_code == 200
    assert second.status_code == 200
    # Same document → same chunk count both times
    assert first.json()["chunks_indexed"] == second.json()["chunks_indexed"]


# ---------------------------------------------------------------------------
# Query-parameter variants
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_endpoint_accepts_custom_chunk_size(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    doc_id = uploaded_doc["document_id"]
    resp = await client.post(
        f"/api/v1/documents/{doc_id}/index",
        params={"chunk_size": 200, "overlap": 20},
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] >= 1


@pytest.mark.asyncio
async def test_index_endpoint_rejects_invalid_chunk_size(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    doc_id = uploaded_doc["document_id"]
    # chunk_size below minimum (100)
    resp = await client.post(
        f"/api/v1/documents/{doc_id}/index",
        params={"chunk_size": 10},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_index_endpoint_rejects_invalid_overlap(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    doc_id = uploaded_doc["document_id"]
    # overlap above maximum (500)
    resp = await client.post(
        f"/api/v1/documents/{doc_id}/index",
        params={"overlap": 9999},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_nonexistent_document_returns_404(
    client: AsyncClient,
) -> None:
    resp = await client.post("/api/v1/documents/99999/index")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_index_pipeline_error_returns_500(
    client: AsyncClient, uploaded_doc: dict
) -> None:
    """If the indexing pipeline raises an unexpected exception, expect 500."""
    doc_id = uploaded_doc["document_id"]

    with patch(
        "app.api.routes.indexing.index_document",
        new_callable=AsyncMock,
        side_effect=RuntimeError("embedding service unavailable"),
    ):
        resp = await client.post(f"/api/v1/documents/{doc_id}/index")

    assert resp.status_code == 500
    assert "Indexing failed" in resp.json()["detail"]