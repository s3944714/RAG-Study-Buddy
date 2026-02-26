import io
import struct

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.session import get_session
from app.main import app as fastapi_app
import app.db.models  # noqa: F401


# ---------------------------------------------------------------------------
# Minimal valid PDF helper
# ---------------------------------------------------------------------------

def make_fake_pdf(size_bytes: int | None = None) -> bytes:
    """Return the smallest structurally-valid PDF bytes.

    If size_bytes is given, pad the content stream to reach that size.
    """
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n190\n%%EOF\n"
    )
    if size_bytes and len(body) < size_bytes:
        # Pad with PDF comment bytes (harmless)
        padding = b"%" + b"x" * (size_bytes - len(body) - 1)
        body = body + padding
    return body


# ---------------------------------------------------------------------------
# Fixtures (same pattern as test_workspaces.py)
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
async def client(session_factory, tmp_path, monkeypatch):
    """Client with in-memory DB and temp storage directory."""
    # Redirect file storage to pytest's tmp_path so no real files are created
    import app.services.documents_service as svc
    monkeypatch.setattr(svc, "STORAGE_ROOT", tmp_path)

    async def override_get_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    fastapi_app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def workspace_id(client: AsyncClient) -> int:
    """Create a workspace and return its id."""
    resp = await client.post("/api/v1/workspaces/", json={"name": "Test WS"})
    assert resp.status_code == 201
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_valid_pdf(client: AsyncClient, workspace_id: int) -> None:
    pdf = make_fake_pdf()
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/upload",
        files={"file": ("lecture.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "lecture.pdf"
    assert data["status"] == "uploaded"
    assert "document_id" in data


@pytest.mark.asyncio
async def test_upload_invalid_mime_type(client: AsyncClient, workspace_id: int) -> None:
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 422
    assert "pdf" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_wrong_extension(client: AsyncClient, workspace_id: int) -> None:
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/upload",
        files={"file": ("image.png", io.BytesIO(b"\x89PNG"), "application/pdf")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_exceeds_size_limit(client: AsyncClient, workspace_id: int) -> None:
    # Build a PDF just over 25 MB
    oversized = make_fake_pdf(size_bytes=25 * 1024 * 1024 + 1)
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/upload",
        files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")},
    )
    assert response.status_code == 422
    assert "large" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_to_missing_workspace(client: AsyncClient) -> None:
    pdf = make_fake_pdf()
    response = await client.post(
        "/api/v1/workspaces/9999/documents/upload",
        files={"file": ("x.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert response.status_code == 422
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_documents(client: AsyncClient, workspace_id: int) -> None:
    pdf = make_fake_pdf()
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/upload",
        files={"file": ("a.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    response = await client.get(f"/api/v1/workspaces/{workspace_id}/documents")
    assert response.status_code == 200
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_get_document_not_found(client: AsyncClient) -> None:
    response = await client.get("/api/v1/documents/9999")
    assert response.status_code == 404