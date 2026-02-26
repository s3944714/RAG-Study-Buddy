import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.session import get_session
from app.main import app as fastapi_app  # ← renamed to avoid collision with `app` package
import app.db.models  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session_factory():
    """Spin up a fresh in-memory SQLite DB for each test."""
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
    """AsyncClient wired to the test DB via dependency override."""
    async def override_get_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    fastapi_app.dependency_overrides[get_session] = override_get_session  # ← fixed

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()  # ← fixed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_workspace(client: AsyncClient) -> None:
    response = await client.post("/api/v1/workspaces/", json={"name": "Physics 101"})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Physics 101"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_workspace_empty_name_fails(client: AsyncClient) -> None:
    response = await client.post("/api/v1/workspaces/", json={"name": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_workspaces_empty(client: AsyncClient) -> None:
    response = await client.get("/api/v1/workspaces/")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_workspaces_returns_all(client: AsyncClient) -> None:
    await client.post("/api/v1/workspaces/", json={"name": "Maths"})
    await client.post("/api/v1/workspaces/", json={"name": "Chemistry"})

    response = await client.get("/api/v1/workspaces/")
    assert response.status_code == 200
    names = [ws["name"] for ws in response.json()]
    assert "Maths" in names
    assert "Chemistry" in names


@pytest.mark.asyncio
async def test_get_workspace_by_id(client: AsyncClient) -> None:
    created = (await client.post("/api/v1/workspaces/", json={"name": "Biology"})).json()

    response = await client.get(f"/api/v1/workspaces/{created['id']}")
    assert response.status_code == 200
    assert response.json()["name"] == "Biology"


@pytest.mark.asyncio
async def test_get_workspace_not_found(client: AsyncClient) -> None:
    response = await client.get("/api/v1/workspaces/9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()