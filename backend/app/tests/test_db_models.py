import json

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Chunk, Document, DocumentStatus, Workspace


@pytest.fixture
async def session():
    """In-memory SQLite session — isolated per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_workspace(session: AsyncSession) -> None:
    ws = Workspace(name="Physics 101")
    session.add(ws)
    await session.commit()
    await session.refresh(ws)

    assert ws.id is not None
    assert ws.name == "Physics 101"
    assert ws.created_at is not None


@pytest.mark.asyncio
async def test_create_document_and_chunk(session: AsyncSession) -> None:
    ws = Workspace(name="Maths")
    session.add(ws)
    await session.flush()

    doc = Document(
        workspace_id=ws.id,
        filename="calculus.pdf",
        storage_path="/uploads/calculus.pdf",
        file_hash="abc123",
        status=DocumentStatus.uploaded,
    )
    session.add(doc)
    await session.flush()

    chunk = Chunk(
        document_id=doc.id,
        chunk_index=0,
        content="A limit is the value a function approaches...",
        metadata_json=json.dumps({"page": 1, "heading": "Limits"}),
        embedding_id=None,
    )
    session.add(chunk)
    await session.commit()
    await session.refresh(chunk)

    assert chunk.id is not None
    assert json.loads(chunk.metadata_json)["heading"] == "Limits"


@pytest.mark.asyncio
async def test_document_status_default(session: AsyncSession) -> None:
    ws = Workspace(name="Chemistry")
    session.add(ws)
    await session.flush()

    doc = Document(
        workspace_id=ws.id,
        filename="organic.pdf",
        storage_path="/uploads/organic.pdf",
        file_hash="def456",
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    assert doc.status == DocumentStatus.uploaded


@pytest.mark.asyncio
async def test_cascade_delete_workspace(session: AsyncSession) -> None:
    ws = Workspace(name="To Delete")
    session.add(ws)
    await session.flush()

    doc = Document(
        workspace_id=ws.id,
        filename="temp.pdf",
        storage_path="/uploads/temp.pdf",
        file_hash="000",
    )
    session.add(doc)
    await session.flush()

    await session.delete(ws)
    await session.commit()

    result = await session.get(Document, doc.id)
    assert result is None  # cascade deleted