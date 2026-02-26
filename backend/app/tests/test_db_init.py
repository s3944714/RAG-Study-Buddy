import asyncio
import os
import tempfile

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.init_db import init_db


@pytest.fixture
def tmp_db_path() -> str:
    """Return a path to a temporary SQLite file, cleaned up after the test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.mark.asyncio
async def test_init_db_creates_tables(tmp_db_path: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_db_path}", future=True)

    await init_db(engine)

    async with engine.connect() as conn:
        tables = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )

    await engine.dispose()

    assert "workspaces" in tables
    assert "documents" in tables
    assert "chunks" in tables


@pytest.mark.asyncio
async def test_init_db_idempotent(tmp_db_path: str) -> None:
    """Calling init_db twice must not raise — CREATE TABLE IF NOT EXISTS."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_db_path}", future=True)

    await init_db(engine)
    await init_db(engine)  # second call should be a no-op

    await engine.dispose()


@pytest.mark.asyncio
async def test_init_db_tables_are_queryable(tmp_db_path: str) -> None:
    """Tables created by init_db should accept a basic SELECT without error."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_db_path}", future=True)
    await init_db(engine)

    async with engine.connect() as conn:
        for table in ("workspaces", "documents", "chunks"):
            result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            assert result.scalar() == 0

    await engine.dispose()