import logging

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import settings
from app.db.base import Base
import app.db.models  # noqa: F401 — registers all models with Base metadata

logger = logging.getLogger(__name__)


async def init_db(engine: AsyncEngine | None = None) -> None:
    """Create all tables if they don't already exist.

    Accepts an optional engine so tests can inject an in-memory SQLite engine
    without touching the real database URL from settings.
    """
    target = engine or create_async_engine(settings.database_url, future=True)

    async with target.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("init_db: tables created / verified against %s", target.url)