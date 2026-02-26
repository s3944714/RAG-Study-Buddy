import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Workspace
from app.api.schemas import WorkspaceCreate

logger = logging.getLogger(__name__)


async def create_workspace(session: AsyncSession, data: WorkspaceCreate) -> Workspace:
    ws = Workspace(name=data.name)
    session.add(ws)
    await session.commit()
    await session.refresh(ws)
    logger.info("Created workspace id=%s name=%r", ws.id, ws.name)
    return ws


async def list_workspaces(session: AsyncSession) -> list[Workspace]:
    result = await session.execute(select(Workspace).order_by(Workspace.created_at.desc()))
    return list(result.scalars().all())


async def get_workspace(session: AsyncSession, workspace_id: int) -> Workspace | None:
    return await session.get(Workspace, workspace_id)