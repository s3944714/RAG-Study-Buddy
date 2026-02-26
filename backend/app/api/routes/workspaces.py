from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import WorkspaceCreate, WorkspaceOut
from app.db.session import get_session
from app.services.workspaces_service import (
    create_workspace,
    get_workspace,
    list_workspaces,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("/", response_model=WorkspaceOut, status_code=status.HTTP_201_CREATED)
async def create_workspace_endpoint(
    data: WorkspaceCreate,
    session: AsyncSession = Depends(get_session),
) -> WorkspaceOut:
    ws = await create_workspace(session, data)
    return WorkspaceOut.model_validate(ws)


@router.get("/", response_model=list[WorkspaceOut])
async def list_workspaces_endpoint(
    session: AsyncSession = Depends(get_session),
) -> list[WorkspaceOut]:
    workspaces = await list_workspaces(session)
    return [WorkspaceOut.model_validate(ws) for ws in workspaces]


@router.get("/{workspace_id}", response_model=WorkspaceOut)
async def get_workspace_endpoint(
    workspace_id: int,
    session: AsyncSession = Depends(get_session),
) -> WorkspaceOut:
    ws = await get_workspace(session, workspace_id)
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )
    return WorkspaceOut.model_validate(ws)