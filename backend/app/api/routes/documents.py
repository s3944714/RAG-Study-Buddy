from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import DocumentOut, UploadResponse
from app.db.session import get_session
from app.services.documents_service import (
    get_document,
    list_documents,
    upload_document,
)

router = APIRouter(tags=["documents"])


@router.post(
    "/workspaces/{workspace_id}/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document_endpoint(
    workspace_id: int,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> UploadResponse:
    try:
        return await upload_document(session, workspace_id, file)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.get(
    "/workspaces/{workspace_id}/documents",
    response_model=list[DocumentOut],
)
async def list_documents_endpoint(
    workspace_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[DocumentOut]:
    docs = await list_documents(session, workspace_id)
    return [DocumentOut.model_validate(d) for d in docs]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentOut,
)
async def get_document_endpoint(
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> DocumentOut:
    doc = await get_document(session, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    return DocumentOut.model_validate(doc)