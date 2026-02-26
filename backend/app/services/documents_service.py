import logging
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import UploadResponse
from app.core.security import (
    compute_sha256,
    sanitize_filename,
    validate_extension,
    validate_file_size,
    validate_mime_type,
)
from app.db.models import Document, DocumentStatus, Workspace

logger = logging.getLogger(__name__)

# Base storage directory — override via env if needed
STORAGE_ROOT = Path("storage")


def _workspace_dir(workspace_id: int) -> Path:
    path = STORAGE_ROOT / str(workspace_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


async def upload_document(
    session: AsyncSession,
    workspace_id: int,
    file: UploadFile,
) -> UploadResponse:
    # --- workspace must exist ---
    ws = await session.get(Workspace, workspace_id)
    if ws is None:
        raise ValueError(f"Workspace {workspace_id} not found")

    # --- validations ---
    if not validate_mime_type(file.content_type):
        raise ValueError(f"Unsupported file type: {file.content_type}. Only PDF is allowed.")

    safe_name = sanitize_filename(file.filename or "upload.pdf")

    if not validate_extension(safe_name):
        raise ValueError(f"Unsupported file extension for: {safe_name}")

    contents = await file.read()

    if not validate_file_size(len(contents)):
        raise ValueError(
            f"File too large: {len(contents)} bytes. Maximum is 25 MB."
        )

    file_hash = compute_sha256(contents)

    # --- persist to disk ---
    dest_dir = _workspace_dir(workspace_id)
    dest_path = dest_dir / f"{file_hash[:8]}_{safe_name}"
    dest_path.write_bytes(contents)
    logger.info("Saved file to %s", dest_path)

    # --- persist to DB ---
    doc = Document(
        workspace_id=workspace_id,
        filename=safe_name,
        storage_path=str(dest_path),
        file_hash=file_hash,
        status=DocumentStatus.uploaded,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    logger.info("Created document id=%s filename=%r", doc.id, doc.filename)

    return UploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status.value,
        message="File uploaded successfully.",
    )


async def list_documents(session: AsyncSession, workspace_id: int) -> list[Document]:
    result = await session.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def get_document(session: AsyncSession, document_id: int) -> Document | None:
    return await session.get(Document, document_id)