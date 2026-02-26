import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.indexing_service import index_document
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["indexing"])


# ---------------------------------------------------------------------------
# Response schema (local — not shared; only this route uses it)
# ---------------------------------------------------------------------------

class IndexResponse(BaseModel):
    document_id: int
    chunks_indexed: int
    status: str
    message: str = ""


# ---------------------------------------------------------------------------
# Dependency providers — injectable so tests can swap implementations
# ---------------------------------------------------------------------------

def _get_embeddings_client() -> EmbeddingsClient:
    return get_embeddings_client()


def _get_vector_store() -> VectorStoreClient:
    return get_vector_store_client()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/documents/{document_id}/index",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger indexing for a document",
    description=(
        "Synchronously extracts text, chunks, embeds, and stores vectors for "
        "the given document. Idempotent — calling again re-indexes the document. "
        "Designed so the body of this handler can be dispatched to a background "
        "worker (e.g. Celery, ARQ) in a future iteration without changing callers."
    ),
)
async def index_document_endpoint(
    document_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    embeddings_client: Annotated[EmbeddingsClient, Depends(_get_embeddings_client)],
    vector_store: Annotated[VectorStoreClient, Depends(_get_vector_store)],
    chunk_size: Annotated[int, Query(ge=100, le=4000, description="Characters per chunk")] = 500,
    overlap: Annotated[int, Query(ge=0, le=500, description="Overlap characters between chunks")] = 100,
) -> IndexResponse:
    # NOTE: To move this to a background job later, extract the block below
    # into a standalone task function and replace with:
    #   task_id = await job_queue.enqueue(run_index_task, document_id, ...)
    #   return IndexResponse(document_id=document_id, status="queued", ...)
    try:
        chunks_indexed = await index_document(
            document_id,
            session,
            embeddings_client=embeddings_client,
            vector_store=vector_store,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except ValueError as exc:
        # Document not found or no extractable content
        detail = str(exc)
        logger.warning("index_document ValueError document_id=%d: %s", document_id, detail)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    except Exception as exc:
        # Pipeline failure (bad PDF, embedding error, etc.)
        logger.error("index_document failed document_id=%d: %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {exc}",
        )

    return IndexResponse(
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        status="indexed",
        message=f"Successfully indexed {chunks_indexed} chunk(s).",
    )