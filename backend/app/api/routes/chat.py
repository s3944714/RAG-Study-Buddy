import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ChatRequest, ChatResponse
from app.db.session import get_session
from app.services.chat_service import run_chat
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Injectable dependency providers (swappable in tests)
# ---------------------------------------------------------------------------

def _get_embeddings() -> EmbeddingsClient:
    return get_embeddings_client()


def _get_vector_store() -> VectorStoreClient:
    return get_vector_store_client()


def _get_llm() -> LLMClient:
    return get_llm_client()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/workspaces/{workspace_id}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with study material",
    description=(
        "Retrieves the most relevant chunks from the workspace, builds a "
        "grounded prompt, and returns an LLM-generated answer with citations. "
        "Answers are strictly grounded in the uploaded documents."
    ),
)
async def chat_endpoint(
    workspace_id: int,
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    embeddings_client: Annotated[EmbeddingsClient, Depends(_get_embeddings)],
    vector_store: Annotated[VectorStoreClient, Depends(_get_vector_store)],
    llm_client: Annotated[LLMClient, Depends(_get_llm)],
) -> ChatResponse:
    # Validate workspace_id consistency between path and body
    if request.workspace_id != workspace_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Path workspace_id ({workspace_id}) does not match "
                f"body workspace_id ({request.workspace_id})."
            ),
        )

    try:
        return await run_chat(
            workspace_id=workspace_id,
            question=request.question,
            embeddings_client=embeddings_client,
            vector_store=vector_store,
            llm_client=llm_client,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "chat_endpoint error workspace_id=%d: %s", workspace_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline error: {exc}",
        )