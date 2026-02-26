import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field

from app.api.schemas import FlashcardSetResponse
from app.db.session import get_session
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.flashcards_service import generate_flashcards
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStoreClient, get_vector_store_client
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["flashcards"])


# ---------------------------------------------------------------------------
# Request schema (local — only this route uses it)
# ---------------------------------------------------------------------------

class FlashcardsRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Subject or query to generate flashcards about.",
    )
    number_of_cards: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of flashcards to generate (1–50).",
    )
    doc_ids: list[int] | None = Field(
        default=None,
        description="Optional document allow-list. None = whole workspace.",
    )


# ---------------------------------------------------------------------------
# Injectable providers (swappable in tests)
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
    "/workspaces/{workspace_id}/flashcards",
    response_model=FlashcardSetResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate flashcards from study material",
    description=(
        "Retrieves the most relevant passages for *topic*, then uses an LLM "
        "to produce question/answer flashcards strictly grounded in the "
        "uploaded documents. Suitable for Anki export or in-app review."
    ),
)
async def generate_flashcards_endpoint(
    workspace_id: Annotated[int, Path(description="Target workspace")],
    request: Annotated[FlashcardsRequest, Body()],
    session: Annotated[AsyncSession, Depends(get_session)],
    embeddings_client: Annotated[EmbeddingsClient, Depends(_get_embeddings)],
    vector_store: Annotated[VectorStoreClient, Depends(_get_vector_store)],
    llm_client: Annotated[LLMClient, Depends(_get_llm)],
) -> FlashcardSetResponse:
    try:
        return await generate_flashcards(
            workspace_id=workspace_id,
            topic=request.topic,
            number_of_cards=request.number_of_cards,
            doc_ids=request.doc_ids,
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
            "flashcards_endpoint error workspace_id=%d: %s",
            workspace_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Flashcard generation error: {exc}",
        )