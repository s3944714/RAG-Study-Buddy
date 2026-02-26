import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.llm_client import LLMClient, get_llm_client
from app.services.quiz_service import next_question, start_quiz, submit_answer
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["quiz"])


# ---------------------------------------------------------------------------
# Request / response schemas (quiz-local)
# ---------------------------------------------------------------------------

class StartQuizRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    num_questions: int = Field(default=5, ge=1, le=20)


class AnswerRequest(BaseModel):
    chosen_index: int = Field(
        ..., ge=0, le=3, description="0-based index of the selected option"
    )


# ---------------------------------------------------------------------------
# Injectable dependency providers
# ---------------------------------------------------------------------------

def _get_embeddings() -> EmbeddingsClient:
    return get_embeddings_client()


def _get_vector_store() -> VectorStoreClient:
    return get_vector_store_client()


def _get_llm() -> LLMClient:
    return get_llm_client()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/workspaces/{workspace_id}/quiz/start",
    status_code=status.HTTP_201_CREATED,
    summary="Start a new quiz session",
)
async def start_quiz_endpoint(
    workspace_id: Annotated[int, Path()],
    request: Annotated[StartQuizRequest, Body()],
    db: Annotated[AsyncSession, Depends(get_session)],
    embeddings_client: Annotated[EmbeddingsClient, Depends(_get_embeddings)],
    vector_store: Annotated[VectorStoreClient, Depends(_get_vector_store)],
    llm_client: Annotated[LLMClient, Depends(_get_llm)],
) -> dict[str, Any]:
    try:
        return await start_quiz(
            db=db,
            workspace_id=workspace_id,
            topic=request.topic,
            num_questions=request.num_questions,
            embeddings_client=embeddings_client,
            vector_store=vector_store,
            llm_client=llm_client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error("start_quiz error ws=%d: %s", workspace_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz generation error: {exc}",
        )


@router.post(
    "/quiz/{session_id}/next",
    status_code=status.HTTP_200_OK,
    summary="Get the next question (or completion summary)",
)
async def next_question_endpoint(
    session_id: Annotated[int, Path()],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    try:
        return await next_question(db=db, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("next_question error session=%d: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz error: {exc}",
        )


@router.post(
    "/quiz/{session_id}/answer",
    status_code=status.HTTP_200_OK,
    summary="Submit an answer and advance the session",
)
async def submit_answer_endpoint(
    session_id: Annotated[int, Path()],
    request: Annotated[AnswerRequest, Body()],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    try:
        return await submit_answer(
            db=db,
            session_id=session_id,
            chosen_index=request.chosen_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error("submit_answer error session=%d: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz error: {exc}",
        )