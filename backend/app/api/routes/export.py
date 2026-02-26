import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.export_service import CSV_ENCODING, export_flashcards_csv
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["export"])


# ---------------------------------------------------------------------------
# Injectable providers
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

@router.get(
    "/workspaces/{workspace_id}/export/flashcards.csv",
    summary="Export flashcards as Anki-compatible CSV",
    description=(
        "Generates flashcards for *topic* using retrieved study material and "
        "streams the result as a semicolon-delimited CSV file compatible with "
        "Anki's text-file import.  Columns: front, back, tags, source."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "Anki-compatible semicolon-delimited CSV file.",
        }
    },
)
async def export_flashcards_csv_endpoint(
    workspace_id: int,
    topic: Annotated[
        str,
        Query(min_length=1, max_length=500, description="Topic to generate cards for"),
    ],
    session: Annotated[AsyncSession, Depends(get_session)],
    embeddings_client: Annotated[EmbeddingsClient, Depends(_get_embeddings)],
    vector_store: Annotated[VectorStoreClient, Depends(_get_vector_store)],
    llm_client: Annotated[LLMClient, Depends(_get_llm)],
    number_of_cards: Annotated[
        int, Query(ge=1, le=50, description="Number of cards to generate")
    ] = 20,
    tags: Annotated[
        str | None,
        Query(description="Space-separated extra tags to add to every card"),
    ] = None,
) -> StreamingResponse:
    extra_tags = tags.split() if tags else None

    try:
        csv_content = await export_flashcards_csv(
            workspace_id=workspace_id,
            topic=topic,
            number_of_cards=number_of_cards,
            extra_tags=extra_tags,
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
            "export_flashcards_csv error workspace_id=%d: %s",
            workspace_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export error: {exc}",
        )

    filename = f"flashcards_ws{workspace_id}_{topic[:30].replace(' ', '_')}.csv"

    return StreamingResponse(
        iter([csv_content.encode(CSV_ENCODING)]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )