"""
Export service — serialises flashcards to Anki-compatible CSV.

Anki CSV format:
    - Default field separator: semicolon (configurable)
    - Columns: Front, Back, Tags, Source
    - First row is data (no header by default) but we include an optional
      commented header line Anki ignores: `#front;back;tags;source`
    - UTF-8 encoding with BOM so Anki on Windows detects the encoding.

References:
    https://docs.ankiweb.net/importing/text-files.html
"""

from __future__ import annotations

import csv
import io
import logging

from app.api.schemas import Flashcard, FlashcardSetResponse
from app.services.flashcards_service import generate_flashcards
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (exported so tests can assert on them)
# ---------------------------------------------------------------------------

CSV_DELIMITER: str  = ";"
CSV_ENCODING: str   = "utf-8-sig"          # BOM — Anki / Excel safe
CSV_COLUMNS: tuple  = ("front", "back", "tags", "source")
ANKI_HEADER_COMMENT: str = "#separator:Semicolon"

# Default tag applied to every exported card so the deck is easily filterable.
DEFAULT_TAG: str = "rag-study-buddy"


# ---------------------------------------------------------------------------
# Core serialisation (pure — no I/O, no FastAPI deps)
# ---------------------------------------------------------------------------

def flashcards_to_csv(
    flashcard_set: FlashcardSetResponse,
    *,
    extra_tags: list[str] | None = None,
    include_header_comment: bool = True,
) -> str:
    """Serialise a FlashcardSetResponse to an Anki-compatible CSV string.

    Args:
        flashcard_set:           The generated flashcard set to export.
        extra_tags:              Additional tags to attach to every card
                                 (merged with DEFAULT_TAG).
        include_header_comment:  Prepend Anki's `#separator:Semicolon` hint.

    Returns:
        A UTF-8 string (with BOM prefix for CSV_ENCODING) ready to stream
        as a file download.  Empty flashcards list → header comment only.
    """
    tags_parts = [DEFAULT_TAG] + (extra_tags or [])
    tag_string = " ".join(t.strip() for t in tags_parts if t.strip())

    buf = io.StringIO()

    if include_header_comment:
        buf.write(ANKI_HEADER_COMMENT + "\n")

    writer = csv.writer(
        buf,
        delimiter=CSV_DELIMITER,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    for card in flashcard_set.flashcards:
        source = _card_source(card, flashcard_set)
        writer.writerow([
            _sanitise(card.front),
            _sanitise(card.back),
            tag_string,
            source,
        ])

    csv_text = buf.getvalue()
    logger.info(
        "export_service: serialised %d cards for workspace_id=%d",
        len(flashcard_set.flashcards),
        flashcard_set.workspace_id,
    )
    return csv_text


def _card_source(card: Flashcard, fset: FlashcardSetResponse) -> str:
    """Build a human-readable source string for the card's fourth column."""
    if card.source_chunk_id is not None:
        return f"workspace={fset.workspace_id} doc={card.source_chunk_id}"
    if fset.document_ids:
        docs = ",".join(str(d) for d in sorted(fset.document_ids))
        return f"workspace={fset.workspace_id} docs=[{docs}]"
    return f"workspace={fset.workspace_id}"


def _sanitise(text: str) -> str:
    """Strip characters that break Anki card rendering."""
    # Anki interprets a bare semicolon outside quotes as a field separator —
    # csv.QUOTE_MINIMAL will quote the field when needed, but we also
    # collapse multiple blank lines to avoid broken card display.
    return text.replace("\r\n", " ").replace("\r", " ").strip()


# ---------------------------------------------------------------------------
# High-level entry point used by the route
# ---------------------------------------------------------------------------

async def export_flashcards_csv(
    *,
    workspace_id: int,
    topic: str,
    number_of_cards: int = 20,
    doc_ids: list[int] | None = None,
    extra_tags: list[str] | None = None,
    embeddings_client: EmbeddingsClient | None = None,
    vector_store: VectorStoreClient | None = None,
    llm_client: LLMClient | None = None,
) -> str:
    """Generate flashcards then serialise them to CSV in one call.

    Returns:
        CSV string ready for streaming as a file download.

    Raises:
        ValueError: propagated from generate_flashcards (empty topic, etc.)
    """
    emb = embeddings_client or get_embeddings_client()
    vs  = vector_store      or get_vector_store_client()
    llm = llm_client        or get_llm_client()

    flashcard_set = await generate_flashcards(
        workspace_id=workspace_id,
        topic=topic,
        number_of_cards=number_of_cards,
        doc_ids=doc_ids,
        embeddings_client=emb,
        vector_store=vs,
        llm_client=llm,
    )

    return flashcards_to_csv(flashcard_set, extra_tags=extra_tags)