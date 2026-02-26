"""
Flashcard generation service.

Strategy:
  1. Retrieve top-k chunks for the requested topic.
  2. Ask the LLM to return a JSON array of {front, back} objects.
  3. Parse the JSON; fall back to chunk-derived stub cards on parse failure
     (keeps DummyLLMClient fully usable in tests without special-casing).
  4. Attach citation metadata from the source chunks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.api.schemas import Citation, Flashcard, FlashcardSetResponse
from app.retrieval.retriever import RetrievedChunk, retrieve
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt constants (exported so tests can assert on them)
# ---------------------------------------------------------------------------

FLASHCARD_SYSTEM_PROMPT: str = (
    "You are a study-card generator. "
    "Given source passages, produce ONLY a valid JSON array — no prose, no markdown fences. "
    "Each element must have exactly two string keys: \"front\" (a question or term) "
    "and \"back\" (the concise answer or definition). "
    "Base every card strictly on the provided passages. "
    "Do NOT invent facts. If the passages do not support a card, omit it."
)

_CARD_USER_TEMPLATE: str = """\
Generate exactly {n} flashcards from the passages below.
Topic focus: {topic}

{context}

Return ONLY a JSON array, e.g.:
[{{"front": "What is X?", "back": "X is ..."}}]
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        heading = f" | {c.heading}" if c.heading else ""
        parts.append(
            f"[{i}] (Doc {c.doc_id}, p.{c.page}{heading})\n{c.text[:600]}"
        )
    return "\n\n".join(parts)


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array from the LLM response.

    Tries three strategies in order:
      1. Direct json.loads on the full response.
      2. Extract the first [...] block via regex (handles leading prose).
      3. Return an empty list (caller will use fallback cards).
    """
    text = raw.strip()

    # Strategy 1 — clean JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract embedded array
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []


def _cards_from_chunks(
    chunks: list[RetrievedChunk],
    n: int,
) -> list[dict[str, Any]]:
    """Fallback: derive minimal cards directly from chunk text when the LLM
    response is unparseable (e.g. DummyLLMClient echo output)."""
    cards: list[dict[str, Any]] = []
    for chunk in chunks[:n]:
        # Use heading as the 'front' when available, else a generic stub
        front = (
            f"What does the passage about '{chunk.heading}' say?"
            if chunk.heading
            else f"Summarise the following passage (Doc {chunk.doc_id}, p.{chunk.page}):"
        )
        back = chunk.text[:300]
        cards.append({"front": front, "back": back})
    return cards


def _validate_card(card: Any) -> dict[str, str] | None:
    """Return the card dict if it has non-empty 'front' and 'back' strings."""
    if not isinstance(card, dict):
        return None
    front = str(card.get("front", "")).strip()
    back  = str(card.get("back",  "")).strip()
    if not front or not back:
        return None
    return {"front": front, "back": back}


def _chunk_citation(chunk: RetrievedChunk) -> Citation:
    return Citation(
        document_id=chunk.doc_id,
        filename=chunk.filename,
        chunk_index=0,
        page=chunk.page or None,
        heading=chunk.heading or None,
        excerpt=chunk.text[:200],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_flashcards(
    *,
    workspace_id: int,
    topic: str,
    number_of_cards: int = 5,
    doc_ids: list[int] | None = None,
    embeddings_client: EmbeddingsClient | None = None,
    vector_store: VectorStoreClient | None = None,
    llm_client: LLMClient | None = None,
) -> FlashcardSetResponse:
    """Generate flashcards for *topic* using retrieved context + LLM.

    Args:
        workspace_id:     Scope retrieval to this workspace.
        topic:            The subject or query to generate cards about.
        number_of_cards:  How many cards to request (1–50).
        doc_ids:          Optional document allow-list.
        embeddings_client / vector_store / llm_client: Injectable for tests.

    Returns:
        FlashcardSetResponse with cards and per-card citations.
    """
    if not topic.strip():
        raise ValueError("topic must not be empty")
    number_of_cards = max(1, min(50, number_of_cards))

    emb = embeddings_client or get_embeddings_client()
    vs  = vector_store      or get_vector_store_client()
    llm = llm_client        or get_llm_client()

    # 1. Retrieve relevant chunks
    chunks = await retrieve(
        workspace_id,
        topic,
        top_k=min(number_of_cards * 2, 20),
        embeddings_client=emb,
        vector_store=vs,
        doc_ids=doc_ids,
    )
    logger.info(
        "flashcards workspace_id=%d topic=%r → %d chunks, requesting %d cards",
        workspace_id, topic[:60], len(chunks), number_of_cards,
    )

    # 2. Build user prompt
    context = _build_context(chunks) if chunks else "(No relevant passages found.)"
    user_prompt = _CARD_USER_TEMPLATE.format(
        n=number_of_cards,
        topic=topic.strip(),
        context=context,
    )

    # 3. Call LLM
    raw = await llm.generate(user_prompt, system_prompt=FLASHCARD_SYSTEM_PROMPT)

    # 4. Parse JSON → fallback to chunk-derived cards if needed
    raw_cards = _parse_llm_json(raw)
    if not raw_cards:
        logger.warning(
            "LLM response was not parseable JSON for workspace_id=%d — using chunk fallback",
            workspace_id,
        )
        raw_cards = _cards_from_chunks(chunks, number_of_cards)

    # 5. Validate and cap
    valid_cards = [_validate_card(c) for c in raw_cards]
    clean_cards = [c for c in valid_cards if c is not None][:number_of_cards]

    # 6. Build Flashcard objects with citations mapped by position
    citation_map: dict[int, Citation] = {
        i: _chunk_citation(c) for i, c in enumerate(chunks)
    }

    flashcards: list[Flashcard] = []
    for idx, card in enumerate(clean_cards):
        # Attach citation from the chunk at the same index when available
        source_chunk_id: int | None = None
        if idx < len(chunks):
            source_chunk_id = chunks[idx].doc_id  # doc_id as stable ref

        flashcards.append(
            Flashcard(
                front=card["front"],
                back=card["back"],
                source_chunk_id=source_chunk_id,
            )
        )

    logger.info(
        "Generated %d/%d flashcards for workspace_id=%d",
        len(flashcards), number_of_cards, workspace_id,
    )

    return FlashcardSetResponse(
        workspace_id=workspace_id,
        document_ids=list({c.doc_id for c in chunks}),
        flashcards=flashcards,
        model=llm.model_name,
    )