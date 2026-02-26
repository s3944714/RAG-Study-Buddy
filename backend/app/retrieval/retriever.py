import logging
from dataclasses import dataclass

from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.vector_store import QueryResult, VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single chunk returned from the retrieval pipeline."""

    embedding_id: str        # vector store ID
    text: str                # chunk content
    doc_id: int              # Document.id FK
    page: int                # source page number (1-based)
    heading: str             # section heading or "" if none detected
    score: float             # cosine similarity in [0, 1]; higher = more relevant
    filename: str = ""       # original filename for citation display


# ---------------------------------------------------------------------------
# Collection naming (must stay in sync with indexer.py)
# ---------------------------------------------------------------------------

def _collection_name(workspace_id: int) -> str:
    return f"workspace_{workspace_id}"


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

async def retrieve(
    workspace_id: int,
    query: str,
    *,
    top_k: int = 5,
    embeddings_client: EmbeddingsClient | None = None,
    vector_store: VectorStoreClient | None = None,
    score_threshold: float = 0.0,
    doc_ids: list[int] | None = None,
) -> list[RetrievedChunk]:
    """Embed *query* then fetch the top-k most similar chunks for *workspace_id*.

    Args:
        workspace_id:      Scope the search to this workspace's collection.
        query:             Natural-language question to embed and search with.
        top_k:             Maximum number of chunks to return.
        embeddings_client: Injectable for tests; defaults to get_embeddings_client().
        vector_store:      Injectable for tests; defaults to get_vector_store_client().
        score_threshold:   Discard results with score strictly below this value.
                           Default 0.0 means keep everything.
        doc_ids:           Optional allow-list of document IDs to restrict search.
                           None means search across the whole workspace.

    Returns:
        List of RetrievedChunk, sorted descending by score (most relevant first).
        Empty list if the workspace collection has no content yet.
    """
    if not query.strip():
        logger.warning("retrieve called with empty query for workspace_id=%d", workspace_id)
        return []

    emb = embeddings_client or get_embeddings_client()
    vs = vector_store or get_vector_store_client()

    # 1. Embed the query
    query_vector = await emb.embed_one(query)
    logger.debug(
        "Embedded query for workspace_id=%d top_k=%d threshold=%.3f",
        workspace_id, top_k, score_threshold,
    )

    # 2. Build optional metadata filter
    where: dict | None = None
    if doc_ids is not None:
        if len(doc_ids) == 0:
            # Empty allow-list → nothing can match
            return []
        if len(doc_ids) == 1:
            # Chroma / InMemory support simple equality filter
            where = {"document_id": doc_ids[0]}
        # NOTE: multi-doc filtering via $in is Chroma-specific; for the
        # InMemory store we post-filter below. Keep where=None for >1 doc_ids
        # and filter after the query so both backends work correctly.

    # 3. Query the vector store
    collection = _collection_name(workspace_id)
    raw_results: list[QueryResult] = await vs.query(
        collection,
        query_vector,
        top_k=top_k if doc_ids is None or len(doc_ids) == 1 else top_k * 4,
        where=where,
    )

    # 4. Parse, post-filter, and convert
    chunks: list[RetrievedChunk] = []
    for result in raw_results:
        meta = result.metadata

        # Post-filter for multi-doc allow-list
        if doc_ids is not None and len(doc_ids) > 1:
            if meta.get("document_id") not in doc_ids:
                continue

        # Score threshold gate
        if result.score < score_threshold:
            continue

        try:
            chunk = RetrievedChunk(
                embedding_id=result.id,
                text=result.content,
                doc_id=int(meta["document_id"]),
                page=int(meta.get("page", 0)),
                heading=meta.get("heading", ""),
                score=result.score,
                filename=meta.get("filename", ""),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "Skipping malformed vector store result id=%s: %s", result.id, exc
            )
            continue

        chunks.append(chunk)

    # 5. Sort descending by score, cap at top_k
    chunks.sort(key=lambda c: c.score, reverse=True)
    chunks = chunks[:top_k]

    logger.info(
        "retrieve workspace_id=%d query=%r → %d chunks (top_k=%d)",
        workspace_id, query[:60], len(chunks), top_k,
    )
    return chunks