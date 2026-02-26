import json
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunking import ChunkCandidate, chunk_pages
from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.services.embeddings_client import EmbeddingsClient
from app.services.vector_store import VectorDocument, VectorStoreClient

logger = logging.getLogger(__name__)


def _collection_name(workspace_id: int) -> str:
    """Each workspace gets its own Chroma collection."""
    return f"workspace_{workspace_id}"


async def run_indexing_pipeline(
    *,
    document_id: int,
    workspace_id: int,
    storage_path: str,
    filename: str,
    embeddings_client: EmbeddingsClient,
    vector_store: VectorStoreClient,
    session: AsyncSession,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Core indexing pipeline: extract → chunk → embed → upsert.

    Returns a list of dicts ready to be persisted as Chunk rows:
        {chunk_index, content, metadata_json, embedding_id}

    This function has no direct DB imports so it stays easily testable.
    All side-effects (DB writes, status updates) are handled by the caller
    (indexing_service.py).
    """
    # 1. Extract text from PDF
    logger.info("Extracting text from document_id=%d path=%s", document_id, storage_path)
    pages = extract_text_from_pdf(Path(storage_path))

    if not any(p["text"] for p in pages):
        raise ValueError(f"No extractable text found in document {filename!r}")

    # 2. Chunk pages
    candidates: list[ChunkCandidate] = chunk_pages(
        pages, chunk_size=chunk_size, overlap=overlap
    )
    logger.info("Produced %d chunks for document_id=%d", len(candidates), document_id)

    if not candidates:
        raise ValueError(f"Chunking produced zero candidates for document {filename!r}")

    # 3. Embed all chunks in one batch
    texts = [c["content"] for c in candidates]
    vectors = await embeddings_client.embed_many(texts)

    # 4. Build VectorDocuments and upsert to vector store
    collection = _collection_name(workspace_id)
    vector_docs = [
        VectorDocument(
            embedding=vector,
            content=candidate["content"],
            metadata={
                "document_id": document_id,
                "chunk_index": candidate["chunk_index"],
                "page": candidate["page"],
                "heading": candidate["heading"] or "",
                "filename": filename,
            },
        )
        for candidate, vector in zip(candidates, vectors)
    ]

    stored_ids = await vector_store.upsert(collection, vector_docs)
    logger.info(
        "Upserted %d vectors to collection %r for document_id=%d",
        len(stored_ids),
        collection,
        document_id,
    )

    # 5. Return chunk rows to be persisted by caller
    return [
        {
            "chunk_index": candidate["chunk_index"],
            "content": candidate["content"],
            "metadata_json": json.dumps(
                {"page": candidate["page"], "heading": candidate["heading"] or ""}
            ),
            "embedding_id": embedding_id,
        }
        for candidate, embedding_id in zip(candidates, stored_ids)
    ]