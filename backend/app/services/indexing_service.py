import logging

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStatus
from app.ingestion.indexer import _collection_name, run_indexing_pipeline
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)


async def index_document(
    document_id: int,
    session: AsyncSession,
    *,
    embeddings_client: EmbeddingsClient | None = None,
    vector_store: VectorStoreClient | None = None,
    chunk_size: int = 500,
    overlap: int = 100,
) -> int:
    """Index (or re-index) a document end-to-end.

    Steps:
        1. Load document from DB — raise if not found.
        2. Purge any existing Chunk rows + vector store entries (idempotent).
        3. Run the extract → chunk → embed → upsert pipeline.
        4. Persist new Chunk rows.
        5. Update Document.status to 'indexed' (or 'failed' on error).

    Args:
        document_id:       PK of the Document row to index.
        session:           Active async DB session (caller owns commit/rollback).
        embeddings_client: Injectable for tests; defaults to get_embeddings_client().
        vector_store:      Injectable for tests; defaults to get_vector_store_client().
        chunk_size:        Characters per chunk.
        overlap:           Overlap characters between consecutive chunks.

    Returns:
        Number of chunks indexed.

    Raises:
        ValueError: If the document is not found or produces no content.
    """
    emb = embeddings_client or get_embeddings_client()
    vs = vector_store or get_vector_store_client()

    # --- 1. Load document ---
    doc: Document | None = await session.get(Document, document_id)
    if doc is None:
        raise ValueError(f"Document {document_id} not found")

    logger.info(
        "Starting indexing: document_id=%d filename=%r workspace_id=%d",
        document_id,
        doc.filename,
        doc.workspace_id,
    )

    try:
        # --- 2. Purge old chunks (idempotent re-index) ---
        await _purge_existing_chunks(
            document_id=document_id,
            workspace_id=doc.workspace_id,
            session=session,
            vector_store=vs,
        )

        # --- 3. Run pipeline ---
        chunk_rows = await run_indexing_pipeline(
            document_id=document_id,
            workspace_id=doc.workspace_id,
            storage_path=doc.storage_path,
            filename=doc.filename,
            embeddings_client=emb,
            vector_store=vs,
            session=session,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # --- 4. Persist Chunk rows ---
        for row in chunk_rows:
            session.add(
                Chunk(
                    document_id=document_id,
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    metadata_json=row["metadata_json"],
                    embedding_id=row["embedding_id"],
                )
            )

        # --- 5. Mark document as indexed ---
        doc.status = DocumentStatus.indexed
        await session.commit()

        logger.info(
            "Indexed document_id=%d → %d chunks (status=indexed)",
            document_id,
            len(chunk_rows),
        )
        return len(chunk_rows)

    except Exception as exc:
        await session.rollback()
        # Best-effort: mark document as failed
        try:
            doc.status = DocumentStatus.failed
            await session.commit()
        except Exception:
            pass
        logger.error("Indexing failed for document_id=%d: %s", document_id, exc)
        raise


async def _purge_existing_chunks(
    *,
    document_id: int,
    workspace_id: int,
    session: AsyncSession,
    vector_store: VectorStoreClient,
) -> None:
    """Delete all existing Chunk rows and their vectors for a document."""
    # Collect embedding_ids to remove from the vector store
    result = await session.execute(
        select(Chunk.embedding_id).where(
            Chunk.document_id == document_id,
            Chunk.embedding_id.is_not(None),
        )
    )
    embedding_ids = [row[0] for row in result.fetchall()]

    if embedding_ids:
        collection = _collection_name(workspace_id)
        await vector_store.delete(collection, embedding_ids)
        logger.debug(
            "Purged %d vectors from collection %r for document_id=%d",
            len(embedding_ids),
            collection,
            document_id,
        )

    # Delete DB rows
    await session.execute(
        delete(Chunk).where(Chunk.document_id == document_id)
    )
    await session.flush()
    logger.debug("Purged Chunk rows for document_id=%d", document_id)