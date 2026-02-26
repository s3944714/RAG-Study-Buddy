import json
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 — registers models
from app.db.base import Base
from app.db.models import Chunk, Document, DocumentStatus, Workspace
from app.services.embeddings_client import DummyEmbeddingsClient
from app.services.indexing_service import index_document
from app.services.vector_store import InMemoryVectorStoreClient


# ---------------------------------------------------------------------------
# Minimal valid PDF (same helper pattern as test_upload.py)
# ---------------------------------------------------------------------------

def _make_pdf(content: str = "Introduction\nThis is test content for RAG indexing.") -> bytes:
    stream = f"BT\n/F1 12 Tf\n72 720 Td\n({content}) Tj\nET".encode()
    stream_obj = (
        b"4 0 obj\n<< /Length " + str(len(stream)).encode() + b" >>\n"
        b"stream\n" + stream + b"\nendstream\nendobj\n"
    )
    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R\n"
        b"   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >> >> >> >>\n"
        b">>\nendobj\n"
    )
    body = header + obj1 + obj2 + obj3 + stream_obj

    def _off(full: bytes, chunk: bytes) -> bytes:
        return f"{full.find(chunk):010d} 00000 n \n".encode()

    xref_pos = len(body)
    xref = (
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        + _off(body, obj1)
        + _off(body, obj2)
        + _off(body, obj3)
        + _off(body, stream_obj)
    )
    trailer = (
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return body + xref + trailer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def workspace(session: AsyncSession) -> Workspace:
    ws = Workspace(name="Test Workspace")
    session.add(ws)
    await session.flush()
    return ws


@pytest_asyncio.fixture
async def pdf_document(
    session: AsyncSession,
    workspace: Workspace,
    tmp_path: Path,
) -> Document:
    """Write a real PDF to tmp_path and create a Document row pointing to it."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(_make_pdf())

    doc = Document(
        workspace_id=workspace.id,
        filename="test.pdf",
        storage_path=str(pdf_path),
        file_hash="abc123",
        status=DocumentStatus.uploaded,
    )
    session.add(doc)
    await session.flush()
    return doc


@pytest_asyncio.fixture
def embeddings() -> DummyEmbeddingsClient:
    return DummyEmbeddingsClient(dimensions=8)


@pytest_asyncio.fixture
def vector_store() -> InMemoryVectorStoreClient:
    return InMemoryVectorStoreClient()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_document_returns_chunk_count(
    session: AsyncSession,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    count = await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )
    assert count >= 1


@pytest.mark.asyncio
async def test_index_document_sets_status_indexed(
    session: AsyncSession,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )
    await session.refresh(pdf_document)
    assert pdf_document.status == DocumentStatus.indexed


@pytest.mark.asyncio
async def test_index_document_writes_chunk_rows(
    session: AsyncSession,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    count = await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )
    result = await session.execute(
        select(Chunk).where(Chunk.document_id == pdf_document.id)
    )
    chunks = result.scalars().all()
    assert len(chunks) == count


@pytest.mark.asyncio
async def test_chunk_rows_have_embedding_id(
    session: AsyncSession,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )
    result = await session.execute(
        select(Chunk).where(Chunk.document_id == pdf_document.id)
    )
    for chunk in result.scalars().all():
        assert chunk.embedding_id is not None
        assert len(chunk.embedding_id) > 0


@pytest.mark.asyncio
async def test_chunk_rows_have_valid_metadata_json(
    session: AsyncSession,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )
    result = await session.execute(
        select(Chunk).where(Chunk.document_id == pdf_document.id)
    )
    for chunk in result.scalars().all():
        meta = json.loads(chunk.metadata_json)
        assert "page" in meta
        assert "heading" in meta


@pytest.mark.asyncio
async def test_chunk_indices_are_sequential(
    session: AsyncSession,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )
    result = await session.execute(
        select(Chunk.chunk_index)
        .where(Chunk.document_id == pdf_document.id)
        .order_by(Chunk.chunk_index)
    )
    indices = [r[0] for r in result.fetchall()]
    assert indices == list(range(len(indices)))


@pytest.mark.asyncio
async def test_vectors_stored_in_correct_collection(
    session: AsyncSession,
    workspace: Workspace,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )
    # Query the collection that should have been populated
    collection = f"workspace_{workspace.id}"
    dummy_query = await embeddings.embed_one("test query")
    results = await vector_store.query(collection, dummy_query, top_k=10)
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Idempotent re-index
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reindex_replaces_old_chunks(
    session: AsyncSession,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    # First indexing
    first_count = await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )

    # Second indexing of the same document
    second_count = await index_document(
        pdf_document.id,
        session,
        embeddings_client=embeddings,
        vector_store=vector_store,
    )

    # Chunk count should be stable (not doubled)
    result = await session.execute(
        select(Chunk).where(Chunk.document_id == pdf_document.id)
    )
    chunks_in_db = result.scalars().all()
    assert len(chunks_in_db) == second_count
    assert len(chunks_in_db) == first_count  # same document = same chunks


@pytest.mark.asyncio
async def test_reindex_does_not_duplicate_vectors(
    session: AsyncSession,
    workspace: Workspace,
    pdf_document: Document,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    collection = f"workspace_{workspace.id}"
    dummy_query = await embeddings.embed_one("query")

    await index_document(
        pdf_document.id, session,
        embeddings_client=embeddings, vector_store=vector_store,
    )
    first_results = await vector_store.query(collection, dummy_query, top_k=100)

    await index_document(
        pdf_document.id, session,
        embeddings_client=embeddings, vector_store=vector_store,
    )
    second_results = await vector_store.query(collection, dummy_query, top_k=100)

    # Vector count must stay the same after re-index
    assert len(first_results) == len(second_results)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_nonexistent_document_raises(
    session: AsyncSession,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        await index_document(
            9999,
            session,
            embeddings_client=embeddings,
            vector_store=vector_store,
        )


@pytest.mark.asyncio
async def test_index_sets_status_failed_on_bad_path(
    session: AsyncSession,
    workspace: Workspace,
    embeddings: DummyEmbeddingsClient,
    vector_store: InMemoryVectorStoreClient,
) -> None:
    """If storage_path points to a missing file, status should become 'failed'."""
    doc = Document(
        workspace_id=workspace.id,
        filename="missing.pdf",
        storage_path="/nonexistent/path/missing.pdf",
        file_hash="000",
        status=DocumentStatus.uploaded,
    )
    session.add(doc)
    await session.flush()

    with pytest.raises(Exception):
        await index_document(
            doc.id,
            session,
            embeddings_client=embeddings,
            vector_store=vector_store,
        )

    # Re-fetch in a fresh query since session may have been rolled back
    result = await session.execute(
        select(Document).where(Document.id == doc.id)
    )
    refreshed = result.scalar_one_or_none()
    if refreshed:
        assert refreshed.status == DocumentStatus.failed