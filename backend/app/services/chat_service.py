import logging
import re

from app.api.schemas import ChatResponse, Citation
from app.retrieval.retriever import RetrievedChunk, retrieve
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.llm_client import LLMClient, get_llm_client
from app.services.prompting import build_chat_prompt
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Convert RetrievedChunks into Citation objects (deduped, ordered)."""
    seen: set[str] = set()
    citations: list[Citation] = []

    for chunk in chunks:
        key = f"{chunk.doc_id}:{chunk.page}:{chunk.embedding_id}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document_id=chunk.doc_id,
                filename=chunk.filename,
                chunk_index=0,          # embedding_id is the stable ref
                page=chunk.page if chunk.page else None,
                heading=chunk.heading or None,
                excerpt=chunk.text[:200],
            )
        )
    return citations


def _inject_citation_numbers(answer: str, citations: list[Citation]) -> str:
    """If the LLM referenced [Doc N, p.X] style markers, remap them to
    [1], [2] … indices that align with the citations array.

    Also appends a formatted references block when citations exist.
    """
    if not citations:
        return answer

    # Build a compact references block appended after the answer
    ref_lines = ["", "**References**"]
    for i, cit in enumerate(citations, start=1):
        page_part = f", p.{cit.page}" if cit.page else ""
        ref_lines.append(f"[{i}] {cit.filename}{page_part}")

    return answer + "\n" + "\n".join(ref_lines)


async def run_chat(
    *,
    workspace_id: int,
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    doc_ids: list[int] | None = None,
    embeddings_client: EmbeddingsClient | None = None,
    vector_store: VectorStoreClient | None = None,
    llm_client: LLMClient | None = None,
) -> ChatResponse:
    """Full RAG chat pipeline: retrieve → prompt → generate → respond.

    Args:
        workspace_id:      Scope all retrieval to this workspace.
        question:          The student's question.
        top_k:             Maximum chunks to retrieve.
        score_threshold:   Minimum relevance score for retrieved chunks.
        doc_ids:           Optional document allow-list for scoped chat.
        embeddings_client: Injectable for tests.
        vector_store:      Injectable for tests.
        llm_client:        Injectable for tests.

    Returns:
        ChatResponse with answer text and a citations list.
    """
    emb = embeddings_client or get_embeddings_client()
    vs  = vector_store      or get_vector_store_client()
    llm = llm_client        or get_llm_client()

    # 1. Retrieve relevant chunks
    chunks: list[RetrievedChunk] = await retrieve(
        workspace_id,
        question,
        top_k=top_k,
        embeddings_client=emb,
        vector_store=vs,
        score_threshold=score_threshold,
        doc_ids=doc_ids,
    )
    logger.info(
        "chat workspace_id=%d question=%r → %d chunks retrieved",
        workspace_id, question[:60], len(chunks),
    )

    # 2. Build hardened prompt pair
    built = build_chat_prompt(question, chunks)

    # 3. Generate answer
    raw_answer = await llm.generate(
        built.user_prompt,
        system_prompt=built.system_prompt,
    )

    # 4. Build citations (deduplicated, ordered by retrieval rank)
    citations = _build_citations(chunks)

    # 5. Append reference block to answer
    final_answer = _inject_citation_numbers(raw_answer, citations)

    return ChatResponse(
        answer=final_answer,
        citations=citations,
        model=llm.model_name,
    )