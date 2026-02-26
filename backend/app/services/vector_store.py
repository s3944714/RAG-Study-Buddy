import logging
import os
from abc import ABC, abstractmethod
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

class VectorDocument:
    """A single unit to upsert or retrieve from the vector store."""

    __slots__ = ("id", "embedding", "content", "metadata")

    def __init__(
        self,
        embedding: list[float],
        content: str,
        metadata: dict | None = None,
        id: str | None = None,
    ) -> None:
        self.id: str = id or str(uuid4())
        self.embedding = embedding
        self.content = content
        self.metadata: dict = metadata or {}


class QueryResult:
    """A single result returned from a similarity search."""

    __slots__ = ("id", "content", "metadata", "score")

    def __init__(
        self,
        id: str,
        content: str,
        metadata: dict,
        score: float,
    ) -> None:
        self.id = id
        self.content = content
        self.metadata = metadata
        self.score = score  # lower = more similar for L2; higher for cosine


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class VectorStoreClient(ABC):
    """Provider-agnostic interface for a vector store."""

    @abstractmethod
    async def upsert(self, collection: str, documents: list[VectorDocument]) -> list[str]:
        """Insert or update documents. Returns the list of IDs stored."""

    @abstractmethod
    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[QueryResult]:
        """Return the top_k most similar documents.

        Args:
            collection:      Name of the Chroma collection (= workspace scope).
            query_embedding: The query vector.
            top_k:           Maximum number of results.
            where:           Optional Chroma metadata filter dict.
        """

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> None:
        """Delete documents by ID from the collection."""

    @abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Drop an entire collection (e.g. when a workspace is removed)."""


# ---------------------------------------------------------------------------
# InMemoryVectorStore — deterministic, no deps, for tests and offline dev
# ---------------------------------------------------------------------------

class InMemoryVectorStoreClient(VectorStoreClient):
    """Simple in-memory vector store using cosine similarity.

    No Chroma dependency required. Safe for unit tests and CI.
    """

    def __init__(self) -> None:
        # { collection_name: { id: VectorDocument } }
        self._store: dict[str, dict[str, VectorDocument]] = {}

    # ------------------------------------------------------------------
    def _collection(self, name: str) -> dict[str, VectorDocument]:
        return self._store.setdefault(name, {})

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    async def upsert(self, collection: str, documents: list[VectorDocument]) -> list[str]:
        col = self._collection(collection)
        for doc in documents:
            col[doc.id] = doc
        logger.debug("InMemory upsert: %d docs into %r", len(documents), collection)
        return [doc.id for doc in documents]

    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[QueryResult]:
        col = self._collection(collection)
        scored: list[tuple[float, VectorDocument]] = []

        for doc in col.values():
            # Apply simple equality metadata filter if provided
            if where:
                if not all(doc.metadata.get(k) == v for k, v in where.items()):
                    continue
            score = self._cosine(query_embedding, doc.embedding)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            QueryResult(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=score,
            )
            for score, doc in scored[:top_k]
        ]

    async def delete(self, collection: str, ids: list[str]) -> None:
        col = self._collection(collection)
        for id_ in ids:
            col.pop(id_, None)
        logger.debug("InMemory delete: %d ids from %r", len(ids), collection)

    async def delete_collection(self, collection: str) -> None:
        self._store.pop(collection, None)
        logger.debug("InMemory delete_collection: %r", collection)


# ---------------------------------------------------------------------------
# ChromaVectorStoreClient — real persistent store
# ---------------------------------------------------------------------------

class ChromaVectorStoreClient(VectorStoreClient):
    """Vector store backed by ChromaDB (HTTP client mode).

    Config via env vars — no secrets in code:
        CHROMA_HOST     default: localhost
        CHROMA_PORT     default: 8000
        CHROMA_TENANT   optional (Chroma Cloud)
        CHROMA_DATABASE optional (Chroma Cloud)
    """

    def __init__(self) -> None:
        try:
            import chromadb  # lazy import — chromadb is optional
        except ImportError as exc:
            raise ImportError(
                "chromadb package is required for ChromaVectorStoreClient. "
                "Run: poetry add chromadb"
            ) from exc

        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        tenant = os.environ.get("CHROMA_TENANT")
        database = os.environ.get("CHROMA_DATABASE")

        kwargs: dict = {"host": host, "port": port}
        if tenant:
            kwargs["tenant"] = tenant
        if database:
            kwargs["database"] = database

        self._client = chromadb.HttpClient(**kwargs)
        logger.info("ChromaVectorStoreClient connected to %s:%s", host, port)

    def _get_or_create(self, collection: str):  # returns chroma Collection
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, collection: str, documents: list[VectorDocument]) -> list[str]:
        col = self._get_or_create(collection)
        col.upsert(
            ids=[d.id for d in documents],
            embeddings=[d.embedding for d in documents],
            documents=[d.content for d in documents],
            metadatas=[d.metadata for d in documents],
        )
        logger.debug("Chroma upsert: %d docs into %r", len(documents), collection)
        return [d.id for d in documents]

    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[QueryResult]:
        col = self._get_or_create(collection)
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        output: list[QueryResult] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity score in [0, 1].
            score = 1.0 - (dist / 2.0)
            output.append(QueryResult(id=id_, content=doc, metadata=meta, score=score))

        return output

    async def delete(self, collection: str, ids: list[str]) -> None:
        col = self._get_or_create(collection)
        col.delete(ids=ids)
        logger.debug("Chroma delete: %d ids from %r", len(ids), collection)

    async def delete_collection(self, collection: str) -> None:
        self._client.delete_collection(collection)
        logger.debug("Chroma delete_collection: %r", collection)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store_client() -> VectorStoreClient:
    """Return the appropriate vector store client based on env vars.

    VECTOR_STORE_PROVIDER controls which backend is used:
        "chroma"   → ChromaVectorStoreClient (requires chromadb + running server)
        "memory"   → InMemoryVectorStoreClient (default; no deps)
    """
    provider = os.environ.get("VECTOR_STORE_PROVIDER", "memory").lower()

    if provider == "chroma":
        logger.info("Using ChromaVectorStoreClient")
        return ChromaVectorStoreClient()

    logger.info("Using InMemoryVectorStoreClient")
    return InMemoryVectorStoreClient()