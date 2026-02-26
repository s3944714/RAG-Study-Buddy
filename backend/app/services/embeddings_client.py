import hashlib
import logging
import math
import os
import struct
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class EmbeddingsClient(ABC):
    """Provider-agnostic interface for generating text embeddings."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The fixed vector size this client produces."""

    @abstractmethod
    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Returns a float vector of length self.dimensions."""

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings. Default: calls embed_one in sequence.

        Concrete implementations should override this with batched API calls.
        """
        return [await self.embed_one(t) for t in texts]

    def validate_vector(self, vector: list[float], *, source: str = "") -> None:
        """Raise ValueError if vector length doesn't match self.dimensions."""
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected vector of dimension {self.dimensions}, "
                f"got {len(vector)}{f' from {source}' if source else ''}"
            )


# ---------------------------------------------------------------------------
# DummyEmbeddings — deterministic, no network, safe for tests and CI
# ---------------------------------------------------------------------------

class DummyEmbeddingsClient(EmbeddingsClient):
    """Deterministic hash-based embeddings — useful for unit tests and CI.

    Produces a stable float vector from the SHA-256 of the input text,
    so identical inputs always produce identical vectors.
    No API key or network call required.
    """

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()  # always 32 bytes

        # Calculate exactly how many bytes we need for `dimensions` floats
        # (each float is 4 bytes), then tile the digest to cover that length.
        bytes_needed = self._dimensions * 4
        repeats = math.ceil(bytes_needed / len(digest))
        repeated = (digest * repeats)[:bytes_needed]

        raw = list(struct.unpack(f"{self._dimensions}f", repeated))
        max_abs = max(abs(v) for v in raw) or 1.0
        normalised = [v / max_abs for v in raw]

        self.validate_vector(normalised)
        return normalised


# ---------------------------------------------------------------------------
# OpenAI-compatible real client (lazy import — openai is optional)
# ---------------------------------------------------------------------------

class OpenAIEmbeddingsClient(EmbeddingsClient):
    """Real embeddings via the OpenAI API (or any compatible endpoint).

    Config via env vars — no secrets in code:
        OPENAI_API_KEY          required
        EMBEDDING_MODEL         default: text-embedding-3-small
        EMBEDDING_BASE_URL      optional override (e.g. Azure, local proxy)
        EMBEDDING_DIMENSIONS    optional int (model default used if not set)
    """

    def __init__(self) -> None:
        try:
            from openai import AsyncOpenAI  # lazy import — openai is optional
            self._AsyncOpenAI = AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingsClient. "
                "Run: poetry add openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("EMBEDDING_BASE_URL") or None
        self._model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = self._AsyncOpenAI(api_key=api_key, base_url=base_url)

        raw_dim = os.environ.get("EMBEDDING_DIMENSIONS")
        self._dimensions: int = int(raw_dim) if raw_dim else 1536

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_batch(texts)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": self._model, "input": texts}
        if os.environ.get("EMBEDDING_DIMENSIONS"):
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)
        vectors = [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]

        for i, v in enumerate(vectors):
            self.validate_vector(v, source=f"item[{i}]")

        logger.debug("Embedded %d texts via %s", len(texts), self._model)
        return vectors


# ---------------------------------------------------------------------------
# Factory — resolves which client to use from env
# ---------------------------------------------------------------------------

def get_embeddings_client() -> EmbeddingsClient:
    """Return the appropriate embeddings client based on env vars.

    EMBEDDINGS_PROVIDER controls which backend is used:
        "dummy"  → DummyEmbeddingsClient (tests / offline dev)
        "openai" → OpenAIEmbeddingsClient (default when OPENAI_API_KEY is set)

    EMBEDDINGS_DUMMY_DIMENSIONS controls DummyEmbeddingsClient vector size.
    """
    provider = os.environ.get("EMBEDDINGS_PROVIDER", "").lower()

    if provider == "dummy" or (not provider and not os.environ.get("OPENAI_API_KEY")):
        dim = int(os.environ.get("EMBEDDINGS_DUMMY_DIMENSIONS", "8"))
        logger.info("Using DummyEmbeddingsClient (dimensions=%d)", dim)
        return DummyEmbeddingsClient(dimensions=dim)

    logger.info(
        "Using OpenAIEmbeddingsClient (model=%s)",
        os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    return OpenAIEmbeddingsClient()