import math
import os
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embeddings_client import (
    DummyEmbeddingsClient,
    get_embeddings_client,
)


# ---------------------------------------------------------------------------
# Helper — build a fake `openai` module so we never need it installed
# ---------------------------------------------------------------------------

def _make_fake_openai() -> ModuleType:
    """Return a minimal mock of the openai module for patching sys.modules."""
    fake = ModuleType("openai")
    fake.AsyncOpenAI = MagicMock()  # type: ignore[attr-defined]
    return fake


# ---------------------------------------------------------------------------
# DummyEmbeddingsClient
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dummy_embed_one_returns_correct_dimensions() -> None:
    client = DummyEmbeddingsClient(dimensions=16)
    vector = await client.embed_one("hello world")
    assert len(vector) == 16


@pytest.mark.asyncio
async def test_dummy_embed_one_is_deterministic() -> None:
    client = DummyEmbeddingsClient(dimensions=8)
    v1 = await client.embed_one("deterministic input")
    v2 = await client.embed_one("deterministic input")
    assert v1 == v2


@pytest.mark.asyncio
async def test_dummy_different_inputs_produce_different_vectors() -> None:
    client = DummyEmbeddingsClient(dimensions=8)
    v1 = await client.embed_one("text A")
    v2 = await client.embed_one("text B")
    assert v1 != v2


@pytest.mark.asyncio
async def test_dummy_embed_many_returns_one_vector_per_text() -> None:
    client = DummyEmbeddingsClient(dimensions=8)
    texts = ["alpha", "beta", "gamma"]
    vectors = await client.embed_many(texts)
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)


@pytest.mark.asyncio
async def test_dummy_vectors_are_normalised() -> None:
    client = DummyEmbeddingsClient(dimensions=8)
    vector = await client.embed_one("normalisation check")
    assert all(-1.0 <= v <= 1.0 for v in vector)


@pytest.mark.asyncio
async def test_dummy_empty_string_embeds_without_error() -> None:
    client = DummyEmbeddingsClient(dimensions=8)
    vector = await client.embed_one("")
    assert len(vector) == 8


@pytest.mark.asyncio
async def test_dummy_large_dimensions() -> None:
    """Ensure byte-tiling works correctly for dimensions > 32."""
    client = DummyEmbeddingsClient(dimensions=64)
    vector = await client.embed_one("large dim test")
    assert len(vector) == 64


def test_dummy_invalid_dimensions_raises() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        DummyEmbeddingsClient(dimensions=0)


# ---------------------------------------------------------------------------
# validate_vector
# ---------------------------------------------------------------------------

def test_validate_vector_passes_correct_dimension() -> None:
    client = DummyEmbeddingsClient(dimensions=4)
    client.validate_vector([0.1, 0.2, 0.3, 0.4])  # no exception


def test_validate_vector_raises_on_wrong_dimension() -> None:
    client = DummyEmbeddingsClient(dimensions=4)
    with pytest.raises(ValueError, match="dimension"):
        client.validate_vector([0.1, 0.2])


# ---------------------------------------------------------------------------
# OpenAIEmbeddingsClient (fake openai module injected via sys.modules)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_client_embed_one_calls_api() -> None:
    mock_item = MagicMock()
    mock_item.index = 0
    mock_item.embedding = [0.1] * 1536

    mock_response = MagicMock()
    mock_response.data = [mock_item]

    mock_async_openai_instance = MagicMock()
    mock_async_openai_instance.embeddings.create = AsyncMock(return_value=mock_response)

    fake_openai = _make_fake_openai()
    fake_openai.AsyncOpenAI.return_value = mock_async_openai_instance

    env = {"OPENAI_API_KEY": "sk-test", "EMBEDDINGS_PROVIDER": "openai"}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            # Re-import to pick up the patched module
            from app.services.embeddings_client import OpenAIEmbeddingsClient
            client = OpenAIEmbeddingsClient()
            vector = await client.embed_one("test text")

    assert len(vector) == 1536
    mock_async_openai_instance.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_openai_client_validates_returned_dimensions() -> None:
    """If API returns wrong-sized vector, validate_vector should raise."""
    mock_item = MagicMock()
    mock_item.index = 0
    mock_item.embedding = [0.1] * 512  # wrong — expected 1536

    mock_response = MagicMock()
    mock_response.data = [mock_item]

    mock_async_openai_instance = MagicMock()
    mock_async_openai_instance.embeddings.create = AsyncMock(return_value=mock_response)

    fake_openai = _make_fake_openai()
    fake_openai.AsyncOpenAI.return_value = mock_async_openai_instance

    env = {"OPENAI_API_KEY": "sk-test", "EMBEDDING_DIMENSIONS": "1536"}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            from app.services.embeddings_client import OpenAIEmbeddingsClient
            client = OpenAIEmbeddingsClient()
            with pytest.raises(ValueError, match="dimension"):
                await client.embed_one("bad response")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_dummy_when_provider_is_dummy() -> None:
    with patch.dict(os.environ, {"EMBEDDINGS_PROVIDER": "dummy"}, clear=False):
        client = get_embeddings_client()
    assert isinstance(client, DummyEmbeddingsClient)


def test_factory_returns_dummy_when_no_api_key() -> None:
    env = {"EMBEDDINGS_PROVIDER": "", "OPENAI_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        client = get_embeddings_client()
    assert isinstance(client, DummyEmbeddingsClient)


def test_factory_returns_openai_when_key_present() -> None:
    fake_openai = _make_fake_openai()
    env = {"EMBEDDINGS_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            from app.services.embeddings_client import OpenAIEmbeddingsClient
            client = get_embeddings_client()
    assert isinstance(client, OpenAIEmbeddingsClient)


def test_factory_dummy_dimensions_from_env() -> None:
    env = {"EMBEDDINGS_PROVIDER": "dummy", "EMBEDDINGS_DUMMY_DIMENSIONS": "32"}
    with patch.dict(os.environ, env, clear=False):
        client = get_embeddings_client()
    assert client.dimensions == 32