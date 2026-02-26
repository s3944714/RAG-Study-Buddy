import os
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_client import DummyLLMClient, get_llm_client


# ---------------------------------------------------------------------------
# Helpers — fake SDK modules so real packages are never required
# ---------------------------------------------------------------------------

def _make_fake_openai(response_text: str = "openai response") -> ModuleType:
    fake = ModuleType("openai")

    mock_message  = MagicMock()
    mock_message.content = response_text

    mock_choice   = MagicMock()
    mock_choice.message = mock_message

    mock_usage    = MagicMock()
    mock_usage.total_tokens = 42

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage   = mock_usage

    mock_instance = MagicMock()
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)

    fake.AsyncOpenAI = MagicMock(return_value=mock_instance)  # type: ignore[attr-defined]
    return fake


def _make_fake_anthropic(response_text: str = "anthropic response") -> ModuleType:
    fake = ModuleType("anthropic")

    mock_content_block     = MagicMock()
    mock_content_block.text = response_text

    mock_response          = MagicMock()
    mock_response.content  = [mock_content_block]
    mock_response.stop_reason = "end_turn"

    mock_instance          = MagicMock()
    mock_instance.messages.create = AsyncMock(return_value=mock_response)

    fake.AsyncAnthropic = MagicMock(return_value=mock_instance)  # type: ignore[attr-defined]
    return fake


# ---------------------------------------------------------------------------
# DummyLLMClient
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dummy_generate_returns_string() -> None:
    client = DummyLLMClient()
    result = await client.generate("What is gravity?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_dummy_generate_echoes_prompt() -> None:
    client = DummyLLMClient()
    result = await client.generate("Tell me about photosynthesis")
    assert "Tell me about photosynthesis" in result


@pytest.mark.asyncio
async def test_dummy_generate_includes_system_prompt() -> None:
    client = DummyLLMClient()
    result = await client.generate("hello", system_prompt="You are a pirate")
    assert "You are a pirate" in result


@pytest.mark.asyncio
async def test_dummy_generate_no_system_prompt() -> None:
    client = DummyLLMClient()
    result = await client.generate("hello")
    assert "[system:" not in result
    assert "hello" in result


@pytest.mark.asyncio
async def test_dummy_generate_empty_prompt() -> None:
    client = DummyLLMClient()
    result = await client.generate("")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_dummy_generate_is_deterministic() -> None:
    client = DummyLLMClient()
    r1 = await client.generate("same input", system_prompt="same system")
    r2 = await client.generate("same input", system_prompt="same system")
    assert r1 == r2


def test_dummy_model_name() -> None:
    client = DummyLLMClient()
    assert isinstance(client.model_name, str)
    assert len(client.model_name) > 0


# ---------------------------------------------------------------------------
# OpenAILLMClient (injected via sys.modules — never needs openai installed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_generate_returns_response_text() -> None:
    fake_openai = _make_fake_openai("The answer is 42.")
    env = {"OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "openai"}

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            from app.services.llm_client import OpenAILLMClient
            client = OpenAILLMClient()
            result = await client.generate("What is the answer?")

    assert result == "The answer is 42."


@pytest.mark.asyncio
async def test_openai_generate_sends_system_prompt() -> None:
    fake_openai = _make_fake_openai("ok")
    env = {"OPENAI_API_KEY": "sk-test"}

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            from app.services.llm_client import OpenAILLMClient
            client = OpenAILLMClient()
            await client.generate("hi", system_prompt="Be brief")

    # Extract the messages passed to the create call
    create_mock = fake_openai.AsyncOpenAI.return_value.chat.completions.create
    call_kwargs = create_mock.call_args.kwargs
    messages = call_kwargs["messages"]
    system_messages = [m for m in messages if m["role"] == "system"]

    assert len(system_messages) == 1
    assert system_messages[0]["content"] == "Be brief"


@pytest.mark.asyncio
async def test_openai_generate_uses_default_system_when_none() -> None:
    fake_openai = _make_fake_openai("ok")
    env = {"OPENAI_API_KEY": "sk-test"}

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            from app.services.llm_client import OpenAILLMClient
            client = OpenAILLMClient()
            await client.generate("hi", system_prompt=None)

    create_mock = fake_openai.AsyncOpenAI.return_value.chat.completions.create
    messages = create_mock.call_args.kwargs["messages"]
    system_content = next(m["content"] for m in messages if m["role"] == "system")
    assert len(system_content) > 0  # a non-empty default was used


def test_openai_model_name_from_env() -> None:
    fake_openai = _make_fake_openai()
    env = {"OPENAI_API_KEY": "sk-test", "LLM_MODEL": "gpt-4o"}

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            from app.services.llm_client import OpenAILLMClient
            client = OpenAILLMClient()

    assert client.model_name == "gpt-4o"


def test_openai_missing_package_raises_import_error() -> None:
    env = {"OPENAI_API_KEY": "sk-test"}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": None}):  # type: ignore[dict-item]
            # Force re-evaluation of the import inside __init__
            with pytest.raises((ImportError, TypeError)):
                from app.services.llm_client import OpenAILLMClient
                OpenAILLMClient()


# ---------------------------------------------------------------------------
# AnthropicLLMClient (injected via sys.modules)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_generate_returns_response_text() -> None:
    fake_anthropic = _make_fake_anthropic("Photosynthesis converts light to energy.")
    env = {"ANTHROPIC_API_KEY": "sk-ant-test", "LLM_PROVIDER": "anthropic"}

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            from app.services.llm_client import AnthropicLLMClient
            client = AnthropicLLMClient()
            result = await client.generate("Explain photosynthesis")

    assert result == "Photosynthesis converts light to energy."


@pytest.mark.asyncio
async def test_anthropic_generate_passes_system_prompt() -> None:
    fake_anthropic = _make_fake_anthropic("ok")
    env = {"ANTHROPIC_API_KEY": "sk-ant-test"}

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            from app.services.llm_client import AnthropicLLMClient
            client = AnthropicLLMClient()
            await client.generate("hello", system_prompt="Be concise")

    create_mock = fake_anthropic.AsyncAnthropic.return_value.messages.create
    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs["system"] == "Be concise"


@pytest.mark.asyncio
async def test_anthropic_empty_content_returns_empty_string() -> None:
    fake_anthropic = _make_fake_anthropic("ignored")
    # Override: response.content is empty list
    mock_response = MagicMock()
    mock_response.content = []
    mock_response.stop_reason = "end_turn"
    fake_anthropic.AsyncAnthropic.return_value.messages.create = AsyncMock(
        return_value=mock_response
    )

    env = {"ANTHROPIC_API_KEY": "sk-ant-test", "LLM_PROVIDER": "anthropic"}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            from app.services.llm_client import AnthropicLLMClient
            client = AnthropicLLMClient()
            result = await client.generate("hello")

    assert result == ""


def test_anthropic_model_name_from_env() -> None:
    fake_anthropic = _make_fake_anthropic()
    env = {"ANTHROPIC_API_KEY": "sk-ant-test", "LLM_MODEL": "claude-opus-4-6"}

    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            from app.services.llm_client import AnthropicLLMClient
            client = AnthropicLLMClient()

    assert client.model_name == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_dummy_when_provider_is_dummy() -> None:
    with patch.dict(os.environ, {"LLM_PROVIDER": "dummy"}, clear=False):
        client = get_llm_client()
    assert isinstance(client, DummyLLMClient)


def test_factory_returns_dummy_when_no_keys_configured() -> None:
    clean = {k: v for k, v in os.environ.items()
             if k not in ("LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    with patch.dict(os.environ, clean, clear=True):
        client = get_llm_client()
    assert isinstance(client, DummyLLMClient)


def test_factory_returns_openai_when_provider_set() -> None:
    fake_openai = _make_fake_openai()
    env = {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai}):
            from app.services.llm_client import OpenAILLMClient
            client = get_llm_client()
    assert isinstance(client, OpenAILLMClient)


def test_factory_returns_anthropic_when_provider_set() -> None:
    fake_anthropic = _make_fake_anthropic()
    env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-ant-test"}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            from app.services.llm_client import AnthropicLLMClient
            client = get_llm_client()
    assert isinstance(client, AnthropicLLMClient)


def test_factory_prefers_openai_when_both_keys_set_and_no_provider() -> None:
    fake_openai    = _make_fake_openai()
    fake_anthropic = _make_fake_anthropic()
    env = {
        "LLM_PROVIDER":      "",
        "OPENAI_API_KEY":    "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"openai": fake_openai, "anthropic": fake_anthropic}):
            from app.services.llm_client import OpenAILLMClient
            client = get_llm_client()
    assert isinstance(client, OpenAILLMClient)