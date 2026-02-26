import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Provider-agnostic interface for text generation."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response for *prompt*.

        Args:
            prompt:        The user-facing message / question.
            system_prompt: Optional system-level instruction. If None the
                           concrete implementation uses its own default.

        Returns:
            The model's response as a plain string.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier of the underlying model."""


# ---------------------------------------------------------------------------
# DummyLLMClient — deterministic, no network, safe for tests and CI
# ---------------------------------------------------------------------------

class DummyLLMClient(LLMClient):
    """Echo-based LLM stub for unit tests and offline development.

    Returns a predictable string that embeds both the system prompt and the
    user prompt so tests can assert on content without a real API call.
    """

    _MODEL = "dummy-echo-v1"

    @property
    def model_name(self) -> str:
        return self._MODEL

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        sys_part = f"[system: {system_prompt}] " if system_prompt else ""
        response = f"{sys_part}[echo] {prompt}"
        logger.debug("DummyLLMClient.generate → %r", response[:120])
        return response


# ---------------------------------------------------------------------------
# OpenAILLMClient — real chat completions (lazy import)
# ---------------------------------------------------------------------------

class OpenAILLMClient(LLMClient):
    """Chat completions via the OpenAI API (or any compatible endpoint).

    Config via env vars — no secrets in code:
        OPENAI_API_KEY          required
        LLM_MODEL               default: gpt-4o-mini
        LLM_BASE_URL            optional override (Azure, local proxy, etc.)
        LLM_MAX_TOKENS          default: 1024
        LLM_TEMPERATURE         default: 0.2
        LLM_DEFAULT_SYSTEM      default system prompt used when caller passes None
    """

    _DEFAULT_SYSTEM = (
        "You are a helpful study assistant. "
        "Answer questions clearly and concisely based on the provided context. "
        "If the context does not contain enough information, say so honestly."
    )

    def __init__(self) -> None:
        try:
            from openai import AsyncOpenAI  # lazy — openai is optional
            self._AsyncOpenAI = AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILLMClient. "
                "Run: poetry add openai"
            ) from exc

        api_key  = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL") or None

        self._model       = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self._max_tokens  = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
        self._temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
        self._default_sys = os.environ.get("LLM_DEFAULT_SYSTEM", self._DEFAULT_SYSTEM)

        self._client = self._AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        system = system_prompt if system_prompt is not None else self._default_sys

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        text = response.choices[0].message.content or ""
        logger.debug(
            "OpenAILLMClient.generate model=%s tokens_used=%s",
            self._model,
            getattr(response.usage, "total_tokens", "?"),
        )
        return text


# ---------------------------------------------------------------------------
# AnthropicLLMClient — real messages API (lazy import)
# ---------------------------------------------------------------------------

class AnthropicLLMClient(LLMClient):
    """Messages API via the Anthropic SDK.

    Config via env vars:
        ANTHROPIC_API_KEY       required
        LLM_MODEL               default: claude-haiku-4-5-20251001
        LLM_MAX_TOKENS          default: 1024
        LLM_DEFAULT_SYSTEM      optional system prompt override
    """

    _DEFAULT_SYSTEM = (
        "You are a helpful study assistant. "
        "Answer questions clearly and concisely based on the provided context. "
        "If the context does not contain enough information, say so honestly."
    )

    def __init__(self) -> None:
        try:
            import anthropic  # lazy — anthropic is optional
            self._anthropic = anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicLLMClient. "
                "Run: poetry add anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._model      = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
        self._max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
        self._default_sys = os.environ.get("LLM_DEFAULT_SYSTEM", self._DEFAULT_SYSTEM)
        self._client = self._anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        system = system_prompt if system_prompt is not None else self._default_sys

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text if response.content else ""
        logger.debug(
            "AnthropicLLMClient.generate model=%s stop_reason=%s",
            self._model,
            response.stop_reason,
        )
        return text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client() -> LLMClient:
    """Return the appropriate LLM client based on env vars.

    LLM_PROVIDER controls which backend is used:
        "dummy"     → DummyLLMClient      (no API key needed)
        "anthropic" → AnthropicLLMClient  (requires ANTHROPIC_API_KEY)
        "openai"    → OpenAILLMClient     (default when OPENAI_API_KEY is set)

    Falls back to DummyLLMClient when no provider or key is configured.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "dummy":
        logger.info("Using DummyLLMClient")
        return DummyLLMClient()

    if provider == "anthropic" or (
        not provider and os.environ.get("ANTHROPIC_API_KEY")
        and not os.environ.get("OPENAI_API_KEY")
    ):
        logger.info("Using AnthropicLLMClient (model=%s)", os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001"))
        return AnthropicLLMClient()

    if provider == "openai" or (not provider and os.environ.get("OPENAI_API_KEY")):
        logger.info("Using OpenAILLMClient (model=%s)", os.environ.get("LLM_MODEL", "gpt-4o-mini"))
        return OpenAILLMClient()

    # No provider and no keys → safe fallback
    logger.warning("No LLM_PROVIDER or API key configured — falling back to DummyLLMClient")
    return DummyLLMClient()