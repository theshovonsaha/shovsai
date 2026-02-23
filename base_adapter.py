"""
BaseLLMAdapter — Abstract interface for LLM providers
------------------------------------------------------
Swap LLM backends by implementing this interface.
The rest of the system only depends on this ABC.

Implementations:
  - OllamaAdapter (llm_adapter.py) — local Ollama
  - OpenAIAdapter (openai_adapter.py) — OpenAI / Azure
  - GroqLLMAdapter (groq_adapter.py) — Groq cloud
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class LLMError(Exception):
    """Raised when an LLM call fails after retries."""
    pass


class BaseLLMAdapter(ABC):
    """
    Universal LLM adapter interface.
    Internal protocol: list[{role: system|user|assistant, content: str}]
    """

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> str:
        """Non-streaming completion. Returns full response string."""
        ...

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """Streaming completion — yields string tokens."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return available model names."""
        ...

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the provider is reachable."""
        ...
