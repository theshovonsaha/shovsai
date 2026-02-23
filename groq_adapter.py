"""
Groq LLM Adapter
-----------------
Fast inference via Groq cloud API.
Uses the Groq Python SDK for chat completions (NOT compound-beta — 
that's used for web search in tools_web.py, not as a general LLM).

Requires: pip install groq  (already in requirements.txt)
Env vars: GROQ_API_KEY
"""

import asyncio
import os
from typing import AsyncIterator, Optional

from base_adapter import BaseLLMAdapter, LLMError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class GroqLLMAdapter(BaseLLMAdapter):

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self.api_key, timeout=self.timeout)
        return self._client

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> str:
        client = self._get_client()
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        last_err: Exception = RuntimeError("no attempts")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if i < len(RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        raise LLMError(f"Groq failed after retries: {last_err}")

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            raise LLMError(f"Groq stream failed: {e}") from e

    async def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            client = self._get_client()
            models = await client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    async def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False
