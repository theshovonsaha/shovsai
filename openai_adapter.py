"""
OpenAI-compatible LLM Adapter
------------------------------
Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API
(Together, Fireworks, local vLLM, etc.)

Requires: pip install openai
Env vars: OPENAI_API_KEY, OPENAI_BASE_URL (optional for custom endpoints)
"""

import asyncio
import os
from typing import AsyncIterator, Optional

from base_adapter import BaseLLMAdapter, LLMError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class OpenAIAdapter(BaseLLMAdapter):

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
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
        msgs = self._prepare_messages(messages, images)
        kwargs = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        last_err: Exception = RuntimeError("no attempts made")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if i < len(RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        raise LLMError(f"OpenAI failed after retries: {last_err}")

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        msgs = self._prepare_messages(messages, images)
        kwargs = {
            "model": model,
            "messages": msgs,
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
            raise LLMError(f"OpenAI stream failed: {e}") from e

    async def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            models = await client.models.list()
            return [m.id for m in models.data if "gpt" in m.id or "o1" in m.id or "o3" in m.id]
        except Exception:
            return ["gpt-4o", "gpt-4o-mini", "o3-mini"]

    async def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False

    def _prepare_messages(self, messages: list[dict], images: Optional[list[str]]) -> list[dict]:
        """Convert internal protocol to OpenAI format, including vision if needed."""
        if not images:
            return messages
        # Attach images to the last user message as content parts
        msgs = [m.copy() for m in messages]
        for msg in reversed(msgs):
            if msg["role"] == "user":
                content_parts = [{"type": "text", "text": msg["content"]}]
                for img_b64 in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                msg["content"] = content_parts
                break
        return msgs
