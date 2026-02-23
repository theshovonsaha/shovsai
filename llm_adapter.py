"""
Universal LLM Adapter — Ollama
-------------------------------
All is language and protocols.
Translates the internal message protocol → Ollama API.
To swap providers: rewrite only this file.

Internal protocol: list[{role: system|user|assistant, content: str}]
"""

import httpx
import json
import asyncio
from typing import AsyncIterator, Optional

from base_adapter import BaseLLMAdapter, LLMError  # re-export LLMError for back-compat


OLLAMA_BASE    = "http://localhost:11434"
RETRY_DELAYS   = [0.5, 1.5, 3.0]


class OllamaAdapter(BaseLLMAdapter):


    def __init__(self, base_url: str = OLLAMA_BASE, timeout: float = 120.0):
        self.base_url = base_url
        self.timeout  = timeout

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,  # base64 strings for vision
    ) -> str:
        """Non-streaming completion with retry. Returns full response string."""
        payload = self._payload(model, messages, temperature, max_tokens, stream=False, images=images)
        last_err: Exception = RuntimeError("no attempts made")

        for i, delay in enumerate(RETRY_DELAYS):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    return resp.json()["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise LLMError(f"LLM rejected: {e.response.status_code}") from e
                last_err = e
            except Exception as e:
                last_err = e

            if i < len(RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)

        raise LLMError(f"LLM failed after retries: {last_err}")

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,  # base64 strings for vision
    ) -> AsyncIterator[str]:
        """Streaming completion — yields string tokens."""
        payload = self._payload(model, messages, temperature, max_tokens, stream=True, images=images)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        
                        if token := chunk.get("message", {}).get("content", ""):
                            yield token
                        if chunk.get("done"):
                            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise LLMError(f"LLM rejected stream: {e.response.status_code}") from e
            raise LLMError(f"LLM stream failed: {e}") from e
        except Exception as e:
             raise LLMError(f"LLM stream connection failed: {e}") from e

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def _payload(self, model, messages, temperature, max_tokens, stream, images=None):
        p: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            p["options"]["num_predict"] = max_tokens
        # Ollama vision: attach images to the last user message
        if images:
            for msg in reversed(p["messages"]):
                if msg["role"] == "user":
                    msg["images"] = images
                    break
        return p
