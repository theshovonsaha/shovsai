"""
Google Gemini LLM Adapter
-------------------------
Integrates Google's Generative AI models (Gemini 1.5 Flash/Pro).

Requires: pip install google-genai
Env vars: GEMINI_API_KEY
"""

import asyncio
import os
from typing import AsyncIterator, Optional, Any

from llm.base_adapter import BaseLLMAdapter, LLMError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class GeminiAdapter(BaseLLMAdapter):

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMError("GEMINI_API_KEY not found in environment.")
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise LLMError("google-genai not installed. Run: pip install google-genai")
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
        
        contents = self._convert_messages(messages)
        
        last_err: Exception = RuntimeError("no attempts made")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config={
                        'temperature': temperature,
                        'max_output_tokens': max_tokens
                    }
                )
                return response.text
            except Exception as e:
                last_err = e
                if i < len(RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        raise LLMError(f"Gemini failed after retries: {last_err}")

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        contents = self._convert_messages(messages)

        try:
            # google-genai stream is also synchronous-iterative in many parts,
            # but we can wrap the generation call in a thread if needed,
            # or just iterate since it's a generator.
            response = await asyncio.to_thread(
                client.models.generate_content_stream,
                model=model,
                contents=contents,
                config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens
                }
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(f"Gemini stream failed: {e}") from e

    async def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            models_resp = await asyncio.to_thread(client.models.list)
            # The resp is an iterator/list of model objects
            return [m.name for m in models_resp]
        except Exception:
            return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]

    async def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            # Simple list metadata to check connection
            await asyncio.to_thread(client.models.list, config={'page_size': 1})
            return True
        except Exception:
            return False

    def _convert_messages(self, messages: list[dict]) -> list[Any]:
        """
        Convert internal {role, content} to google-genai contents format.
        Gemini roles: 'user', 'model'. System messages map to 'user'.
        CRITICAL: Gemini rejects consecutive same-role messages, so we merge them.
        """
        from google.genai import types
        
        raw_messages = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            raw_messages.append({"role": role, "content": msg["content"]})
        
        # Merge consecutive same-role messages
        merged = []
        for msg in raw_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append({"role": msg["role"], "content": msg["content"]})
        
        genai_contents = []
        for msg in merged:
            genai_contents.append(
                types.Content(
                    role=msg["role"],
                    parts=[types.Part(text=msg["content"])]
                )
            )
        
        return genai_contents
