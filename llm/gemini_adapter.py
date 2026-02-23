"""
Google Gemini LLM Adapter
-------------------------
Integrates Google's Generative AI models (Gemini 1.5 Flash/Pro).

Requires: pip install google-generativeai
Env vars: GEMINI_API_KEY
"""

import asyncio
import os
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter, LLMError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class GeminiAdapter(BaseLLMAdapter):

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._genai = None

    def _get_genai(self):
        if self._genai is None:
            if not self.api_key:
                raise LLMError("GEMINI_API_KEY not found in environment.")
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai = genai
            except ImportError:
                raise LLMError("google-generativeai not installed. Run: pip install google-generativeai")
        return self._genai

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> str:
        genai = self._get_genai()
        gemini_model = genai.GenerativeModel(model)
        
        # Convert internal protocol to Gemini format
        history, last_msg = self._convert_messages(messages)
        
        chat = gemini_model.start_chat(history=history)
        
        last_err: Exception = RuntimeError("no attempts made")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                # Gemini doesn't have a direct async 'send_message' in the simple chat interface
                # for non-streaming in some SDK versions, but we'll use generate_content for simplicity
                # if images are present or just use send_message.
                response = await asyncio.to_thread(
                    chat.send_message,
                    last_msg,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
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
        genai = self._get_genai()
        
        # Note: Gemini 1.5 models are multimodal by default.
        # If images are provided, we should probably handle them in generate_content
        gemini_model = genai.GenerativeModel(model)
        history, last_msg = self._convert_messages(messages)
        chat = gemini_model.start_chat(history=history)

        try:
            # We use to_thread because the current SDK's send_message is synchronous
            response = await asyncio.to_thread(
                chat.send_message,
                last_msg,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(f"Gemini stream failed: {e}") from e

    async def list_models(self) -> list[str]:
        try:
            genai = self._get_genai()
            models = await asyncio.to_thread(genai.list_models)
            return [m.name.replace("models/", "") for m in models if "generateContent" in m.supported_generation_methods]
        except Exception:
            return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]

    async def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            genai = self._get_genai()
            # Simple list_models test or a small generation could work for health check
            await asyncio.to_thread(genai.list_models)
            return True
        except Exception:
            return False

    def _convert_messages(self, messages: list[dict]):
        """
        Convert internal {role, content} to Gemini {role, parts} history.
        Gemini roles: 'user', 'model'.
        Returns (history, last_message_string).
        """
        gemini_history = []
        # Gemini history shouldn't include the 'system' instruction in the same way,
        # but GenerativeModel supports system_instruction in constructor usually.
        # For simplicity and compatibility with our loop, we fold system into user/model if needed,
        # or just pass it as a user message if it's the first.
        
        system_instruction = ""
        msgs_to_process = messages.copy()
        
        if msgs_to_process and msgs_to_process[0]["role"] == "system":
            system_instruction = msgs_to_process.pop(0)["content"]

        for msg in msgs_to_process[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        
        last_msg = msgs_to_process[-1]["content"]
        if system_instruction:
            # Prepend system instruction to the last message or first user message
            # Better: many recent Gemini versions support system_instruction in GenerativeModel
            pass

        return gemini_history, last_msg
