"""
Adapter Factory
----------------
Instantiates the correct LLM adapter based on environment configuration.

Priority order (first configured wins):
  1. OLLAMA_BASE_URL set or Ollama reachable → OllamaAdapter (default)
  2. OPENAI_API_KEY set → OpenAIAdapter
  3. GROQ_API_KEY set → GroqLLMAdapter

Override with: LLM_PROVIDER=ollama|openai|groq
"""

import os
from llm.base_adapter import BaseLLMAdapter


def create_adapter(provider: str = None) -> BaseLLMAdapter:
    """
    Create and return the appropriate LLM adapter.
    """
    if provider and ":" in provider:
        p_part = provider.split(":")[0].lower()
        if p_part in ["ollama", "openai", "groq"]:
            provider = p_part

    # Ensure provider is a string and not None for further comparisons
    provider_str: str = provider or os.getenv("LLM_PROVIDER", "auto")

    if provider_str == "openai" or (provider_str == "auto" and os.getenv("OPENAI_API_KEY")):
        from llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter()

    if provider_str == "groq" or (provider_str == "auto" and os.getenv("GROQ_API_KEY")):
        from llm.groq_adapter import GroqLLMAdapter
        return GroqLLMAdapter()

    # Default: Ollama (local, no key required)
    from llm.llm_adapter import OllamaAdapter
    return OllamaAdapter()


def strip_provider_prefix(model_name: str) -> str:
    """
    Removes the "provider:" prefix if present.
    Example: "groq:llama-..." -> "llama-..."
    """
    if not model_name or ":" not in model_name:
        return model_name
    
    parts = model_name.split(":", 1)
    if parts[0].lower() in ["ollama", "openai", "groq"]:
        return parts[1]
    return model_name


def get_default_model(adapter: BaseLLMAdapter) -> str:
    """Return sensible default model for each provider type."""
    from llm.llm_adapter import OllamaAdapter
    from llm.openai_adapter import OpenAIAdapter
    from llm.groq_adapter import GroqLLMAdapter

    if isinstance(adapter, OllamaAdapter):
        return os.getenv("DEFAULT_MODEL", "llama3.2")
    if isinstance(adapter, OpenAIAdapter):
        return os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    if isinstance(adapter, GroqLLMAdapter):
        return os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
    return "llama3.2"
