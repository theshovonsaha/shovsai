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
from base_adapter import BaseLLMAdapter


def create_adapter(provider: str = None) -> BaseLLMAdapter:
    """
    Create and return the appropriate LLM adapter.
    
    Args:
        provider: Force a specific provider. If None, auto-detect from env.
                  Values: "ollama", "openai", "groq"
    """
    provider = provider or os.getenv("LLM_PROVIDER", "auto")

    if provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        from openai_adapter import OpenAIAdapter
        print("[AdapterFactory] Using OpenAI adapter")
        return OpenAIAdapter()

    if provider == "groq" or (provider == "auto" and os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY")):
        from groq_adapter import GroqLLMAdapter
        print("[AdapterFactory] Using Groq adapter")
        return GroqLLMAdapter()

    # Default: Ollama (local, no key required)
    from llm_adapter import OllamaAdapter
    print("[AdapterFactory] Using Ollama adapter (local)")
    return OllamaAdapter()


def get_default_model(adapter: BaseLLMAdapter) -> str:
    """Return sensible default model for each provider type."""
    from llm_adapter import OllamaAdapter
    from openai_adapter import OpenAIAdapter
    from groq_adapter import GroqLLMAdapter

    if isinstance(adapter, OllamaAdapter):
        return os.getenv("DEFAULT_MODEL", "llama3.2")
    if isinstance(adapter, OpenAIAdapter):
        return os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    if isinstance(adapter, GroqLLMAdapter):
        return os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
    return "llama3.2"
