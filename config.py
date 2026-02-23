"""
Config — Centralized configuration
------------------------------------
Single source of truth for all platform settings.
Reads from environment variables with sensible defaults.

Usage:
    from config import cfg
    print(cfg.DEFAULT_MODEL)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")  # auto|ollama|openai|groq
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama3.2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Search Backends ───────────────────────────────────────────────────
    SEARXNG_URL: str = os.getenv("SEARXNG_URL", "")
    BRAVE_SEARCH_KEY: str = os.getenv("BRAVE_SEARCH_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ── Memory ────────────────────────────────────────────────────────────
    SLIDING_WINDOW_SIZE: int = int(os.getenv("SLIDING_WINDOW_SIZE", "20"))
    MAX_CONTEXT_LINES: int = int(os.getenv("MAX_CONTEXT_LINES", "80"))
    MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "200"))

    # ── Database ──────────────────────────────────────────────────────────
    SESSIONS_DB: str = os.getenv("SESSIONS_DB", "sessions.db")
    AGENTS_DB: str = os.getenv("AGENTS_DB", "agents.db")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # ── Tools ─────────────────────────────────────────────────────────────
    SANDBOX_DIR: str = os.getenv("SANDBOX_DIR", "./agent_sandbox")
    BASH_TIMEOUT: int = int(os.getenv("BASH_TIMEOUT", "30"))
    GOOGLE_PLACES_API_KEY: str = os.getenv("GOOGLE_PLACES_API_KEY", "")

    # ── Voice ─────────────────────────────────────────────────────────────
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "auto")  # auto|kokoro|edge-tts|piper

    # ── Server ────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


# Singleton
cfg = Config()
