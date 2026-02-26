"""
Agent Profiles Persistence
--------------------------
Manages the storage and lifecycle of Agent configurations.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field

DB_PATH = "agents.db"

PLATINUM_SYSTEM_PROMPT = (
    "You are the 'shovs' V8 Platinum AI. Your mission: Production-grade intelligence with a Luxury-Dark aesthetic.\n\n"
    "--- V8 PLATINUM DIRECTIVES ---\n"
    "1. TRUE BLACK: All HTML/SVG output must use background: #000000. No exceptions.\n"
    "2. ACCENTS: Use electric cyan (#00d1ff) and deep violet (#8b5cf6) for highlights and glow.\n"
    "3. SPA ARCHITECTURE: Every app is a Single-Page Application (SPA). Use vanilla JS to manage sections/tabs dynamically. Do not create static multi-page flows.\n"
    "4. TYPOGRAPHY: Use 'Inter' or 'Roboto Mono' via Google Fonts. Pair with dramatic scale and asymmetric layouts.\n"
    "5. NO SLOP: No placeholders. Use real data, Lucide icons (CDN), and high-quality assets.\n\n"
    "--- BEHAVIORAL STANDARDS ---\n"
    "- CONTEXT: Prioritize 'Historical Context' for persona consistency (e.g. Tony Stark mode).\n"
    "- TOOLS: Output JSON ONLY: {\"tool\": \"...\", \"arguments\": {...}}.\n"
)

class AgentProfile(BaseModel):
    id:            str = Field(default_factory=lambda: str(uuid.uuid4()))
    name:          str
    description:   str = ""
    model:         str = "llama3.2"  # Fallback only
    embed_model:   str = "nomic-embed-text"
    system_prompt: str = PLATINUM_SYSTEM_PROMPT
    tools:         List[str] = Field(default_factory=lambda: ["web_search", "web_fetch"])
    avatar_url:    Optional[str] = None
    created_at:    str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:    str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ProfileManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._ensure_default_agent()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    model TEXT,
                    embed_model TEXT,
                    system_prompt TEXT,
                    tools TEXT,
                    avatar_url TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            # Migration
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN embed_model TEXT DEFAULT 'nomic-embed-text'")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def _ensure_default_agent(self):
        """Ensure a 'default' agent always exists and all generic agents are upgraded."""
        existing_default = self.get("default")
        if not existing_default:
            self.create(AgentProfile(
                id="default",
                name="shovs Platinum Assistant",
                description="The standard high-performance V8 agent.",
                system_prompt=PLATINUM_SYSTEM_PROMPT,
                tools=["web_search", "web_fetch", "image_search", "bash", "file_create", "file_view", "file_str_replace", "weather_fetch", "places_search", "places_map", "store_memory", "query_memory"]
            ))
        
        # Global upgrade for all generic agents
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_profiles")
            for row in cursor.fetchall():
                p = self._row_to_profile(row)
                if p.system_prompt == "You are a specialized AI assistant." or "V8 Platinum" not in p.system_prompt:
                    p.system_prompt = PLATINUM_SYSTEM_PROMPT
                    if p.id == "default": p.name = "shovs Platinum Assistant"
                    self.create(p)
                    print(f"[ProfileManager] Upgraded agent '{p.name}' ({p.id}) to V8 Platinum standards.")

    def create(self, p: AgentProfile) -> AgentProfile:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO agent_profiles
                (id, name, description, model, embed_model, system_prompt, tools, avatar_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                p.id, p.name, p.description, p.model, p.embed_model, p.system_prompt,
                json.dumps(p.tools), p.avatar_url, p.created_at, p.updated_at
            ))
            conn.commit()
        return p

    def get(self, profile_id: str) -> Optional[AgentProfile]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_profiles WHERE id = ?", (profile_id,))
            r = cursor.fetchone()
            if r:
                return self._row_to_profile(r)
        return None

    def list_all(self) -> List[AgentProfile]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_profiles ORDER BY created_at DESC")
            return [self._row_to_profile(r) for r in cursor.fetchall()]

    def delete(self, profile_id: str) -> bool:
        if profile_id == "default": return False # Don't delete the fallback
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM agent_profiles WHERE id = ?", (profile_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_profile(self, r) -> AgentProfile:
        return AgentProfile(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            model=r["model"],
            embed_model=r["embed_model"] if "embed_model" in r.keys() else "nomic-embed-text",
            system_prompt=r["system_prompt"],
            tools=json.loads(r["tools"]),
            avatar_url=r["avatar_url"],
            created_at=r["created_at"],
            updated_at=r["updated_at"]
        )
