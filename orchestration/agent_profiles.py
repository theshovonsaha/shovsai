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

class AgentProfile(BaseModel):
    id:            str = Field(default_factory=lambda: str(uuid.uuid4()))
    name:          str
    description:   str = ""
    model:         str = "llama3.2"
    embed_model:   str = "nomic-embed-text"
    system_prompt: str = "You are a specialized AI assistant."
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
        """Ensure a 'default' agent always exists for backward compatibility."""
        if not self.get("default"):
            self.create(AgentProfile(
                id="default",
                name="Global Assistant",
                description="The standard all-purpose agent.",
                tools=["web_search", "web_fetch", "image_search", "bash", "file_create", "file_view", "file_str_replace", "weather_fetch", "places_search", "places_map", "store_memory", "query_memory"]
            ))

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
