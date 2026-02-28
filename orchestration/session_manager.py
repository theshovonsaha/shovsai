"""
Session Manager (FIXED)
-----------------------
BUG FIXED: SLIDING_WINDOW_SIZE was 6 (3 turns). After 4+ exchanges,
the first message fell completely out of the window AND was never
compressed (trivial skip). Model had zero access to early messages
and hallucinated answers to "what was my first message?".

FIXES:
  1. SLIDING_WINDOW_SIZE raised from 6 → 20 (10 turns in view at all times)
  2. Session stores first_message explicitly — always available regardless
     of window size or compression. Never lost.
  3. append_message now passes is_first_exchange=True signal via return value
     so AgentCore can tell context_engine to never skip the first compression.
"""

import json
import sqlite3
import uuid
import asyncio
from datetime import datetime, timezone
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


from config.config import cfg

MAX_SESSIONS        = cfg.MAX_SESSIONS
SLIDING_WINDOW_SIZE = cfg.SLIDING_WINDOW_SIZE  # Centralized — no more mismatch
DB_PATH             = cfg.SESSIONS_DB


@dataclass
class Session:
    id:                  str
    agent_id:            str = "default"  # Scoping key
    created_at:          str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:          str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model:               str = "llama3.2" # Fallback
    system_prompt:       str = ""
    compressed_context:  str = ""
    sliding_window:      list[dict] = field(default_factory=list)
    full_history:        list[dict] = field(default_factory=list)
    title:               Optional[str] = None
    first_message:       Optional[str] = None
    parent_id:           Optional[str] = None
    lock:                asyncio.Lock = field(default_factory=asyncio.Lock)
    message_count:       int = 0


class SessionManager:

    def __init__(self, max_sessions: int = MAX_SESSIONS, db_path: str = DB_PATH):
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._max       = max_sessions
        self.db_path    = db_path
        self._init_db()
        self._load_from_db()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT DEFAULT 'default',
                    created_at TEXT,
                    updated_at TEXT,
                    model TEXT,
                    system_prompt TEXT,
                    compressed_context TEXT,
                    sliding_window TEXT,
                    full_history TEXT,
                    title TEXT,
                    first_message TEXT,
                    parent_id TEXT,
                    message_count INTEGER
                )
            ''')
            # Migrations for existing DBs
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN first_message TEXT")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN agent_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN parent_id TEXT")
            except sqlite3.OperationalError: pass
            conn.commit()

    def _load_from_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (self._max,))
            for r in reversed(cursor.fetchall()):
                s = self._row_to_session(r)
                self._sessions[s.id] = s

    def _row_to_session(self, r) -> Session:
        return Session(
            id=r["id"],
            agent_id=r["agent_id"] if "agent_id" in r.keys() else "default",
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            model=r["model"],
            system_prompt=r["system_prompt"],
            compressed_context=r["compressed_context"] or "",
            sliding_window=json.loads(r["sliding_window"]),
            full_history=json.loads(r["full_history"]),
            title=r["title"],
            first_message=r["first_message"] if "first_message" in r.keys() else None,
            parent_id=r["parent_id"] if "parent_id" in r.keys() else None,
            message_count=r["message_count"],
        )

    def _save_to_db(self, s: Session):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO sessions
                (id, agent_id, created_at, updated_at, model, system_prompt, compressed_context,
                 sliding_window, full_history, title, first_message, parent_id, message_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                s.id, s.agent_id, s.created_at, s.updated_at, s.model, s.system_prompt,
                s.compressed_context,
                json.dumps(s.sliding_window),
                json.dumps(s.full_history),
                s.title, s.first_message, s.parent_id, s.message_count,
            ))
            conn.commit()

    def _delete_from_db(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def create(self, model: str, system_prompt: str, agent_id: str = "default", session_id: Optional[str] = None, parent_id: Optional[str] = None) -> Session:
        now = datetime.now(timezone.utc).isoformat()
        sid = session_id or str(uuid.uuid4())
        session = Session(id=sid, agent_id=agent_id, created_at=now, updated_at=now,
                          model=model, system_prompt=system_prompt, parent_id=parent_id)
        self._sessions[sid] = session
        self._save_to_db(session)
        self._evict_if_needed()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            r = cursor.fetchone()
            if r:
                s = self._row_to_session(r)
                self._sessions[s.id] = s
                self._evict_if_needed()
                return s
        return None

    def get_or_create(self, session_id: Optional[str], model: str, system_prompt: str, agent_id: str = "default", parent_id: Optional[str] = None) -> Session:
        if session_id:
            s = self.get(session_id)
            if s:
                return s
        return self.create(model=model, system_prompt=system_prompt, agent_id=agent_id, session_id=session_id, parent_id=parent_id)

    def delete(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._delete_from_db(session_id)
        return existed

    def update_model(self, session_id: str, model: str):
        """Update the model for an existing session (when user switches models mid-chat)."""
        s = self._sessions.get(session_id)
        if s:
            s.model = model
            self._save_to_db(s)

    # ── Mutations ─────────────────────────────────────────────────────────────

    def update_context(self, session_id: str, context: str):
        if s := self.get(session_id):
            s.compressed_context = context
            s.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_to_db(s)

    def append_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Append message. Returns True if this was the FIRST user message
        so AgentCore knows to pass is_first_exchange=True to ContextEngine.
        """
        s = self.get(session_id)
        if not s:
            return False

        is_first = False
        msg = {"role": role, "content": content}
        s.full_history.append(msg)
        s.sliding_window.append(msg)

        if len(s.sliding_window) > SLIDING_WINDOW_SIZE:
            s.sliding_window = s.sliding_window[-SLIDING_WINDOW_SIZE:]

        s.message_count += 1
        s.updated_at = datetime.now(timezone.utc).isoformat()

        # Store first user message permanently
        if role == "user" and s.first_message is None:
            s.first_message = content
            is_first = True

        # Auto-title
        if role == "user" and not s.title:
            s.title = content[:60] + ("…" if len(content) > 60 else "")

        self._save_to_db(s)
        return is_first

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_sessions(self, agent_id: Optional[str] = None) -> list[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if agent_id:
                    cursor.execute(
                        "SELECT id, title, created_at, updated_at, model, message_count "
                        "FROM sessions WHERE agent_id = ? AND parent_id IS NULL ORDER BY updated_at DESC", (agent_id,)
                    )
                else:
                    cursor.execute(
                        "SELECT id, title, created_at, updated_at, model, message_count "
                        "FROM sessions WHERE parent_id IS NULL ORDER BY updated_at DESC"
                    )
                return [
                    {"id": r["id"], "title": r["title"] or "New Chat",
                     "created_at": r["created_at"], "updated_at": r["updated_at"],
                     "model": r["model"], "message_count": r["message_count"]}
                    for r in cursor.fetchall()
                ]
        except Exception as e:
            print(f"[SessionManager] DB read failed: {e}")
            return [
                {"id": s.id, "title": s.title or "New Chat", "created_at": s.created_at,
                 "updated_at": s.updated_at, "model": s.model, "message_count": s.message_count}
                for s in reversed(list(self._sessions.values()))
            ]

    def _evict_if_needed(self):
        while len(self._sessions) > self._max:
            evicted_id, _ = self._sessions.popitem(last=False)
            print(f"[SessionManager] Evicted: {evicted_id}")