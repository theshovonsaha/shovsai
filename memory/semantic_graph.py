"""
Semantic Graph DB
-----------------
A lightweight Hybrid SQLite + Vector Knowledge Graph.
Stores Subject-Predicate-Object triplets and their vector embeddings
for fuzzy retrieval ("Deep Memory").

Uses Ollama 'nomic-embed-text' for embedding generation.
"""

import sqlite3
import json
import asyncio
import numpy as np
import httpx
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from config.config import cfg


class SemanticGraph:
    def __init__(self, db_path: str = "memory_graph.db", embedding_model: str = "nomic-embed-text"):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            # We store the embedding as a JSON string for simplicity,
            # since a personal agent DB will easily fit in memory for numpy cosine sim.
            # In a production scaled system, we would use sqlite-vec or chroma,
            # but for a portable agent OS, native SQLite + numpy is 0-dependency exact math.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            conn.commit()

    async def _get_embedding(self, text: str) -> List[float]:
        """Fetch an embedding from the provider."""
        # For simplicity, default to Ollama. If user uses OpenAI, fallback to OpenAI embeddings.
        if cfg.LLM_PROVIDER == "openai":
            headers = {"Authorization": f"Bearer {cfg.OPENAI_API_KEY}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    json={"model": "text-embedding-3-small", "input": text},
                    headers=headers
                )
                res.raise_for_status()
                return res.json()["data"][0]["embedding"]
        else:
            # Default to Ollama
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    f"{cfg.OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text}
                )
                res.raise_for_status()
                return res.json()["embedding"]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def add_triplet(self, subject: str, predicate: str, object_: str) -> int:
        """
        Embed the relationship and store the triplet.
        We embed the string: "subject predicate object"
        """
        text_to_embed = f"{subject} {predicate} {object_}"
        try:
            vector = await self._get_embedding(text_to_embed)
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {e}")

        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memories (subject, predicate, object, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (subject.strip(), predicate.strip(), object_.strip(), json.dumps(vector), now))
            conn.commit()
            return cursor.lastrowid

    async def traverse(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """
        Perform a semantic traversal:
        1. Embed the query.
        2. Calculate cosine similarity against all stored memories in memory.
        3. Return the top_k matching relationships.
        """
        try:
            query_vector = await self._get_embedding(query)
        except Exception as e:
            print(f"[SemanticGraph] Embedding error: {e}")
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, subject, predicate, object, embedding, created_at FROM memories")
            all_memories = cursor.fetchall()

            for row in all_memories:
                m_id, sub, pred, obj, emb_json, created = row
                try:
                    db_vector = json.loads(emb_json)
                    sim = self._cosine_similarity(query_vector, db_vector)
                    if sim >= threshold:
                        results.append({
                            "id": m_id,
                            "subject": sub,
                            "predicate": pred,
                            "object": obj,
                            "similarity": round(sim, 3),
                            "created_at": created
                        })
                except Exception:
                    continue  # Skip corrupted rows

        # Sort by most similar first
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def clear(self):
        """Wipe the graph."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories")
            conn.commit()
