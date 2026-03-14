import os
import hashlib
import httpx
import chromadb
from typing import List, Optional, Dict

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DB_PATH     = "./chroma_db"

class VectorEngine:
    def __init__(self, session_id: str, agent_id: str = "default", model: str = "nomic-embed-text"):
        from config.config import cfg
        self.session_id = session_id
        self.agent_id   = agent_id
        self.model      = model
        self.base_url   = cfg.OLLAMA_BASE_URL
        self.client     = chromadb.PersistentClient(path=DB_PATH)
        self._ensure_collection()

    def _ensure_collection(self):
        # Isolation: agent_{agent_id}_session_{session_id}
        safe_agent = self.agent_id.replace("-", "_")
        safe_sid   = self.session_id.replace("-", "_")
        self.collection = self.client.get_or_create_collection(
            name=f"agent_{safe_agent}_session_{safe_sid}"
        )

    async def _get_embedding(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    def _generate_id(self, key: str, anchor: str) -> str:
        return hashlib.sha256(f"{key}\n{anchor}".encode()).hexdigest()

    async def index(self, key: str, anchor: str, metadata: Optional[dict] = None):
        doc_id = self._generate_id(key, anchor)
        embedding = await self._get_embedding(key)
        meta = metadata or {}
        meta["key"] = key
        meta["anchor"] = anchor
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[anchor]
        )

    async def query(self, text: str, limit: int = 3) -> List[dict]:
        embedding = await self._get_embedding(text)
        results = self.collection.query(query_embeddings=[embedding], n_results=limit)
        parsed = []
        if not results or not results["ids"]: return parsed
        for i in range(len(results["ids"][0])):
            parsed.append({
                "id": results["ids"][0][i],
                "key": results["metadatas"][0][i].get("key"),
                "anchor": results["metadatas"][0][i].get("anchor"),
                "metadata": results["metadatas"][0][i]
            })
        return parsed

    async def count(self) -> int:
        try:
            return self.collection.count()
        except:
            self._ensure_collection()
            return self.collection.count()

    async def clear(self):
        try:
            self.client.delete_collection(self.collection.name)
        except:
            pass
        self._ensure_collection()
