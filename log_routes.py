"""
Add these routes to main.py to expose the internal log stream.

1. Add to imports at top of main.py:
   from logger import get_logger, log

2. Add after app = FastAPI(...):
   from log_routes import setup_log_routes
   setup_log_routes(app)

3. In chat_stream(), after agent_instance is created, add:
   log("agent", "system", f"Request: agent={agent_id} model={model or 'default'}")
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Optional
import json


def setup_log_routes(app: FastAPI):
    from logger import get_logger

    @app.get("/logs/stream")
    async def log_stream(session_id: Optional[str] = None, category: Optional[str] = None):
        """
        SSE stream of internal log entries.
        Optional filters: ?session_id=xxx&category=tool
        """
        logger = get_logger()

        async def generate():
            # Send recent history first so panel isn't blank on connect
            for entry in logger.recent(limit=80):
                if session_id and entry.session != session_id and entry.session != "system":
                    continue
                if category and entry.category != category:
                    continue
                yield entry.to_sse()

            # Then stream live
            q = logger.subscribe()
            try:
                async for chunk in logger.stream(q):
                    # Parse and filter
                    try:
                        data = json.loads(chunk.replace("data: ", "").strip())
                        if session_id and data.get("session") not in (session_id, "system"):
                            continue
                        if category and data.get("category") != category:
                            continue
                    except Exception:
                        pass
                    yield chunk
            finally:
                logger.unsubscribe(q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.get("/logs/recent")
    async def log_recent(limit: int = 100, session_id: Optional[str] = None, category: Optional[str] = None):
        """REST endpoint for recent logs — useful for initial load."""
        logger = get_logger()
        entries = logger.recent(limit=limit)
        if session_id:
            entries = [e for e in entries if e.session in (session_id, "system")]
        if category:
            entries = [e for e in entries if e.category == category]
        from dataclasses import asdict
        return {"logs": [asdict(e) for e in entries]}