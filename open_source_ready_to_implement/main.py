"""
Agent Platform — FastAPI Backend v0.6.0
---------------------------------------
Changes from v0.5.0:
  - Docker sandbox for bash execution
  - MCP client support (mcp_servers.json)
  - Per-session RAG with ChromaDB
  - rag_search as a callable agent tool
  - File upload endpoint for RAG ingestion
  - Researcher agent profile seeded at startup
  - Bug fixes: delegation 404, hallucinated URLs, context truncation
"""

import json
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List

from engine.core import AgentCore
from orchestration.agent_profiles import ProfileManager, AgentProfile
from orchestration.orchestrator import AgenticOrchestrator
from llm.llm_adapter import OllamaAdapter
from engine.context_engine import ContextEngine
from orchestration.session_manager import SessionManager
from llm.adapter_factory import create_adapter
from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools
from plugins.tools_web import register_web_tools
from engine.file_processor import FileProcessor
from orchestration.agent_manager import AgentManager
from config.logger import get_logger, log
from api.log_routes import setup_log_routes
from api.voice_endpoint import setup_voice_routes
from api.rag_routes import make_rag_router            # ← NEW

from guardrails import GuardrailMiddleware
from guardrails.api_routes import make_guardrail_router

# ── Config ────────────────────────────────────────────────────────────────────
FALLBACK_CHAT_MODEL = "llama3.2"

# ── Singletons ────────────────────────────────────────────────────────────────
adapter         = OllamaAdapter()
tool_registry   = ToolRegistry()
session_manager = SessionManager(max_sessions=200)
context_engine  = ContextEngine(adapter=adapter, compression_model=FALLBACK_CHAT_MODEL)
file_processor  = FileProcessor()
profile_manager = ProfileManager()
orchestrator    = AgenticOrchestrator(adapter=adapter)

# ── Guardrails ────────────────────────────────────────────────────────────────
guardrail_middleware = GuardrailMiddleware(
    registry                 = tool_registry,
    require_confirmation_for = "confirm_and_above",
    log_path                 = "./logs/tool_audit.jsonl",
)

agent_manager = AgentManager(
    profiles             = profile_manager,
    sessions             = session_manager,
    context_engine       = context_engine,
    adapter              = adapter,
    global_registry      = tool_registry,
    orchestrator         = orchestrator,
    guardrail_middleware = guardrail_middleware,
)

# ── Register tools ────────────────────────────────────────────────────────────
register_all_tools(tool_registry, agent_manager=agent_manager)
register_web_tools(tool_registry)


# ── MCP Client (optional — loads mcp_servers.json if present) ─────────────────
mcp_manager = None

async def _init_mcp():
    global mcp_manager
    try:
        from plugins.mcp_client import MCPClientManager
        mcp_manager = MCPClientManager(tool_registry)
        count = await mcp_manager.load_from_config("mcp_servers.json")
        if count:
            log("mcp", "startup", f"Loaded {count} MCP tools", level="ok")
    except Exception as e:
        log("mcp", "startup", f"MCP init skipped: {e}", level="warn")


# ── Standard agent profiles (seeded at startup if missing) ────────────────────
def _seed_standard_profiles():
    standard = [
        AgentProfile(
            id="researcher",
            name="Research Specialist",
            model="groq:llama-3.3-70b-versatile",
            tools=["web_search", "web_fetch", "rag_search", "query_memory", "store_memory"],
            system_prompt=(
                "You are a meticulous research agent. Always verify claims across multiple sources. "
                "CRITICAL: Only call web_fetch with URLs returned by a prior web_search result. "
                "Never invent or guess URLs. Cite sources. Never fabricate data."
            ),
        ),
        AgentProfile(
            id="analyst",
            name="Data Analyst Agent",
            model="groq:llama-3.3-70b-versatile",
            tools=["file_create", "file_view", "file_str_replace", "rag_search", "bash"],
            system_prompt=(
                "You are a data analyst. Write clean, well-structured markdown reports and Python scripts. "
                "Save all outputs to files in the sandbox. Use rag_search to recall prior research."
            ),
        ),
        AgentProfile(
            id="coder",
            name="Coder Extraordinaire",
            model="groq:llama-3.3-70b-versatile",
            tools=["bash", "file_create", "file_view", "file_str_replace"],
            system_prompt=(
                "You are an expert programmer. Write clean, tested, working code. "
                "Always run code with bash after writing it to verify it works."
            ),
        ),
    ]
    for p in standard:
        if not profile_manager.get(p.id):
            profile_manager.create(p)
            log("startup", "profiles", f"Seeded agent profile: {p.id}")


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _seed_standard_profiles()
    await _init_mcp()
    yield
    # Shutdown
    if mcp_manager:
        await mcp_manager.disconnect_all()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="shovs", version="0.6.0", lifespan=lifespan)
setup_log_routes(app)
setup_voice_routes(app, agent_manager)
app.include_router(make_guardrail_router(guardrail_middleware), prefix="/guardrails")
app.include_router(make_rag_router(), prefix="/rag")          # ← NEW
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Static Sandbox Shell ──────────────────────────────────────────────────────
from plugins.tools import SANDBOX_DIR
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/sandbox", StaticFiles(directory=str(SANDBOX_DIR)), name="sandbox")


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:       str  = Field(..., min_length=1, max_length=32_000)
    session_id:    Optional[str] = None
    agent_id:      Optional[str] = "default"
    system_prompt: Optional[str] = None
    force_memory:  Optional[bool] = False
    forced_tools:  Optional[List[str]] = Field(default_factory=list)


# ── Core routes (unchanged from v0.5.0) ──────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "shovs", "version": "0.6.0",
        "docs": "/docs",
        "tools": len(tool_registry.list_tools()),
        "mcp_servers": mcp_manager.list_connected() if mcp_manager else [],
    }

@app.get("/health")
async def health():
    ollama     = create_adapter("ollama")
    groq       = create_adapter("groq")
    openai     = create_adapter("openai")
    gemini     = create_adapter("gemini")
    anthropic  = create_adapter("anthropic")

    ollama_ok    = await ollama.health()
    groq_ok      = await groq.health()
    openai_ok    = await openai.health()
    gemini_ok    = await gemini.health()
    anthropic_ok = await anthropic.health()

    return {
        "status": "ok" if any([ollama_ok, groq_ok, openai_ok, gemini_ok, anthropic_ok]) else "degraded",
        "providers": {
            "ollama": ollama_ok, "groq": groq_ok, "openai": openai_ok,
            "gemini": gemini_ok, "anthropic": anthropic_ok,
        },
        "tools": tool_registry.list_tools(),
    }

@app.get("/models")
async def list_models():
    grouped = {"ollama": [], "groq": [], "openai": [], "gemini": [], "anthropic": []}
    for provider in grouped:
        try:
            a = create_adapter(provider)
            models = await a.list_models()
            if models:
                grouped[provider] = models
        except Exception:
            pass
    return {"models": grouped}

@app.get("/tools")
async def list_tools():
    return {"tools": tool_registry.list_tools()}

@app.post("/chat/stream")
async def chat_stream(
    message:           str              = Form(...),
    session_id:        Optional[str]    = Form(None),
    agent_id:          Optional[str]    = Form("default"),
    model:             Optional[str]    = Form(None),
    system_prompt:     Optional[str]    = Form(None),
    search_backend:    Optional[str]    = Form(None),
    search_engine:     Optional[str]    = Form(None),
    force_memory:      Optional[bool]   = Form(False),
    use_planner:       Optional[bool]   = Form(True),
    planner_model:     Optional[str]    = Form(None),
    context_model:     Optional[str]    = Form(None),
    forced_tools_json: Optional[str]    = Form(None),
    files:             List[UploadFile] = File(default=[]),
):
    forced_tools = []
    if forced_tools_json:
        try:
            forced_tools = json.loads(forced_tools_json)
        except Exception:
            pass

    async def generate():
        try:
            processed_files, image_b64s = [], []
            for upload in files:
                raw = await upload.read()
                pf  = file_processor.process(upload.filename, raw, upload.content_type)
                processed_files.append(pf)
                ev = {
                    "type": "attachment", "filename": upload.filename,
                    "file_type": "image" if pf.is_image else "document", "ok": pf.ok,
                }
                if not pf.ok:
                    ev["error"] = pf.error
                yield f"data: {json.dumps(ev)}\n\n"
                if pf.ok and pf.is_image:
                    image_b64s.append(pf.base64_data)

                # ── NEW: auto-index non-image uploads into session RAG ──────
                if pf.ok and not pf.is_image and session_id and pf.text_content:
                    try:
                        from memory.session_rag import get_session_rag
                        rag = get_session_rag(session_id)
                        chunks = await rag.index_file(upload.filename, pf.text_content)
                        if chunks:
                            yield f"data: {json.dumps({'type': 'rag_indexed', 'filename': upload.filename, 'chunks': chunks})}\n\n"
                    except Exception:
                        pass

            text_injection = file_processor.build_text_injection([f for f in processed_files if f.ok])
            full_message   = f"{text_injection}\n\n{message}" if text_injection else message

            log("agent", "system", f"Incoming request: agent={agent_id} model={model or 'default'}")
            agent_instance = agent_manager.get_agent_instance(agent_id or "default")

            async for event in agent_instance.chat_stream(
                user_message   = full_message,
                session_id     = session_id,
                agent_id       = agent_id,
                model          = model,
                system_prompt  = system_prompt,
                search_backend = search_backend,
                search_engine  = search_engine,
                force_memory   = force_memory,
                use_planner    = use_planner,
                planner_model  = planner_model,
                context_model  = context_model,
                forced_tools   = forced_tools,
                images         = image_b64s or None,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            log("agent", "stream", f"Generator error: {e}", level="error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── All remaining routes identical to v0.5.0 ─────────────────────────────────
# (sessions, memory, tool-results, agents endpoints — no changes needed)

@app.post("/sessions/{session_id}/clear_context")
async def clear_session_context(session_id: str):
    try:
        agent_manager.session_manager.update_context(session_id, "")
        return {"status": "ok", "message": "Context purged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions(agent_id: Optional[str] = None):
    return {"sessions": session_manager.list_sessions(agent_id=agent_id)}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return {
        "id": s.id, "title": s.title or "New Chat", "model": s.model,
        "created_at": s.created_at, "updated_at": s.updated_at,
        "message_count": s.message_count, "compressed_context": s.compressed_context,
        "context_lines": len([l for l in s.compressed_context.split("\n") if l.strip()]),
        "history": s.full_history,
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_manager.delete(session_id):
        raise HTTPException(404, "Session not found")
    # Also clean up RAG collection on session delete
    try:
        from memory.session_rag import cleanup_session_rag
        cleanup_session_rag(session_id)
    except Exception:
        pass
    return {"deleted": session_id}

@app.get("/sessions/{session_id}/context")
async def get_context(session_id: str):
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    lines = [l for l in s.compressed_context.split("\n") if l.strip()]
    return {"session_id": session_id, "lines": len(lines), "context": lines, "raw": s.compressed_context}

@app.get("/memory")
async def list_memories(limit: int = 100):
    from memory.semantic_graph import SemanticGraph
    graph = SemanticGraph()
    return {"memories": graph.list_all(limit=limit), "total": graph.count(), "limit": limit}

@app.post("/memory/search")
async def search_memory(payload: dict):
    from memory.semantic_graph import SemanticGraph
    query = payload.get("query", "")
    top_k = payload.get("top_k", 5)
    if not query:
        raise HTTPException(400, "query is required")
    results = await SemanticGraph().traverse(query, top_k=top_k)
    return {"query": query, "results": results}

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int):
    from memory.semantic_graph import SemanticGraph
    if not SemanticGraph().delete_by_id(memory_id):
        raise HTTPException(404, f"Memory {memory_id} not found")
    return {"deleted": memory_id}

@app.delete("/memory")
async def clear_all_memories():
    from memory.semantic_graph import SemanticGraph
    graph  = SemanticGraph()
    before = graph.count()
    graph.clear()
    return {"cleared": before}

@app.get("/tool-results/{session_id}")
async def get_tool_results(session_id: str, limit: int = 50):
    from memory.tool_results_db import ToolResultsDB
    results = ToolResultsDB().get_by_session(session_id, limit=limit)
    return {"session_id": session_id, "results": results, "count": len(results)}

@app.get("/apps")
async def list_generated_apps(limit: int = 100):
    from memory.tool_results_db import ToolResultsDB
    apps = ToolResultsDB().get_all_apps(limit=limit)
    return {"apps": apps, "count": len(apps)}

@app.get("/apps/{session_id}")
async def get_session_apps(session_id: str):
    from memory.tool_results_db import ToolResultsDB
    apps = ToolResultsDB().get_apps_by_session(session_id)
    return {"session_id": session_id, "apps": apps, "count": len(apps)}

@app.get("/agents")
async def list_agents():
    return {"agents": profile_manager.list_all()}

@app.post("/agents")
async def create_agent(profile: AgentProfile):
    return profile_manager.create(profile)

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    p = profile_manager.get(agent_id)
    if not p:
        raise HTTPException(404, "Agent not found")
    return p

@app.patch("/agents/{agent_id}")
async def update_agent(agent_id: str, payload: dict):
    p = profile_manager.get(agent_id)
    if not p:
        raise HTTPException(404, "Agent not found")
    allowed = {"name", "description", "model", "embed_model", "system_prompt", "tools", "avatar_url"}
    for key, val in payload.items():
        if key in allowed:
            setattr(p, key, val)
    from datetime import datetime, timezone
    p.updated_at = datetime.now(timezone.utc).isoformat()
    profile_manager.create(p)
    agent_manager.invalidate_cache(agent_id)
    return p

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    success = profile_manager.delete(agent_id)
    if not success:
        raise HTTPException(400, "Could not delete agent")
    return {"status": "ok"}

# ── MCP management endpoints ──────────────────────────────────────────────────

@app.get("/mcp/servers")
async def list_mcp_servers():
    """List connected MCP servers."""
    connected = mcp_manager.list_connected() if mcp_manager else []
    return {"connected": connected, "count": len(connected)}

@app.post("/mcp/connect")
async def connect_mcp_server(payload: dict):
    """Dynamically connect a new MCP server at runtime."""
    if not mcp_manager:
        raise HTTPException(503, "MCP manager not initialized")
    server_id = payload.get("id")
    command   = payload.get("command", "npx")
    args      = payload.get("args", [])
    env       = payload.get("env", {})
    if not server_id:
        raise HTTPException(400, "id is required")
    count = await mcp_manager.connect_server(server_id, command, args, env)
    return {"server_id": server_id, "tools_registered": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
