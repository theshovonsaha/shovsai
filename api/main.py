"""
Agent Platform — FastAPI Backend v0.4.0 (Multi-Agent Support)
-----------------------------------------------------------
Supports multiple specialized agent instances with isolated memory and toolsets.
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List

from engine.core import AgentCore
from orchestration.agent_profiles import ProfileManager
from orchestration.orchestrator import AgenticOrchestrator
from llm.llm_adapter import OllamaAdapter
from engine.context_engine import ContextEngine
from orchestration.session_manager import SessionManager
from llm.adapter_factory import create_adapter
from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools, register_tools
from plugins.tools_web import register_web_tools
from engine.file_processor import FileProcessor
from orchestration.agent_profiles import ProfileManager, AgentProfile
from orchestration.agent_manager import AgentManager
from config.logger import get_logger, log
from api.log_routes import setup_log_routes
from api.voice_endpoint import setup_voice_routes

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

agent_manager   = AgentManager(
    profiles=profile_manager,
    sessions=session_manager,
    context_engine=context_engine,
    adapter=adapter,
    global_registry=tool_registry,
    orchestrator=orchestrator
)

# Register tools with manager for delegation
register_all_tools(tool_registry, agent_manager=agent_manager)
register_web_tools(tool_registry)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="shovs", version="0.5.0")
setup_log_routes(app)
setup_voice_routes(app, agent_manager)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Static Sandbox Shell ──────────────────────────────────────────────────────
from plugins.tools import SANDBOX_DIR
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/sandbox", StaticFiles(directory=str(SANDBOX_DIR)), name="sandbox")

class ChatRequest(BaseModel):
    message:       str  = Field(..., min_length=1, max_length=32_000)
    session_id:    Optional[str] = None
    agent_id:      Optional[str] = "default"
    system_prompt: Optional[str] = None
    force_memory:  Optional[bool] = False
    forced_tools:  Optional[List[str]] = Field(default_factory=list)

@app.get("/")
async def root():
    return {"name": "shovs", "version": "0.5.0", "docs": "/docs", "tools": len(tool_registry.list_tools())}

@app.get("/health")
async def health():
    ollama = create_adapter("ollama")
    groq = create_adapter("groq")
    openai = create_adapter("openai")
    gemini = create_adapter("gemini")
    anthropic = create_adapter("anthropic")
    
    ollama_ok = await ollama.health()
    groq_ok = await groq.health()
    openai_ok = await openai.health()
    gemini_ok = await gemini.health()
    anthropic_ok = await anthropic.health()
    
    return {
        "status": "ok" if (ollama_ok or groq_ok or openai_ok or gemini_ok or anthropic_ok) else "degraded",
        "providers": {
            "ollama": ollama_ok,
            "groq": groq_ok,
            "openai": openai_ok,
            "gemini": gemini_ok,
            "anthropic": anthropic_ok
        },
        "tools": tool_registry.list_tools(),
    }

@app.get("/models")
async def list_models():
    ollama = create_adapter("ollama")
    groq = create_adapter("groq")
    openai = create_adapter("openai")
    gemini = create_adapter("gemini")
    anthropic = create_adapter("anthropic")
    
    grouped_models = {
        "ollama": [],
        "groq": [],
        "openai": [],
        "gemini": [],
        "anthropic": []
    }

    try:
        m_local = await ollama.list_models()
        if m_local: grouped_models["ollama"] = m_local
    except: pass
    
    try:
        m_cloud = await groq.list_models()
        if m_cloud: grouped_models["groq"] = m_cloud
    except: pass

    try:
        m_openai = await openai.list_models()
        if m_openai: grouped_models["openai"] = m_openai
    except: pass

    try:
        m_gemini = await gemini.list_models()
        if m_gemini: grouped_models["gemini"] = m_gemini
    except: pass

    try:
        m_anthropic = await anthropic.list_models()
        if m_anthropic: grouped_models["anthropic"] = m_anthropic
    except: pass
    
    return {"models": grouped_models}

@app.get("/tools")
async def list_tools():
    return {"tools": tool_registry.list_tools()}

@app.post("/chat/stream")
async def chat_stream(
    message:       str              = Form(...),
    session_id:    Optional[str]    = Form(None),
    agent_id:      Optional[str]    = Form("default"),
    model:         Optional[str]    = Form(None),
    system_prompt: Optional[str]    = Form(None),
    search_backend: Optional[str]   = Form(None),
    search_engine:  Optional[str]   = Form(None),
    force_memory:   Optional[bool]  = Form(False),
    use_planner:    Optional[bool]  = Form(True),
    planner_model:  Optional[str]   = Form(None),
    context_model:  Optional[str]   = Form("deepseek-r1:8b"),
    forced_tools_json: Optional[str] = Form(None),
    files:         List[UploadFile] = File(default=[]),
):
    forced_tools = []
    if forced_tools_json:
        try:
            forced_tools = json.loads(forced_tools_json)
        except:
            pass
    async def generate():
        try:
            processed_files, image_b64s = [], []
            for upload in files:
                raw = await upload.read()
                pf  = file_processor.process(upload.filename, raw, upload.content_type)
                processed_files.append(pf)
                ev = {"type": "attachment", "filename": upload.filename,
                      "file_type": "image" if pf.is_image else "document", "ok": pf.ok}
                if not pf.ok: ev["error"] = pf.error
                yield f"data: {json.dumps(ev)}\n\n"
                if pf.ok and pf.is_image: image_b64s.append(pf.base64_data)

            text_injection = file_processor.build_text_injection([f for f in processed_files if f.ok])
            full_message   = f"{text_injection}\n\n{message}" if text_injection else message

            log("agent", "system", f"Incoming request: agent={agent_id} model={model or 'default'}")
            agent_instance = agent_manager.get_agent_instance(agent_id or "default")

            # ── Heartbeat Guard ───────────────────────────────────────────
            # Yield intermittent comments to keep SSE connections alive
            # and detect client disconnects promptly.
            
            async for event in agent_instance.chat_stream(
                user_message=full_message, 
                session_id=session_id,
                agent_id=agent_id,
                model=model, 
                system_prompt=system_prompt,
                search_backend=search_backend,
                search_engine=search_engine,
                force_memory=force_memory,
                use_planner=use_planner,
                planner_model=planner_model,
                context_model=context_model,
                forced_tools=forced_tools,
                images=image_b64s or None,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            log("agent", "stream", f"Generator error: {e}", level="error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

@app.post("/sessions/{session_id}/clear_context")
async def clear_session_context(session_id: str):
    """Purge the compressed memory of a specific session."""
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
    if not s: raise HTTPException(404, "Session not found")
    return {
        "id": s.id, "title": s.title or "New Chat", "model": s.model,
        "created_at": s.created_at, "updated_at": s.updated_at,
        "message_count": s.message_count, "compressed_context": s.compressed_context,
        "context_lines": len([l for l in s.compressed_context.split("\n") if l.strip()]),
        "history": s.full_history,
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_manager.delete(session_id): raise HTTPException(404, "Session not found")
    return {"deleted": session_id}

@app.get("/sessions/{session_id}/context")
async def get_context(session_id: str):
    s = session_manager.get(session_id)
    if not s: raise HTTPException(404, "Session not found")
    lines = [l for l in s.compressed_context.split("\n") if l.strip()]
    return {"session_id": session_id, "lines": len(lines), "context": lines, "raw": s.compressed_context}

# ── Memory Management ──────────────────────────────────────────────────────────

@app.get("/memory")
async def list_memories(limit: int = 100):
    """List all stored long-term memories from the semantic graph."""
    from memory.semantic_graph import SemanticGraph
    graph = SemanticGraph()
    memories = graph.list_all(limit=limit)
    total = graph.count()
    return {"memories": memories, "total": total, "limit": limit}

@app.post("/memory/search")
async def search_memory(payload: dict):
    """Semantic search through the memory graph."""
    from memory.semantic_graph import SemanticGraph
    query = payload.get("query", "")
    top_k = payload.get("top_k", 5)
    if not query:
        raise HTTPException(400, "query is required")
    graph = SemanticGraph()
    results = await graph.traverse(query, top_k=top_k)
    return {"query": query, "results": results}

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int):
    """Delete a single memory entry by ID."""
    from memory.semantic_graph import SemanticGraph
    graph = SemanticGraph()
    deleted = graph.delete_by_id(memory_id)
    if not deleted:
        raise HTTPException(404, f"Memory {memory_id} not found")
    return {"deleted": memory_id}

@app.delete("/memory")
async def clear_all_memories():
    """Wipe the entire memory graph. This is irreversible."""
    from memory.semantic_graph import SemanticGraph
    graph = SemanticGraph()
    before = graph.count()
    graph.clear()
    return {"cleared": before, "message": f"Deleted {before} memories"}

# ── Agent Management ──────────────────────────────────────────────────────────

@app.get("/agents")
async def list_agents():
    return {"agents": profile_manager.list_all()}

@app.post("/agents")
async def create_agent(profile: AgentProfile):
    return profile_manager.create(profile)

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    p = profile_manager.get(agent_id)
    if not p: raise HTTPException(404, "Agent not found")
    return p

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    success = profile_manager.delete(agent_id)
    if not success: raise HTTPException(400, "Could not delete agent (default cannot be deleted or not found)")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)