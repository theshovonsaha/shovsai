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
from pydantic import BaseModel, Field
from typing import Optional, List

from engine.core import AgentCore
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
DEFAULT_CHAT_MODEL = "llama3.2"

# ── Singletons ────────────────────────────────────────────────────────────────
adapter         = OllamaAdapter()
tool_registry   = ToolRegistry()
register_all_tools(tool_registry)
register_web_tools(tool_registry)

session_manager = SessionManager(max_sessions=200)
context_engine  = ContextEngine(adapter=adapter, compression_model=DEFAULT_CHAT_MODEL)
file_processor  = FileProcessor()
profile_manager = ProfileManager()

agent_manager   = AgentManager(
    profiles=profile_manager,
    sessions=session_manager,
    context_engine=context_engine,
    adapter=adapter,
    global_registry=tool_registry
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Agent Platform", version="0.5.0")
setup_log_routes(app)
setup_voice_routes(app, agent_manager)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message:       str  = Field(..., min_length=1, max_length=32_000)
    session_id:    Optional[str] = None
    agent_id:      Optional[str] = "default"
    system_prompt: Optional[str] = None
    force_memory:  Optional[bool] = False
    forced_tools:  Optional[List[str]] = Field(default_factory=list)

@app.get("/")
async def root():
    return {"name": "Agent Platform", "version": "0.4.0", "docs": "/docs", "tools": len(tool_registry.list_tools())}

@app.get("/health")
async def health():
    ollama = create_adapter("ollama")
    groq = create_adapter("groq")
    openai = create_adapter("openai")
    gemini = create_adapter("gemini")
    
    ollama_ok = await ollama.health()
    groq_ok = await groq.health()
    openai_ok = await openai.health()
    gemini_ok = await gemini.health()
    
    return {
        "status": "ok" if (ollama_ok or groq_ok or openai_ok or gemini_ok) else "degraded",
        "providers": {
            "ollama": ollama_ok,
            "groq": groq_ok,
            "openai": openai_ok,
            "gemini": gemini_ok
        },
        "tools": tool_registry.list_tools(),
    }

@app.get("/models")
async def list_models():
    ollama = create_adapter("ollama")
    groq = create_adapter("groq")
    openai = create_adapter("openai")
    gemini = create_adapter("gemini")
    
    models = []
    try:
        m_local = await ollama.list_models()
        models.extend([f"ollama:{m}" for m in m_local] if m_local else [])
    except: pass
    
    try:
        m_cloud = await groq.list_models()
        models.extend([f"groq:{m}" for m in m_cloud] if m_cloud else [])
    except: pass

    try:
        m_openai = await openai.list_models()
        models.extend([f"openai:{m}" for m in m_openai] if m_openai else [])
    except: pass

    try:
        m_gemini = await gemini.list_models()
        models.extend([f"gemini:{m}" for m in m_gemini] if m_gemini else [])
    except: pass
    
    return {"models": models}

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
    force_memory:   Optional[bool]  = Form(False),
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

            async for event in agent_instance.chat_stream(
                user_message=full_message, 
                session_id=session_id,
                agent_id=agent_id,
                model=model, 
                system_prompt=system_prompt,
                search_backend=search_backend,
                force_memory=force_memory,
                forced_tools=forced_tools,
                images=image_b64s or None,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

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