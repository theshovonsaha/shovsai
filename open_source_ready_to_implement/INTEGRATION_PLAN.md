# Shovs Platform v0.6.0 — Integration Plan
## Safe step-by-step guide for a junior dev or AI coding agent

This plan is ordered so that each step is independently testable.
No step breaks the running system if done correctly.
If anything goes wrong, the rollback is: `git checkout -- <file>`.

---

## Prerequisites

```bash
# 1. Create a git branch first — always
git checkout -b v06-enhancements

# 2. Install new dependencies
pip install chromadb sentence-transformers docker mcp

# 3. Verify installs
python -c "import chromadb; print('chromadb ok')"
python -c "import sentence_transformers; print('sentence_transformers ok')"
python -c "import docker; print('docker ok')"
python -c "import mcp; print('mcp ok')"

# 4. Make sure Docker Desktop is running (for bash sandbox)
docker ps
```

---

## STEP 1 — Add new files (zero risk, nothing existing is touched)

Copy these files from the patch folder into the repo:

```
plugins/docker_sandbox.py     → copy as-is
plugins/mcp_client.py         → copy as-is
memory/session_rag.py         → copy as-is
api/rag_routes.py             → copy as-is
mcp_servers.json              → copy to project root
```

**Test after Step 1:**
```bash
python -c "from plugins.docker_sandbox import run_in_docker; print('docker_sandbox ok')"
python -c "from plugins.mcp_client import MCPClientManager; print('mcp_client ok')"
python -c "from memory.session_rag import SessionRAG; print('session_rag ok')"
python -c "from api.rag_routes import make_rag_router; print('rag_routes ok')"
```
All should print ok. If any fail, check the import paths match your project structure.

---

## STEP 2 — Fix the delegation 404 bug (highest priority)

**File:** `orchestration/agent_manager.py`

Find the `run_agent_task` method.

**Change 1:** Find where `AgentCore` is instantiated near the bottom of the method:
```python
# FIND:
            default_model=clean_model,
# REPLACE WITH:
            default_model=effective_model,
```

**Change 2:** Find the `chat_stream` call at the very bottom:
```python
# FIND:
        async for event in agent.chat_stream(user_message=task, session_id=child_sid, model=clean_model):
# REPLACE WITH:
        async for event in agent.chat_stream(user_message=task, session_id=child_sid, model=effective_model):
```

**Test after Step 2:**
Start the server. In the UI, select `groq:llama-3.3-70b-versatile` and send:
```
Delegate to the analyst agent: write a file called test.md with the text "delegation works"
```
In the logs you should see:
```
[AGENT] · [delegate] Turn started · model=groq:llama-3.3-70b-versatile   ← model has prefix now
[LLM  ] ✓ [delegate] Turn 0 complete                                      ← no 404
```
NOT:
```
[AGENT] · [delegate] Model switched: groq:llama-3.3-70b-versatile → llama-3.3-70b-versatile
[LLM  ] ✗ [delegate] LLM error: LLM rejected stream: 404
```

---

## STEP 3 — Fix tool_result not emitted to frontend

**File:** `engine/core.py`

Find this block (around line 575):
```python
                if not self.middleware:
                    yield _ev("tool_result", tool_name=call.tool_name, success=result.success, content=result.content)
```

Replace with:
```python
                # Always yield tool_result so frontend receives it
                yield _ev("tool_result", tool_name=call.tool_name, success=result.success, content=result.content)
```

**Test after Step 3:**
In the browser frontend, after a tool runs, you should see the tool result
rendered in the UI. Previously it was invisible when middleware was active.

---

## STEP 4 — Add URL hallucination guard to system prompt

**File:** `engine/core.py`

Find `DEFAULT_SYSTEM_PROMPT`, find this line:
```python
- ACCURACY: Never fabricate tool results. If a tool fails, explain the limitation.
```

Add this line directly after it:
```python
- WEB FETCH RULE: You may ONLY call web_fetch with URLs that appeared in a prior web_search result in this conversation. NEVER invent or guess URLs. If you need content from a site, run web_search first, then use a real URL from the results.
```

**Test after Step 4:**
Send: `"fetch the homepage of example.com/FrameworkA"`
The agent should respond saying it needs to search first rather than
calling `web_fetch("https://example.com/FrameworkA")`.

---

## STEP 5 — Add model-aware context truncation

**File:** `engine/core.py`

Add this dict and function near the top of the file, after the imports:

```python
# Model context limits — prevents 413 "Request too large" errors
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "moonshotai/kimi-k2-instruct": 8_000,
    "llama-3.2-3b-preview":        6_000,
    "llama-3.2-1b-preview":        4_000,
    "qwen2.5-coder:7b":            10_000,
    "qwen2.5-coder:3b":            6_000,
    "gemma2:2b":                   6_000,
    "_default":                    40_000,
}

def _truncate_for_model(content: str, model: str) -> str:
    limit = MODEL_CONTEXT_LIMITS.get(model, MODEL_CONTEXT_LIMITS["_default"])
    if len(content) <= limit:
        return content
    half = limit // 2
    return (
        f"{content[:half]}\n\n"
        f"[...{len(content) - limit} chars truncated to fit model context...]\n\n"
        f"{content[-half:]}"
    )
```

Then find where `combined_results.append(...)` is called in the tool loop:
```python
# FIND:
                combined_results.append(
                    f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                    f"{result.content}\n"
                    f"</SYSTEM_TOOL_RESULT>"
                    f"{pivot_msg}"
                )
# REPLACE WITH:
                truncated_content = _truncate_for_model(result.content, clean_model)
                combined_results.append(
                    f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                    f"{truncated_content}\n"
                    f"</SYSTEM_TOOL_RESULT>"
                    f"{pivot_msg}"
                )
```

**Test after Step 5:**
Switch to `groq:moonshotai/kimi-k2-instruct` and run a web search.
Previously it crashed with 413. Now it should succeed with truncated results.

---

## STEP 6 — Replace bash handler with Docker sandbox

**File:** `plugins/tools.py`

Find the `_bash` async function. The entire function body should be replaced:

```python
async def _bash(command: str, timeout: int = BASH_TIMEOUT, workdir: str = None) -> str:
    from plugins.docker_sandbox import run_in_docker
    return await run_in_docker(command, timeout=timeout, workdir=workdir)
```

Keep the `BASH_TOOL = Tool(...)` definition exactly as-is.

**Test after Step 6:**
```bash
# Make sure Docker Desktop is running first
docker ps

# Test with DOCKER_DISABLED=true first (subprocess fallback)
DOCKER_DISABLED=true python -c "
import asyncio
from plugins.tools import _bash
result = asyncio.run(_bash('echo hello from sandbox'))
print(result)
"
# Should print: hello from sandbox

# Test with Docker
python -c "
import asyncio
from plugins.docker_sandbox import run_in_docker
result = asyncio.run(run_in_docker('echo hello from docker'))
print(result)
"
# Should print: hello from docker
```

If Docker isn't running, the code gracefully falls back to subprocess.
Set `DOCKER_DISABLED=true` in `.env` for development without Docker.

---

## STEP 7 — Add rag_search tool to tools.py

**File:** `plugins/tools.py`

Add this function and Tool object before the `ALL_TOOLS` list:

```python
async def _rag_search(query: str, top_k: int = 5, **kwargs) -> str:
    """Search everything retrieved in this conversation session."""
    session_id = kwargs.get("_session_id")
    if not session_id:
        return "rag_search requires a session context. Pass _session_id in context."
    from memory.session_rag import get_session_rag
    rag = get_session_rag(session_id)
    results = await rag.query(query, top_k=top_k)
    return rag.format_results_for_llm(results, query)

RAG_SEARCH_TOOL = Tool(
    name="rag_search",
    description=(
        "Search everything retrieved earlier in this conversation — web pages fetched, "
        "files created, tool results, uploaded documents. "
        "Use this BEFORE web_search to avoid redundant fetches."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in session memory"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)"},
        },
        "required": ["query"],
    },
    handler=_rag_search,
    tags=["memory", "rag"],
)
```

Then add `RAG_SEARCH_TOOL` to the `ALL_TOOLS` list:
```python
ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    WEB_FETCH_TOOL,
    IMAGE_SEARCH_TOOL,
    BASH_TOOL,
    FILE_CREATE_TOOL,
    FILE_VIEW_TOOL,
    FILE_STR_REPLACE_TOOL,
    WEATHER_TOOL,
    PLACES_SEARCH_TOOL,
    PLACES_MAP_TOOL,
    STORE_MEMORY_TOOL,
    QUERY_MEMORY_TOOL,
    GENERATE_APP_TOOL,
    PDF_PROCESSOR_TOOL,
    RAG_SEARCH_TOOL,        # ← NEW
    DELEGATE_TO_AGENT_TOOL,
]
```

---

## STEP 8 — Auto-index tool results into session RAG

**File:** `engine/core.py`

In the tool execution loop, find this line:
```python
                if result.success:
                    log("tool", sid, f"{call.tool_name} → {len(result.content)} chars returned", level="ok")
```

Add this block immediately after:
```python
                # Auto-index successful results into per-session RAG
                if result.success and len(result.content) > 100:
                    try:
                        from memory.session_rag import get_session_rag
                        _session_rag = get_session_rag(sid)
                        asyncio.create_task(
                            _session_rag.index(
                                content=result.content,
                                source=call.tool_name,
                                tool_name=call.tool_name,
                            )
                        )
                    except Exception:
                        pass  # RAG indexing never blocks the main flow
```

---

## STEP 9 — Update main.py

Replace `main.py` entirely with the provided `main.py` from the patch folder.
This adds:
- `lifespan` handler for MCP init and shutdown
- `make_rag_router()` mounted at `/rag`
- Standard profile seeding (`researcher`, `analyst`, `coder`)
- MCP management endpoints (`/mcp/servers`, `/mcp/connect`)
- Auto-index file uploads into session RAG

**Test after Step 9:**
```bash
# Restart the server
# Check all routes are present
curl http://localhost:8000/
# Should show version: 0.6.0 and mcp_servers: []

curl http://localhost:8000/agents
# Should show researcher, analyst, coder profiles

# Test RAG upload
curl -X POST http://localhost:8000/rag/test-session/upload \
  -F "files=@README.md"
# Should return chunks_indexed > 0
```

---

## STEP 10 — Final integration test

Run the three-step prompt from earlier:

**Step 1:**
```
Search the web for the top 3 AI agent frameworks trending right now in 2026.
Web fetch the GitHub README for each one. Then delegate to the analyst agent
to write a structured markdown file called agent_frameworks.md comparing their
architecture, tools support, and safety model.
```

**Expected log pattern (success):**
```
[TOOL  ] ✓ web_search → results
[TOOL  ] ✓ web_fetch  → real GitHub content (not example.com)
[TOOL  ] ✓ delegate_to_agent → approved
[AGENT ] · [delegate] Turn started · model=groq:llama-3.3-70b-versatile  ← prefix kept
[LLM  ] ✓ [delegate] Turn 0 complete                                      ← no 404
[TOOL  ] ✓ file_create → agent_frameworks.md written
```

**Step 2:**
```
Use rag_search to find everything you know about safety models from the
frameworks you just researched. Then summarize the key differences.
```

**Expected:** Agent calls `rag_search`, gets back content from the web_fetch
results indexed in Step 1, no redundant web requests made.

---

## Rollback procedure

If any step breaks the server:
```bash
git diff --name-only          # see what changed
git checkout -- <broken_file> # restore single file
# or
git checkout -- .             # restore everything
```

New files (Step 1) can just be deleted — they don't affect existing behavior
until they're imported.

---

## Environment variables to add to .env

```bash
# Docker sandbox
DOCKER_SANDBOX_IMAGE=python:3.11-slim
DOCKER_SANDBOX_MEMORY=256m
DOCKER_SANDBOX_TIMEOUT=30
DOCKER_DISABLED=false          # set true if Docker Desktop not running

# Per-session RAG
CHROMA_DIR=./data/chroma
SESSION_RAG_EMBED_MODEL=all-MiniLM-L6-v2
SESSION_RAG_CHUNK_SIZE=800
SESSION_RAG_CHUNK_OVERLAP=100

# MCP
# (no env vars needed — all config in mcp_servers.json)
```

---

## For AI coding agents (Cursor, Windsurf, etc.)

Paste this as the task description:

```
Apply the following changes to the shovs agent platform in this exact order.
After each step, run the test command shown and confirm it passes before
proceeding to the next step. Do not modify any file not listed in the step.
Create a git commit after each step with the message format: "fix(step N): description"

[paste the steps above]
```
