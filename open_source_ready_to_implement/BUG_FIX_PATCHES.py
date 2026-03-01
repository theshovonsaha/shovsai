"""
BUG FIX PATCHES
===============
These are the exact changes needed in existing files.
Each section shows the OLD code and the NEW code to replace it with.
Apply these one at a time.

────────────────────────────────────────────────────────────────────────────────
FIX 1: orchestration/agent_manager.py
BUG:   Delegation 404 — clean model (no prefix) passed to AgentCore, then
       create_adapter(clean_model) resolves to wrong provider
ROOT:  Line `default_model=clean_model` strips the `groq:` prefix, so
       core.py's `create_adapter(resolved_model)` can't find the Groq adapter
────────────────────────────────────────────────────────────────────────────────

FIND THIS in run_agent_task():

        agent = AgentCore(
            adapter=delegation_adapter,
            context_engine=delegate_ctx,
            session_manager=self.sessions,
            tool_registry=filtered_registry,
            middleware=self.guardrail_middleware,
            orchestrator=delegate_orch,
            default_model=clean_model,        # ← BUG: no prefix, adapter won't resolve
            embed_model=config.embed_model,
            default_system_prompt=config.system_prompt,
        )

REPLACE WITH:

        agent = AgentCore(
            adapter=delegation_adapter,
            context_engine=delegate_ctx,
            session_manager=self.sessions,
            tool_registry=filtered_registry,
            middleware=self.guardrail_middleware,
            orchestrator=delegate_orch,
            default_model=effective_model,    # ← FIX: keep full prefix e.g. groq:llama-3.3-70b-versatile
            embed_model=config.embed_model,
            default_system_prompt=config.system_prompt,
        )

ALSO FIND in run_agent_task() where the session is initialized:

        self.sessions.get_or_create(
            session_id=child_sid,
            model=effective_model,
            system_prompt=config.system_prompt,
            agent_id=agent_id,
            parent_id=parent_id
        )

AND change the chat_stream call at the bottom:

        async for event in agent.chat_stream(user_message=task, session_id=child_sid, model=clean_model):

REPLACE WITH:

        async for event in agent.chat_stream(user_message=task, session_id=child_sid, model=effective_model):

────────────────────────────────────────────────────────────────────────────────
FIX 2: engine/core.py
BUG A: tool_result not yielded to frontend when middleware is active
BUG B: LLM hallucinating URLs for web_fetch (example.com/FrameworkA etc.)
BUG C: Context window explosion on large search results
────────────────────────────────────────────────────────────────────────────────

FIX 2A — tool_result always emitted:

FIND THIS (around line 575):

                if not self.middleware:
                    yield _ev("tool_result", tool_name=call.tool_name, success=result.success, content=result.content)

REPLACE WITH:

                # Always yield tool_result — frontend needs it regardless of middleware
                yield _ev("tool_result", tool_name=call.tool_name, success=result.success, content=result.content)

────────────────────────────────────────────────────────────────────────────────

FIX 2B — Add URL hallucination guard to DEFAULT_SYSTEM_PROMPT:

FIND THIS in DEFAULT_SYSTEM_PROMPT:

- ACCURACY: Never fabricate tool results. If a tool fails, explain the limitation.

REPLACE WITH:

- ACCURACY: Never fabricate tool results. If a tool fails, explain the limitation.
- WEB FETCH RULE: You may ONLY call web_fetch with URLs that appeared in a prior web_search result in this conversation. NEVER invent or guess URLs. If you need content from a site, run web_search first, then web_fetch the real URL from the results.

────────────────────────────────────────────────────────────────────────────────

FIX 2C — Model-aware search result truncation:
Add this dict near the top of core.py (after imports):

# Max chars to inject per tool result based on model context limits
# Prevents 413 "Request too large" errors on small-context models
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "moonshotai/kimi-k2-instruct": 8_000,
    "llama-3.2-3b-preview":        6_000,
    "llama-3.2-1b-preview":        4_000,
    "qwen2.5-coder:7b":            10_000,
    "qwen2.5-coder:3b":            6_000,
    "gemma2:2b":                   6_000,
    # Default for unlisted models:
    "_default":                    40_000,
}

def _truncate_for_model(content: str, model: str) -> str:
    \"\"\"Truncate tool result to fit within the model's practical context limit.\"\"\"
    limit = MODEL_CONTEXT_LIMITS.get(model, MODEL_CONTEXT_LIMITS["_default"])
    if len(content) <= limit:
        return content
    half = limit // 2
    return f"{content[:half]}\n\n[...{len(content) - limit} chars truncated to fit model context...]\n\n{content[-half:]}"

Then in the tool execution loop, FIND:

                combined_results.append(
                    f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                    f"{result.content}\n"
                    f"</SYSTEM_TOOL_RESULT>"
                    f"{pivot_msg}"
                )

REPLACE WITH:

                truncated_content = _truncate_for_model(result.content, clean_model)
                combined_results.append(
                    f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                    f"{truncated_content}\n"
                    f"</SYSTEM_TOOL_RESULT>"
                    f"{pivot_msg}"
                )

────────────────────────────────────────────────────────────────────────────────
FIX 3: plugins/tools.py
Replace the bash handler with Docker sandbox
────────────────────────────────────────────────────────────────────────────────

FIND the _bash handler function and replace entirely:

async def _bash(command: str, timeout: int = BASH_TIMEOUT, workdir: str = None) -> str:
    from plugins.docker_sandbox import run_in_docker
    result = await run_in_docker(command, timeout=timeout, workdir=workdir)
    return result

The BASH_TOOL Tool definition stays identical — only the handler changes.

────────────────────────────────────────────────────────────────────────────────
FIX 4: Add rag_search tool to plugins/tools.py
Add this after the DELEGATE_TO_AGENT_TOOL definition:
────────────────────────────────────────────────────────────────────────────────

async def _rag_search(query: str, top_k: int = 5, **kwargs) -> str:
    \"\"\"Search everything retrieved in this conversation session.\"\"\"
    session_id = kwargs.get("_session_id")
    if not session_id:
        return "rag_search requires a session context."
    from memory.session_rag import get_session_rag
    rag = get_session_rag(session_id)
    results = await rag.query(query, top_k=top_k)
    return rag.format_results_for_llm(results, query)

RAG_SEARCH_TOOL = Tool(
    name="rag_search",
    description=(
        "Search everything retrieved earlier in this conversation — web pages fetched, "
        "files created, tool results, uploaded documents. Use this BEFORE web_search "
        "to avoid redundant fetches. Returns the most relevant passages."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in session memory",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
            },
        },
        "required": ["query"],
    },
    handler=_rag_search,
    tags=["memory", "rag"],
)

Then add RAG_SEARCH_TOOL to ALL_TOOLS list:

ALL_TOOLS = [
    ...existing tools...,
    RAG_SEARCH_TOOL,    # ← add this
    DELEGATE_TO_AGENT_TOOL,
]

────────────────────────────────────────────────────────────────────────────────
FIX 5: engine/core.py — Auto-index tool results into session RAG
Add this block AFTER the tool result is obtained (after line ~556):
────────────────────────────────────────────────────────────────────────────────

FIND in the tool execution loop (after result.success check):

                if result.success:
                    log("tool", sid, f"{call.tool_name} → {len(result.content)} chars returned", level="ok")

ADD AFTER that block (before the failed_tools logic):

                # Auto-index successful tool results into per-session RAG
                if result.success and len(result.content) > 100:
                    try:
                        from memory.session_rag import get_session_rag
                        session_rag = get_session_rag(sid)
                        asyncio.create_task(
                            session_rag.index(
                                content   = result.content,
                                source    = call.tool_name,
                                tool_name = call.tool_name,
                            )
                        )
                    except Exception:
                        pass  # RAG indexing never blocks the main flow

────────────────────────────────────────────────────────────────────────────────
FIX 6: Add missing 'researcher' agent profile
Add to wherever profiles are seeded (ProfileManager init or main.py startup):
────────────────────────────────────────────────────────────────────────────────

In main.py, after the agent_manager is created, add:

# Seed missing standard agent profiles if they don't exist
_standard_profiles = [
    AgentProfile(
        id="researcher",
        name="Research Specialist",
        model="groq:llama-3.3-70b-versatile",
        tools=["web_search", "web_fetch", "rag_search", "query_memory", "store_memory"],
        system_prompt=(
            "You are a meticulous research agent. Always verify claims across multiple sources. "
            "Only call web_fetch with URLs returned by prior web_search results. Never invent URLs. "
            "Cite sources. Never fabricate data."
        ),
    ),
    AgentProfile(
        id="analyst",
        name="Data Analyst Agent",
        model="groq:llama-3.3-70b-versatile",
        tools=["file_create", "file_view", "file_str_replace", "rag_search", "bash"],
        system_prompt=(
            "You are a data analyst. You write clean, well-structured markdown reports and Python scripts. "
            "Save all outputs to files in the sandbox."
        ),
    ),
    AgentProfile(
        id="coder",
        name="Coder Extraordinaire",
        model="groq:llama-3.3-70b-versatile",
        tools=["bash", "file_create", "file_view", "file_str_replace"],
        system_prompt=(
            "You are an expert programmer. Write clean, working code. "
            "Always test code with bash after writing it. "
            "Never use delegate_to_agent — handle all coding tasks yourself."
        ),
    ),
]

for profile in _standard_profiles:
    if not profile_manager.get(profile.id):
        profile_manager.create(profile)
        print(f"[Startup] Seeded missing agent profile: {profile.id}")
"""
