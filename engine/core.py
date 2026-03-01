"""
AgentCore v3 — intelligence fixes + structured tracing
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter
from llm.llm_adapter     import OllamaAdapter, LLMError
from engine.context_engine  import ContextEngine
from orchestration.session_manager import SessionManager
from plugins.tool_registry   import ToolRegistry
from guardrails.middleware import GuardrailMiddleware
from memory.vector_engine   import VectorEngine
from memory.semantic_graph  import SemanticGraph
from llm.adapter_factory    import create_adapter, strip_provider_prefix
from orchestration.orchestrator import AgenticOrchestrator
from config.logger          import log

# ── Trace Logger ─────────────────────────────────────────────────────────────
_TRACE_DIR = os.getenv("TRACE_DIR", "./logs")
os.makedirs(_TRACE_DIR, exist_ok=True)
_TRACE_PATH = os.path.join(_TRACE_DIR, "agent_trace.jsonl")

def _trace(agent_id: str, session_id: str, event_type: str, data: dict):
    """Append structured trace event to JSONL file for debugging."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent_id": agent_id,
        "session_id": session_id,
        "event_type": event_type,
        "data": data,
    }
    try:
        with open(_TRACE_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never crash on trace failure

DEFAULT_SYSTEM_PROMPT = """\
You are a highly advanced AI agent running on the [Shovs Agent Platform].

Core Directives:
- CONTEXT: Prioritize "Historical Context" and "Session Memory" for persona consistency. If a specific persona is detected in memory, adopt it fully.
- MEMORY: Use `query_memory` early to re-discover user preferences and past interactions.
- VISUALS: Wrap visual data in ```html or ```svg blocks for the Live View.
- TOOLS: To use a tool, output ONLY the JSON call: {"tool": "<name>", "arguments": {<args>}}. You may chain multiple tool calls.
- ACCURACY: Never fabricate tool results. If a tool fails, explain the limitation.
- DELEGATION: Use `delegate_to_agent` when a task is better handled by a specialized agent.
"""


MAX_TOOL_TURNS = 6

def _ev(type_: str, **kw) -> dict:
    return {"type": type_, **kw}

def _to_bool(val) -> bool:
    if isinstance(val, bool): return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)

class AgentCore:

    def __init__(
        self,
        adapter:         OllamaAdapter,
        context_engine:  ContextEngine,
        session_manager: SessionManager,
        tool_registry:   ToolRegistry,
        middleware:      Optional[GuardrailMiddleware] = None,
        orchestrator:    Optional[AgenticOrchestrator] = None,
        default_model:   str = "llama3.2",
        embed_model:     str = "nomic-embed-text",
        default_system_prompt: str = None,
    ):
        self.adapter   = adapter
        self.ctx_eng   = context_engine
        self.sessions  = session_manager
        self.tools     = tool_registry
        self.middleware = middleware
        self.orch      = orchestrator
        self.def_model = default_model
        self.embed_model = embed_model
        self.def_system_prompt = default_system_prompt or DEFAULT_SYSTEM_PROMPT

    async def chat_stream(
        self,
        user_message:   str,
        session_id:     Optional[str] = None,
        agent_id:       str = "default",
        model:          Optional[str] = None,
        system_prompt:  Optional[str] = None,
        search_backend: Optional[str] = None,
        search_engine:  Optional[str] = None, # Added param
        images:         Optional[list[str]] = None,
        force_memory:   bool = False,
        forced_tools:   Optional[list[str]] = None,
        **kw
    ) -> AsyncIterator[dict]:

        model         = model or self.def_model
        system_prompt = system_prompt or self.def_system_prompt

        session = self.sessions.get_or_create(
            session_id=session_id,
            model=model,
            system_prompt=system_prompt,
            agent_id=agent_id,
        )
        sid = session.id
        
        # New: Layer settings (passed via kwargs or extracted from request)
        use_planner    = _to_bool(kw.get("use_planner", True))
        planner_model  = kw.get("planner_model")
        context_model  = kw.get("context_model", "deepseek-r1:8b")

        # BUG FIX: Update session model when user switches models mid-conversation
        if session.model != model:
            log("agent", sid, f"Model switched: {session.model} → {model}")
            session.model = model
            self.sessions.update_model(sid, model)

        yield _ev("session", session_id=sid, agent_id=session.agent_id)

        log("agent", sid, f"Turn started · agent={agent_id} · model={model}")

        full_response = ""
        error_msg     = None

        async with session.lock:
            # ── Dynamic Adapter Selection ─────────────────────────────────────
            # Smart switching logic with resolution priority:
            # 1. Explicit model request
            # 2. Session model
            # 3. Agent default model
            # 4. System default (llama3.2)
            
            resolved_model = model or session.model or self.def_model or "llama3.2"
            current_use_adapter = create_adapter(provider=resolved_model)
            
            # Use clean model name for API calls
            clean_model = strip_provider_prefix(resolved_model)
            
            # ── CRITICAL: Propagate dynamic adapter to subsystems ─────────
            # Without this, ContextEngine and Orchestrator remain stuck on 
            # the original OllamaAdapter even when user selects Groq/Claude/Gemini.
            self.ctx_eng.set_adapter(current_use_adapter)
            if self.orch:
                self.orch.set_adapter(current_use_adapter)
            
            # ── Vector RAG ────────────────────────────────────────────────────
            ve = VectorEngine(sid, agent_id=session.agent_id, model=self.embed_model)
            historical_anchors = await ve.query(user_message, limit=3)

            if historical_anchors:
                keys = [a.get("key", "?") for a in historical_anchors]
                log("rag", sid, f"Retrieved {len(historical_anchors)} anchors: {', '.join(keys)}", level="ok")
            else:
                log("rag", sid, "No relevant history anchors found")

            current_facts = SemanticGraph().get_current_facts(sid)

            # ── Agentic Planning (Manager Agent) ──────────────────────────────
            if self.orch and use_planner and not forced_tools:
                log("agent", sid, "Planner active · determining strategy...", level="info")
                # Use override planner_model if provided
                use_p_model = planner_model or clean_model
                planned_tools = await self.orch.plan(user_message, self.tools.list_tools(), model=use_p_model)
                if planned_tools:
                    strategy = f"Strategy: Using {', '.join(planned_tools)} to resolve query."
                    yield _ev("plan", strategy=strategy, tools=planned_tools)
                    forced_tools = planned_tools
            elif not use_planner:
                log("agent", sid, "Planner bypassed (Direct Mode)")

            messages = self._build_messages(
                system_prompt=session.system_prompt,
                context=session.compressed_context,
                sliding_window=session.sliding_window,
                user_message=user_message,
                first_message=session.first_message,
                message_count=session.message_count,
                historical_anchors=historical_anchors,
                force_memory=force_memory,
                forced_tools=forced_tools,
                current_facts=current_facts,
            )

            prompt_tokens_est = sum(len(m["content"]) for m in messages) // 4
            log("llm", sid, f"Prompt built · {len(messages)} messages · ~{prompt_tokens_est} tokens est.")

            # Trace: log the actual prompt chain sent to LLM with flags
            _trace(agent_id, sid, "prompt_chain", {
                "model": model,
                "message_count": len(messages),
                "estimated_tokens": prompt_tokens_est,
                "has_rag": len(historical_anchors) > 0,
                "has_facts": len(current_facts) > 0,
                "has_memory": len(session.compressed_context) > 0,
                "system_prompt_preview": messages[0]["content"][:300] if messages else "",
                "user_message": user_message[:200],
            })

            try:
                async for event in self._agent_loop(
                    model=clean_model,
                    messages=messages,
                    adapter=current_use_adapter,
                    search_backend=search_backend,
                    search_engine=search_engine, # Pass down
                    images=images,
                    agent_id=session.agent_id,
                    session_id=sid,
                ):
                    yield event
                    if event["type"] == "token":
                        full_response += event["content"]
                    elif event["type"] == "_retract_response":
                        # BUG FIX: clean tool-call JSON from full_response
                        full_response = event.get("clean_response", full_response)
                    elif event["type"] == "error":
                        error_msg = event["message"]
            except LLMError as e:
                log("llm", sid, f"LLM error: {e}", level="error")
                yield _ev("error", message=str(e))
                return
            except asyncio.TimeoutError:
                log("llm", sid, "LLM generation timed out after 60s", level="error")
                yield _ev("error", message="Generation timed out. The model might be overloaded or the connection was lost.")
                return
            except Exception as e:
                log("agent", sid, f"Unexpected loop error: {e}", level="error")
                yield _ev("error", message=f"Internal Error: {e}")
                return

        if error_msg:
            return

        # BUG FIX: strip any remaining tool-call JSON and internal reasoning from full_response before storing
        clean_response = self._strip_tool_json(self._strip_reasoning(full_response))
        
        # Guard: If response is only tool calls, we still want a placeholder for continuity
        if not clean_response and full_response:
            clean_response = "[Tool Execution Turn]"

        is_first_exchange = self.sessions.append_message(sid, "user", user_message)
        self.sessions.append_message(sid, "assistant", clean_response)

        log("agent", sid, f"Response complete · {len(clean_response)} chars · first_exchange={is_first_exchange}")

        # Trace: log the stored response
        _trace(agent_id, sid, "assistant_response", {
            "content": clean_response[:500],
            "raw_length": len(full_response),
            "clean_length": len(clean_response),
            "was_cleaned": len(full_response) != len(clean_response),
        })

        yield _ev("compressing")
        try:
            # Decouple: Background compression uses the provided context model layer
            # Fallback logic: Ensure the target model is compatible with the current adapter
            from llm.adapter_factory import get_default_model
            default_fallback = get_default_model(current_use_adapter)
            
            # Smart Fallback: If context_model is deepseek but adapter is NOT ollama, fallback.
            adapter_name = current_use_adapter.__class__.__name__.lower()
            is_cloud = any(p in adapter_name for p in ["groq", "openai", "gemini", "anthropic"])
            
            compression_target = context_model or model or default_fallback
            
            if is_cloud and "deepseek" in compression_target.lower():
                # Force fallback for cloud regions to the main model (which we know is cloud-safe)
                compression_target = model or default_fallback
            
            log("ctx", sid, f"Compressing exchange with {compression_target} (adapter: {adapter_name})...")
            updated_ctx, keyed_facts, voids = await self.ctx_eng.compress_exchange(
                user_message=user_message,
                assistant_response=clean_response,
                current_context=session.compressed_context,
                is_first_exchange=is_first_exchange,
                model=compression_target, 
            )
            self.sessions.update_context(sid, updated_ctx)
            lines = len([l for l in updated_ctx.split("\n") if l.strip()])
            log("ctx", sid, f"Context updated · {lines} lines · {len(keyed_facts)} new facts, {len(voids)} voids", level="ok")

            try:
                sg = SemanticGraph()
                for void in voids:
                    sg.void_temporal_fact(sid, void["subject"], void["predicate"], session.message_count)
                    log("ctx", sid, f"Voided fact: {void['subject']} | {void['predicate']}", level="ok")
                for item in keyed_facts:
                    # Save explicitly mapped [FACT]s to the deterministic table
                    if item.get("subject") and item.get("predicate") and item.get("subject") != "General":
                        sg.add_temporal_fact(sid, item["subject"], item["predicate"], item.get("object", ""), session.message_count)
            except Exception as e:
                log("ctx", sid, f"Database deterministic facts error: {e}", level="error")

            if keyed_facts:
                ve = VectorEngine(sid, agent_id=session.agent_id)
                anchor_text = f"User: {user_message}\nAssistant: {clean_response}"  # Use cleaned
                for item in keyed_facts:
                    await ve.index(key=item["key"], anchor=anchor_text, metadata={"fact": item["fact"]})
                log("rag", sid, f"Indexed {len(keyed_facts)} facts into vector store", level="ok")

                # Trace: log indexed facts
                _trace(agent_id, sid, "facts_indexed", {
                    "facts": [f["key"] for f in keyed_facts],
                    "context_lines": lines,
                })
        except Exception as e:
            log("ctx", sid, f"Compression/indexing error: {e}", level="error")
            lines = 0

        yield _ev("context_updated", lines=lines)
        yield _ev("done")
        log("agent", sid, "Turn complete")

    @staticmethod
    def _strip_tool_json(text: str) -> str:
        """Remove tool-call JSON from response text before storing. Upgraded for V10 logic."""
        if not text: return ""
        
        # Use the same brace-counting scanner from ToolRegistry to find all JSON objects
        # This handles nested braces correctly (which regex can't)
        from plugins.tool_registry import _extract_json_objects
        candidates = _extract_json_objects(text)
        
        # Remove any JSON block that looks like a tool call
        cleaned = text
        for obj, raw_str in candidates:
            if "tool" in obj and "arguments" in obj:
                cleaned = cleaned.replace(raw_str, "")
        
        # Cleanup whitespace artifacts
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove internal reasoning blocks wrapped in <think> tags."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _verify_citations(self, response: str, anchors: list[dict]) -> bool:
        """
        Verify that the response is faithful to the provided anchors.
        Returns True if key facts from anchors are reflected accurately in the response.
        Catches blatant contradictions where the response states opposite values.
        """
        if not anchors:
            return True

        import re

        for anchor in anchors:
            anchor_text = anchor.get("anchor", "")
            anchor_key = anchor.get("key", "").lower()
            
            # Extract what the user stated (User: ... lines)
            user_statements = re.findall(r"User:\s*(.+?)(?:\s*Assistant:|\n|$)", anchor_text, re.DOTALL)
            for stmt in user_statements:
                # Extract ALL meaningful words from the user statement
                stop_words = {
                    "the", "and", "that", "this", "with", "from", "have", "will",
                    "been", "they", "would", "could", "should", "about", "like",
                    "want", "only", "ever", "just", "also", "into", "your",
                    "what", "when", "where", "how", "who", "which", "does",
                    "are", "was", "were", "not", "but", "for", "all", "can",
                }
                words = [
                    w.lower().strip(".,!?\"'")
                    for w in stmt.split()
                    if w.lower().strip(".,!?\"'") not in stop_words
                    and len(w.strip(".,!?\"'")) >= 2
                ]
                if not words:
                    continue

                resp_lower = response.lower()
                content_words = [w for w in words if len(w) >= 3]
                if not content_words:
                    continue

                # Check each content word
                for word in content_words:
                    if word in resp_lower:
                        continue  # word is present in response — no contradiction here
                    
                    # Word is missing from response. Is the response still 
                    # discussing the same topic?
                    other_words = [w for w in content_words if w != word]
                    
                    if other_words:
                        # Multiple keywords: check if others appear in response
                        others_found = sum(1 for w in other_words if w in resp_lower)
                        if others_found >= len(other_words) * 0.3:
                            return False  # Same topic, different value
                    else:
                        # Single keyword case (e.g., "I like blue" → words=["blue"])
                        # Check if the anchor key topic is referenced in the response
                        key_words = [
                            w.lower().strip(".,!?\"'") for w in anchor_key.split()
                            if len(w.strip(".,!?\"'")) >= 3
                            and w.lower().strip(".,!?\"'") not in stop_words
                        ]
                        key_in_resp = any(kw in resp_lower for kw in key_words)
                        
                        # Also check if the original statement's stop words hint at topic
                        # e.g., "likes" (from "The user likes") is a topic indicator
                        stmt_lower = stmt.lower()
                        topic_verbs = ["like", "prefer", "love", "hate", "want", "use", "need"]
                        topic_match = any(
                            v in stmt_lower and v in resp_lower
                            for v in topic_verbs
                        )
                        
                        if key_in_resp or topic_match:
                            return False  # Response discusses same topic but different value

        return True

    async def _agent_loop(
        self,
        model:          str,
        messages:       list[dict],
        adapter:        Optional[BaseLLMAdapter] = None,
        search_backend: Optional[str] = None,
        search_engine:  Optional[str] = None,
        images:         Optional[list[str]] = None,
        agent_id:       str = "default",
        session_id:     str = "unknown",
    ) -> AsyncIterator[dict]:
        tool_turn = 0
        failed_tools: dict[str, int] = {}
        sid = session_id
        current_adapter = adapter or self.adapter

        while True:
            turn_buffer = ""
            turn_images = images if tool_turn == 0 else None

            log("llm", sid, f"Streaming turn {tool_turn} · model={model}")

            # Trace the full prompt for debugging (raw context the LLM sees)
            _trace(agent_id or "default", sid, "llm_prompt", {
                "turn": tool_turn,
                "model": model,
                "message_count": len(messages),
                "estimated_tokens": sum(len(m.get("content", "")) for m in messages) // 4,
                "messages": messages,  # Full raw context
            })

            # Watchdog: ensure the adapter stream doesn't hang forever
            # Python 3.10 compatible — asyncio.timeout() requires 3.11+
            try:
                stream_coro = current_adapter.stream(model=model, messages=messages, images=turn_images)
                token_buffer = []
                
                async def _consume_stream():
                    async for token in stream_coro:
                        token_buffer.append(token)
                
                await asyncio.wait_for(_consume_stream(), timeout=60.0)
                
                for token in token_buffer:
                    turn_buffer += token
                    yield _ev("token", content=token)
                    
            except asyncio.TimeoutError:
                log("llm", sid, "Stream watchdog triggered: model stalled", level="error")
                yield _ev("error", message="LLM Stream stalled. Consider trying a different model or context window.")
                break

            log("llm", sid, f"Turn {tool_turn} complete · {len(turn_buffer)} chars generated", level="ok")

            if not self.tools.has_tools() or tool_turn >= MAX_TOOL_TURNS:
                if tool_turn >= MAX_TOOL_TURNS:
                    log("agent", sid, f"Max tool turns ({MAX_TOOL_TURNS}) reached", level="warn")
                break

            calls = self.tools.detect_tool_calls(turn_buffer)
            if not calls:
                log("agent", sid, "No tool call detected — turn complete")
                break

            for call in calls:
                yield _ev("tool_call", tool_name=call.tool_name, arguments=call.arguments)
            
            yield _ev("retract_last_tokens")

            # BUG FIX: emit internal event so chat_stream can clean full_response
            # Strip the tool-call JSONs that were already streamed to the UI
            clean = turn_buffer
            for call in calls:
                if call.raw_json in clean:
                    clean = clean[:clean.index(call.raw_json)].rstrip()
            yield {"type": "_retract_response", "clean_response": clean}

            tool_turn += 1

            # Execute all tools concurrently or sequentially
            combined_results = []
            
            for call in calls:
                if call.tool_name == "web_search":
                    if search_backend and search_backend != "auto":
                        call.arguments["backend"] = search_backend
                    if search_engine:
                        call.arguments["search_engine"] = search_engine # Inject into kwargs

                args_summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in call.arguments.items())
                log("tool", sid, f"Calling {call.tool_name}({args_summary})")

                yield _ev("tool_running", tool_name=call.tool_name)
                # Pass context (session_id, agent_id) to tool execution for hierarchy support
                context = {"_session_id": sid, "_agent_id": agent_id}
                
                # Use GuardrailMiddleware if available
                if self.middleware:
                    async for event in self.middleware.execute_stream(
                        call,
                        session_id = sid,
                        agent_id   = agent_id,
                        context    = context,
                    ):
                        # Forward middleware events (confirmations, blocks, etc.)
                        # We yield them all, but the tool_result specifically will also 
                        # be used to populate our 'result' object for unified yielding below.
                        if event["type"] != "tool_result":
                            yield event
                        
                        if event["type"] in ("tool_blocked", "confirmation_denied", "confirmation_timeout"):
                            from plugins.tool_registry import ToolResult
                            result = ToolResult(
                                tool_name = call.tool_name,
                                success   = False,
                                content   = event.get("reason", event["type"]),
                            )
                            break
                        
                        if event["type"] == "tool_result":
                            from plugins.tool_registry import ToolResult
                            result = ToolResult(
                                tool_name = call.tool_name,
                                success   = event["success"],
                                content   = event["content"],
                            )
                            break
                else:
                    result = await self.tools.execute(call, context=context)

                if result.success:
                    log("tool", sid, f"{call.tool_name} → {len(result.content)} chars returned", level="ok")
                else:
                    log("tool", sid, f"{call.tool_name} failed: {result.content[:120]}", level="error")

                if not result.success:
                    failed_tools[call.tool_name] = failed_tools.get(call.tool_name, 0) + 1
                else:
                    failed_tools[call.tool_name] = 0

                pivot_msg = ""
                if failed_tools.get(call.tool_name, 0) >= 3:
                    log("agent", sid, f"Solvability guard: hard pivot on {call.tool_name}", level="warn")
                    pivot_msg = (
                        f"\n\n[SYSTEM: Tool '{call.tool_name}' failed 3 times. "
                        "Stop attempting it. Answer with what you know and explain the limitation.]"
                    )

                # Unified yield for tool result - handles both middleware and direct execution
                yield _ev("tool_result", tool_name=call.tool_name, success=result.success, content=result.content)

                # ── Persist tool result to dedicated DB ───────────────────
                try:
                    from memory.tool_results_db import ToolResultsDB
                    result_type = "text"
                    if call.tool_name == "generate_app":
                        result_type = "app_view"
                    elif call.tool_name in ("web_search", "image_search"):
                        result_type = "search"
                    elif call.tool_name in ("file_create", "file_view", "file_str_replace"):
                        result_type = "file_op"
                    elif call.tool_name == "bash":
                        result_type = "bash"
                    
                    ToolResultsDB().store(
                        session_id=sid,
                        tool_name=call.tool_name,
                        arguments=call.arguments,
                        result=result.content[:10000],  # Cap at 10KB per result
                        success=result.success,
                        result_type=result_type,
                        agent_id=agent_id,
                    )
                except Exception as e:
                    log("tool", sid, f"Failed to persist tool result: {e}", level="error")

                
                # Assemble the XML result block for this specific tool call
                combined_results.append(
                    f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                    f"{result.content}\n"
                    f"</SYSTEM_TOOL_RESULT>"
                    f"{pivot_msg}"
                )

            # Append the agent's raw JSONs back into the message history
            messages.append({"role": "assistant", "content": "\n".join(c.raw_json for c in calls)})
            
            # Inject the combined tool results block for the next turn
            messages.append({
                "role": "user",
                "content": (
                    "\n\n".join(combined_results) +
                    "\n\nRead the result(s) above and write your response to the user. Do not repeat the JSON tool call."
                ),
            })

    def _build_messages(
        self,
        system_prompt:      str,
        context:            str,
        sliding_window:     list[dict],
        user_message:       str,
        first_message:      Optional[str] = None,
        message_count:      int = 0,
        historical_anchors: Optional[list[dict]] = None,
        force_memory:       bool = False,
        forced_tools:       Optional[list[str]] = None,
        current_facts:      Optional[list[tuple[str, str, str]]] = None,
    ) -> list[dict]:
        now = datetime.now().strftime("%A, %B %d, %Y")
        parts = [
            f"--- Runtime Metadata ---\n"
            f"Current Date: {now}\n"
            f"---\n\n"
            f"{system_prompt}"
        ]

        if forced_tools:
            tools_str = ", ".join(forced_tools)
            parts.append(
                "--- TOOL OVERRIDE ---\n"
                f"COMMAND: You MUST use the following tools immediately to answer this query: {tools_str}\n"
                "Do not provide a final answer until you have processed the results from these tools.\n"
                "--- END OVERRIDE ---"
            )

        if force_memory:
            parts.append(
                "--- PRIORITY INSTRUCTION ---\n"
                "CRITICAL: The user has requested FORCED MEMORY. "
                "You MUST use the `query_memory` tool immediately to search for any relevant past context "
                "before providing a finalized answer. Do not skip this step.\n"
                "--- END PRIORITY ---"
            )

        if first_message is not None:
            total_turns = max(1, (message_count + 1) // 2)
            parts.append(
                f"--- Session Anchor ---\n"
                f"First message: \"{first_message}\"\n"
                f"Total turns so far: {total_turns}\n"
                f"--- End Anchor ---"
            )

        if current_facts:
            fact_lines = [f"FACT: {s} {p} {o}".strip() for (s, p, o) in current_facts]
            parts.append(
                "--- Deterministic Facts ---\n"
                "The following facts are currently true and override any prior memory:\n"
                + "\n".join(fact_lines)
                + "\n--- End Facts ---"
            )

        if historical_anchors:
            anchor_parts = []
            total_rag_chars = 0
            MAX_RAG_CHARS = 4000 # Hard cap for ALL RAG content combined
            
            for a in historical_anchors:
                key  = a.get("key", "Context")
                fact = a.get("metadata", {}).get("fact", "")
                ctx  = a.get("anchor", "")
                
                # Context Guard: Individual anchor truncation
                if len(ctx) > 600:
                    ctx = f"{ctx[:400]}\n[...truncated...]\n{ctx[-200:]}"
                
                part = f"CONCEPT: {key}\nFACT: {fact}\nEXCHANGE:\n{ctx}"
                
                # Context Guard: Global RAG cap
                if total_rag_chars + len(part) > MAX_RAG_CHARS:
                    anchor_parts.append(f"[...{len(historical_anchors)-len(anchor_parts)} more historical anchors omitted to save context...]")
                    break
                    
                anchor_parts.append(part)
                total_rag_chars += len(part)

            parts.append(
                "--- Historical Context ---\n"
                + "\n\n---\n".join(anchor_parts)
                + "\n--- End Historical Context ---"
            )

        ctx_block = self.ctx_eng.build_context_block(context)
        if ctx_block:
            parts.append(ctx_block)

        tools_block = self.tools.build_tools_block()
        if tools_block:
            parts.append(tools_block)

        messages = [{"role": "system", "content": "\n\n".join(parts)}]
        
        # SLIDING WINDOW with size-limit guard
        max_chars_per_window_msg = 4000
        for m in sliding_window:
            if m["role"] in ("user", "assistant"):
                content = m["content"]
                if len(content) > max_chars_per_window_msg:
                    # Truncate mid-conversation blocks (e.g. huge code dumps) to keep prompt manageable
                    half = max_chars_per_window_msg // 2
                    content = f"{content[:half]}\n\n[...large block truncated...]\n\n{content[-half:]}"
                messages.append({"role": m["role"], "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages