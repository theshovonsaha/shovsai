"""
AgentCore v3 — intelligence fixes + structured tracing
"""

import asyncio
import json
import os
from datetime import datetime
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter
from llm.llm_adapter     import OllamaAdapter, LLMError
from engine.context_engine  import ContextEngine
from orchestration.session_manager import SessionManager
from plugins.tool_registry   import ToolRegistry
from memory.vector_engine   import VectorEngine
from llm.adapter_factory    import create_adapter, strip_provider_prefix
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
You are a highly advanced AI agent running on the [Antigravity Agent Platform].

Architectural Self-Awareness:
- Patching & Memory: I use a dual-layer memory system. Short-term context is managed by "ContextEngine v2" which compresses and ranks facts using a rolling window. Long-term memory is "Semantic Graph v3" (RAG), which stores and retrieves declarative facts via `query_memory` and `store_memory` tools.
- Vector RAG: Historical anchors are automatically retrieved and injected into my prompt to maintain consistency across sessions.

Persona & Style Rules:
- CRITICAL: Always prioritize the "Historical Context" and "Session Memory" blocks when determining your persona, tone, and style. If a user preference or persona (e.g. "Tony Stark") is stored, follow it religiously over any default behavior.
- Use `query_memory` early in a conversation to re-discover user preferences if they aren't already in context.

Tool Use Rules:
- Output ONLY JSON calls for tools: {"tool": "<n>", "arguments": {<args>}}
- You MAY output multiple tools back-to-back.
- Live View: Wrap visual data in ```html or ```svg blocks.
"""

MAX_TOOL_TURNS = 6

def _ev(type_: str, **kw) -> dict:
    return {"type": type_, **kw}


class AgentCore:

    def __init__(
        self,
        adapter:         OllamaAdapter,
        context_engine:  ContextEngine,
        session_manager: SessionManager,
        tool_registry:   ToolRegistry,
        default_model:   str = "llama3.2",
        embed_model:     str = "nomic-embed-text",
        default_system_prompt: str = None,
    ):
        self.adapter   = adapter
        self.ctx_eng   = context_engine
        self.sessions  = session_manager
        self.tools     = tool_registry
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
        images:         Optional[list[str]] = None,
        force_memory:   bool = False,
        forced_tools:   Optional[list[str]] = None,
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
            # If the requested model suggests a different provider than current adapter
            current_use_adapter = self.adapter
            
            # Smart switching logic:
            if model:
                # 1. Explicit prefix takes priority
                if ":" in model:
                    current_use_adapter = create_adapter(provider=model)
                # 2. Known model patterns for cloud providers
                elif any(p in model.lower() for p in ["gpt-", "llama-3.3", "mixtral", "groq/", "gemini-"]):
                    provider = "openai"
                    if "groq/" in model or "llama-3.3" in model or "mixtral" in model:
                        provider = "groq"
                    elif "gemini-" in model:
                        provider = "gemini"
                    current_use_adapter = create_adapter(provider=provider)
            
            # Use clean model name for API calls
            clean_model = strip_provider_prefix(model)
            
            # ── Vector RAG ────────────────────────────────────────────────────
            ve = VectorEngine(sid, agent_id=session.agent_id, model=self.embed_model)
            historical_anchors = await ve.query(user_message, limit=3)

            if historical_anchors:
                keys = [a.get("key", "?") for a in historical_anchors]
                log("rag", sid, f"Retrieved {len(historical_anchors)} anchors: {', '.join(keys)}", level="ok")
            else:
                log("rag", sid, "No relevant history anchors found")

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
            )

            prompt_tokens_est = sum(len(m["content"]) for m in messages) // 4
            log("llm", sid, f"Prompt built · {len(messages)} messages · ~{prompt_tokens_est} tokens est.")

            # Trace: log the actual prompt chain sent to LLM
            _trace(agent_id, sid, "prompt_chain", {
                "model": model,
                "message_count": len(messages),
                "system_prompt_preview": messages[0]["content"][:300] if messages else "",
                "user_message": user_message[:200],
            })

            try:
                async for event in self._agent_loop(
                    model=clean_model,
                    messages=messages,
                    adapter=current_use_adapter,
                    search_backend=search_backend,
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
            except Exception as e:
                log("agent", sid, f"Unexpected error: {e}", level="error")
                yield _ev("error", message=f"Unexpected: {e}")
                return

        if error_msg:
            return

        # BUG FIX: strip any remaining tool-call JSON from full_response before storing
        clean_response = self._strip_tool_json(full_response)

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
            log("ctx", sid, f"Compressing exchange with {model}...")
            updated_ctx, keyed_facts = await self.ctx_eng.compress_exchange(
                user_message=user_message,
                assistant_response=clean_response,  # Use cleaned response, not raw
                current_context=session.compressed_context,
                is_first_exchange=is_first_exchange,
                model=model,
            )
            self.sessions.update_context(sid, updated_ctx)
            lines = len([l for l in updated_ctx.split("\n") if l.strip()])
            log("ctx", sid, f"Context updated · {lines} lines · {len(keyed_facts)} new facts", level="ok")

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
        """Remove tool-call JSON from response text before storing.
        The model sometimes emits {"tool": ...} inline before writing prose or chains them.
        We need to strip this so session history stays clean."""
        import re
        # Remove ALL JSON tool calls that appear in the text, including sequences 
        # like `{"tool": "..."}; {"tool": "..."}`
        cleaned = text
        pattern = r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}[\n\s;]*'
        cleaned = re.sub(pattern, '', cleaned).strip()
        
        # Sometimes models output standalone semicolons from chained sequences
        cleaned = re.sub(r'^;+\s*', '', cleaned)
        return cleaned if cleaned else text

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

            async for token in current_adapter.stream(model=model, messages=messages, images=turn_images):
                turn_buffer += token
                yield _ev("token", content=token)

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
                if call.tool_name == "web_search" and search_backend and search_backend != "auto":
                    call.arguments["backend"] = search_backend
                    log("tool", sid, f"Overriding search backend → {search_backend}")

                args_summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in call.arguments.items())
                log("tool", sid, f"Calling {call.tool_name}({args_summary})")

                yield _ev("tool_running", tool_name=call.tool_name)
                result = await self.tools.execute(call)

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

                yield _ev("tool_result", tool_name=call.tool_name, success=result.success)
                
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
    ) -> list[dict]:
        parts = [system_prompt]

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

        if historical_anchors:
            anchor_parts = []
            for a in historical_anchors:
                key  = a.get("key", "Context")
                fact = a.get("metadata", {}).get("fact", "")
                ctx  = a.get("anchor", "")
                anchor_parts.append(f"CONCEPT: {key}\nFACT: {fact}\nEXCHANGE:\n{ctx}")
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
        for m in sliding_window:
            if m["role"] in ("user", "assistant"):
                messages.append(m)
        messages.append({"role": "user", "content": user_message})
        return messages