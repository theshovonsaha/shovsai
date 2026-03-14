import json
import re
from typing import List, Dict, Any, Optional
from llm.base_adapter import BaseLLMAdapter
from config.logger import log

PLANNING_PROMPT = """\
You are the [Shovs Orchestrator]. Analyze the user query and decide which tools are needed.

Available Tools:
{tools_docs}

Session Signals:
- session_has_history: {session_has_history}
- current_fact_count: {current_fact_count}
- recently_failed_tools: {failed_tools}

Rules:
- Conversational queries ("hi", "how are you", opinions) → return []
- Factual/current data → ["web_search"]
- URL reading → ["web_fetch"]
- File creation/editing → ["file_create"] or ["file_str_replace"]
- Image lookup → ["image_search"]
- Weather → ["weather_fetch"]
- Multi-step tasks → list all required tools in execution order
- Memory recall → ["query_memory"]
- Delegation → ["delegate_to_agent"]

Return ONLY JSON (no markdown). Preferred format:
{{
  "strategy": "one-line plan",
  "tools": [
    {{"name": "tool_name", "priority": "high|medium|low", "reason": "short reason"}}
  ],
  "force_memory": true/false,
  "memory_topic": "topic or empty",
  "confidence": 0.0-1.0
}}

Legacy fallback format allowed: ["web_search", "web_fetch"]

User Query: "{query}"
"""

class AgenticOrchestrator:
    def __init__(self, adapter: BaseLLMAdapter):
        self.adapter = adapter

    def set_adapter(self, adapter: BaseLLMAdapter):
        """Hot-swap the underlying adapter when user switches providers."""
        self.adapter = adapter

    async def plan_with_context(
        self,
        query: str,
        tools_list: List[Dict],
        model: str = "llama3.1:8b",
        session_has_history: bool = False,
        current_fact_count: int = 0,
        failed_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze query and return structured execution guidance."""
        tools_docs = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
        known_tools = {t["name"] for t in tools_list if isinstance(t, dict) and t.get("name")}
        failed_set = set(failed_tools or [])
        prompt = PLANNING_PROMPT.format(
            tools_docs=tools_docs,
            query=query,
            session_has_history=str(bool(session_has_history)).lower(),
            current_fact_count=current_fact_count,
            failed_tools=", ".join(sorted(failed_set)) if failed_set else "none",
        )
        
        from llm.adapter_factory import create_adapter, strip_provider_prefix
        current_adapter = create_adapter(provider=model) if ":" in model else self.adapter
        clean_model = strip_provider_prefix(model)
        
        try:
            response = await current_adapter.complete(
                model=clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            def _normalize_tools(entries: List[Any]) -> List[Dict[str, str]]:
                normalized = []
                for entry in entries:
                    if isinstance(entry, str):
                        name = entry
                        priority = "medium"
                        reason = "Planner-selected tool"
                    elif isinstance(entry, dict):
                        name = entry.get("name")
                        priority = str(entry.get("priority", "medium")).lower()
                        reason = str(entry.get("reason", "Planner-selected tool"))
                    else:
                        continue
                    if not isinstance(name, str) or name not in known_tools or name in failed_set:
                        continue
                    if priority not in {"high", "medium", "low"}:
                        log("orch", "plan", f"Invalid planner priority '{priority}' for tool '{name}'. Using medium.", level="warn")
                        priority = "medium"
                    normalized.append({"name": name, "priority": priority, "reason": reason})
                priority_rank = {"high": 0, "medium": 1, "low": 2}
                normalized.sort(key=lambda item: (priority_rank[item["priority"]], item["name"]))
                return normalized

            payload: Dict[str, Any] = {}
            obj_match = re.search(r'\{.*\}', response, re.DOTALL)
            if obj_match:
                maybe_obj = json.loads(obj_match.group(0))
                if isinstance(maybe_obj, dict):
                    payload = maybe_obj
            else:
                arr_match = re.search(r'\[.*\]', response, re.DOTALL)
                if arr_match:
                    maybe_list = json.loads(arr_match.group(0))
                    if isinstance(maybe_list, list):
                        payload = {"tools": maybe_list}

            tools = _normalize_tools(payload.get("tools", []))
            # If we have history/facts and no explicit tool choice, bias toward memory lookup.
            if (session_has_history or current_fact_count > 0) and not any(t["name"] == "query_memory" for t in tools):
                if "query_memory" in known_tools and "query_memory" not in failed_set:
                    tools.insert(0, {
                        "name": "query_memory",
                        "priority": "high",
                        "reason": "Session has prior context worth checking."
                    })

            structured = {
                "strategy": str(payload.get("strategy", "Use selected tools to gather evidence before final answer.")),
                "tools": tools,
                "force_memory": bool(payload.get("force_memory", session_has_history or current_fact_count > 0)),
                "memory_topic": str(payload.get("memory_topic", query[:80])).strip(),
                "confidence": float(payload.get("confidence", 0.5)),
            }
            log("orch", "plan", f"Orchestrator strategy: {[t['name'] for t in tools]}")
            return structured
        except Exception as e:
            log("orch", "plan", f"Orchestrator failed: {e}", level="error")
            return {
                "strategy": "Planner failed; continue with direct reasoning.",
                "tools": [],
                "force_memory": False,
                "memory_topic": "",
                "confidence": 0.0,
            }

    async def plan(self, query: str, tools_list: List[Dict], model: str = "llama3.1:8b") -> List[str]:
        """
        Backward-compatible planning API: returns tool names only.
        """
        structured = await self.plan_with_context(query=query, tools_list=tools_list, model=model)
        return [t["name"] for t in structured.get("tools", []) if isinstance(t, dict) and isinstance(t.get("name"), str)]
