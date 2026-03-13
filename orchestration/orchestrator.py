import json
import re
from typing import List, Dict
from llm.base_adapter import BaseLLMAdapter
from config.logger import log

PLANNING_PROMPT = """\
You are the [Shovs Orchestrator]. Analyze the user query and decide which tools are needed.

Available Tools:
{tools_docs}

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

Return ONLY a JSON array of tool name strings. No explanation.
Example: ["web_search", "web_fetch"]

User Query: "{query}"
"""

class AgenticOrchestrator:
    def __init__(self, adapter: BaseLLMAdapter):
        self.adapter = adapter

    def set_adapter(self, adapter: BaseLLMAdapter):
        """Hot-swap the underlying adapter when user switches providers."""
        self.adapter = adapter

    async def plan(self, query: str, tools_list: List[Dict], model: str = "llama3.1:8b") -> List[str]:
        """
        Analyze query and return a list of required tool names.
        """
        tools_docs = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
        prompt = PLANNING_PROMPT.format(tools_docs=tools_docs, query=query)
        
        from llm.adapter_factory import create_adapter, strip_provider_prefix
        current_adapter = create_adapter(provider=model) if ":" in model else self.adapter
        clean_model = strip_provider_prefix(model)
        
        try:
            # Use the specified model and its correct adapter for planning
            response = await current_adapter.complete(
                model=clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Extract JSON list
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                planned_tools = json.loads(match.group(0))
                if isinstance(planned_tools, list):
                    log("orch", "plan", f"Orchestrator strategy: {planned_tools}")
                    return [t for t in planned_tools if isinstance(t, str)]
            
            return []
        except Exception as e:
            log("orch", "plan", f"Orchestrator failed: {e}", level="error")
            return []

