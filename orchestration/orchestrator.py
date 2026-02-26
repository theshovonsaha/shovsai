import json
from typing import List, Dict
from llm.base_adapter import BaseLLMAdapter
from config.logger          import log

PLANNING_PROMPT = """\
You are the [Antigravity Orchestrator]. Your job is to analyze a user query and determine which tools (if any) are absolutely necessary to resolve it efficiently.

Available Tools:
{tools_docs}

Guidelines:
- If the query is simple prose (e.g. "hi", "how are you"), return an empty list.
- If the query requires factual lookup, use 'web_search'.
- If the query involves creating or modifying files, use 'file_create' or 'file_str_replace'.
- If the query involves images, use 'image_search'.
- return ONLY a JSON list of tool names. Example: ["web_search", "file_create"]

User Query: "{query}"
"""

class AgenticOrchestrator:
    def __init__(self, adapter: BaseLLMAdapter):
        self.adapter = adapter

    async def plan(self, query: str, tools_list: List[Dict], model: str = "llama3.1:8b") -> List[str]:
        """
        Analyze query and return a list of required tool names.
        """
        tools_docs = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
        prompt = PLANNING_PROMPT.format(tools_docs=tools_docs, query=query)
        
        try:
            # Use the faster model for planning
            response = await self.adapter.complete(
                model=model,
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

import re # needed for search
