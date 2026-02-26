import asyncio
from engine.core import AgentCore
from orchestration.orchestrator import AgenticOrchestrator
from llm.llm_adapter import OllamaAdapter
from engine.context_engine import ContextEngine
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolRegistry

async def test():
    adapter = OllamaAdapter()
    tr = ToolRegistry()
    ce = ContextEngine(adapter=adapter)
    sm = SessionManager()
    orch = AgenticOrchestrator(adapter=adapter)
    
    core = AgentCore(adapter, ce, sm, tr, orchestrator=orch)
    
    print("--- Test with use_planner=True ---")
    async for ev in core.chat_stream("hello", use_planner=True):
        if ev['type'] == 'plan':
            print("PLAN EVENT DETECTED")
            
    print("\n--- Test with use_planner=False ---")
    async for ev in core.chat_stream("hello", use_planner=False):
        if ev['type'] == 'plan':
            print("PLAN EVENT DETECTED (SHOULD NOT HAPPEN)")

    print("\n--- Test with use_planner='false' (String) ---")
    async for ev in core.chat_stream("hello", use_planner='false'):
        if ev['type'] == 'plan':
            print("PLAN EVENT DETECTED (BUG: STRING 'false' IS TRUTHY)")

if __name__ == "__main__":
    asyncio.run(test())
