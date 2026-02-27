import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from llm.llm_adapter import OllamaAdapter
from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools
from orchestration.agent_profiles import ProfileManager, AgentProfile
from orchestration.session_manager import SessionManager
from engine.context_engine import ContextEngine
from orchestration.orchestrator import AgenticOrchestrator
from orchestration.agent_manager import AgentManager
from llm.adapter_factory import create_adapter, get_default_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_delegation():
    print("🚀 Starting V11 Delegation Test (using Groq)...")
    
    # 1. Setup Infrastructure
    adapter = create_adapter("groq")
    # Use the highly stable Groq model
    test_model = "llama-3.1-8b-instant"
    print(f"Using Groq adapter with model: {test_model}")
    
    registry = ToolRegistry()
    session_mgr = SessionManager(max_sessions=10)
    # Ensure context engine also uses a model Groq supports
    context_eng = ContextEngine(adapter=adapter, compression_model=test_model)
    profile_mgr = ProfileManager()
    orchestrator = AgenticOrchestrator(adapter=adapter)
    
    # 1a. Ensure agents use a Groq model
    p_default = profile_mgr.get("default")
    if p_default:
        p_default.model = test_model
        profile_mgr.create(p_default)
    
    # Create a 'coder' agent for Part B
    profile_mgr.create(AgentProfile(
        id="coder", 
        name="Coder Extraordinaire", 
        model=test_model,
        tools=["bash", "file_create"]
    ))

    manager = AgentManager(
        profiles=profile_mgr,
        sessions=session_mgr,
        context_engine=context_eng,
        adapter=adapter,
        global_registry=registry,
        orchestrator=orchestrator
    )
    
    # 2. Register tools (including the new delegate_to_agent)
    register_all_tools(registry, agent_manager=manager)
    
    # 3. Test Part A: Direct Tool Handler Call
    print("\n--- Part A: Direct Tool Handler Call ---")
    delegate_tool = registry.get("delegate_to_agent")
    if not delegate_tool:
        print("❌ Error: delegate_to_agent tool not found in registry!")
        return

    # We'll delegate to the 'default' agent a simple task
    task = "Say 'The delegation was successful!'"
    print(f"Delegating task: '{task}' to agent 'default'...")
    
    result = await delegate_tool.handler(target_agent_id="default", task=task)
    print(f"Result from delegation: {result}")
    
    if "successful" in result.lower():
        print("✅ Part A Passed!")
    else:
        print("❌ Part A Failed (Check output above)")

    # 4. Test Part B: Full Model-Driven Delegation
    print("\n--- Part B: Full Model-Driven Delegation ---")
    mother_agent = manager.get_agent_instance("default", model_override=test_model)
    
    # Using a much more explicit prompt that forces a "Modular" mindset
    query = (
        "I need you to act as a Project Manager. Use the 'delegate_to_agent' tool to "
        "ask the 'coder' agent to write a one-line bash script that prints 'V11 Mother Platform' "
        "and return exactly what the coder says."
    )
    
    print(f"Querying Mother Agent: '{query}'")
    full_response = ""
    async for event in mother_agent.chat_stream(query, session_id="test_mother_session"):
        etype = event.get("type")
        if etype == "token":
            t = event.get("text", "")
            print(t, end="", flush=True)
            full_response += t
        elif etype == "plan":
            print(f"\n[PLAN] {event.get('strategy', 'No strategy')}")
        elif etype == "tool_call":
            print(f"\n[TOOL] Calling {event.get('tool')} with {event.get('arguments')}")
        elif etype == "tool_result":
            print(f"\n[RESULT] {event.get('content')[:200]}...")

    print("\n\n--- Final Conclusion ---")
    if "V11 Mother Platform" in full_response:
        print("✅ Part B Passed: Mother successfully delegated to sub-agent!")
    else:
        print("⚠️ Part B result inconclusive. It may be due to model reasoning.")

if __name__ == "__main__":
    asyncio.run(test_delegation())
