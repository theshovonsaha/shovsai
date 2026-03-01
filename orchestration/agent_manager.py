"""
Agent Manager
-------------
Orchestrates the loading of AgentProfiles and instantiation of specialized AgentCore engines.
"""

from typing import Optional
from orchestration.agent_profiles import ProfileManager, AgentProfile
from engine.core import AgentCore
from plugins.tool_registry import ToolRegistry, Tool
from orchestration.session_manager import SessionManager
from engine.context_engine import ContextEngine
from llm.base_adapter import BaseLLMAdapter
from orchestration.orchestrator import AgenticOrchestrator
from guardrails.middleware import GuardrailMiddleware

class AgentManager:
    def __init__(
        self,
        profiles:        ProfileManager,
        sessions:        SessionManager,
        context_engine:  ContextEngine,
        adapter:         BaseLLMAdapter,
        global_registry: ToolRegistry,
        orchestrator:    Optional[AgenticOrchestrator] = None,
        guardrail_middleware: Optional[GuardrailMiddleware] = None,
    ):
        self.profiles        = profiles
        self.sessions        = sessions
        self.ctx_eng         = context_engine
        self.adapter         = adapter
        self.global_registry = global_registry
        self.orch            = orchestrator
        self.guardrail_middleware = guardrail_middleware
        self._agent_cache: dict[str, AgentCore] = {}  # Cache agent instances

    def get_agent_instance(self, agent_id: str = "default", model_override: Optional[str] = None) -> AgentCore:
        """
        Creates (or returns cached) specialized AgentCore instance based on profile config.
        Filters the global registry down to only allowed tools.
        """
        # Return cached instance if available - but not if model is overridden
        if not model_override and agent_id in self._agent_cache:
            return self._agent_cache[agent_id]

        config = self.profiles.get(agent_id)
        if not config:
            # Fallback to default if requested agent doesn't exist
            config = self.profiles.get("default")
            print(f"[AgentManager] WARNING: Agent '{agent_id}' not found. Falling back to default.")

        # Create a filtered registry
        filtered_registry = ToolRegistry()
        for tool_name in config.tools:
            tool = self.global_registry.get(tool_name)
            if tool:
                filtered_registry.register(tool)
            else:
                print(f"[AgentManager] WARNING: Tool '{tool_name}' requested by agent '{config.name}' but not found in global registry.")

        # Return a specialized core
        print(f"[AgentManager] Specialized Agent '{config.name}' instantiated with {len(filtered_registry.list_tools())} tools.")
        instance = AgentCore(
            adapter=self.adapter,
            context_engine=self.ctx_eng,
            session_manager=self.sessions,
            tool_registry=filtered_registry,
            middleware=self.guardrail_middleware,
            orchestrator=self.orch,
            default_model=model_override or config.model,
            embed_model=config.embed_model,
            default_system_prompt=config.system_prompt,
        )

        self._agent_cache[agent_id] = instance
        return instance

    async def run_agent_task(self, agent_id: str, task: str, parent_id: Optional[str] = None) -> str:
        """
        Runs an agent to completion for a specific task and returns the final response string.
        Creates a isolated child session if a parent_id is provided.
        Inherits the parent session's active model/provider so cloud adapters work correctly.
        """
        # Create a unique, private session for this specific delegation
        import uuid
        child_sid = f"delegated_{uuid.uuid4().hex[:8]}"
        
        # Get target agent config to know its model/prompt
        config = self.profiles.get(agent_id) or self.profiles.get("default")
        
        # ── Inherit parent session's active model ──────────────────────────
        # This fixes the critical bug where delegated agents use `llama3.2`
        # (from the profile) even when the parent is running on Groq/Claude/Gemini.
        effective_model = config.model  # fallback
        delegation_adapter = self.adapter  # fallback
        
        if parent_id:
            parent_session = self.sessions.get(parent_id)
            if parent_session and parent_session.model:
                effective_model = parent_session.model
                # Create the correct provider adapter for the inherited model
                from llm.adapter_factory import create_adapter
                delegation_adapter = create_adapter(provider=effective_model)
        
        from llm.adapter_factory import strip_provider_prefix
        clean_model = strip_provider_prefix(effective_model)
        
        # Initialize the session
        self.sessions.get_or_create(
            session_id=child_sid,
            model=effective_model,
            system_prompt=config.system_prompt,
            agent_id=agent_id,
            parent_id=parent_id
        )

        # Build a fresh agent instance using the inherited adapter + model
        from engine.core import AgentCore
        from engine.context_engine import ContextEngine
        from plugins.tool_registry import ToolRegistry

        filtered_registry = ToolRegistry()
        for tool_name in config.tools:
            tool = self.global_registry.get(tool_name)
            if tool:
                filtered_registry.register(tool)

        delegate_ctx = ContextEngine(adapter=delegation_adapter, compression_model=clean_model)

        # ── Create a fresh orchestrator wired to the delegation adapter ────────
        # BUG FIX: passing self.orch here still used the original adapter (e.g. Ollama),
        # causing the planner to 404 even after fixing the main agent's adapter.
        from orchestration.orchestrator import AgenticOrchestrator
        delegate_orch = AgenticOrchestrator(adapter=delegation_adapter)

        agent = AgentCore(
            adapter=delegation_adapter,
            context_engine=delegate_ctx,
            session_manager=self.sessions,
            tool_registry=filtered_registry,
            middleware=self.guardrail_middleware, # Propagate to delegated agents too
            orchestrator=delegate_orch,      # ← fresh orchestrator with correct adapter
            default_model=effective_model,   # ← FIX: preserve prefix for adapter resolution
            embed_model=config.embed_model,
            default_system_prompt=config.system_prompt,
        )

        full_response = ""
        
        async for event in agent.chat_stream(user_message=task, session_id=child_sid, model=effective_model):
            if event["type"] == "token":
                full_response += event.get("content", "")
            elif event["type"] == "error":
                err_msg = event.get("message", event.get("text", "Unknown error"))
                return f"Error from agent '{agent_id}': {err_msg}"
        
        return full_response.strip()


    def invalidate_cache(self, agent_id: str = None):
        """Clear agent cache — call when profile is updated."""
        if agent_id:
            self._agent_cache.pop(agent_id, None)
        else:
            self._agent_cache.clear()
