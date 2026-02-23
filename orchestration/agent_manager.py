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
from llm.llm_adapter import OllamaAdapter

class AgentManager:
    def __init__(
        self,
        profiles:        ProfileManager,
        sessions:        SessionManager,
        context_engine:  ContextEngine,
        adapter:         OllamaAdapter,
        global_registry: ToolRegistry,
    ):
        self.profiles        = profiles
        self.sessions        = sessions
        self.ctx_eng         = context_engine
        self.adapter         = adapter
        self.global_registry = global_registry
        self._agent_cache: dict[str, AgentCore] = {}  # Cache agent instances

    def get_agent_instance(self, agent_id: str = "default") -> AgentCore:
        """
        Creates (or returns cached) specialized AgentCore instance based on profile config.
        Filters the global registry down to only allowed tools.
        """
        # Return cached instance if available
        if agent_id in self._agent_cache:
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
            default_model=config.model,
            embed_model=config.embed_model,
            default_system_prompt=config.system_prompt,
        )

        self._agent_cache[agent_id] = instance
        return instance

    def invalidate_cache(self, agent_id: str = None):
        """Clear agent cache — call when profile is updated."""
        if agent_id:
            self._agent_cache.pop(agent_id, None)
        else:
            self._agent_cache.clear()
