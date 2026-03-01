"""
MCP Client
----------
Connects to any MCP (Model Context Protocol) server and auto-registers
its tools into your existing ToolRegistry as native Tool objects.

This makes every MCP server in the ecosystem (GitHub, Slack, Linear,
Postgres, browser automation, etc.) instantly available to your agents
without writing a single tool handler.

Requirements:
    pip install mcp

Usage in main.py:
    from plugins.mcp_client import MCPClientManager
    mcp_manager = MCPClientManager(tool_registry)
    await mcp_manager.connect_server("github", "npx", ["-y", "@modelcontextprotocol/server-github"])
    await mcp_manager.connect_server("filesystem", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])

Or via config file (mcp_servers.json):
    await mcp_manager.load_from_config("mcp_servers.json")

mcp_servers.json format:
    {
      "servers": [
        {
          "id": "github",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "env": {"GITHUB_TOKEN": "your-token"}
        }
      ]
    }
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from plugins.tool_registry import Tool, ToolRegistry
from config.logger import log


class MCPClientManager:
    """
    Manages connections to multiple MCP servers.
    Each server's tools are registered into the shared ToolRegistry.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry   = registry
        self._sessions: dict[str, Any] = {}   # server_id → active session
        self._contexts: dict[str, Any] = {}   # server_id → context managers

    async def connect_server(
        self,
        server_id: str,
        command:   str,
        args:      list[str],
        env:       Optional[dict] = None,
    ) -> int:
        """
        Connect to an MCP server via stdio and register all its tools.
        Returns the number of tools registered.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            log("mcp", server_id, "MCP SDK not installed. Run: pip install mcp", level="error")
            return 0

        server_env = {**os.environ, **(env or {})}
        params = StdioServerParameters(command=command, args=args, env=server_env)

        try:
            read, write = await stdio_client(params).__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            # Store for cleanup
            self._sessions[server_id] = session

            # List and register all tools
            tools_response = await session.list_tools()
            count = 0
            for mcp_tool in tools_response.tools:
                tool = self._wrap_mcp_tool(server_id, session, mcp_tool)
                self.registry.register(tool)
                count += 1

            log("mcp", server_id, f"Connected · {count} tools registered", level="ok")
            return count

        except Exception as e:
            log("mcp", server_id, f"Failed to connect: {e}", level="error")
            return 0

    def _wrap_mcp_tool(self, server_id: str, session: Any, mcp_tool: Any) -> Tool:
        """
        Wraps an MCP tool definition into a native Tool object.
        The handler calls back to the MCP session when invoked.
        """
        # Capture by closure
        _session    = session
        _tool_name  = mcp_tool.name
        _server_id  = server_id

        async def handler(**kwargs) -> str:
            # Strip internal context keys before passing to MCP
            clean_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            try:
                result = await _session.call_tool(_tool_name, clean_kwargs)
                # MCP returns a list of content blocks
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif hasattr(block, "data"):
                        parts.append(f"[binary data: {len(block.data)} bytes]")
                return "\n".join(parts) if parts else "Tool returned no output."
            except Exception as e:
                return f"MCP tool error ({_server_id}/{_tool_name}): {e}"

        # Use MCP tool's own JSON schema for parameters
        parameters = {}
        if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
            parameters = mcp_tool.inputSchema
        else:
            parameters = {"type": "object", "properties": {}}

        return Tool(
            name        = f"{server_id}__{mcp_tool.name}",   # namespaced to avoid collisions
            description = f"[MCP:{server_id}] {mcp_tool.description or mcp_tool.name}",
            parameters  = parameters,
            handler     = handler,
            tags        = ["mcp", server_id],
        )

    async def load_from_config(self, config_path: str = "mcp_servers.json") -> int:
        """
        Load and connect all MCP servers defined in a JSON config file.
        Returns total tools registered across all servers.
        """
        path = Path(config_path)
        if not path.exists():
            log("mcp", "config", f"Config file not found: {config_path}", level="warn")
            return 0

        try:
            config = json.loads(path.read_text())
        except Exception as e:
            log("mcp", "config", f"Failed to parse {config_path}: {e}", level="error")
            return 0

        total = 0
        for server in config.get("servers", []):
            server_id = server.get("id", "unknown")
            command   = server.get("command", "npx")
            args      = server.get("args", [])
            env       = server.get("env", {})

            if not server.get("enabled", True):
                log("mcp", server_id, "Skipped (disabled in config)")
                continue

            count = await self.connect_server(server_id, command, args, env)
            total += count

        return total

    async def disconnect_all(self):
        """Clean up all active MCP sessions. Call on app shutdown."""
        for server_id, session in self._sessions.items():
            try:
                await session.__aexit__(None, None, None)
                log("mcp", server_id, "Disconnected")
            except Exception:
                pass
        self._sessions.clear()

    def list_connected(self) -> list[str]:
        return list(self._sessions.keys())
