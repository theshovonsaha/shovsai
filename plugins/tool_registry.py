"""
Tool Registry
-------------
Modular plugin system. Register tools; AgentCore routes to them.

Protocol:
  LLM emits:    {"tool": "name", "arguments": {...}}
  System calls: handler(**arguments)
  Result injects back as tool_result in next user message.
  LLM continues reasoning with result in context.

FIX v2: Replaced fragile regex with proper JSON scanner that handles:
  - Nested objects in arguments (bash commands, complex queries)
  - Multiple JSON objects in the same output (takes first valid tool call)
  - Whitespace variations and model quirks

To add a tool:
  1. Write an async handler function
  2. Define its JSON Schema parameters
  3. registry.register(Tool(...))
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Tool:
    name:        str
    description: str
    parameters:  dict
    handler:     Callable
    tags:        list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    tool_name:  str
    arguments:  dict
    raw_json:   str


@dataclass
class ToolResult:
    tool_name: str
    success:   bool
    content:   str


def _extract_json_objects(text: str) -> list[tuple[dict, str]]:
    """
    Extract all valid JSON objects from text using a brace-counting scanner.
    Handles nested objects, escaped strings — things regex cannot.
    Returns list of (parsed_dict, original_substring) tuples.
    """
    results = []
    i = 0
    while i < len(text):
        if text[i] != '{':
            i += 1
            continue
        # Found a potential object start — scan to find matching close brace
        depth = 0
        in_string = False
        escape_next = False
        start = i
        j = i
        while j < len(text):
            ch = text[j]
            if escape_next:
                escape_next = False
                j += 1
                continue
            if ch == '\\' and in_string:
                escape_next = True
                j += 1
                continue
            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                results.append((parsed, candidate))
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i = j + 1
    return results


class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        print(f"[ToolRegistry] Registered: {tool.name}")

    def unregister(self, name: str):
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has_tools(self) -> bool:
        return bool(self._tools)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def build_tools_block(self) -> str:
        """Inject tool descriptions into system prompt."""
        if not self._tools:
            return ""
            
        tool_docs = []
        for t in self._tools.values():
            params = t.parameters.get("properties", {})
            required = t.parameters.get("required", [])
            
            param_str = []
            for name, prop in params.items():
                req_mark = "(REQUIRED)" if name in required else "(optional)"
                desc = prop.get("description", "")
                ptype = prop.get("type", "string")
                param_str.append(f'      "{name}": <{ptype}> {req_mark} - {desc}')
                
            args_block = "{\n" + ",\n".join(param_str) + "\n    }" if params else "{}"
            
            tool_docs.append(
                f"  Tool: {t.name}\n"
                f"  Description: {t.description}\n"
                f"  Arguments Schema:\n    {args_block}"
            )
            
        doc_string = "\n\n".join(tool_docs)

        return (
            "--- Available Tools ---\n"
            "To use a tool, you MUST output ONLY the following JSON on its own line and then STOP:\n"
            '{"tool": "<tool_name>", "arguments": {<args>}}\n\n'
            "CRITICAL: Do NOT output the tool's schema definition. You must output the actual invocation with real values.\n\n"
            f"{doc_string}\n"
            "--- End Tools ---"
        )

    def detect_tool_call(self, text: str) -> Optional[ToolCall]:
        """
        Parse LLM output for a tool call JSON block.
        Uses brace-counting JSON scanner — handles nested objects correctly.
        Returns first valid tool call found, or None.
        """
        calls = self.detect_tool_calls(text)
        return calls[0] if calls else None

    def detect_tool_calls(self, text: str) -> list[ToolCall]:
        """
        Parse LLM output for all tool call JSON blocks.
        Returns a list of all valid tool calls found.
        """
        calls = []
        candidates = _extract_json_objects(text)
        for obj, original_text in candidates:
            tool_name = obj.get("tool")
            arguments = obj.get("arguments")
            if (
                tool_name
                and isinstance(tool_name, str)
                and isinstance(arguments, dict)
                and tool_name in self._tools
            ):
                calls.append(ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_json=original_text,
                ))
        return calls


    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return a ToolResult."""
        tool = self._tools.get(call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=f"Tool '{call.tool_name}' not found.",
            )
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                raw = await tool.handler(**call.arguments)
            else:
                raw = tool.handler(**call.arguments)
            content = raw if isinstance(raw, str) else json.dumps(raw)
            return ToolResult(tool_name=call.tool_name, success=True, content=content)
        except TypeError as e:
            # Argument mismatch — give model useful error to self-correct
            import inspect
            sig = inspect.signature(tool.handler)
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=(
                    f"Tool argument error: {e}\n"
                    f"Expected signature: {sig}\n"
                    f"Received arguments: {list(call.arguments.keys())}"
                ),
            )
        except Exception as e:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=f"Tool error: {e}",
            )