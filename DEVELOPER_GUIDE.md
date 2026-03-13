# Shovs Platform: Developer Implementation Guide (2026)

This guide is for engineers building specialized autonomous agents, custom tools, or entire vertical applications on top of the Shovs Platform.

## 🏗️ Architecture for Builders

The platform follows a **Modular Intelligence** architecture:
1.  **Transport Layer**: FastAPI / WebSockets for streaming turns.
2.  **Orchestration Layer**: `AgenticOrchestrator` manages the loop of "Plan → Act → Observe".
3.  **Tool Layer**: Extensible `Tool` class allows adding bash handlers, web consumers, or custom internal APIs.
4.  **Memory Layer**: Convergence of V2 Graph (determinism) and Vector RAG (semantic lookup).

## 🚀 Quick Start: Building Your First Agent

### 1. Define your Profile
Add a new profile to `orchestration/agent_profiles.py`. You can explicitly list the tools and the model (including provider tags like `groq:` or `openai:`).

```python
AgentProfile(
    id="logistics_bot",
    name="Logistics Optimizer",
    model="groq:llama-3.3-70b-versatile",
    tools=["web_search", "bash", "rag_search"],
    system_prompt="You are an expert in supply chain optimization..."
)
```

### 2. Create Custom Tools
Add your tools to `plugins/tools.py`. Tools must be `async` and return a `str`.

```python
async def _my_custom_tool(param1: str) -> str:
    # Your logic here
    return f"Processed {param1}"

MY_TOOL = Tool(
    name="my_custom_tool",
    description="Does something specialized.",
    parameters={
        "type": "object",
        "properties": {"param1": {"type": "string"}},
        "required": ["param1"]
    },
    handler=_my_custom_tool
)
```

### 3. Connect MCP Servers
2026 is the year of standardized tooling. If your service supports MCP, simply add it to `mcp_servers.json`:

```json
{
  "my-service": {
    "command": "npx",
    "args": ["-y", "@vendor/mcp-server-name", "--api-key", "..."]
  }
}
```

## 🎨 Platinum UI Standards (2026)
Agents are seeded with the `PLATINUM_SYSTEM_PROMPT` which enforces:
- **True Black**: `#000000` backgrounds for all generated UI.
- **SPA Architecture**: All generated apps must be Single-Page Applications using vanilla JS.
- **Real Assets**: No placeholders; use CDN-linked icons (Lucide) and real data.

## 🔒 Security Best Practices
- **Sandbox Everything**: Always use the `bash` tool (which routes through the Docker sandbox) for code execution. Avoid `subprocess.run` on the host.
- **Guardrail Middleware**: Use `guardrail_middleware.py` to require manual confirmation for destructive actions.
- **Memory Isolation**: The platform automatically handles per-session RAG. Ensure your front-end passes the same `session_id` to maintain context.

## 📈 Leveraging the 2026 Context
- **Graph Convergent Memory**: Don't just rely on chat history. Query the `SemanticGraph` for deterministic facts stored during previous sessions.
- **Trace Audit**: Use `agent_trace.jsonl` to debug agent logic—crucial for 2026 compliance and explainability.

---
For deep API docs, run the app and visit `http://localhost:8000/docs`.
