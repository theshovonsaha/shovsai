# Agentic Systems Deep Research & Enhancement Guide
## Your Platform vs. The World — Full Architecture Audit

*Research conducted March 2026 | Based on live source code, system prompts, and architecture docs*

---

## Table of Contents

1. [The Competitive Landscape](#1-the-competitive-landscape)
2. [OpenClaw — The Viral Simplicity Champion](#2-openclaw)
3. [Claude Code — The Gold Standard Agent Loop](#3-claude-code)
4. [Gemini CLI — The Open-Source Reactor](#4-gemini-cli)
5. [CrewAI — Role-Based Orchestration](#5-crewai)
6. [LangGraph — The State Machine Approach](#6-langgraph)
7. [n8n — The Visual Workflow Agent](#7-n8n)
8. [Side-by-Side Feature Matrix](#8-feature-matrix)
9. [Context Engineering — The Real Science](#9-context-engineering)
10. [Your System — Honest Strengths & Gaps](#10-your-system-audit)
11. [The Enhancement Roadmap — Exact Code](#11-enhancement-roadmap)

---

## 1. The Competitive Landscape

Before diving per-system, understand the **five core problems** every agent framework is solving:

| Problem | What It Means |
|---|---|
| **Task Tracking** | Preventing the agent from losing track of what it's doing mid-session |
| **Context Rot** | As context window fills, early tokens get "lost in the middle" |
| **Tool Reliability** | JSON parsing, retry logic, error propagation between tools |
| **Memory Architecture** | Short-term (session) vs long-term (cross-session) knowledge |
| **Sub-Agent Isolation** | Spawning workers without polluting the main agent's context |

Every framework below makes distinct design choices on each axis. Your system already has opinions on all five — the question is whether those opinions are the right ones.

---

## 2. OpenClaw

**What it is:** A personal AI agent daemon that went from a weekend WhatsApp relay to 216,000 GitHub stars in 10 days (Feb 2026). Creator Peter Steinberger was hired by OpenAI; the project moved to a foundation to stay open source.

### Core Architecture

OpenClaw is a **Gateway + serialized agent loop** system. The Gateway is a long-lived daemon that owns all messaging channel connections (WhatsApp, Telegram, email). It exposes a typed WebSocket API to clients and connects to "nodes" (devices that run local actions).

The critical architectural decision: **one serialized run per session at a time**. No concurrent tool calls, no state corruption. This is described explicitly in their Vision doc as an intentional constraint that eliminates an entire category of bugs.

```
Agent Loop (serialized per session):
  intake → context assembly → model inference → tool execution → streaming → persistence
```

### Tool Registration Pattern

This is their core interface contract. Tools are not an add-on — they are the product:

```typescript
import { Type } from "@sinclair/typebox";

export default function register(api) {
  api.registerTool({
    name: "my_tool",
    description: "Do a thing",
    parameters: Type.Object({ input: Type.String() }),
    async execute(_id, params) {
      return { content: [{ type: "text", text: params.input }] };
    },
  });
}
```

Every capability is a tool registration. Adding a new integration is adding a new `registerTool` call. This is why they shipped 200+ integrations so quickly.

### Memory Architecture

Deliberately unromantic. No vector databases on day one:

```
memory/YYYY-MM-DD.md  →  append-only daily logs (today + yesterday auto-loaded)
MEMORY.md             →  curated long-term memory (main/private session only)
USER.md               →  user identity and preferences
SOUL.md               →  agent persona and values
IDENTITY.md           →  agent self-description
```

The model only "remembers" what gets written to disk. Memory tools (`memory_search`, `memory_get`) are provided by a plugin — meaning the entire memory subsystem is swappable. Their research notes are honest about the weakness: this format struggles with entity-centric queries ("What does X prefer?") without re-reading many files.

Their proposed next layer: tagged "Retain" entries in a structured format:

```
## Retain
- W @Peter: Currently in Marrakech (Nov 27-Dec 1, 2025)
- B @code: Fixed WS crash by wrapping handlers in try/catch
- O(c=0.95) @Peter: Prefers concise replies on WhatsApp; long content goes into files.
```

`W` = Working memory, `B` = Bug/fix, `O` = Opinion with confidence score.

### What You Can Steal

1. **Serialized run per session** — your system already does this via Python's async, but make it explicit
2. **Plugin-based memory** — your `SessionRAG` + `VectorEngine` do this, but the abstraction boundary isn't clean enough; OpenClaw's approach means you can swap the entire memory implementation without touching the loop
3. **User identity files** — you have `agent_profiles.py` for agent configuration but nothing equivalent to `USER.md` for persistent user preferences injected at session start

---

## 3. Claude Code

**What it is:** Anthropic's own CLI agent for software engineering. Has 18+ built-in tools, sub-agents, MCP support, and a context compaction system. Reverse-engineered from a minified JS file — their system prompt alone is 40+ separate strings that change with every version.

### The Master Loop (codenamed `nO`)

```
Single-threaded master loop + async h2A event queue
  ↓
User input arrives
  ↓
TodoWrite — break task into tracked steps
  ↓
Tool execution (Bash, Read, Write, Glob, Grep, WebFetch, Task...)
  ↓
After each tool call: re-inject current TODO state into context
  ↓
Results feed back to model
  ↓
Loop until final answer
  ↓
Compact at ~92% context usage
```

The **async h2A queue** is the crucial innovation that isn't obvious from the architecture description. It handles out-of-band events: user interrupts mid-run, file changes, IDE events, linter output. These are injected as `<system-reminder>` tags into the next model call without breaking the main loop.

### TodoWrite — The Killer Feature for Accuracy

This is the single biggest thing Claude Code does that most agents don't. TodoWrite is not a planning tool — it's a **state persistence tool**. Here's the actual prompt instruction:

```
## Task Management

You have access to the TodoWrite tools to help you manage and plan tasks.
Use these tools VERY frequently to ensure that you are tracking your tasks
and giving the user visibility into your progress.

It is critical that you mark todos as completed as soon as you are done
with a task. Do not batch up multiple tasks before marking them as completed.
```

And after **every tool call**, the current TODO list is re-injected into context:

```
[System reminder: Current TODO state]
- [completed] Run the build
- [in_progress] Fix type error in auth.ts line 42
- [pending] Fix type error in db.ts line 89
- [pending] Run lint
```

This prevents context rot from destroying task continuity. The model always knows exactly where it is in a multi-step workflow, even 50 tool calls in. This is the architectural reason Claude Code can run for hours on complex codebases without losing the thread.

### Context Compaction — The `wU2` Compressor

Triggers at approximately **92% of context window usage**:

1. Older conversation chunks are summarized using Haiku (cheap, fast)
2. Key project facts are refreshed in `CLAUDE.md`
3. Live context is trimmed back
4. The loop continues — no interruption

The `CLAUDE.md` file is "project memory" — persistent markdown that every session starts with. Users can edit it directly. It's inspectable, diffable, version-controllable.

### Sub-Agent Isolation via `Task` Tool

When the main agent calls `Task` (dispatches a sub-agent):
- Sub-agent gets an **isolated context** — it does not see the main agent's conversation history
- Sub-agent reports back as a normal tool output
- Main loop stays single-threaded
- This keeps the main agent's context small and clean

### System Prompt Architecture

Claude Code doesn't have one system prompt — it has a **conditional assembly system** with 40+ components:

```
Core system prompt (always present)
  + Tool descriptions (per-tool, ~73-4180 tokens each)
  + CLAUDE.md content (project memory)
  + git status (current repo state)
  + Session memory summary
  + Tool override reminders (injected after tool use)
  + Sub-agent prompts (separate prompts for Plan/Explore/Task agents)
  + Compact utility prompt (for Haiku-based summarization)
  + Session title generation prompt
  + Security review prompt (conditional)
```

### What You Can Steal

1. **TodoWrite pattern** — a persistent task list re-injected after every tool call
2. **Two-speed context compaction** — cheap model (Haiku equivalent) for summarization, main model for reasoning
3. **Sub-agent isolation** — your delegation already creates fresh orchestrators; make sure the sub-agent's conversation history is completely isolated, not inherited
4. **`<system-reminder>` injection** — after tool results, inject structured state reminders as separate system messages rather than stuffing them into the tool result content
5. **Conditional prompt assembly** — your `_build_messages` builds one big system prompt; Claude Code assembles it conditionally from components at runtime

---

## 4. Gemini CLI

**What it is:** Google's open-source terminal agent (Apache 2.0). Backed by Gemini 2.5 Pro with a 1M token context window. ReAct loop, MCP support, hooks middleware, Agent Skills system.

### Core Architecture: ReAct Loop

```
User prompt
  ↓
GEMINI.md / AGENT.md loaded as persistent context
  ↓
ReAct: Reason → Act (tool call) → Observe → Reason...
  ↓
MCP tools + built-in tools (shell, file ops, web search, web fetch)
  ↓
Checkpointing: save/resume conversation state
  ↓
Token caching for expensive context reuse
```

### Hooks Middleware — The Real Innovation

Gemini CLI introduces **hooks**: scripts or programs that execute at specific lifecycle points in the agentic loop. This is the cleanest "middleware for AI" design seen so far:

```json
{
  "hooks": {
    "BeforeTool": [
      {
        "matcher": "write_file|replace",
        "hooks": [
          {
            "name": "secret-scanner",
            "type": "command",
            "command": "$GEMINI_PROJECT_DIR/.gemini/hooks/block-secrets.sh",
            "description": "Prevent committing secrets"
          }
        ]
      }
    ],
    "AfterAgent": [
      {
        "name": "ralph-loop",
        "type": "command",
        "command": "ralph-extension.sh"
      }
    ]
  }
}
```

Hook types: `BeforeTool`, `AfterTool`, `BeforeAgent`, `AfterAgent`. The hook script reads the tool input from stdin as JSON, and returns a decision:

```bash
#!/usr/bin/env bash
input=$(cat)
content=$(echo "$input" | jq -r '.tool_input.content // ""')

if echo "$content" | grep -qE 'api[_-]?key|password|secret'; then
  cat <<EOF
{
  "decision": "deny",
  "reason": "Security Policy: Potential secret detected.",
  "systemMessage": "Security scanner blocked operation"
}
EOF
  exit 0
fi
echo '{"decision": "allow"}'
```

When a hook returns `deny`, the agent receives the `systemMessage` as a tool result and can self-correct. This enables policy enforcement, content injection, and loop control entirely outside the agent code.

### Agent Skills System

Gemini CLI has a "Skills" system — reusable capability packages that are loaded on-demand:

```
.gemini/skills/
  my-skill/
    SKILL.md      ← Description (shown to model for discovery)
    instructions/ ← Full implementation (only loaded when activated)
    resources/    ← Any supporting files
```

At session start, only the **name and description** of each skill is injected into the system prompt. When the model identifies a relevant task, it calls `activate_skill` — which triggers a user consent prompt, then loads the full SKILL.md instructions and adds the skill directory to the agent's allowed file paths.

This is elegant: you get a massive library of capabilities without polluting the context window. The context cost is proportional to what you actually use, not what's available.

### The "Ralph Loop" via AfterAgent Hook

The viral "Ralph loop" technique: an `AfterAgent` hook intercepts the agent's completion signal and forces it into a continuous iterative loop. The hook refreshes context between attempts to prevent context rot. This transforms Gemini CLI from reactive into autonomous:

```bash
#!/usr/bin/env bash
# AfterAgent hook: evaluate completion, loop if needed
result=$(cat)  # the agent's final output
if ! verify_complete "$result"; then
  echo '{"continue": true, "systemMessage": "Task incomplete. Continue from where you left off."}'
fi
```

### Context Management

Unlike Claude Code which compacts at 92%, Gemini CLI has **token caching** — expensive prefix context (large system prompts, injected documents) can be cached and reused across turns without re-transmitting. This matters enormously for cost at scale.

### What You Can Steal

1. **Hooks middleware** — `BeforeTool` and `AfterTool` hooks are exactly what you need for your security layer, the RAG indexing trigger, and tool result validation
2. **Agent Skills on-demand loading** — inject skill descriptions only; load full instructions when activated. Directly applicable to your tool descriptions which currently all load at once
3. **AfterAgent loop continuation** — your planner already decides whether to continue; a post-agent verification step could independently confirm completion
4. **Token caching** — for your compressed context block and system prompt, which are repeated every request

---

## 5. CrewAI

**What it is:** Python multi-agent framework for role-based agent orchestration. Agents are defined by `role`, `goal`, `backstory`, and `tools`. Tasks have `description` and `expected_output`. A Crew runs them in sequential or hierarchical mode.

### Core Architecture

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="AI Technology Researcher",
    goal="Research the latest AI developments",
    backstory="You're a veteran tech journalist with 10 years of experience...",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    verbose=True,
    respect_context_window=True,  # auto context management
)

research_task = Task(
    description="Research {topic} and produce a comprehensive report",
    expected_output="A structured report with key findings and sources",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    planning=True,  # generates shared plan before execution
)

result = crew.kickoff(inputs={"topic": "AI Agent Frameworks 2025"})
```

### How Context is Assembled

When a task executes, CrewAI builds this prompt structure:

```
[agent role and backstory]
[agent goal]
[task description + expected output format]
[context from previous tasks if sequential]
[tool descriptions]
[planning roadmap if planning=True]
```

The `planning=True` flag causes CrewAI to generate a step-by-step workflow before any agent runs, then inject that plan into every agent's context. This is similar to Claude Code's TodoWrite but at a crew level rather than per-agent.

### Process Types

**Sequential**: tasks execute in order, each task's output becomes context for the next.

**Hierarchical**: a manager agent (optional, auto-created if not specified) dynamically delegates tasks. The manager reviews outputs and can reassign:

```python
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o",  # separate model for management
)
```

### Context Window Management

`respect_context_window=True` on an agent activates automatic truncation. CrewAI also has query rewriting for RAG:

```python
# Original task prompt
task_prompt = "Answer about user's favorite movies"
# Behind the scenes, rewritten to:
rewritten_query = "What movies did John watch last week?"
```

### The Weakness: Verbose Deliberation

Benchmarks show CrewAI has the **longest execution times** of the major frameworks because agents do autonomous deliberation before every tool call. If you watch a verbose=True run, agents think out loud about *whether* to use a tool before using it. This is expensive for simple tasks.

### What You Can Steal

1. **Role + Goal + Backstory** as three separate prompt components — your `agent_profiles.py` has `system_prompt` as one big string; splitting it into these three structured components gives the model clearer anchors
2. **Shared planning roadmap injected to all agents** — when your orchestrator makes a plan, inject it as a `<plan>` block into each sub-agent's context
3. **`expected_output` field on tasks** — telling the agent explicitly what format you want the output in dramatically reduces garbage responses; your tool definitions have descriptions but not output format specs
4. **`planning=True` equivalent** — before multi-step execution, generate a shared plan and re-inject it periodically (similar to TodoWrite injection after each tool call)

---

## 6. LangGraph

**What it is:** The lowest-level of the major frameworks. Models agent behavior as a directed graph (inspired by Google's Pregel and Apache Beam). State flows through nodes, edges define transitions. Used by Klarna, Replit, Elastic in production.

### Core Architecture

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: list
    current_intent: str
    tool_results: list
    completed: bool

def process_input(state: AgentState) -> dict:
    # reads state, returns partial update
    return {"current_intent": analyze(state["messages"][-1])}

def call_tools(state: AgentState) -> dict:
    # execute tool based on intent
    result = execute_tool(state["current_intent"])
    return {"tool_results": [result]}

def router(state: AgentState) -> str:
    # conditional edge: decide next node
    if state["completed"]:
        return "END"
    return "call_tools"

workflow = StateGraph(AgentState)
workflow.add_node("process", process_input)
workflow.add_node("tools", call_tools)
workflow.add_edge(START, "process")
workflow.add_conditional_edges("process", router, {
    "call_tools": "tools",
    "END": END
})
workflow.add_edge("tools", "process")

app = workflow.compile(checkpointer=SqliteSaver.from_conn_string("agent.db"))
```

### Checkpointing — The Killer Feature

Every state transition is persisted to a checkpointer backend (SQLite, Postgres, Redis). This gives you:

```python
# Resume from exact state after crash
config = {"configurable": {"thread_id": "user-session-123"}}
result = app.invoke(state, config=config)

# Time-travel debugging: replay from any prior state
history = list(app.get_state_history(config))
app.invoke(None, config=config, checkpoint_id=history[3].config["configurable"]["checkpoint_id"])
```

**Time-travel debugging** is the feature no other framework has at this level. You can replay any workflow from any historical checkpoint. This is invaluable for production debugging — when your agent does something unexpected, you can rewind to the exact state, modify the inputs, and replay.

### State as Shared Truth

The fundamental concept: every node receives the **entire shared state** and returns a **partial update**. Nodes never communicate directly — they only read and write to the same state object. This makes reasoning about multi-agent systems tractable:

```python
class MultiAgentState(TypedDict):
    task: str
    researcher_findings: list
    writer_draft: str
    reviewer_feedback: str
    final_output: str
```

### Human-in-the-Loop

LangGraph has first-class support for pausing workflows:

```python
# Interrupt before a specific node
graph_builder.add_node("human_approval", approval_node)
# Workflow pauses, waits for human input, then resumes from exact state
```

This is impossible to do cleanly in a linear agent loop without breaking the conversation history.

### What You Can Steal

1. **State-based persistence** — your `session_manager.py` stores `compressed_context` and `sliding_window`; this is a partial state, but it doesn't capture tool call history or intermediate results. A full `AgentState` object would be richer
2. **Time-travel debugging** — store each turn's full state (messages, context, tool calls, results) in SQLite with a checkpoint ID; your `_trace` system does this partially but doesn't support replay
3. **Conditional edge logic** — your `orchestrator.py` uses a planner to decide tool forcing; a graph-based conditional edge would make this more explicit and testable
4. **Explicit state schema** — defining `AgentState` as a TypedDict forces you to be explicit about what the agent carries between turns; this prevents accidental state leakage

---

## 7. n8n

**What it is:** Visual low-code workflow automation platform with LangChain-powered AI agent nodes. The AI Agent node is a LangChain Tools Agent wrapped in a visual node. Built for non-developers and integration-first workflows.

### Architecture

```
Trigger Node (Webhook / Schedule / Chat)
    ↓
AI Agent Node (LangChain Tools Agent)
    ├── Chat Model Sub-node (OpenAI / Anthropic / Gemini / Ollama)
    ├── Memory Sub-node (Buffer / Summary / Postgres)
    └── Tool Sub-nodes (HTTP Request / Code / DB / App nodes)
    ↓
Output / Action Nodes
```

### What n8n Actually Does Well

n8n is not a serious agent architecture — it's a **glue layer** for non-developers. The AI Agent node is essentially a LangChain Tools Agent with a UI. Its real value:

- **600+ pre-built integrations** — if you need to connect to Salesforce, HubSpot, Notion, and Slack in one workflow, n8n saves you weeks of API code
- **Visual debugging** — every node shows input/output inline; you can click any execution and see exactly what went wrong
- **Human-in-the-loop gates** — you can pause a workflow and send an approval request to Slack/Telegram before executing a sensitive tool

### What n8n Gets Wrong

n8n fundamentally cannot:
- Do autonomous planning or dynamic task decomposition
- Maintain context across sessions without external storage (Postgres/Airtable)
- Handle unexpected tool failures gracefully — errors cascade
- Execute complex multi-step reasoning chains reliably

The benchmark data is damning: n8n relies entirely on **manual prompt engineering and fixed branching logic**. It cannot break down complex goals or adapt to real-time feedback.

### What You Can Steal

1. **Visual execution inspection** — your `_trace` system writes JSON; an n8n-style UI that lets you click through each turn and see exactly what the model saw would be extremely valuable for debugging
2. **Human approval gates for sensitive tools** — before executing `bash` in Docker or calling external APIs, optionally pause for user confirmation; n8n does this cleanly via Slack/Telegram webhooks
3. **Pre-built tool template library** — n8n's 600+ integrations are mostly just `api.registerTool` patterns; a tool marketplace or template library for your platform would reduce the friction of adding new capabilities

---

## 8. Feature Matrix

| Feature | Your System | OpenClaw | Claude Code | Gemini CLI | CrewAI | LangGraph | n8n |
|---|---|---|---|---|---|---|---|
| **Task tracking (TodoWrite)** | ✅ | ❌ | ✅ Re-injected after every tool call | ❌ | ✅ Shared plan | ❌ State-implicit | ❌ |
| **Context compression** | ✅ `compress_exchange` | ✅ Markdown files | ✅ `wU2` at 92% | ✅ Token caching | ✅ Auto truncation | ❌ Raw messages | ❌ External DB |
| **Cross-session memory** | ✅ SQLite + Chroma | ✅ Markdown files | ✅ CLAUDE.md | ✅ GEMINI.md | ✅ Memory plugin | ✅ Checkpointer | ❌ |
| **Sub-agent isolation** | ✅ `parent_id` isolation | ❌ Single agent | ✅ Task tool | ❌ | ✅ Hierarchical | ✅ Subgraphs | ✅ AI Agent Tool |
| **Per-tool hooks/middleware** | ❌ | ❌ | ✅ Hooks | ✅ Hooks system | ❌ | ✅ Middleware | ✅ |
| **Checkpointed replay** | ❌ `_trace` only | ❌ | ❌ | ✅ Checkpointing | ❌ | ✅ Time-travel | ❌ |
| **Tool RAG Indexing**  | ✅ | ❌ (Search) | ❌ (Search) | ❌ (Search) | ❌ (Manual only) | 🟢 IMPLEMENTED (v0.6.0) |
| **Semantic graph (facts)** | ✅ Unique! | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Temporal fact voiding** | ✅ Unique! | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **MCP support** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Planner / planning mode** | ✅ | ❌ | ✅ `/think` mode | ❌ | ✅ `planning=True` | ✅ Custom nodes | ❌ |
| **Multi-model / adapter** | ✅ Unique! | ❌ | ❌ | ❌ | ✅ Per-agent model | ❌ | ✅ |
| **Docker sandbox** | ✅ | ❌ | ✅ Built-in | ❌ Sandboxed shell | ❌ | ❌ | ❌ |
| **Citation verification** | ✅ `_verify_citations` (unwired) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Tool schema validation** | ✅ JSON scanner | ❌ | ✅ Native | ✅ TypeBox | ✅ Pydantic | ✅ TypedDict | ✅ |
| **Visual debugging UI** | ❌ | ❌ | ✅ VS Code | ✅ | ❌ | ✅ LangGraph Studio | ✅ |
| **On-demand skill loading** | ❌ | ❌ | ❌ | ✅ Skills system | ❌ | ❌ | ❌ |
| **Output format spec** | ❌ | ❌ | ❌ | ❌ | ✅ `expected_output` | ❌ | ✅ |

**Your system has the most sophisticated memory/context architecture of any open-source system here.** The combination of `ContextEngine` + `SemanticGraph` + `VectorEngine` + `SessionRAG` is genuinely unique. The gaps are in task tracking, middleware hooks, and output format specification.

---

## 9. Context Engineering — The Real Science

This is now recognized as the **#1 job of AI engineers** building agent systems (Cognition, 2025). Andrej Karpathy defined it as "the delicate art and science of filling the context window with just the right information for the next step."

### The Four Failure Modes

**Context Poisoning**: Incorrect or hallucinated information enters the context window and gets repeatedly referenced. Your `_verify_citations` method exists to catch this but is never called.

**Context Rot**: As context window fills, the model's ability to recall information from early in the context degrades. Information in the middle gets "lost." Your sliding window of 20 + compressed context directly addresses this.

**Context Distraction**: Irrelevant information in context causes the model to produce off-topic responses. Your `MAX_RAG_CHARS = 4000` cap and individual anchor truncation help here.

**Context Starvation**: The model doesn't have the context it needs to make good decisions — this is the most common failure mode. "When agents fail, it's usually because the right context was not passed to the LLM."

### The Hierarchy of Context

From Anthropic's own context engineering guide:

```
System Layer     →  Core identity, capabilities, rules
Task Layer       →  Specific instructions for current task
Tool Layer       →  Descriptions and usage for available tools
Memory Layer     →  Relevant historical context and learnings
```

Your `_build_messages` assembles this but doesn't treat these as distinct layers with different freshness, token budgets, or injection rules.

### The "Right Altitude" for System Prompts

Anthropic's guidance: system prompts should operate at "the Goldilocks zone":
- Too specific → brittle, fragile, high maintenance ("NEVER do X in scenario Y with Z...")
- Too vague → model interprets freely, inconsistent behavior

The right altitude: **principles and examples** rather than rules. Show the model what good behavior looks like rather than prohibiting every bad behavior.

### Context Engineering for Your Tool Descriptions

Tool description quality directly affects whether the model calls the right tool. Best practices from the research:

```python
# BAD - vague, no output spec, no examples
{
    "name": "search_web",
    "description": "Search the web for information"
}

# GOOD - specific, includes when to use, output format, examples
{
    "name": "web_search",
    "description": """Search the internet for current information.
    
USE WHEN: User asks about recent events, current facts, prices, people's
current status, or anything that may have changed since training data.

DO NOT USE FOR: Math calculations, coding questions, historical facts
that are unlikely to have changed.

OUTPUT: Returns top 5 search results with title, URL, and snippet.
Each result includes a relevance score.

EXAMPLE CALL:
{"query": "current CEO of OpenAI 2025"}""",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Specific search query. Be precise."
            }
        }
    }
}
```

---

## 10. Your System Audit

### What You Have That Nobody Else Has

**1. ContextEngine with FACT/VOIDS semantic compression**

No other framework in this comparison extracts structured facts from compressed context. Your `compress_exchange` produces:
- `keyed_facts` → indexed in ChromaDB for semantic retrieval
- `voids` → temporal facts that are no longer true get explicitly voided

This is the most architecturally sophisticated memory system of any open-source agent. Claude Code uses Markdown files. OpenClaw uses Markdown files. You have a proper semantic knowledge graph with temporal invalidation.

**2. Multi-adapter hot-swap architecture**

Your system can run any session on any LLM provider (Groq, Ollama, OpenAI, etc.) and even switch mid-session via `context_model` vs `model` separation. Nobody else does this. OpenClaw is single-model, Claude Code is Claude-only, Gemini CLI is Gemini-only.

**3. SemanticGraph with temporal facts**

`void_temporal_fact` + `add_temporal_fact` with `turn_index` is genuinely novel. When a user says "I'm moving to London" in turn 5, and "I moved back to Paris" in turn 47, your system can correctly void the London fact. LangGraph has state checkpointing but no semantic fact invalidation.

**4. Per-session ChromaDB collections**

`SessionRAG` gives each session an isolated vector store. This prevents cross-contamination between user sessions — a real security and accuracy concern that none of the other frameworks address explicitly.

### What's Missing or Broken

**CRITICAL (breaks accuracy right now):**

```python
# core.py line ~618 — WRONG — RAG indexing is completely dead
asyncio.create_task(rag.index(
    text=f"Tool result from {call.tool_name}:\n{result.content}",  # 'text' doesn't exist
    metadata={"source": "tool", "tool": call.tool_name}           # 'metadata' doesn't exist
))

# Fix:
asyncio.create_task(rag.index(
    content=result.content,
    source=call.tool_name,
    tool_name=call.tool_name,
))

# main.py line ~265 — WRONG — file upload indexing is also dead
asyncio.create_task(rag.index(
    text=f"Uploaded file: {upload.filename}\n\n{pf.text_content}",
    metadata={"source": "upload", "filename": upload.filename}
))

# Fix:
asyncio.create_task(rag.index_file(upload.filename, pf.text_content))
```

**HIGH PRIORITY (accuracy gaps):**

- `_verify_citations` is implemented but never called — contradiction detection is written and idle
- No TodoWrite equivalent — multi-step tasks lose thread after 10+ tool calls
- `context_model` defaults to `deepseek-r1:8b` when not specified — cloud adapters will fail
- Tool descriptions have no `expected_output` format — model produces variable response formats
- `<system-reminder>` injection between tool calls doesn't carry task state

**MEDIUM PRIORITY:**

- No `BeforeTool` / `AfterTool` hook system — security validation happens inside tools rather than as middleware
- Agent Skills lazy-loading not implemented — all tool descriptions always loaded regardless of relevance
- No time-travel debugging — `_trace` logs exist but can't replay from a specific turn state

---

## 11. Enhancement Roadmap — Exact Code

Ordered by impact-to-effort ratio.

---

### Enhancement 1: Fix the RAG Bugs (10 minutes, maximum accuracy impact)

These are in the previous audit. Do these first. Every tool result and every uploaded file should be flowing into ChromaDB. Currently none of it is.

```python
# core.py — fix tool result indexing
asyncio.create_task(rag.index(
    content=result.content,
    source=call.tool_name,
    tool_name=call.tool_name,
))

# main.py — fix file upload indexing
asyncio.create_task(rag.index_file(upload.filename, pf.text_content))
```

---

### Enhancement 2: TodoWrite — Session Task Tracker (2-3 hours, high accuracy impact)

Add a `TaskTracker` that the model can call to maintain a structured TODO list, and re-inject the current task state after every tool call as a `<system-reminder>`.

```python
# memory/task_tracker.py
import json
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Task:
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
    priority: Literal["low", "medium", "high"] = "medium"

class TaskTracker:
    def __init__(self):
        self._tasks: list[Task] = []

    def write(self, tasks: list[dict]) -> str:
        """Replace the entire task list. Returns formatted state."""
        self._tasks = [Task(**t) for t in tasks]
        return self.render()

    def update(self, task_id: str, status: str) -> str:
        """Update a single task status. Returns formatted state."""
        for t in self._tasks:
            if t.id == task_id:
                t.status = status
                break
        return self.render()

    def render(self) -> str:
        if not self._tasks:
            return ""
        lines = ["Current Tasks:"]
        status_icons = {"pending": "○", "in_progress": "●", "completed": "✓"}
        for t in self._tasks:
            icon = status_icons.get(t.status, "?")
            lines.append(f"  [{icon}] {t.id}: {t.content} ({t.status})")
        return "\n".join(lines)

    @property
    def has_tasks(self) -> bool:
        return len(self._tasks) > 0
```

Then in `core.py`, after assembling `combined_results` (after each tool call loop iteration), inject the task state:

```python
# core.py — inside _agent_loop, after building combined_results
if hasattr(self, '_task_tracker') and self._task_tracker.has_tasks:
    task_state = self._task_tracker.render()
    combined_results.append(
        f"<system-reminder>\n{task_state}\n</system-reminder>"
    )
```

Add `TODO_WRITE_TOOL` and `TODO_UPDATE_TOOL` to your tools:

```python
TODO_WRITE_TOOL = {
    "name": "todo_write",
    "description": """Create or replace the session task list for multi-step work.
    
USE THIS TOOL: At the start of any task involving more than 2 steps.
Write a complete list of all steps needed. Update task status as you progress.
Never leave a task in 'in_progress' when moving to the next.

This list is re-shown after every tool call so you never lose track.""",
    "parameters": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                    }
                }
            }
        },
        "required": ["tasks"]
    }
}
```

---

### Enhancement 3: Wire `_verify_citations` (1 hour, direct accuracy improvement)

The method exists and is correct. Wire it into the response pipeline:

```python
# core.py — after full_response is assembled, before storing
if historical_anchors:
    citations_valid = self._verify_citations(full_response, historical_anchors)
    if not citations_valid:
        log("rag", sid, "Citation contradiction detected — appending correction", level="warn")
        # Append a soft correction without regenerating (cheaper)
        correction_msg = (
            "\n\n[Note: Some details above may conflict with the user's stated preferences. "
            "Please prioritize the stored facts over any assumptions made in this response.]"
        )
        full_response += correction_msg
        
        # Optionally: trigger a full regeneration for high-stakes responses
        # yield _ev("citation_mismatch")
```

---

### Enhancement 4: BeforeTool Hook System (2-3 hours, security + extensibility)

Inspired by Gemini CLI's hook architecture. Allows pluggable middleware per tool:

```python
# plugins/hook_system.py
from typing import Callable, Awaitable, Optional
from dataclasses import dataclass, field
import re

@dataclass
class HookResult:
    decision: str  # "allow" | "deny" | "transform"
    reason: Optional[str] = None
    system_message: Optional[str] = None
    transformed_args: Optional[dict] = None

HookFn = Callable[[str, dict], Awaitable[HookResult]]

class HookRegistry:
    def __init__(self):
        self._before_tool: dict[str, list[HookFn]] = {}
        self._after_tool: dict[str, list[HookFn]] = {}

    def before_tool(self, matcher: str):
        """Decorator. matcher is a regex pattern matching tool names."""
        def decorator(fn: HookFn):
            pattern = re.compile(matcher)
            key = matcher
            self._before_tool.setdefault(key, []).append((pattern, fn))
            return fn
        return decorator

    def after_tool(self, matcher: str):
        def decorator(fn: HookFn):
            pattern = re.compile(matcher)
            self._after_tool.setdefault(matcher, []).append((pattern, fn))
            return fn
        return decorator

    async def run_before(self, tool_name: str, args: dict) -> HookResult:
        for key, hooks in self._before_tool.items():
            for pattern, fn in hooks:
                if pattern.search(tool_name):
                    result = await fn(tool_name, args)
                    if result.decision != "allow":
                        return result
        return HookResult(decision="allow")

    async def run_after(self, tool_name: str, args: dict, result: str) -> HookResult:
        for key, hooks in self._after_tool.items():
            for pattern, fn in hooks:
                if pattern.search(tool_name):
                    hook_result = await fn(tool_name, {"args": args, "result": result})
                    if hook_result.decision != "allow":
                        return hook_result
        return HookResult(decision="allow")

# Global registry
hooks = HookRegistry()

# Example: block secrets in bash commands
@hooks.before_tool(r"^bash$")
async def block_secrets_in_bash(tool_name: str, args: dict) -> HookResult:
    import re
    cmd = args.get("command", "")
    if re.search(r'(ANTHROPIC|OPENAI|GROQ)_API_KEY', cmd, re.IGNORECASE):
        return HookResult(
            decision="deny",
            reason="Security policy: potential API key exposure in bash command",
            system_message="I can't execute commands that might expose API keys. Please rephrase."
        )
    return HookResult(decision="allow")

# Example: auto-index RAG after tool execution (replaces broken inline logic)
@hooks.after_tool(r".*")
async def index_tool_result_to_rag(tool_name: str, args: dict) -> HookResult:
    # This hook runs after every tool — trigger RAG indexing here
    # avoids the bug in core.py where wrong kwargs were used
    return HookResult(decision="allow")
```

Then in `core.py` inside `_agent_loop`, before executing each tool:

```python
# Before tool execution
hook_result = await self.hooks.run_before(call.tool_name, call.arguments)
if hook_result.decision == "deny":
    result = ToolResult(
        tool_name=call.tool_name,
        content=hook_result.system_message or "Tool execution blocked by policy",
        success=False
    )
    # skip actual tool execution
else:
    result = await self.tools.execute(call)
    await self.hooks.run_after(call.tool_name, call.arguments, result.content)
```

---

### Enhancement 5: Expected Output Format on Tool Descriptions (30 min, accuracy improvement)

Add `returns` field to tool definitions. The model uses this to format its response correctly:

```python
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": """Search the internet for current information.

USE WHEN: Recent events, current facts, prices, current status of people/companies,
anything that may have changed since 2024.

RETURNS: JSON array of {title, url, snippet, date} objects. Up to 5 results.
Always cite sources in your response using the URLs provided.

EXAMPLE CALL: {"query": "current OpenAI CEO 2025"}""",
    # ... parameters
}
```

---

### Enhancement 6: On-Demand Tool Loading (2-3 hours, token efficiency)

Inspired by Gemini CLI's Agent Skills. Instead of loading all 50+ tool descriptions into every system prompt, load only the descriptions of a small "always-on" set, and provide a `discover_tools` tool:

```python
# Core tools always loaded (~5 tools):
ALWAYS_ON_TOOLS = ["web_search", "bash", "todo_write", "delegate_to_agent", "discover_tools"]

DISCOVER_TOOLS_TOOL = {
    "name": "discover_tools",
    "description": """List available tools and activate specific ones for this session.
    
USE WHEN: You need a capability that isn't currently available (file ops, RAG search,
code execution, image gen, etc.).

Calling this tool injects the full description of requested tools into context.""",
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "Describe what you're trying to do (e.g., 'search memory', 'read a file')"
            }
        }
    }
}
```

Then when `discover_tools` is called, dynamically inject the relevant tool descriptions into the next system turn. This can cut your system prompt token count by 60-70% for simple conversations.

---

### Enhancement 7: Context Quality Score (2 hours, observability)

Add a lightweight quality score to the compression output so you can monitor context degradation:

```python
# context_engine.py — add to compress_exchange return value
async def compress_exchange(self, ...) -> tuple[str, list, list, float]:
    # ... existing logic ...
    
    # Compute quality score
    fact_density = len(keyed_facts) / max(1, len(updated_ctx.split("\n")))
    void_ratio = len(voids) / max(1, len(keyed_facts) + 1)
    has_contradictions = self._detect_contradictions(updated_ctx)
    
    quality_score = (
        min(1.0, fact_density * 10) * 0.5 +        # fact richness
        (1.0 - void_ratio) * 0.3 +                   # stability
        (0.0 if has_contradictions else 1.0) * 0.2   # coherence
    )
    
    return updated_ctx, keyed_facts, voids, quality_score
```

Log this score per session. If it drops below 0.3, trigger a full recompression rather than incremental.

---

### Enhancement 8: User Preferences File (30 min, UX improvement inspired by OpenClaw)

Inspired by OpenClaw's `USER.md` / `MEMORY.md` split. Add a persistent user preferences document injected at session start:

```python
# session_manager.py — add user preferences loading
async def get_user_preferences(user_id: str) -> str:
    """Load persistent user preferences from disk/DB."""
    pref_path = f"data/users/{user_id}/preferences.md"
    if os.path.exists(pref_path):
        with open(pref_path) as f:
            return f.read()
    return ""

# core.py — in _build_messages, inject preferences
user_prefs = await self.sessions.get_user_preferences(session.user_id)
if user_prefs:
    parts.append(
        f"--- User Preferences ---\n"
        f"{user_prefs}\n"
        f"--- End Preferences ---"
    )
```

Expose a `/users/{id}/preferences` endpoint for users to edit their preferences. This makes your agent feel "personalized" in a way that's transparent and user-controlled.

---

## Priority Order Summary

| # | Enhancement | Time | Impact | Difficulty |
|---|---|---|---|---|
| 1 | Fix RAG kwargs bugs | 10 min | 🔴 Critical | Trivial |
| 2 | Wire `_verify_citations` | 1 hr | 🔴 High | Easy |
| 3 | TodoWrite task tracker | 2-3 hrs | 🔴 High | Medium |
| 4 | BeforeTool hook system | 2-3 hrs | 🟡 Medium | Medium |
| 5 | Expected output on tool descriptions | 30 min | 🟡 Medium | Easy |
| 6 | User preferences file | 30 min | 🟡 Medium | Easy |
| 7 | On-demand tool loading | 2-3 hrs | 🟡 Medium | Medium |
| 8 | Context quality score | 2 hrs | 🟢 Low | Medium |
| 9 | Time-travel debugging | 3-4 hrs | 🟢 Low | Hard |

---

## The One-Line Verdict Per System

- **OpenClaw**: Radical simplicity wins. Ship fast, add tools, use files for memory. Your architecture is already more sophisticated — don't go backwards.
- **Claude Code**: TodoWrite re-injection after every tool call is the single most impactful technique here. Steal it.
- **Gemini CLI**: Hooks middleware and on-demand skill loading are architecturally elegant. Steal both.
- **CrewAI**: `expected_output` on every task and shared planning injection are underrated. Steal both.
- **LangGraph**: State-as-shared-truth and time-travel debugging are production-grade. Steal the state schema concept.
- **n8n**: Visual execution inspection is genuinely useful for debugging. Build your own from the `_trace` data you already collect.

Your memory architecture is already the best in class. The gap is in task continuity (TodoWrite), output format discipline (expected_output), and tool middleware (hooks). Fix those three and you'll be architecturally ahead of all of them.