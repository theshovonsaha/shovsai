# Shovs Agent Platform: Features & 2026 Roadmap

The Shovs Agent Platform is a high-autonomy, isolated execution ecosystem designed for the next generation of "Write-Back" AI agents.

## 🌟 Core Features (Implemented)

### 1. Unified Intelligence Orchestration
- **Agentic Planning**: Every turn begins with a Manager-level planning phase to select optimal tools.
- **Specialized Profiles**: Native `Researcher`, `Analyst`, and `Coder` agents with fine-tuned system prompts.
- **Dynamic Adapter Factory**: Switch providers (Ollama, Groq, OpenAI, Gemini, Anthropic) per-session or per-task.
- **Platinum UI Standards**: Built-in directives for "True Black" SPA generation with high-signal animations and real data (not placeholders).

### 2. The "Convergent Graph" Context Engine (V2)
- **Fact Discovery**: Automatically extracts Subject-Predicate-Object triplets and stores them in a persistable SQLite semantic graph.
- **Temporal Validity**: Supports "Fact Voiding" (invalidating facts that are no longer true over time).
- **Context Compression**: Uses high-density summarization to maintain infinite-feeling history without bloating the token window.
- **Goal Persistence**: Agents track "Active Goals" so they don't get sidetracked during long multi-turn tasks.

### 3. Native Model Context Protocol (MCP)
- **Universal Tooling**: First-class support for MCP servers. Connect to any compliant toolset by adding its command to `mcp_servers.json`.
- **Dynamic Discovery**: Agents can introspect and use new tools added to the ecosystem at runtime.

### 4. Zero-Trust Docker Sandbox
- **Hardened Isolation**: All code execution happens in ephemeral, memory-capped Docker containers.
- **Kill-Switch Logic**: Automatic force-removal of runaway processes and PID-limit protection against fork bombs.
- **Host Protection**: No direct filesystem access; communication happens through a strict `/sandbox` mount.

---

## 📈 2026 Market Perspective: Where We Fit

As of early 2026, the AI landscape has moved beyond simple "Chat with your PDF" tools into **Autonomous Operations**.

| The 2026 Gap | How Shovs Fills It |
| :--- | :--- |
| **Context Poisoning** | We use a **Convergent Graph** to verify new info against established session facts, preventing hallucinations. |
| **Centralized Risk** | High-profile leaks from "Agent-as-a-Service" clouds have made **Self-Hosted Isolation** (Docker-native) the gold standard. |
| **Interop Fatigue** | 2026 is the year of **MCP Ubiquity**. We treat MCP as a core primitive, not a plugin. |

---

## 🗺️ Roadmap (Upcoming)

- **[Q2 2026] Distributed Agent Swarms**: Allow multiple Docker instances to coordinate on large-scale engineering tasks.
- **[Q3 2026] Neural Fact Verification**: Automated cross-referencing between the Semantic Graph and the Session RAG to flag contradictions.
- **[Q4 2026] Multi-Modal Operations**: Direct grounding of video and audio streams into the temporal fact engine.
