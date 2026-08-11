---
trigger: always_on
glob:
description: >
  Tech context for the hey-neo project. Covers LangChain, LangGraph, and
  DeepAgent/deepagents. All info verified from live web sources — June 2026.
---

# Hey-Neo Tech Context
> Last updated: 2026-06-11 | Sources: langchain.com, github.com, plainenglish.io, and web research

---

## 1. LangChain (v1.0 — Stable)

### Version & Stability
- **LangChain 1.0** was released in late 2025 — first major stable release.
- Commitment: **no breaking changes until v2.0**.
- Requires **Python 3.10+** (Python 3.9 reached EOL in October 2025 and is no longer supported).
- The ecosystem is a **three-pillar structure**:
  - `langchain` — components: models, prompts, retrievers, memory
  - `langgraph` — control flow: stateful agents, graphs, loops
  - `langsmith` — observability: tracing, evaluation, deployment

### Core Packages
```
langchain          # Core abstractions, chains, agents
langchain-core     # Base types: BaseMessage, Tool, Runnable etc.
langchain-ollama   # ChatOllama integration (local models via Ollama)
langchain-community # Third-party integrations (Tavily, Qdrant, etc.)
```

### Key APIs (2025/2026)

#### Tools — `@tool` decorator
```python
from langchain_core.tools import tool

@tool
def bm25_search_tool(query: str) -> list:
    """
    Searches the BM25 index for exact keyword/package/path matches.
    Use this for specific package names, file paths, or shell commands.
    """
    return bm25_search(query)
```
- **Docstrings are the LLM's routing instructions** — must be precise and complete.
- **Type hints are mandatory** — used to auto-generate the JSON schema the LLM sends.
- Use `args_schema` (Pydantic model) for complex multi-field inputs.
- Keep tools focused: one concern per tool, avoid "god-tools".

#### ToolMessage
```python
from langchain_core.messages import ToolMessage

msg = ToolMessage(
    content="result string",
    tool_call_id="call_abc123"   # must match AIMessage.tool_calls[i].id
)
```
- `tool_call_id` correlation is **mandatory** especially for parallel tool calling.
- Use `artifact=` for large objects that shouldn't clutter the context window.
- Always catch exceptions inside tools and return a descriptive error string as `content`.

#### ChatOllama (langchain-ollama)
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:14b",
    reasoning=True,     # enables reasoning_content extraction (qwen3 think mode)
    temperature=0.2
)
```
- **`reasoning=True`**: separates `<think>` tokens into `chunk.additional_kwargs["reasoning_content"]`; `chunk.content` = clean final answer only.
- **`reasoning=False`**: disables thinking entirely.
- **`reasoning=None`** (default): raw model behaviour — `<think>` tags may appear inside `content`.
- Update `langchain-ollama` regularly; the `reasoning` vs `reasoning_content` field mapping is version-sensitive.

#### create_agent (replaces create_react_agent)
```python
from langchain.agents import create_agent   # ✅ current API

agent = create_agent(
    llm,
    tools,
    system_prompt="You are Neo, a local Linux intelligence assistant..."
)
```
- `create_react_agent` from `langgraph.prebuilt` is **DEPRECATED** as of v1.0.
- `create_agent` from `langchain.agents` is the current standard.
- Still backed by LangGraph under the hood (compiled graph).

### LangSmith — Observability
- Tracing, debugging, evaluation — considered **non-optional for production**.
- Features: Trace Mode, end-to-end OpenTelemetry support, cost tracking.
- "Time-travel" debugging: inspect and rewind agent execution steps.
- No-code agent building (preview) and UI-based evaluation.

### MCP (Model Context Protocol)
- LangChain now supports `llms.txt` and **MCP server** tool interoperability.
- Allows standardized tool discovery across agents and IDEs.

---

## 2. LangGraph (v1.0 — Stable)

### What It Is
- A **graph-based orchestration layer** for stateful, multi-actor agentic workflows.
- Reached **v1.0 stable** in late 2025.
- Sits on top of LangChain; provides durable, cyclical execution vs. LangChain's linear chains.

### LangChain vs LangGraph

| Feature | LangChain | LangGraph |
|---|---|---|
| Workflow | Linear / sequential | Cyclical / graph-based |
| State | Stateless by default | Stateful with checkpoints |
| Best For | Prototyping, simple pipelines | Complex long-running agents |
| Streaming | Basic | Advanced: tokens, tools, custom |

### Core Concepts

#### State
```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```
- Uses `TypedDict` + `Annotated` reducers.
- `add_messages` appends instead of overwriting — standard for chat state.

#### Building a Graph
```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=checkpointer)
```

#### Streaming — All Modes
```python
# Single mode
for chunk, metadata in agent.stream(inputs, stream_mode="messages"):
    ...

# Multiple modes simultaneously (v1.2+)
async for chunk in graph.astream(inputs, stream_mode=["messages", "custom"]):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(token.content, end="")
    elif chunk["type"] == "custom":
        print(f"[status] {chunk['data']}")
```

**stream_mode options:**
| Mode | What it yields |
|---|---|
| `"messages"` | Raw LLM token chunks + metadata tuple; `(AIMessageChunk, metadata)` |
| `"updates"` | State diff after each node completes |
| `"values"` | Full state snapshot after each node |
| `"custom"` | Arbitrary data emitted by `get_stream_writer()` inside nodes |

**Emitting custom events from inside nodes:**
```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer({"status": "Searching BM25 index..."})
    results = bm25_search(state["query"])
    return {"results": results}
```

#### Reasoning Token Extraction (qwen3)
```python
# In stream loop (stream_mode="messages"):
for chunk, metadata in stream_iter:
    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
    if reasoning:
        reasoning_text += reasoning
    content = getattr(chunk, "content", "") or ""
    if content and isinstance(chunk, AIMessageChunk) and not chunk.tool_call_chunks:
        # This is the final answer, not a tool call
        accumulated += content
```

#### Tool Call Detection in Stream
```python
tc_chunks = getattr(chunk, "tool_call_chunks", None) or []
for tc in tc_chunks:
    tc_id = tc.get("id", "") or str(tc.get("index", ""))
    name  = (tc.get("name") or "").strip()
    if tc_id and tc_id not in seen_tool_ids and name:
        seen_tool_ids.add(tc_id)
        # record tool being called
```

### Memory & Checkpointing

| Checkpointer | Use Case |
|---|---|
| `MemorySaver` | Dev/testing only — all state lost on restart |
| `SqliteSaver` | Local persistence, small projects. Pin `langgraph-checkpoint-sqlite >= 3.0` (CVE-2025-67644 patched) |
| `PostgresSaver` | Production — horizontal scaling, crash recovery |

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-session-001"}}
graph.invoke({"messages": [...]}, config=config)
```

**Checkpointer vs Store:**
- **Checkpointer** = execution state for a single thread/conversation.
- **BaseStore** = long-term memory across threads (user preferences, facts, etc.).

### Human-in-the-Loop (HITL)

```python
from langgraph.types import interrupt, Command

def human_review_node(state):
    decision = interrupt("Please review this tool call...")
    if decision.get("approved"):
        return execute_action(state)
    return {"status": "rejected"}

# Resume after human input:
graph.invoke(Command(resume={"approved": True}), config=config)
```

**Patterns:**
- `interrupt_before=["tool_node"]` in `compile()` — declarative pause before a node.
- `interrupt()` inside a node — dynamic pause mid-execution.
- Always use a persistent checkpointer (SQLite/Postgres) so state survives the pause.

### Multi-Agent / Supervisor Pattern

```python
# Supervisor uses tool-calling for handoffs
@tool
def route_to_researcher(task: str) -> Command:
    """Delegate a research task to the researcher sub-agent."""
    return Command(goto="researcher_agent", update={"task": task})
```

- **Supervisor as router**: define handoff tools → supervisor calls them → `Command(goto=...)` routes to subgraph.
- **Subgraphs as nodes**: compile child graph, add it as a node in parent graph.
- Use `withStructuredOutput()` on supervisor for deterministic routing.
- Monitor handoffs with LangSmith — most multi-agent bugs are misrouted context.

---

## 3. DeepAgent / `deepagents` Library

### What It Is
- An open-source **agent harness** built by LangChain on top of LangGraph.
- Install: `pip install deepagents`
- Factory: `create_deep_agent(model, tools, system_prompt)`

### Architecture
```
create_deep_agent
    └── LangGraph compiled graph
        └── Middleware stack (composable plugins):
            ├── TodoListMiddleware    — task planning, write_todos tool
            ├── FilesystemMiddleware — ls, read, write, grep tools
            ├── SkillsMiddleware     — reusable skill injection
            ├── SummarizationMiddleware — context compression
            └── SubAgentMiddleware   — delegates to child agents
```

### Known Issues ⚠️
| Issue | Details |
|---|---|
| **40+ minute hangs with local models** | Middleware stack (especially SubAgentMiddleware + SummarizationMiddleware) overwhelms local/small models. This is why `deepagents.create_deep_agent` was **removed from hey-neo**. |
| **~1,600 token overhead** | Default middleware stack consumes significant context before the user query even reaches the agent. |
| **Rapid API churn** | 80+ releases in 8 months — pin your version, test upgrades carefully. |
| **`KeyError: 'system_prompt'`** | Known bug with certain Ollama/custom LLM configs when SubAgentMiddleware auto-configures sub-agents. |
| **State loss in nested agents** | When nesting `create_deep_agent` inside hierarchical graphs with custom `AgentState` schemas. |

### Decision for hey-neo
`create_deep_agent` was **explicitly removed** because:
1. Its middleware overhead caused 40+ minute hangs with `qwen3:14b` running locally via Ollama.
2. Local models lack the throughput to service the token-heavy middleware pipeline efficiently.
3. `create_agent` (plain `langchain.agents`) with a hand-crafted system prompt provides equivalent routing with zero overhead.

### When to Use deepagents
- **Cloud models** (GPT-4o, Claude, Gemini) where latency is acceptable.
- Tasks needing genuine **planning/todos, filesystem delegation, sub-agent orchestration**.
- Do **NOT** use with local Ollama models on consumer hardware.

---

## 4. Hey-Neo Specific Patterns

### Agent Build Pattern (current)
```python
# agents.py
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from tools import bm25_search_tool, similarity_search_tool, package_search_tool, web_search_tool

def build_agent():
    llm = ChatOllama(model="qwen3:14b", temperature=0.2)
    tools = [bm25_search_tool, similarity_search_tool, package_search_tool, web_search_tool]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
```

### Tool Priority Order (system prompt enforced)
1. `bm25_search_tool` — exact keyword, packages, paths, aliases
2. `similarity_search_tool` — hardware, system descriptions
3. `package_search_tool` — apt package semantic search
4. `web_search_tool` — last resort, unknown topics only

### Two-Phase Streaming (ui.py pattern)
```python
stream_iter = iter(agent.stream(inputs, stream_mode="messages"))

# Phase A: Thinking trace — commits permanently with transient=False
with Live(..., transient=False) as live_trace:
    for chunk, metadata in stream_iter:
        # collect reasoning_content → display as sliding window
        # collect tool_call_chunks → display tool names
        if first_answer_token:
            break   # hand off to Phase B

# Phase B: Response — same iterator, continues from where A stopped
with Live(..., transient=False) as live_resp:
    for chunk, metadata in stream_iter:
        # stream final answer tokens
```
- **Shared generator** = Phase B resumes exactly where Phase A stopped.
- Both panels use `transient=False` so they stay permanently printed.

### Query Rewriting
```python
# tools.py
from langchain_ollama import ChatOllama

rewriter = ChatOllama(model="qwen2.5:1.5b")

def rewrite_query(raw: str) -> str:
    """Fast pre-processing: convert casual user input into clean search query."""
    ...
```
- Uses a **lightweight model** (1.5b params) so it's near-instant.
- Runs before the main agent is built, inside a `console.status()` spinner.

---

## 5. Quick Reference — Deprecations & Gotchas

| Old / Wrong | New / Correct |
|---|---|
| `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` |
| `reasoning_content` field mismatch with older langchain-ollama | Use `langchain-ollama >= 0.3` with `reasoning=True` |
| `deepagents.create_deep_agent` with local Ollama | Use `create_agent` directly |
| `MemorySaver` in production | `SqliteSaver` (local) or `PostgresSaver` (prod) |
| Generic tool docstrings | Precise, routing-focused docstrings — LLM routes by reading them |
| `collection_exists` guard in Qdrant | Delete-then-recreate is fine for dev; use `collection_exists` for prod safety |

---

## 6. Ecosystem Versions (as of June 2026)

| Package | Stable Version | Notes |
|---|---|---|
| `langchain` | 1.x | Stable, no breaking changes until v2.0 |
| `langgraph` | 1.x | Stable |
| `langchain-ollama` | 0.3+ | Required for `reasoning=True` support |
| `langchain-core` | 1.x | |
| `langchain-community` | 0.3+ | Tavily, Qdrant integrations |
| `langgraph-checkpoint-sqlite` | 3.0+ | CVE-2025-67644 patched in 3.0 |
| `deepagents` | rapidly evolving | Pin version; avoid with local LLMs |
| Python | 3.10+ | 3.9 EOL since Oct 2025 |
