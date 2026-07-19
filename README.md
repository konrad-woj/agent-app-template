# agent-app-template

A collection of best practices and patterns for Python AI/ML agent projects — designed to learn from or use as a starting point.

The repo is a monorepo with three packages:

| Package | Description |
|---------|-------------|
| [`agent-lib/`](agent-lib/README.md) | LangGraph agent library — reusable graph primitives and Postgres checkpointing |
| [`agent-app/`](agent-app/README.md) | LangGraph research-assistant reference implementation — FastAPI + Postgres checkpointing |

## Highlights

### LangGraph agent (`agent-lib` / `agent-app`)

- **LangGraph patterns** — subgraph, fan-out/fan-in (`Send` API), ReAct, human-in-the-loop (`interrupt`), reflection, MCP, guardrails, dead letter, time-travel, token streaming
- **Postgres checkpointing** — every state snapshot persisted; full time-travel and replay via API
- **Three-layer guardrails** — regex → GLiGuard → LLM topic check on input; GLiGuard PII redaction → LLM grounding check on output
- **MCP integration** — `fastmcp` server exposes tools; `langchain-mcp-adapters` binds them to the ReAct researcher node
- **Reliability safeguards** — configurable ceilings for reflection loops, ReAct steps, LLM timeouts, retries, and total pipeline supersteps
- **SSE streaming** — writer-node tokens streamed as Server-Sent Events; keepalive ping prevents proxy idle-timeout
- **Langfuse observability** — optional tracing; configure via `LANGFUSE_*` env vars