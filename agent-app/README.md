# agent-app

LangGraph research-assistant reference implementation using FastAPI + Postgres checkpointing.

Each node in the graph is a vehicle for exactly one LangGraph or agentic pattern — the business logic is deliberately simple so the mechanics are easy to follow.

## Quick start

Four things need to be running at once: **Postgres**, an **LLM backend**, the **MCP tool server**, and the **app** itself. Start them in this order.

```bash
# 1. Postgres (checkpointer) — one-time container, port 5433
docker run --name langgraph-db \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=langgraph -e POSTGRES_USER=postgres \
  -p 5433:5432 -d postgres:17

# 2. LLM backend — pick one, then point AGENT_LLM_BASE_URL/AGENT_LLM_MODEL at it in .env
#    Unsloth Studio (GUI, recommended): see "LLM setup" below — serves on :8888
#    llama.cpp (headless): see "LLM setup" below — serves on 8889

# 3. MCP tool server — separate terminal, port 8001
cd agent-app
uv sync
uv run task mcp

# 4. The app — separate terminal, port 8000
cd agent-app
cp .env.example .env   # fill in AGENT_LLM_API_KEY / AGENT_LLM_MODEL / AGENT_LLM_BASE_URL for step 2
uv run python -m app
```

Once `/ready` returns `200`, you're live:

```bash
curl -s http://localhost:8000/ready | jq

curl -s -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-1", "message": "Research recent advances in LLM agents"}' | jq
```

That first call pauses at a human-in-the-loop interrupt with a research plan — approve it to let the agent continue:

```bash
curl -s -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-1", "message": "approve", "approve": true}' | jq
```

If anything doesn't come up, check [Ports used](#ports-used) below for what should be listening where, and [Local setup](#local-setup) for the full step-by-step (Docker/Postgres, Unsloth vs. llama.cpp, Langfuse, LangGraph Studio). Once the API is live, browse it interactively at `http://localhost:8000/docs`.

## Ports used

| Port   | What                                                      | Started by                                          |
|--------|-----------------------------------------------------------|-----------------------------------------------------|
| `5433` | Postgres (LangGraph checkpointer)                         | `docker run ... -p 5433:5432` (step 1)              |
| `8000` | This app (FastAPI)                                        | `uv run python -m app` (step 4)                     |
| `8001` | MCP tool server (`web_search`, `fetch_url`, `fact_check`) | `uv run task mcp` (step 3)                          |
| `8888` | LLM backend — Unsloth Studio (Option A)                   | `unsloth studio -H 127.0.0.1 -p 8888`               |
| `8889` | LLM backend — llama.cpp server (Option B)                 | `llama-server ... --port 8889`                      |
| `3000` | Langfuse UI (optional observability)                      | `docker compose up -d` in a cloned `langfuse/` repo |

`AGENT_APP_PORT` / `AGENT_MCP_PORT` change 8000/8001 if either is taken; `AGENT_DB_URI` and `AGENT_LLM_BASE_URL` change 5433/8888 the same way. Port 5433 (not 5432) is deliberate — it avoids clashing with any Postgres you already have running locally, including a Langfuse stack.

## Patterns demonstrated

| Pattern | Where | Mechanism |
|---|---|---|
| **Subgraph** | `verify_subgraph`, `reflection_subgraph` | Compiled `StateGraph` added as a single node via wrapper |
| **Fan-out / Fan-in** | Inside `verify_subgraph` | `Send` API spawns parallel claim verifiers; `operator.add` reducer collects results |
| **ReAct** | `react_researcher` | Model ↔ `ToolNode` loop; exits when model emits no `tool_calls` |
| **Human-in-the-loop** | `plan_review` (plan generated + guarded by `planner`) | `interrupt()` pauses graph; resumed with `Command(resume=True/False)` |
| **Reflection** | `reflection_subgraph` | Critic → Refiner loop until quality criteria are met |
| **MCP** | `react_researcher` (consumer) + `app/mcp/server.py` (server) | `fastmcp` exposes tools; `langchain-mcp-adapters` binds them to LangGraph |
| **Guardrails** | `input_guard`, `resume_guard`, `output_guard` | Three-layer input check (regex → GLiGuard → LLM topic); resume message checked by dedicated node; two-layer output check (GLiGuard PII redaction → LLM grounding) |
| **Dead letter** | `dead_letter` terminal node | Any unhandled node exception writes `DeadLetterInfo` to state and routes here instead of crashing |
| **Time-travel** | `GET /v1/threads/{id}/history`, `POST /v1/threads/{id}/replay` | Postgres checkpointer stores every state snapshot; replay re-invokes from any checkpoint |
| **Token streaming** | `POST /v1/chat/stream` | `astream_events` pipes writer-node tokens as SSE frames |
| **Prompt externalization (i18n-ready)** | `app/prompts/en/**/*.md`, loaded via `app/prompts/loader.py` | Every node prompt lives in an external `.md` file keyed by `(locale, node, name)` instead of a Python string constant; `AGENT_LOCALE` selects the locale directory, so adding a language is a new sibling directory, not a code change |

## Reliability safeguards

All limits are configurable via `AGENT_` env vars.

| Safeguard | Env var | Default | What it prevents |
|---|---|---|---|
| Reflection ceiling | `AGENT_MAX_REFLECTION_ATTEMPTS` | `5` | Critic/refiner loop running indefinitely when quality bar is never met |
| ReAct ceiling | `AGENT_MAX_REACT_STEPS` | `10` | Model emitting tool calls forever without self-terminating |
| LLM timeout | `AGENT_LLM_TIMEOUT_SECONDS` | `60` | Single LLM call blocking the event loop indefinitely |
| Retry with backoff | `AGENT_LLM_MAX_RETRIES` | `3` | Transient rate-limit / 5xx errors surfacing as failures immediately |
| Global pipeline cap | `AGENT_MAX_PIPELINE_STEPS` | `50` | Routing bugs or unexpected loops consuming unbounded supersteps (`GraphRecursionError` on breach) |
| MCP tool call timeout | `AGENT_MCP_TOOL_CALL_TIMEOUT_SECONDS` | `30` | A hung/slow MCP tool call blocking a branch indefinitely — applied to `fact_check` in `verify_subgraph` directly, and to every ReAct tool call (`web_search`, `fetch_url`, `fact_check`) via `ToolNode(awrap_tool_call=...)` in `react_researcher`'s `tools` node |
| GLiGuard concurrency cap | `AGENT_GUARD_MAX_CONCURRENCY` | `4` | Unbounded concurrent forward passes through the single shared GLiGuard model exhausting host RAM or GPU VRAM under concurrent requests |
| Postgres connection pool | `AGENT_DB_POOL_MAX_SIZE` | `20` | Concurrent requests serializing through a single checkpointer DB connection |
| Guard input/output call timeout | `AGENT_GUARD_TIMEOUT_SECONDS` | `10` | A hung/slow GLiGuard classification call blocking a request indefinitely |
| Guard model load retry | `AGENT_GUARD_LOAD_RETRIES` / `AGENT_GUARD_LOAD_TIMEOUT_SECONDS` | `3` / `120` | A stuck or transiently-failing HuggingFace model download hanging app startup forever with no diagnostic |
| Graceful shutdown timeout | `AGENT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS` | `25` | In-flight requests being cut off mid-response on SIGTERM, or the process hanging past the deployment's grace period and being SIGKILLed |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- Python 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- LLM inference server — see [LLM setup](#2-llm-setup)

## Local setup

### 1. Postgres

We use port 5433 to avoid conflicts with any existing Postgres instances you might have running on the default port.

```bash
docker run --name langgraph-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=langgraph \
  -e POSTGRES_USER=postgres \
  -p 5433:5432 \
  -d postgres:17
```

### 2. LLM setup

**Option A — Unsloth Studio** (recommended; GUI, easier model management):

See https://unsloth.ai/docs/get-started/install

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
unsloth studio -H 127.0.0.1 -p 8888
# Open http://127.0.0.1:8888, download unsloth/Qwen3.6-35B-A3B-MTP-GGUF (UD-Q4_K_XL quant), click Start.
# Set AGENT_LLM_BASE_URL=http://127.0.0.1:8888/v1 to the port shown in Studio.
```

**Option B — llama.cpp server** (headless):

```bash
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "*UD-Q4_K_XL*"
./llama.cpp/llama-server \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --alias "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" \
  --ctx-size 16384 --port 8889
# Set AGENT_LLM_BASE_URL=http://127.0.0.1:8889/v1
```

### 3. MCP server (separate terminal)

```bash
cd agent-app
uv run task mcp   # serves on http://localhost:8001
```

### 4. App

```bash
cd agent-app
cp .env.example .env   # fill in LANGFUSE_* keys for observability (optional)
uv sync
uv run python -m app
# → http://localhost:8000/docs
```

### 5. Langfuse (optional — observability)

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
# UI at http://localhost:3000 — default login: admin@langfuse.com / password
# Create a project and copy the keys to .env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
```

### 6. LangGraph Studio

Download the free Mac desktop app from https://studio.langchain.com.
Open the `agent-app/` directory — Studio reads `langgraph.json` and starts a dev server automatically.

## Environment variables

All agent variables use the `AGENT_` prefix. Defaults work for local development.

| Variable | Default | Description |
|---|---|---|
| `AGENT_LOCALE` | `en` | Locale directory under `app/prompts/` used to load node prompts |
| `AGENT_DB_URI` | `postgresql://postgres:postgres@localhost:5433/langgraph` | Postgres connection string |
| `AGENT_DB_POOL_MAX_SIZE` | `20` | Max connections in the checkpointer's AsyncConnectionPool |
| `AGENT_LLM_MODEL` | `openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` | LiteLLM model identifier |
| `AGENT_LLM_BASE_URL` | `http://127.0.0.1:8888/v1` | LLM provider base URL (Unsloth Studio) |
| `AGENT_LLM_API_KEY` | `None` | API key for the LLM provider (any OpenAI-compatible backend) |
| `AGENT_LLM_THINKING` | `false` | Enable Qwen3 chain-of-thought mode |
| `AGENT_LLM_TIMEOUT_SECONDS` | `60` | Per-call LLM timeout in seconds |
| `AGENT_LLM_MAX_RETRIES` | `3` | Retries for transient LLM errors (exponential backoff) |
| `AGENT_MCP_SERVER_URL` | `http://localhost:8001/mcp` | MCP tool server URL |
| `AGENT_MCP_CONNECT_TIMEOUT_SECONDS` | `10` | Per-attempt timeout for connecting to the MCP server and listing tools |
| `AGENT_MCP_TOOL_CALL_TIMEOUT_SECONDS` | `30` | Per-call timeout for invoking an MCP tool (e.g. `fact_check`) from within a node |
| `AGENT_MAX_REFLECTION_ATTEMPTS` | `5` | Hard ceiling on reflection critic/refiner iterations |
| `AGENT_MAX_REACT_STEPS` | `10` | Hard ceiling on ReAct tool-call iterations |
| `AGENT_MAX_PIPELINE_STEPS` | `50` | LangGraph `recursion_limit`: total supersteps across the whole pipeline per invocation |
| `AGENT_GUARD_MODEL` | `fastino/gliguard-LLMGuardrails-300M` | HuggingFace model for GLiGuard (prompt injection, jailbreak, PII) |
| `AGENT_GUARD_DEVICE` | `cpu` | Inference device for GLiGuard: `cpu`, `cuda`, or `mps` |
| `AGENT_GUARD_TIMEOUT_SECONDS` | `10` | Per-call timeout for GLiGuard classification calls |
| `AGENT_GUARD_MAX_CONCURRENCY` | `4` | Max concurrent GLiGuard inference calls; excess calls queue on a semaphore |
| `AGENT_GUARD_LOAD_TIMEOUT_SECONDS` | `120` | Per-attempt timeout for downloading/loading the GLiGuard model at startup |
| `AGENT_GUARD_LOAD_RETRIES` | `3` | Attempts to load the GLiGuard model at startup before startup fails |
| `AGENT_READINESS_CHECK_TIMEOUT_SECONDS` | `3` | Timeout for the `/ready` endpoint's database connectivity check |
| `AGENT_WEB_SEARCH_MAX_RESULTS` | `10` | Server-side cap on `max_results` for `web_search` and `fact_check` MCP tools |
| `AGENT_API_KEY` | `None` | X-API-Key header value; when unset, auth is disabled (local dev). App logs a startup warning if unset. |
| `AGENT_RATE_LIMIT` | `None` | slowapi limit string, e.g. `20/minute`; when unset, rate limiting is disabled. App logs a startup warning if unset. |
| `AGENT_SSE_KEEPALIVE_SECONDS` | `15` | SSE ping interval to prevent proxy idle-timeout on long graph runs |
| `AGENT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS` | `25` | On SIGTERM, max seconds uvicorn waits for in-flight requests before cancelling them; keep below the deployment's `terminationGracePeriodSeconds` |
| `AGENT_APP_HOST` | `0.0.0.0` | Bind host for the FastAPI app |
| `AGENT_APP_PORT` | `8000` | Bind port for the FastAPI app |
| `AGENT_MCP_HOST` | `0.0.0.0` | Bind host for the MCP tool server |
| `AGENT_MCP_PORT` | `8001` | Bind port for the MCP tool server |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse project public key (observability) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse project secret key (observability) |
| `LANGFUSE_BASE_URL` | `http://localhost:3000` | Langfuse server URL |

## API

| Method | Path                       | Description                                                                    |
|--------|----------------------------|--------------------------------------------------------------------------------|
| `GET`  | `/health`                  | Liveness probe — static, no dependency checks                                  |
| `GET`  | `/ready`                   | Readiness probe — checks GLiGuard loaded, Postgres reachable, MCP tools loaded |
| `GET`  | `/metrics/dead-letter`     | In-process count of dead-lettered runs since this pod started, by failed node  |
| `POST` | `/v1/chat`                 | Invoke the agent (first turn or interrupt resume); returns full response       |
| `POST` | `/v1/chat/stream`          | Same as `/v1/chat` but streams writer tokens as SSE                            |
| `GET`  | `/v1/threads/{id}/history` | Full checkpoint list for a thread (time-travel)                                |
| `POST` | `/v1/threads/{id}/replay`  | Re-invoke from a named checkpoint                                              |

`/v1/chat` and `/v1/chat/stream` classify LLM/DB errors identically via the shared
`_classify_error` helper (429 rate limit, 503 service unavailable, 502 service error,
500 recursion-limit exceeded) — `/v1/chat` returns these as an `HTTPException`,
`/v1/chat/stream` emits them as an `error` SSE frame with the same code/detail.

### SSE event types (`POST /v1/chat/stream`)

| Event       | Payload                                    | When                                      |
|-------------|--------------------------------------------|-------------------------------------------|
| `token`     | `{"token": "..."}`                         | Each writer-node LLM token                |
| `interrupt` | `{"interrupt_value": {...}}`               | Graph paused at planner human-in-the-loop |
| `done`      | `{"status": "...", "final_answer": "..."}` | Graph reached END                         |
| `error`     | `{"code": 4xx/5xx, "detail": "..."}`       | Unhandled exception escaped the graph     |

### Example

```bash
# Start a research session
curl -s -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-1", "message": "Research recent advances in LLM agents"}' | jq

# Resume after human-in-the-loop interrupt (approve the plan)
curl -s -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-1", "message": "approve", "approve": true}' | jq

# Stream writer tokens as SSE
curl -s -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-2", "message": "Research recent advances in LLM agents"}'

# Retrieve checkpoint history
curl -s http://localhost:8000/v1/threads/session-1/history | jq

# Replay from a specific checkpoint
curl -s -X POST http://localhost:8000/v1/threads/session-1/replay \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_id": "<id_from_history>"}' | jq
```

## Tests

```bash
# Full suite (requires Postgres on localhost:5433)
uv run pytest
# or via taskipy
uv run task test

# With coverage report
uv run task test-cov

# Models only — no Postgres needed
uv run pytest tests/test_models.py

# API routes
uv run pytest tests/test_routers.py

# Graph checkpointing and time-travel
uv run pytest tests/graph/

# Individual node tests (includes resume_guard, prompt_utils, writer)
uv run pytest tests/graph/nodes/

# GLiGuard and guard utilities
uv run pytest tests/guards/

# MCP server tools
uv run pytest tests/mcp/

# Externalized prompt files (Phase 6) — no server/Postgres/MCP needed
uv run pytest tests/test_prompts.py

# Node-level evals (Phase 5a) — one node factory at a time, no server/Postgres/MCP needed
uv run pytest tests/evals/test_node_tasks.py

# Eval harness unit tests (Phase 5d) — trace assertions, evaluators, dataset validation; zero network calls
uv run pytest tests/evals/test_trace_assertions.py tests/evals/test_evaluators.py tests/evals/test_dataset_validation.py

# Smoke test (requires app + Postgres; LLM-dependent tests auto-skip if unreachable)
uv run python evals/smoke_test.py
uv run python evals/smoke_test.py --base-url http://localhost:8010  # custom port

# HTTP-driven experiment (Phase 5d; requires a running app + Postgres; Langfuse upload is optional)
uv run task experiment
uv run python evals/run_experiment.py evals/configs/exp_baseline.yaml
```
