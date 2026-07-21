# Agent UI — Implementation Plan

## Overview

Build AG-UI protocol support in `agent-app` and a companion web UI (`agent-ui/`) that consumes it.

**AG-UI** (Agent-User Interaction protocol) is an open, event-based protocol for connecting AI agent backends to frontends over SSE. The agent streams typed events; the UI renders them without knowing agent internals.

---

## Architecture

```
agent-ui/ (Next.js)            agent-app/ (FastAPI + LangGraph)
┌─────────────────┐   POST      ┌───────────────────────────────┐
│  Chat panel     │ ──────────► │  POST /v1/agui                │
│  History panel  │ ◄── SSE ─── │  AG-UI adapter                │
│  Interrupt UI   │             │    ↕                          │
└─────────────────┘             │  existing graph / checkpointer│
                                └───────────────────────────────┘
```

The adapter translates agent-app's internal SSE format (`token`, `interrupt`, `done`, `error`) into the AG-UI event envelope so the UI stays protocol-agnostic.

---

## AG-UI Event Mapping

| agent-app event | AG-UI event(s)                                                                     |
|-----------------|------------------------------------------------------------------------------------|
| stream start    | `RUN_STARTED`                                                                      |
| `token`         | `TEXT_MESSAGE_START` (first), `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` (on done) |
| `interrupt`     | `STEP_FINISHED` + custom `INTERRUPT` state snapshot                                |
| `done`          | `RUN_FINISHED`                                                                     |
| `error`         | `RUN_ERROR`                                                                        |

AG-UI state snapshot carries `thread_id`, `status`, `interrupt_value`, and `final_answer` so the UI can render the full conversation state without extra REST calls.

---

## Phase 1 — AG-UI Adapter in `agent-app`

**Goal:** Add a single AG-UI-compatible endpoint. No changes to existing endpoints.

### 1.1 — Models (`app/models.py`)

Add AG-UI request/response models:

```python
class AGUIRunInput(BaseModel):
    thread_id: str          # maps to ChatRequest.thread_id
    message: str            # maps to ChatRequest.message
    approve: bool | None    # maps to ChatRequest.approve
    run_id: str             # AG-UI run identifier (UUID, client-generated)
```

### 1.2 — AG-UI event helpers (`app/agui.py`)

Stateless functions that format AG-UI SSE frames:

```python
def run_started(run_id: str, thread_id: str) -> str: ...
def text_message_start(message_id: str) -> str: ...
def text_message_content(message_id: str, delta: str) -> str: ...
def text_message_end(message_id: str) -> str: ...
def state_snapshot(thread_id: str, status: str, **kwargs) -> str: ...
def run_finished(run_id: str, thread_id: str) -> str: ...
def run_error(run_id: str, code: int, message: str) -> str: ...
```

Each function returns a fully formatted `data: <json>\n\n` SSE string with `event: <TYPE>`.

### 1.3 — Endpoint (`app/routers.py`)

```
POST /v1/agui
```

- Accepts `AGUIRunInput` (JSON body)
- Returns `text/event-stream`
- Internally calls the same graph logic as `/v1/chat/stream`
- Wraps `_generate()` output in AG-UI event envelope via `app/agui.py`
- Auth: same `verify_api_key` dependency

### 1.4 — Tests (`tests/test_agui.py`)

- Unit test each event helper (output format)
- Integration test: mock graph → verify SSE frames parse as valid AG-UI events

**Deliverable:** `uv run pytest tests/test_agui.py` passes; `/v1/agui` documented in Swagger.

---

## Phase 2 — `agent-ui/` Web App

**Goal:** Simple, functional chat UI that speaks AG-UI to `agent-app`.

### 2.1 — Project Setup

```
agent-ui/
├── package.json            # Next.js 15, React 19, TypeScript
├── .env.local.example      # NEXT_PUBLIC_AGENT_URL, NEXT_PUBLIC_AGENT_API_KEY
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx        # renders <ChatShell />
│   ├── components/
│   │   ├── ChatShell.tsx   # top-level layout (sidebar + main)
│   │   ├── MessageList.tsx # renders assistant/user messages
│   │   ├── InputBar.tsx    # message input + send button
│   │   ├── InterruptCard.tsx # approve/reject planner plan UI
│   │   └── HistoryPanel.tsx  # thread checkpoint list
│   ├── hooks/
│   │   ├── useAGUIStream.ts  # SSE client, AG-UI event parsing
│   │   └── useThreadHistory.ts # GET /v1/threads/:id/history
│   └── lib/
│       ├── agui-client.ts    # typed AG-UI event types + fetch wrapper
│       └── api.ts            # REST calls (history, replay)
```

Stack: **Next.js 15** (App Router), **React 19**, **TypeScript**, **Tailwind CSS v4**. No heavy component libraries — keep it minimal.

### 2.2 — AG-UI Client (`src/lib/agui-client.ts`)

Typed AG-UI event union, SSE parser, and `runAgent()` function:

```ts
type AGUIEvent =
  | { type: 'RUN_STARTED'; run_id: string; thread_id: string }
  | { type: 'TEXT_MESSAGE_START'; message_id: string }
  | { type: 'TEXT_MESSAGE_CONTENT'; message_id: string; delta: string }
  | { type: 'TEXT_MESSAGE_END'; message_id: string }
  | { type: 'STATE_SNAPSHOT'; thread_id: string; status: string; interrupt_value?: object; final_answer?: string }
  | { type: 'RUN_FINISHED'; run_id: string }
  | { type: 'RUN_ERROR'; run_id: string; code: number; message: string };

async function* runAgent(input: RunInput): AsyncGenerator<AGUIEvent>
```

### 2.3 — Core Hook (`src/hooks/useAGUIStream.ts`)

```ts
function useAGUIStream(): {
  messages: Message[];
  isStreaming: boolean;
  interruptValue: object | null;
  sendMessage: (text: string) => void;
  resolveInterrupt: (approve: boolean) => void;
  threadId: string;
}
```

State machine: `idle → streaming → interrupted | done | error`.

### 2.4 — UI Components

- **MessageList**: user bubbles (right), assistant bubbles (left), streaming tokens appended in real time
- **InputBar**: textarea + send; disabled while streaming; shows spinner
- **InterruptCard**: shown when `interruptValue != null`; renders plan as markdown; Approve / Reject buttons
- **HistoryPanel**: collapsible sidebar listing checkpoints; clicking a checkpoint calls `POST /v1/threads/:id/replay`

### 2.5 — Configuration

`.env.local`:
```
NEXT_PUBLIC_AGENT_URL=http://localhost:8000
NEXT_PUBLIC_AGENT_API_KEY=<key>
```

API key is sent as `X-API-Key` header from the browser (acceptable for local POC; note: not safe for production).

**Deliverable:** `npm run dev` in `agent-ui/` opens a working chat UI; streaming tokens render live; planner interrupt shows approve/reject card.

---

## Phase 3 — Polish & DX

- Add `README.md` section to root documenting how to start both services
- Add `docker-compose.yml` service for `agent-ui` (optional)
- `npm run build` passes with no TypeScript errors
- Lighthouse accessibility score ≥ 80

---

## Out of Scope

- Multi-user / session persistence in the browser
- Authentication in the UI beyond passing the API key as a header
- Production deployment hardening
- Unit tests for React components (Playwright e2e is sufficient for a POC)

---

## Task Breakdown

| # | Task | Phase |
|---|------|-------|
| 1 | Add `AGUIRunInput` model to `app/models.py` | 1.1 |
| 2 | Implement `app/agui.py` event helpers | 1.2 |
| 3 | Add `POST /v1/agui` endpoint to `app/routers.py` | 1.3 |
| 4 | Write `tests/test_agui.py` | 1.4 |
| 5 | Scaffold Next.js app (`npx create-next-app`) | 2.1 |
| 6 | Implement `agui-client.ts` and `api.ts` | 2.2 |
| 7 | Implement `useAGUIStream` hook | 2.3 |
| 8 | Build `MessageList`, `InputBar`, `InterruptCard` | 2.4 |
| 9 | Build `HistoryPanel` + replay flow | 2.4 |
| 10 | Wire `.env.local` config + smoke test end-to-end | 2.5 |
| 11 | README update + optional docker-compose | 3 |
