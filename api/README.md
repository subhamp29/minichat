# Bhavyam AI — FastAPI Backend (`api/`)

A standalone FastAPI service that exposes the chat / memory / export / stats
logic of the existing Streamlit app (`app.py`) over HTTP, so a Next.js
frontend can consume it. This is an **additive** change: `app.py` is
untouched and still runnable.

The shared, non-Streamlit logic lives in [`chat_core.py`](./chat_core.py)
and reuses the existing standalone modules (`trust_resolver/`) directly — no
duplication of the trust logic. Message cleaning, SQLite persistence,
local-GGUF loading, and the Groq/Gemini fallback are mirrored from `app.py`
so behaviour stays identical during the migration.

## Run locally

From the **`MiniChat/`** directory (so `chat_history.db` and `.env` are
resolved correctly):

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs:
`http://127.0.0.1:8000/docs`.

CORS is open to all origins (`allow_origins=["*"]`) for development. Lock
this down to the Vercel domain before deploying.

## Endpoints

All paths are prefixed with `/api`.

### 1. `GET /api/models`
List available models (same source as `MODEL_OPTIONS` in `app.py`).

**Response** `200`:
```json
[
  {
    "id": "Groq + Gemini Fallback (Cloud)",
    "display_name": "Groq + Gemini Fallback (Cloud)",
    "backend": "cloud",
    "description": "Primary: Groq llama-3.3-70b | Fallback: Google Gemini 3.5 Flash-Lite"
  },
  {
    "id": "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast",
    "display_name": "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast",
    "backend": "local",
    "description": "⚡ Ultra fast (~400MB RAM). Works smoothly on any device without memory errors."
  }
]
```
`backend` is coarse: `local` for GGUF models, `cloud` for cloud-based ones.

### 2. `GET /api/conversations`
List saved conversations, newest first.

**Response** `200`:
```json
[
  {
    "id": "1e86c880-...",
    "title": "hi",
    "preview": "hi",
    "created_at": "2026-08-19T14:45:13.886183",
    "updated_at": "2026-08-19T14:45:13.886183"
  }
]
```
> The DB has a single `created_at` column (no separate `updated_at`);
> `updated_at` currently mirrors `created_at`.

### 3. `GET /api/conversations/{id}`
Full message history for one conversation. Cleaning-on-load is applied
(identical to `app.py`).

**Response** `200`:
```json
{
  "id": "1e86c880-...",
  "title": "hi",
  "created_at": "2026-08-19T14:45:13.886183",
  "messages": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello! ..."}
  ]
}
```
`404` if the conversation does not exist.

### 4. `POST /api/conversations`
Create a new empty conversation.

**Response** `201`:
```json
{ "id": "f47ac10b-..." }
```

### 5. `DELETE /api/conversations/{id}`
Delete a conversation (mirrors "New Chat" / delete in `app.py`).

**Response** `200`:
```json
{ "id": "f47ac10b-...", "deleted": true }
```
`404` if not found.

### 6. `POST /api/chat`
Stream the assistant reply as **Server-Sent Events**. Reuses the Groq/Gemini
fallback for cloud models and the llama-cpp-python generator for local GGUF;
applies the same per-chunk `_clean_streaming` sanitization and final
`clean_response` cleanup as `app.py`, and persists the full exchange to
`chat_history.db` once streaming completes.

**Request body**:
```json
{
  "conversation_id": "f47ac10b-...",   // existing id, or "" to start a new one
  "message": "Hello!",
  "model_id": "Groq + Gemini Fallback (Cloud)",
  "temperature": 0.7,                   // optional
  "top_p": 0.95,                        // optional
  "max_tokens": 512                     // optional
}
```

**Response** `200` — `text/event-stream`. Each event is one JSON object per
line, prefixed with `data: `:
```
data: {"delta": "Hello"}
data: {"delta": " there!"}
data: {"done": true, "conversation_id": "f47ac10b-...", "model_id": "..."}
```
On error:
```
data: {"error": "..."}
```
> The client should accumulate `delta` chunks to render the answer, and treat
> a `done` event as the end of the stream (the exchange is already saved), or
> an `error` event as a failure (nothing is saved).

### 7. `GET /api/conversations/{id}/export`
Returns the same plain-text export `app.py` generates (sidebar **Export**
button): messages joined as `User: ...` / `Assistant: ...`, separated by
blank lines, each message run through the same cleaning logic.

**Response** `200` — `text/plain; charset=utf-8`,
`Content-Disposition: attachment; filename="chat_history.txt"`.
`404` if not found.

### 8. `GET /api/stats`
Aggregate data for the future 3D dashboard, read from the normalized
`messages` table. Per-model counts, token usage and response times are
**real** (captured by `/api/chat`). Messages with no `model_id` (saved before
the migration, or by the Streamlit `app.py`) are grouped under the
`"unknown"` bucket rather than dropped.

The DB also keeps the original `conversations` table (with the `messages`
JSON blob) for Streamlit `app.py` compatibility; a separate `messages` table
is the source of truth for stats:

```sql
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model_id TEXT,            -- NULL for pre-migration / Streamlit messages
    created_at TEXT NOT NULL,
    token_count INTEGER,      -- completion tokens, when the provider returns usage
    response_ms INTEGER       -- time from request start to stream completion
);
```

The migration (`chat_core.migrate_messages_table`, run automatically on
import) creates this table and backfills existing JSON-blob messages with
`model_id = NULL`. It is idempotent (deterministic row ids + `INSERT OR
IGNORE`), so it is safe to run on existing databases and on fresh installs.

**Response** `200`:
```json
{
  "total_conversations": 14,
  "total_messages": 71,
  "messages_by_role": {"user": 35, "assistant": 36},
  "messages_by_model": {
    "unknown": 68,
    "Groq + Gemini Fallback (Cloud)": 2,
    "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast": 1
  },
  "messages_per_day": [
    {"date": "2026-08-19", "messages": 24},
    {"date": "2026-08-18", "messages": 0}
  ],
  "tokens": {
    "tracked": true,
    "total_completion_tokens": 49,
    "avg_completion_tokens": 24.5,
    "by_model": {
      "Groq + Gemini Fallback (Cloud)": {
        "total_completion_tokens": 49,
        "avg_completion_tokens": 24.5,
        "messages": 2
      }
    }
  },
  "avg_response_time_ms": {
    "tracked": true,
    "overall": 2031.0,
    "by_model": {
      "Groq + Gemini Fallback (Cloud)": {"avg_response_ms": 1416.0, "messages": 2},
      "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast": {"avg_response_ms": 2646.0, "messages": 1}
    }
  }
}
```
> `token_count` is captured from providers that return usage (Groq/Gemini);
> local GGUF responses currently leave it `NULL`, so `tokens.by_model` only
> lists models with captured usage. `response_ms` is recorded for every
> assistant message.

### Extras
- `GET /api/health` — `{ "status": "ok", "models": [...] }`.
- `GET /api/trust` — trust_resolver decision for the current working directory
  (reuses `trust_resolver/`).

## Notes / transition plan
- `chat_core.py` mirrors `app.py`'s cleaning/DB/model logic verbatim because
  `app.py` cannot be imported (it runs Streamlit at module load). Once the
  Streamlit app is retired, `app.py` should import these helpers from
  `chat_core.py` instead of keeping its own copy.
- The API and the Streamlit app share the **same** `chat_history.db` file, so
  conversations created in either surface in the other.
