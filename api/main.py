"""Bhavyam AI — FastAPI backend.

Exposes the chat/memory/export/stats logic of the existing Streamlit app
(app.py) as a standalone HTTP API so a Next.js frontend can consume it.
All Streamlit-coupled code stays in app.py (untouched); the shared,
non-Streamlit logic lives in api/chat_core.py, which reuses
router_client.py and trust_resolver/ directly.

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
import json
import time
import uuid
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import jwt
from jwt.algorithms import ECAlgorithm
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field

# Make this package importable regardless of CWD.
_API_DIR = str(Path(__file__).resolve().parent)
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from chat_core import (  # noqa: E402
    MODEL_OPTIONS,
    SYSTEM_PROMPT,
    CHAT_TEMPLATES,
    generate,
    build_prompt,
    trim_history,
    load_model,
    get_conversation,
    list_conversations,
    save_conversation,
    delete_conversation,
    insert_message,
    generate_conversation_title,
    export_conversation_text,
    list_model_catalogue,
    _clean_streaming,
    clean_response,
    truncate_text,
    check_trust,
    get_trust_events,
    RouterConfigurationError,
    RouterError,
    N8nOrchestrationError,
    _N8N_CLIENT_AVAILABLE,
    send_to_n8n_webhook,
)


app = FastAPI(
    title="Bhavyam AI API",
    description="Standalone FastAPI backend extracted from the MiniChat Streamlit app.",
    version="0.1.0",
)

# CORS — must come before routes/auth so preflight is handled by middleware,
# not rejected by the route layer.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8080",
        "https://bhavyam-frontend.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ---------------------------------------------------------------------------
# Supabase JWT auth
# ---------------------------------------------------------------------------
import os as _os

load_dotenv()

_SUPABASE_URL = _os.environ.get("SUPABASE_URL", "").rstrip("/")
_SUPABASE_KEY = _os.environ.get("SUPABASE_KEY", "")
_JWKS_URL = f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json" if _SUPABASE_URL else ""
_jwks_cache: dict[str, dict] = {}
_jwks_cache_ts: float = 0.0
_JWKS_TTL_SECONDS = 3600  # 1 hour


async def _fetch_jwks() -> dict:
    """Fetch Supabase JWKS from .well-known endpoint with apikey header."""
    if not _JWKS_URL or not _SUPABASE_KEY:
        raise HTTPException(status_code=401, detail="Server misconfigured: SUPABASE_URL or SUPABASE_KEY missing")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_JWKS_URL, headers={"apikey": _SUPABASE_KEY})
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to fetch JWKS")
        data = resp.json()
    return {
        key["kid"]: ECAlgorithm.from_jwk(json.dumps(key))
        for key in data.get("keys", [])
    }


async def _get_jwks() -> dict:
    """Get JWKS cache, re-fetching if stale (>1 hour old)."""
    global _jwks_cache, _jwks_cache_ts
    now = time.time()
    if _jwks_cache and (now - _jwks_cache_ts) < _JWKS_TTL_SECONDS:
        return _jwks_cache
    _jwks_cache = await _fetch_jwks()
    _jwks_cache_ts = now
    return _jwks_cache


async def get_current_user(request: Request) -> dict:
    """FastAPI dependency: validate Supabase JWT from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header[7:]
    try:
        unverified = jwt.get_unverified_header(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    kid = unverified.get("kid", "")
    jwks = await _get_jwks()
    key = jwks.get(kid)
    if not key:
        # Re-fetch once in case of key rotation
        jwks = await _fetch_jwks()
        key = jwks.get(kid)
        if not key:
            raise HTTPException(status_code=401, detail="Key not found")
    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=["ES256"],
            audience="authenticated",
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"payload": payload, "token": token}


async def _supabase_rest_get(path: str, user_token: str, params: dict | None = None) -> dict | list:
    """Make an authenticated GET request to the Supabase REST API."""
    url = f"{_SUPABASE_URL}/rest/v1/{path.lstrip('/')}"
    headers = {
        "apikey": _os.environ.get("SUPABASE_KEY", ""),
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Supabase auth failed")
        if resp.status_code == 404:
            return [] if "messages" in path or "conversations" in path else {}
        resp.raise_for_status()
        return resp.json()


async def _supabase_rest(method: str, path: str, user_token: str, json_data: dict | None = None, params: dict | None = None) -> dict | list:
    """Make an authenticated request to the Supabase REST API."""
    url = f"{_SUPABASE_URL}/rest/v1/{path.lstrip('/')}"
    headers = {
        "apikey": _os.environ.get("SUPABASE_KEY", ""),
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(method, url, headers=headers, json=json_data, params=params)
        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Supabase auth failed")
        if resp.status_code == 404:
            return [] if "messages" in path or "conversations" in path else {}
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    conversation_id: str = Field(default="", description="Existing conversation id, or empty to start a new one.")
    message: str = Field(..., min_length=1, description="The user's message.")
    model_id: str = Field(..., description="Model id from GET /api/models.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class ConversationCreateResponse(BaseModel):
    id: str


# ---------------------------------------------------------------------------
# 1. GET /api/models
# ---------------------------------------------------------------------------
@app.get("/api/models")
def get_models():
    """List available models (id, display_name, backend, description)."""
    return list_model_catalogue()


# ---------------------------------------------------------------------------
# 2. GET /api/conversations
# ---------------------------------------------------------------------------
@app.get("/api/conversations")
async def get_conversations(user: dict = Depends(get_current_user)):
    """List saved conversations for the authenticated user from Supabase."""
    user_token = user["token"]
    user_id = user["payload"].get("sub")
    rows = await _supabase_rest_get(
        "conversations",
        user_token,
        params={"select": "id,title,created_at,updated_at", "user_id": f"eq.{user_id}", "order": "updated_at.desc"},
    )
    if not isinstance(rows, list):
        rows = []
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "preview": "",
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# 3. GET /api/conversations/{id}
# ---------------------------------------------------------------------------
@app.get("/api/conversations/{conversation_id}")
async def get_conversation_by_id(conversation_id: str, user: dict = Depends(get_current_user)):
    """Return full message history for one conversation from Supabase."""
    user_token = user["token"]
    user_id = user["payload"].get("sub")

    convs = await _supabase_rest_get(
        "conversations",
        user_token,
        params={"select": "id,title,created_at", "id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
    )
    if not isinstance(convs, list) or len(convs) == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = convs[0]

    messages = await _supabase_rest_get(
        "messages",
        user_token,
        params={"select": "role,content", "conversation_id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}", "order": "created_at.asc"},
    )
    if not isinstance(messages, list):
        messages = []

    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# 4. POST /api/conversations
# ---------------------------------------------------------------------------
@app.post("/api/conversations", status_code=201)
async def create_conversation(user: dict = Depends(get_current_user)):
    """Create a new empty conversation in Supabase and return its id."""
    user_token = user["token"]
    user_id = user["payload"].get("sub")
    conv_id = str(uuid.uuid4())
    await _supabase_rest(
        "POST",
        "conversations",
        user_token,
        json_data={"id": conv_id, "user_id": user_id, "title": "New Chat"},
    )
    return ConversationCreateResponse(id=conv_id)


# ---------------------------------------------------------------------------
# 5. DELETE /api/conversations/{id}
# ---------------------------------------------------------------------------
@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_by_id(conversation_id: str, user: dict = Depends(get_current_user)):
    """Delete a conversation from Supabase."""
    user_token = user["token"]
    user_id = user["payload"].get("sub")

    # Verify conversation exists and belongs to user.
    convs = await _supabase_rest_get(
        "conversations",
        user_token,
        params={"select": "id", "id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
    )
    if not isinstance(convs, list) or len(convs) == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await _supabase_rest(
        "DELETE",
        "conversations",
        user_token,
        params={"id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
    )
    return {"id": conversation_id, "deleted": True}


# ---------------------------------------------------------------------------
# 6. POST /api/chat  (Server-Sent Events)
# ---------------------------------------------------------------------------
def _stream_chat_events(
    conversation_id: str,
    user_message: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    user_token: str | None = None,
    user_id: str | None = None,
):
    """Yield SSE event dicts while streaming the model response.

    Mirrors app.py's per-backend streaming generators exactly, applying the
    same ``_clean_streaming`` per-chunk cleanup and ``clean_response`` final
    cleanup.
    """
    state = {"captured_tokens": [], "error_occurred": False, "error_message": "", "token_count": None}
    emitted_error = False
    start_time = time.time()

    # Load conversation history from Supabase if user credentials provided,
    # otherwise fall back to SQLite (Streamlit app.py compatibility).
    history = []
    if user_token and user_id:
        try:
            import asyncio
            convs = asyncio.get_event_loop().run_until_complete(_supabase_rest_get(
                "conversations",
                user_token,
                params={"select": "messages", "id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
            ))
            if isinstance(convs, list) and len(convs) > 0:
                raw_messages = convs[0].get("messages", [])
                if isinstance(raw_messages, str):
                    raw_messages = json.loads(raw_messages)
                history = raw_messages
        except Exception:
            history = []
    else:
        conv = get_conversation(conversation_id)
        history = conv["messages"] if conv else []

    model_cfg = MODEL_OPTIONS.get(model_id)
    if model_cfg is None:
        yield {"error": f"Unknown model_id '{model_id}'. Valid ids: {list(MODEL_OPTIONS.keys())}"}
        return

    backend = model_cfg.get("backend", "local_gguf")
    template = model_cfg.get("template", "phi3")
    model_slug = model_cfg.get("slug", "")
    stop_sequences = CHAT_TEMPLATES[template]["stop"]

    augmented_user_msg = user_message
    history_for_prompt = trim_history(history, SYSTEM_PROMPT, augmented_user_msg, template)

    # ----- remote (router_client.stream_chat) -----
    if backend == "remote":
        try:
            remote_messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + history_for_prompt
                + [{"role": "user", "content": augmented_user_msg}]
            )
            chunks = []
            for chunk in stream_chat(
                messages=remote_messages,
                model_slug=model_slug,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ):
                state["captured_tokens"].append(chunk)
                chunks.append(chunk)

            if len(chunks) == 1 and chunks[0]:
                # Single-shot response: emit the cleaned full text as one delta
                # (no artificial typewriter delay — the frontend can animate).
                cleaned_full = _clean_streaming(chunks[0])
                if cleaned_full:
                    yield {"delta": cleaned_full}
            else:
                prev_chunk = ""
                for chunk in chunks:
                    if (
                        prev_chunk
                        and not prev_chunk.endswith((" ", "\n", "\t"))
                        and chunk
                        and not chunk.startswith((" ", "\n", "\t"))
                    ):
                        chunk = " " + chunk
                    prev_chunk = chunk
                    cleaned_chunk = _clean_streaming(chunk)
                    if cleaned_chunk:
                        yield {"delta": cleaned_chunk}
        except (RouterConfigurationError, RouterError) as exc:
            state["error_occurred"] = True
            state["error_message"] = str(exc)
            emitted_error = True
            yield {"error": _clean_streaming(str(exc))}

    # ----- n8n orchestrated -----
    elif backend == "n8n_orchestrated":
        if not _N8N_CLIENT_AVAILABLE:
            state["error_occurred"] = True
            state["error_message"] = (
                "n8n_orchestrated backend is not available: n8n_supabase_client module is missing."
            )
            emitted_error = True
            yield {"error": _clean_streaming(state["error_message"])}
        else:
            try:
                n8n_res = send_to_n8n_webhook(
                    message=augmented_user_msg,
                    conversation_id=conversation_id,
                    model=model_slug or "claude-opus-free",
                    backend="n8n_orchestrated",
                )
                answer = n8n_res.get("response", "")
                if not answer:
                    answer = "n8n completed workflow successfully but returned empty response."
                state["captured_tokens"].append(answer)
                yield {"delta": _clean_streaming(answer)}
            except N8nOrchestrationError as exc:
                state["error_occurred"] = True
                state["error_message"] = str(exc)
                emitted_error = True
                yield {"error": _clean_streaming(str(exc))}
            except Exception as exc:  # noqa: BLE001
                state["error_occurred"] = True
                state["error_message"] = f"n8n Orchestration Error: {exc}"
                emitted_error = True
                yield {"error": _clean_streaming(str(exc))}

    # ----- groq + gemini fallback -----
    elif backend == "groq_gemini":
        try:
            from chat_core import get_chat_response

            api_messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + history_for_prompt
                + [{"role": "user", "content": augmented_user_msg}]
            )
            full_text, usage = get_chat_response(api_messages, temperature=temperature)
            full_text = _clean_streaming(full_text)
            if not full_text:
                full_text = "*(no response)*"
            state["captured_tokens"].append(full_text)
            # Real token usage from the provider (completion tokens), if returned.
            state["token_count"] = (
                (usage or {}).get("completion_tokens") if usage else None
            )
            yield {"delta": full_text}
        except Exception as exc:  # noqa: BLE001
            state["error_occurred"] = True
            state["error_message"] = str(exc)
            emitted_error = True
            yield {"error": _clean_streaming(str(exc))}

    # ----- local GGUF (llama-cpp-python) -----
    else:
        try:
            model = load_model(model_id)
            if model is None:
                state["error_occurred"] = True
                state["error_message"] = "Local model unavailable."
                emitted_error = True
                yield {"error": state["error_message"]}
            else:
                prompt = build_prompt(SYSTEM_PROMPT, history_for_prompt, augmented_user_msg, template)
                for token in generate(model, prompt, temperature, top_p, max_tokens, stop_sequences):
                    if token.startswith("\n\n*[Generation error:"):
                        state["error_occurred"] = True
                        state["error_message"] = token
                        return
                    if "does not support image input" in token or "Cannot read" in token:
                        state["error_occurred"] = True
                        state["error_message"] = token
                        return
                    state["captured_tokens"].append(token)
                    cleaned_token = _clean_streaming(token)
                    if cleaned_token:
                        yield {"delta": cleaned_token}
        except Exception as e:  # noqa: BLE001
            state["error_occurred"] = True
            state["error_message"] = clean_response(f"\n\n*[Generation error: {e}]*")
            emitted_error = True
            yield {"error": state["error_message"]}

    # Terminal event — mirrors app.py's post-stream processing.
    if not state["error_occurred"]:
        raw_response = "".join(state["captured_tokens"])
        final_response = clean_response(raw_response)
        # Note: conversation persistence is now handled by the frontend via Supabase.
        # The Streamlit app (app.py) still uses save_conversation/insert_message
        # from chat_core.py for its own SQLite storage.
        yield {"done": True, "conversation_id": conversation_id, "model_id": model_id}
    elif not emitted_error:
        yield {"error": state.get("error_message", "Unknown error")}


@app.post("/api/chat")
def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    """Stream the assistant's reply as Server-Sent Events."""
    conversation_id = req.conversation_id.strip() or str(uuid.uuid4())

    def event_generator():
        for event in _stream_chat_events(
            conversation_id=conversation_id,
            user_message=req.message,
            model_id=req.model_id,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# 7. GET /api/conversations/{id}/export
# ---------------------------------------------------------------------------
@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str, user: dict = Depends(get_current_user)):
    """Return the conversation in app.py's Export format (plain text)."""
    user_token = user["token"]
    user_id = user["payload"].get("sub")

    convs = await _supabase_rest_get(
        "conversations",
        user_token,
        params={"select": "id,title,created_at", "id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
    )
    if not isinstance(convs, list) or len(convs) == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = convs[0]

    messages = await _supabase_rest_get(
        "messages",
        user_token,
        params={"select": "role,content", "conversation_id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}", "order": "created_at.asc"},
    )
    if not isinstance(messages, list):
        messages = []

    conv_data = {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "messages": messages,
    }
    text = export_conversation_text(conv_data)
    return PlainTextResponse(
        text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="chat_history.txt"'},
    )


# ---------------------------------------------------------------------------
# 8. GET /api/stats
# ---------------------------------------------------------------------------
@app.get("/api/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    """Aggregate stats for the future 3D dashboard.

    Reads from Supabase (per-user conversations/messages).
    """
    user_token = user["token"]
    user_id = user["payload"].get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user token")

    # Fetch conversations count.
    convs = await _supabase_rest_get(
        "conversations",
        user_token,
        params={"select": "id", "user_id": f"eq.{user_id}"},
    )
    total_conversations = len(convs) if isinstance(convs, list) else 0

    # Fetch all messages for this user.
    messages = await _supabase_rest_get(
        "messages",
        user_token,
        params={"select": "role,model,token_count,response_ms,created_at", "user_id": f"eq.{user_id}"},
    )
    if not isinstance(messages, list):
        messages = []
    total_messages = len(messages)

    # Messages by role.
    messages_by_role: dict[str, int] = {}
    for m in messages:
        role = m.get("role", "unknown")
        messages_by_role[role] = messages_by_role.get(role, 0) + 1

    # Messages by model.
    messages_by_model: dict[str, int] = {}
    for m in messages:
        model = m.get("model") or "unknown"
        messages_by_model[model] = messages_by_model.get(model, 0) + 1

    # Token usage (completion tokens).
    token_rows = [m for m in messages if m.get("token_count") is not None]
    tokens_tracked = len(token_rows) > 0
    tokens = {
        "tracked": tokens_tracked,
        "total_completion_tokens": sum(m["token_count"] for m in token_rows) if tokens_tracked else None,
        "avg_completion_tokens": round(sum(m["token_count"] for m in token_rows) / len(token_rows), 1) if tokens_tracked else None,
        "by_model": {},
    }
    for m in token_rows:
        model = m.get("model") or "unknown"
        if model not in tokens["by_model"]:
            tokens["by_model"][model] = {"total_completion_tokens": 0, "avg_completion_tokens": 0, "messages": 0}
        entry = tokens["by_model"][model]
        entry["total_completion_tokens"] += m["token_count"]
        entry["messages"] += 1
    for model, entry in tokens["by_model"].items():
        entry["avg_completion_tokens"] = round(entry["total_completion_tokens"] / entry["messages"], 1)

    # Response time (ms).
    rt_rows = [m for m in messages if m.get("response_ms") is not None]
    rt_tracked = len(rt_rows) > 0
    avg_response_time_ms = {
        "tracked": rt_tracked,
        "overall": round(sum(m["response_ms"] for m in rt_rows) / len(rt_rows), 1) if rt_tracked else None,
        "by_model": {},
    }
    for m in rt_rows:
        model = m.get("model") or "unknown"
        if model not in avg_response_time_ms["by_model"]:
            avg_response_time_ms["by_model"][model] = {"avg_response_ms": 0, "messages": 0}
        entry = avg_response_time_ms["by_model"][model]
        entry["avg_response_ms"] = round((entry["avg_response_ms"] * entry["messages"] + m["response_ms"]) / (entry["messages"] + 1), 1)
        entry["messages"] += 1

    # Messages per day (last 30 days).
    today = datetime.utcnow().date()
    per_day: dict[str, dict] = {
        (today - timedelta(days=i)).isoformat(): {
            "date": (today - timedelta(days=i)).isoformat(),
            "messages": 0,
        }
        for i in range(29, -1, -1)
    }
    for m in messages:
        day = (m.get("created_at") or "")[:10]
        if day in per_day:
            per_day[day]["messages"] += 1

    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "db_size_mb": None,
        "messages_by_role": messages_by_role,
        "messages_by_model": messages_by_model,
        "messages_per_day": list(per_day.values()),
        "tokens": tokens,
        "avg_response_time_ms": avg_response_time_ms,
    }


# ---------------------------------------------------------------------------
# 9. GET /api/trending
# ---------------------------------------------------------------------------

_trending_cache: dict = {"data": None, "fetched_at": 0}
_TRENDING_CACHE_TTL = 600  # 10 minutes

@app.get("/api/trending")
async def get_trending(limit: int = 6, user: dict = Depends(get_current_user)):
    """Return today's trending searches in India, from Google Trends RSS."""
    now = time.time()
    if _trending_cache["data"] and (now - _trending_cache["fetched_at"] < _TRENDING_CACHE_TTL):
        return {"topics": _trending_cache["data"][:limit]}

    url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=IN"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_bytes = resp.read()

        root = ET.fromstring(xml_bytes)
        ns = {"ht": "https://trends.google.com/trends/trendingsearches/daily"}

        topics = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            traffic_el = item.find("ht:approx_traffic", ns)
            if title_el is None or not title_el.text:
                continue
            traffic_text = traffic_el.text if traffic_el is not None else None
            topics.append({"label": title_el.text.strip(), "traffic": traffic_text})

        _trending_cache["data"] = topics
        _trending_cache["fetched_at"] = now
        return {"topics": topics[:limit]}

    except Exception:
        # Fall back to stale cache if available, else empty list
        if _trending_cache["data"]:
            return {"topics": _trending_cache["data"][:limit]}
        return {"topics": []}


# ---------------------------------------------------------------------------
# Health / trust probe
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "models": [m["id"] for m in list_model_catalogue()]}


@app.get("/api/trust")
def trust_probe():
    """Expose the trust_resolver decision for the current working directory."""
    cwd = str(Path.cwd())
    events = get_trust_events(cwd)
    trusted = check_trust(cwd)
    return {"cwd": cwd, "trusted": trusted, "events": events}
