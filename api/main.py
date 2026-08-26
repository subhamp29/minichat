"""Bhavyam AI — FastAPI backend.

Exposes the chat/memory/export/stats logic of the existing Streamlit app
(app.py) as a standalone HTTP API so a Next.js frontend can consume it.
All Streamlit-coupled code stays in app.py (untouched); the shared,
non-Streamlit logic lives in api/chat_core.py, which reuses
trust_resolver/ directly.

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
import json
import base64
import time
import uuid
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from PIL import Image
import io

import jwt
from jwt.algorithms import ECAlgorithm
import httpx
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

try:
    from docx import Document
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

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
    N8nOrchestrationError,
    _N8N_CLIENT_AVAILABLE,
    send_to_n8n_webhook,
)


app = FastAPI(
    title="Bhavyam AI API",
    description="Standalone FastAPI backend extracted from the MiniChat Streamlit app.",
    version="0.1.0",
)

# CORS — allow all origins during local dev. Lock this down before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Image upload configuration
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Maximum extracted document text sent to the model (safety ceiling).
MAX_DOCUMENT_TEXT_CHARS = 120_000


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
    conversation_id: str = ""
    message: str
    model_id: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512


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
async def _read_image_upload(file: UploadFile) -> tuple[bytes, str]:
    """Read an uploaded image into memory.

    The image is never written to disk or database.
    The returned bytes exist only for the current request.
    """

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Could not determine image type.",
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Please upload JPG, PNG, or WebP.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    return image_bytes, file.content_type


def _normalize_image(file_bytes: bytes) -> tuple[bytes, str]:
    """Normalize uploaded images for faster Gemini Vision processing.

    - Converts any supported image to RGB JPEG.
    - Resizes images so the longest side is at most 2048px.
    - Uses JPEG quality 82 to reduce upload/request size.
    - Keeps smaller images at their original dimensions.
    """
    MAX_DIMENSION = 2048
    JPEG_QUALITY = 82

    try:
        image = Image.open(io.BytesIO(file_bytes))

        # Fully decode/validate the image before processing.
        image.load()

        # Convert to RGB so PNG/WebP/transparency/other modes
        # are safely converted to a Gemini-compatible JPEG.
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize only when necessary.
        width, height = image.size
        longest_side = max(width, height)

        if longest_side > MAX_DIMENSION:
            scale = MAX_DIMENSION / longest_side
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

        # Encode as optimized JPEG.
        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )

        normalized_bytes = output.getvalue()

        print(
            f"[VISION] Image normalized: "
            f"{width}x{height} -> {image.width}x{image.height}, "
            f"{len(file_bytes)} -> {len(normalized_bytes)} bytes, "
            f"MIME=image/jpeg",
            flush=True,
        )

        return normalized_bytes, "image/jpeg"

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode uploaded image: {exc}",
        ) from exc


def _optimize_document_text(text: str) -> str:
    """Reduce unnecessary document text overhead before sending to the model."""
    import re

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    if len(text) > MAX_DOCUMENT_TEXT_CHARS:
        print(
            f"[DOCUMENT] Text truncated: "
            f"{len(text)} → {MAX_DOCUMENT_TEXT_CHARS} characters",
            flush=True,
        )

        text = (
            text[:MAX_DOCUMENT_TEXT_CHARS]
            + "\n\n[Document text truncated for processing.]"
        )

    return text


def _stream_chat_events(
    conversation_id: str,
    user_message: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    user_token: str | None = None,
    user_id: str | None = None,
    file_text: str | None = None,
    file_name: str | None = None,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
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

    if file_text:
        prefix = (
            f"[Attached file: {file_name or 'uploaded_file'}]\n"
            "The following is the extracted text from the uploaded file:\n"
            "```\n"
            f"{file_text}\n"
            "```\n\n"
        )

        augmented_user_msg = prefix + user_message

        # Explicitly discard the temporary extracted text after
        # constructing the model prompt.
        file_text = None
        prefix = None

    history_for_prompt = trim_history(history, SYSTEM_PROMPT, augmented_user_msg, template)

    # ----- n8n orchestrated -----
    if backend == "n8n_orchestrated":
        if image_bytes is not None:
            yield {"error": "Image input is not supported through n8n orchestration."}
            return
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

            if image_bytes is not None:
                print(
                    f"[VISION] Sending image: "
                    f"{len(image_bytes)} bytes, "
                    f"MIME={image_mime_type}",
                    flush=True,
                )

            full_text, usage = get_chat_response(
                api_messages,
                temperature=temperature,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
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
        if image_bytes is not None:
            yield {
                "error": "⚠️ Image input is not supported by local models. "
                         "Please use the Groq + Gemini Fallback model for image analysis."
            }
            return
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
        response_ms = int((time.time() - start_time) * 1000)
        yield {
            "done": True,
            "conversation_id": conversation_id,
            "model_id": model_id,
            "response_ms": response_ms,
            "token_count": state.get("token_count"),
        }
    elif not emitted_error:
        yield {"error": state.get("error_message", "Unknown error")}


@app.post("/api/chat")
async def chat(
    conversation_id: str = Form(default=""),
    message: str = Form(...),
    model_id: str = Form(...),
    temperature: float = Form(default=0.7),
    top_p: float = Form(default=0.95),
    max_tokens: int = Form(default=512),
    file: UploadFile | None = File(default=None),
    user: dict = Depends(get_current_user),
):
    """Stream the assistant's reply.

    Uploaded files are processed only for this request and are never
    persisted to Supabase, SQLite, or disk.
    """

    conversation_id = conversation_id.strip() or str(uuid.uuid4())

    file_text: str | None = None
    file_name: str | None = None
    image_bytes: bytes | None = None
    image_mime_type: str | None = None

    if file is not None:
        file_name = file.filename or "uploaded_file"
        content_type = file.content_type or ""
        lower_name = file_name.lower()

        try:
            # --- IMAGE ---
            if content_type.startswith("image/"):
                image_bytes, image_mime_type = await _read_image_upload(file)

                image_bytes, image_mime_type = _normalize_image(
                    image_bytes
                )

            # --- PDF ---
            elif (
                content_type == "application/pdf"
                or lower_name.endswith(".pdf")
            ):
                file_bytes = await file.read()

                if not file_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded file is empty.",
                    )

                if len(file_bytes) > 20 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail="PDF is too large. Maximum size is 20 MB.",
                    )

                file_text = _extract_pdf_text(file_bytes)

                if not file_text:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not extract readable text from this PDF.",
                    )

                file_text = _optimize_document_text(file_text)

            # --- DOCX ---
            elif (
                content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                or lower_name.endswith(".docx")
            ):
                file_bytes = await file.read()

                if not file_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded file is empty.",
                    )

                file_text = _extract_docx_text(file_bytes)

                if not file_text:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not extract readable text from this DOCX.",
                    )

                file_text = _optimize_document_text(file_text)

            # --- TXT ---
            elif (
                content_type == "text/plain"
                or lower_name.endswith(".txt")
            ):
                file_bytes = await file.read()

                if not file_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded file is empty.",
                    )

                try:
                    file_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    file_text = file_bytes.decode("utf-8", errors="replace")

                file_text = file_text.strip()

                if not file_text:
                    raise HTTPException(
                        status_code=400,
                        detail="The text file is empty.",
                    )

                file_text = _optimize_document_text(file_text)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unsupported file type. "
                        "Supported types: images (JPG, PNG, WebP), PDF, DOCX, TXT."
                    ),
                )

        finally:
            # Explicitly release the raw file buffer (not image_bytes or
            # file_text — those are captured by event_generator below and
            # must remain alive until the streaming generator runs).
            file_bytes = None

            try:
                await file.close()
            except Exception:
                pass

    def event_generator():
        user_token = user.get("token")
        user_id = user.get("payload", {}).get("sub")
        for event in _stream_chat_events(
            conversation_id=conversation_id,
            user_message=message,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            file_text=file_text,
            file_name=file_name,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            user_token=user_token,
            user_id=user_id,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # Do not retain file data after constructing the generator.
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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

def _get_db_size_mb() -> float | None:
    """Return chat_history.db size in MB, or None if missing."""
    candidates = [
        Path("chat_history.db"),
        Path(__file__).resolve().parent.parent / "chat_history.db",
    ]
    for path in candidates:
        if path.exists():
            try:
                return round(path.stat().st_size / (1024 * 1024), 2)
            except OSError:
                return None
    return None


def _extract_pdf_text(file_bytes: bytes) -> str | None:
    """Extract and compact text from a PDF in memory."""
    if not _PYPDF_AVAILABLE:
        return None

    try:
        import io
        import re

        reader = PdfReader(io.BytesIO(file_bytes))

        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""

            # Normalize whitespace.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            if text:
                pages.append(text)

        extracted = "\n\n".join(pages).strip()

        if not extracted:
            return None

        print(
            f"[DOCUMENT] PDF extracted: "
            f"{len(reader.pages)} pages, "
            f"{len(file_bytes)} bytes → "
            f"{len(extracted)} characters",
            flush=True,
        )

        return extracted

    except Exception as exc:
        print(
            f"[DOCUMENT] PDF extraction failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _extract_docx_text(file_bytes: bytes) -> str | None:
    """Extract and compact text from a DOCX in memory."""
    if not _DOCX_AVAILABLE:
        return None

    try:
        import io
        import re

        document = Document(io.BytesIO(file_bytes))

        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                # Collapse unnecessary whitespace.
                text = re.sub(r"[ \t]+", " ", text)
                paragraphs.append(text)

        extracted = "\n\n".join(paragraphs).strip()

        if not extracted:
            return None

        print(
            f"[DOCUMENT] DOCX extracted: "
            f"{len(paragraphs)} paragraphs, "
            f"{len(file_bytes)} bytes → "
            f"{len(extracted)} characters",
            flush=True,
        )

        return extracted

    except Exception as exc:
        print(
            f"[DOCUMENT] DOCX extraction failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


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
        params={"select": "role,model,backend,created_at,response_ms,token_count", "user_id": f"eq.{user_id}"},
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
        "db_size_mb": _get_db_size_mb(),
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

# ---------------------------------------------------------------------------

def _is_english(text: str) -> bool:
    """Return True if *text* contains only ASCII/Latin characters."""
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


@app.get("/api/trending")
async def get_trending(limit: int = 6):
    """Return today's trending searches in India, from Google Trends RSS."""
    now = time.time()

    # Return cached data if still fresh (10-minute TTL).
    if _trending_cache["data"] and (now - _trending_cache["fetched_at"] < _TRENDING_CACHE_TTL):
        return {"topics": _trending_cache["data"][:limit]}

    url = "https://trends.google.com/trending/rss?geo=IN&gl=IN&hl=en"

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            resp.raise_for_status()
            xml_bytes = resp.content

            # Diagnostic: ensure we got XML, not an HTML consent page or redirect.
            content_type = resp.headers.get("content-type", "")
            if "xml" not in content_type.lower():
                snippet = xml_bytes[:200].decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Google Trends returned non-XML content-type="
                    f"{content_type!r}. Body snippet: {snippet!r}"
                )

        root = ET.fromstring(xml_bytes)

        topics = []
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()

            traffic = ""

            for child in item:
                if child.tag.endswith("approx_traffic"):
                    traffic = (child.text or "").strip()
                    break

            if title:
                # Filter to English-language trends only.
                if not _is_english(title):
                    continue
                topics.append({
                    "label": title,
                    "traffic": traffic,
                })

        # Diagnostic: show how many were filtered.
        all_items = root.findall(".//item")
        filtered = len(all_items) - len(topics)
        if filtered:
            print(
                f"[TRENDING] Filtered {filtered} non-English topics "
                f"({len(all_items)} -> {len(topics)} English)",
                flush=True,
            )

        _trending_cache["data"] = topics
        _trending_cache["fetched_at"] = now

        print(
            f"[TRENDING] Fetched {len(topics)} topics from Google Trends "
            f"({len(xml_bytes)} bytes)",
            flush=True,
        )

        return {"topics": topics[:limit]}

    except httpx.TimeoutException as exc:
        print(f"[TRENDING] Google feed timeout after 10s: {exc}", flush=True)
    except httpx.HTTPStatusError as exc:
        print(
            f"[TRENDING] Google feed HTTP {exc.response.status_code}: "
            f"{exc.response.text[:200]}",
            flush=True,
        )
    except ET.ParseError as exc:
        print(f"[TRENDING] Google feed XML parse error: {exc}", flush=True)
    except httpx.RequestError as exc:
        print(
            f"[TRENDING] Google feed request error: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[TRENDING] Unexpected error: {type(exc).__name__}: {exc}",
            flush=True,
        )

    # Fall back to stale cache if available, else empty list.
    if _trending_cache["data"]:
        print(
            f"[TRENDING] Returning stale cache "
            f"({len(_trending_cache['data'])} topics, "
            f"age={int(now - _trending_cache['fetched_at'])}s)",
            flush=True,
        )
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
# test reload
