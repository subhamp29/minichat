"""Shared backend core for the Bhavyam AI FastAPI service.

This module is the non-Streamlit heart of Bhavyam AI, extracted so the new
FastAPI service (api/main.py) and the existing Streamlit app (app.py) can
share identical behaviour.

IMPORTANT (transition note):
  * `app.py` MUST stay untouched and runnable during the Next.js migration.
    Because app.py executes Streamlit at import time, it cannot be imported
    here, so the cleaning / DB / model logic below is mirrored verbatim from
    app.py. Once the Streamlit app is retired, app.py should import these
    helpers from this module instead of keeping its own copy.
  * `trust_resolver/` is imported and reused directly (no duplication) —
    it is a standalone, importable module.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import uuid
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — make the sibling MiniChat modules importable
# (trust_resolver/) without touching app.py.
# ---------------------------------------------------------------------------
MINICHAT_ROOT = Path(__file__).resolve().parent.parent
if str(MINICHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(MINICHAT_ROOT))

# Load the same .env the Streamlit app uses (GROQ_*, GEMINI_*, ...).
load_dotenv(MINICHAT_ROOT / ".env")

# Reuse the existing standalone modules directly (no duplication).
from trust_resolver import TrustConfig, TrustResolver  # noqa: E402

try:
    from n8n_supabase_client import (
        N8nOrchestrationError,
        send_to_n8n_webhook,
    )
    _N8N_CLIENT_AVAILABLE = True
except ImportError:
    class N8nOrchestrationError(Exception):  # pragma: no cover - fallback
        """Fallback when n8n_supabase_client is not installed."""
    _N8N_CLIENT_AVAILABLE = False
    send_to_n8n_webhook = None


# ---------------------------------------------------------------------------
# Configuration (mirrored from app.py)
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "Groq + Gemini Fallback (Cloud)": {
        "backend": "groq_gemini",
        "n_ctx": 8192,
        "template": "phi3",
        "description": "Primary: Groq compound | Fallback: Google Gemini 2.5 Flash-Lite",
    },
    "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast": {
        "backend": "local_gguf",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "n_ctx": 4096,
        "template": "chatml",
        "description": "⚡ Ultra fast (~400MB RAM). Works smoothly on any device without memory errors.",
    },
}

DEFAULT_MODEL_KEY = os.environ.get("DEFAULT_MODEL_KEY", "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast")
if DEFAULT_MODEL_KEY not in MODEL_OPTIONS:
    DEFAULT_MODEL_KEY = "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast"

# Share the EXACT same on-disk database as the Streamlit app so history is unified.
DB_PATH = MINICHAT_ROOT / "chat_history.db"

CACHE_DIR = Path(os.environ.get("HF_HOME", str(MINICHAT_ROOT / ".model_cache")))

N_THREADS = os.cpu_count() or 4
N_GPU_LAYERS = 0
N_BATCH = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are Bhavyam AI, an AI assistant created and owned by Subham Mahapatra, based in Odisha, India. "
    "When asked who created you, who owns you, or what your name is, answer in first person "
    "(for example: 'I was created by Subham Mahapatra' or 'My name is Bhavyam AI'), not second person. "
    "Otherwise, be a helpful, respectful, and honest assistant."
    "\n\nCRITICAL RULES:"
    "\n1. NEVER refuse to answer a question. Always provide a helpful response."
    "\n2. NEVER mention working directory, file paths, OS, Python version, CPU cores, model files, GGUF, context window size, or any system metadata."
    "\n3. NEVER output <environment_details>, <system>, <|im_start|>, <|im_end|>, or any similar tags."
    "\n4. When web search results are provided inside <|web_search|> tags, you MUST use them to answer accurately."
    "\n5. If web search returns no results, answer from your own knowledge without mentioning the search failure."
    "\n6. MATH: You CAN and SHOULD answer all math questions. Do arithmetic, algebra, calculus, and statistics confidently using your training knowledge. When asked ANY math question, give ONLY the mathematical answer in plain text. Do NOT add programming examples, code snippets, implementation notes, or any other subject. If the user asks for integration, differentiation, equations, or any math concept, respond with ONLY the mathematical result and explanation."
    "\n7. Stay strictly on topic. Answer ONLY what the user asks. Do not add unrelated information, side notes, or extra commentary outside the direct answer."
    "\n8. NEVER mix answers from multiple sources into one response. Give one clear, focused answer."
    "\n9. You are intelligent and knowledgeable. Never claim ignorance for questions you were trained on."
)

CHAT_TEMPLATES = {
    "chatml": {
        "system": "<|im_start|>system\n{c}<|im_end|>\n",
        "user": "<|im_start|>user\n{c}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{c}<|im_end|>\n",
        "assistant_open": "<|im_start|>assistant\n",
        "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
    },
    "phi3": {
        "system": "<|system|>\n{c}<|end|>\n",
        "user": "<|user|>\n{c}<|end|>\n",
        "assistant": "<|assistant|>\n{c}<|end|>\n",
        "assistant_open": "<|assistant|>\n",
        "stop": ["<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"],
    },
    "llama3": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{c}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{c}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{c}<|eot_id|>",
        "assistant_open": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "stop": ["<|eot_id|>", "<|end_of_text|>"],
    },
}

STOP_SEQUENCES = ["<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]

# Allow up to 60s for image and uploaded-document analysis.
ANALYSIS_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Remote (Groq + Gemini) fallback — mirrored from app.py get_chat_response
# ---------------------------------------------------------------------------
def get_chat_response(
    messages: list,
    temperature: float = 0.7,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
):
    """Call Groq first, fall back to Gemini if Groq fails or rate-limits.

    When ``image_bytes`` is provided, Groq is skipped (it does not support
    vision) and the image is sent directly to Gemini as an ``inline_data``
    part alongside the last user message's text.

    Returns a tuple ``(text, usage)`` where ``usage`` is
    ``{"prompt_tokens": int|None, "completion_tokens": int|None}`` captured
    from the provider's response (used by /api/chat to record token usage).
    """
    import requests as _requests
    import base64 as _base64

    usage: dict = {"prompt_tokens": None, "completion_tokens": None}

    def call_groq():
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        # Groq compound is text-only; images are routed to Gemini.
        if image_bytes is not None:
            raise RuntimeError("Groq compound does not support image input.")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "groq/compound",
            "messages": messages,
            "temperature": temperature,
        }
        response = _requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        u = data.get("usage") or {}
        usage["prompt_tokens"] = u.get("prompt_tokens")
        usage["completion_tokens"] = u.get("completion_tokens")
        return data["choices"][0]["message"]["content"]

    def call_gemini():
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-3.5-flash-lite:generateContent?key={api_key}"
        )
        contents = []
        system_instruction = None
        last_idx = len(messages) - 1
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text
                continue
            if role == "assistant":
                role = "model"
            parts = [{"text": text}]
            # Attach image to the last user message (if provided).
            if (
                image_bytes is not None
                and image_mime_type is not None
                and role == "user"
                and i == last_idx
            ):
                image_b64 = _base64.b64encode(image_bytes).decode("utf-8")
                parts.insert(0, {
                    "inline_data": {
                        "mime_type": image_mime_type,
                        "data": image_b64,
                    }
                })
            contents.append({"role": role, "parts": parts})
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        response = _requests.post(url, json=payload, timeout=ANALYSIS_TIMEOUT)
        if not response.ok:
            raise RuntimeError(f"Gemini {response.status_code}: {response.text[:300]}")
        data = response.json()
        um = data.get("usageMetadata") or {}
        usage["prompt_tokens"] = um.get("promptTokenCount")
        usage["completion_tokens"] = um.get("candidatesTokenCount")
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # Vision request: skip Groq (text-only), go straight to Gemini.
    if image_bytes is not None:
        try:
            return call_gemini(), dict(usage)
        except Exception as e:  # noqa: BLE001
            error_message = f"{type(e).__name__}: {e}"
            print(f"[CHAT PROVIDER ERROR] Gemini (vision): {error_message}", flush=True)
            raise RuntimeError(
                f"All providers failed. Details: Gemini: {error_message}"
            )

    providers = [("Groq", call_groq), ("Gemini", call_gemini)]
    errors: list[tuple[str, str]] = []
    for name, fn in providers:
        try:
            return fn(), dict(usage)
        except Exception as exc:  # noqa: BLE001
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"[CHAT PROVIDER ERROR] {name}: {error_message}", flush=True)
            errors.append((name, error_message))
            continue
    raise RuntimeError(
        "All providers failed. Details: "
        + " | ".join(f"{name}: {error}" for name, error in errors)
    )


# ---------------------------------------------------------------------------
# Chat Memory / SQLite helpers (mirrored from app.py)
# ---------------------------------------------------------------------------
def init_db():
    """Create conversations table if it does not exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                messages TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"Database initialization issue: {e}")


# ---------------------------------------------------------------------------
# Normalized `messages` table + migration (powers /api/stats per-model,
# token and latency tracking).
#
# Messages are ALSO still stored as a JSON blob on each conversation (so the
# untouched Streamlit app.py keeps working). We additionally keep a
# normalized `messages` table so stats can aggregate per message cheaply
# with plain SQL. Existing JSON-blob messages are backfilled with
# model_id = NULL (the "unknown" bucket). Idempotent on repeated runs and
# on fresh installs.
# ---------------------------------------------------------------------------
MESSAGES_TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        model_id TEXT,
        created_at TEXT NOT NULL,
        token_count INTEGER,
        response_ms INTEGER
    )
"""

# Deterministic namespace so backfilled message ids are stable across runs
# (lets INSERT OR IGNORE make the backfill idempotent).
_MESSAGES_NS = uuid.uuid5(uuid.NAMESPACE_URL, "bhavyam.ai/messages")


def migrate_messages_table():
    """Create the normalized messages table and backfill it from existing
    conversation JSON blobs.

    Safe to run repeatedly: the table is created with IF NOT EXISTS and
    backfilled rows use deterministic ids, so re-running is a no-op.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(MESSAGES_TABLE_SCHEMA)
        rows = conn.execute(
            "SELECT id, created_at, messages FROM conversations"
        ).fetchall()
        for conv_id, created_at, messages_json in rows:
            try:
                msgs = (
                    json.loads(messages_json)
                    if isinstance(messages_json, str)
                    else messages_json
                )
            except Exception:  # noqa: BLE001
                continue
            for idx, m in enumerate(msgs):
                mid = str(uuid.uuid5(_MESSAGES_NS, f"{conv_id}:{idx}"))
                conn.execute(
                    "INSERT OR IGNORE INTO messages "
                    "(id, conversation_id, role, content, model_id, created_at, token_count, response_ms) "
                    "VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL)",
                    (
                        mid,
                        conv_id,
                        m.get("role", "unknown"),
                        _clean_message_content(m.get("content", "")),
                        created_at,
                    ),
                )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"Message table migration issue: {e}")


def insert_message(
    conversation_id: str,
    role: str,
    content: str,
    model_id: Optional[str] = None,
    created_at: Optional[str] = None,
    token_count: Optional[int] = None,
    response_ms: Optional[int] = None,
) -> None:
    """Insert a single message row into the normalized messages table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO messages "
            "(id, conversation_id, role, content, model_id, created_at, token_count, response_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                conversation_id,
                role,
                _clean_message_content(content),
                model_id,
                created_at or datetime.utcnow().isoformat(),
                token_count,
                response_ms,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to insert message: {e}")


def list_conversations():
    """Return all conversations sorted by most recent first."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT id, title, created_at, messages FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [
            {
                "id": row[0],
                "title": _clean_message_content(row[1]),
                "created_at": row[2],
                "messages": [
                    {"role": m["role"], "content": _clean_message_content(m.get("content", ""))}
                    for m in json.loads(row[3])
                ],
            }
            for row in rows
        ]
    except Exception:  # noqa: BLE001
        try:
            DB_PATH.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        init_db()
        return []


def save_conversation(conversation_id, title, messages):
    """Upsert a conversation into SQLite."""
    try:
        cleaned_messages = [
            {"role": m["role"], "content": _clean_message_content(m.get("content", ""))}
            for m in messages
        ]
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO conversations (id, title, created_at, messages) VALUES (?, ?, ?, ?)",
            (
                conversation_id,
                title,
                datetime.utcnow().isoformat(),
                json.dumps(cleaned_messages, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to save conversation: {e}")


def delete_conversation(conversation_id):
    """Delete a conversation from SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to delete conversation: {e}")


def get_conversation(conversation_id):
    """Load a single conversation by ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT id, title, created_at, messages FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        conn.close()
        if row:
            messages = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            cleaned_messages = [
                {"role": m["role"], "content": _clean_message_content(m.get("content", ""))}
                for m in messages
            ]
            return {
                "id": row[0],
                "title": _clean_message_content(row[1]),
                "created_at": row[2],
                "messages": cleaned_messages,
            }
        return None
    except Exception:  # noqa: BLE001
        return None


def get_most_recent_conversation():
    """Return the most recently created conversation, or None."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT id, title, created_at, messages FROM conversations ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            messages = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            cleaned_messages = [
                {"role": m["role"], "content": _clean_message_content(m.get("content", ""))}
                for m in messages
            ]
            return {
                "id": row[0],
                "title": _clean_message_content(row[1]),
                "created_at": row[2],
                "messages": cleaned_messages,
            }
        return None
    except Exception:  # noqa: BLE001
        return None


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to max_length characters, appending '...' if truncated."""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def generate_conversation_title(messages: list) -> str:
    """Generate a title from the first user message in the conversation."""
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            cleaned = _clean_message_content(msg["content"].strip())
            return truncate_text(cleaned)
    return "New Chat"


# ---------------------------------------------------------------------------
# Message cleaning / sanitization (mirrored verbatim from app.py)
# ---------------------------------------------------------------------------
def _strip_environment_details(text: str) -> str:
    """Remove everything between <environment_details> and </environment_details>."""
    text = re.sub(
        r"(?is)<environment_details[^>]*>.*?</environment_details\s*>",
        "",
        text,
    )
    result = []
    i = 0
    text_lower = text.lower()
    while i < len(text):
        start = text_lower.find("<environment_details", i)
        if start == -1:
            result.append(text[i:])
            break
        end = text_lower.find("</environment_details>", start)
        if end == -1:
            break
        result.append(text[i:start])
        i = end + len("</environment_details>")
    return "".join(result)


def _clean_message_content(content: str) -> str:
    """Clean a single message's content for safe storage and display."""
    if not isinstance(content, str):
        return content
    text = _strip_environment_details(content)
    text = re.sub(
        r"(?i)^ERROR:\s*Cannot read\s*['\"][^'\"]+['\"].*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    lines = text.splitlines()
    keep = []
    for line in lines:
        lower = line.strip().lower()
        if any(k in lower for k in [
            "current time:",
            "working directory:",
            "workspace root folder:",
            "environment_details",
            "open tabs:",
            "cannot read",
        ]):
            continue
        keep.append(line)
    text = "\n".join(keep)
    lower_text = text.lower()
    if any(p in lower_text for p in [
        "<environment_details",
        "environment_details>",
        "current time:",
        "working directory:",
        "workspace root folder:",
    ]):
        return "I cannot provide that answer."
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def clean_response(text: str) -> str:
    """Remove leaked system metadata tags from model output."""
    return _clean_message_content(text)


def _clean_error(text: str) -> str:
    """Remove leaked metadata from error messages."""
    text = _strip_environment_details(text)
    text = re.sub(
        r"(?i)^ERROR:\s*Cannot read\s*['\"][^'\"]+['\"].*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        lower = line.strip().lower()
        if any(k in lower for k in [
            "current time:",
            "working directory:",
            "workspace root folder:",
            "environment_details",
            "open tabs:",
            "cannot read",
        ]):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _clean_streaming(text: str) -> str:
    """Lightweight cleanup for real-time streaming display."""
    text = _strip_environment_details(text)
    text = re.sub(
        r"(?i)^ERROR:\s*Cannot read\s*['\"][^'\"]+['\"].*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    lines = text.splitlines()
    keep = []
    for line in lines:
        lower = line.strip().lower()
        if any(k in lower for k in [
            "current time:",
            "working directory:",
            "workspace root folder:",
            "environment_details",
            "open tabs:",
            "cannot read",
        ]):
            continue
        keep.append(line)
    return "\n".join(keep)


# ---------------------------------------------------------------------------
# Prompt construction & context management (mirrored from app.py)
# ---------------------------------------------------------------------------
def build_prompt(system: str, history: list, user_msg: str, template_key: str) -> str:
    t = CHAT_TEMPLATES[template_key]
    clean_system = _clean_streaming(system)
    prompt = t["system"].format(c=clean_system)
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        if role in ("user", "assistant"):
            clean_content = _clean_streaming(content)
            prompt += t[role].format(c=clean_content)
    clean_user = _clean_streaming(user_msg)
    prompt += t["user"].format(c=clean_user) + t["assistant_open"]
    return prompt


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def trim_history(history: list, system: str, user_msg: str, template_key: str, max_history_tokens: int = 1024) -> list:
    """Keep last 10 exchanges and trim until prompt fits within max_history_tokens."""
    trimmed = history[:]
    trimmed = trimmed[-20:]
    while trimmed and estimate_tokens(build_prompt(system, trimmed, user_msg, template_key)) > max_history_tokens:
        trimmed.pop(0)
    return trimmed


# ---------------------------------------------------------------------------
# Local GGUF model engine (mirrored from app.py — Streamlit cache replaced
# by an in-process dict cache so it works under FastAPI/uvicorn).
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()


def ensure_model(model_key: str):
    """Download the model if it is not present locally. Returns the path to the GGUF file."""
    from huggingface_hub import hf_hub_download
    import shutil

    model_config = MODEL_OPTIONS[model_key]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=model_config["repo"],
        filename=model_config["file"],
        cache_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    model_path = Path(model_path)

    if model_path.exists() and model_path.stat().st_size == 0:
        blob_dir = model_path.parent.parent / "blobs"
        if blob_dir.exists():
            for blob_file in blob_dir.iterdir():
                if blob_file.is_file() and not blob_file.name.endswith(".incomplete"):
                    try:
                        shutil.copy2(blob_file, model_path)
                        break
                    except OSError:
                        continue

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    file_size = model_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Model file is empty (0 bytes) at {model_path}")
    if file_size < 100_000_000:
        raise ValueError(f"Model file seems too small ({file_size} bytes) at {model_path}")
    return str(model_path)


def load_model(model_key: str):
    """Load (and cache) a llama-cpp-python model instance for local_gguf models."""
    model_config = MODEL_OPTIONS.get(model_key, {})
    if model_config.get("backend") in ("n8n_orchestrated", "groq_gemini"):
        return None

    if model_key not in MODEL_OPTIONS:
        model_key = DEFAULT_MODEL_KEY
    model_config = MODEL_OPTIONS[model_key]

    with _MODEL_LOCK:
        if model_key in _MODEL_CACHE:
            return _MODEL_CACHE[model_key]

    model_path = ensure_model(model_key)

    fallback_ctxs = [model_config["n_ctx"], 2048, 1024, 512, 256]
    seen = set()
    ctx_candidates = [c for c in fallback_ctxs if not (c in seen or seen.add(c))]

    from llama_cpp import Llama

    last_error = None
    llm = None
    for n_ctx in ctx_candidates:
        for batch_size in [N_BATCH, 256, 128]:
            try:
                llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=N_THREADS,
                    n_gpu_layers=N_GPU_LAYERS,
                    n_batch=batch_size,
                    verbose=False,
                )
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                continue
        if llm is not None:
            break

    if llm is None:
        error_msg = f"Failed to load model '{model_key}' with any context size.\n"
        error_msg += f"Model path: {model_path}\n"
        error_msg += f"Last error: {last_error}"
        raise RuntimeError(error_msg)

    with _MODEL_LOCK:
        _MODEL_CACHE[model_key] = llm
    return llm


def generate(model, prompt, temperature, top_p, max_tokens, stop_sequences=None):
    """Yield tokens one by one from llama.cpp."""
    if stop_sequences is None:
        stop_sequences = STOP_SEQUENCES
    try:
        stream = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.15,
            stop=stop_sequences,
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            yield token
    except Exception as e:  # noqa: BLE001
        yield clean_response(f"\n\n*[Generation error: {e}]*")


# ---------------------------------------------------------------------------
# Trust Resolver Integration (mirrored from app.py)
# ---------------------------------------------------------------------------
try:
    _trust_config = TrustConfig()
    _trust_config.add_allowlisted(os.getcwd())
    _trust_resolver = TrustResolver(_trust_config)
    _TRUST_RESOLVER_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRUST_RESOLVER_AVAILABLE = False


def _policy_name(value):
    from trust_resolver import TrustPolicy
    return TrustPolicy.name(value) if value is not None else None


def _resolution_name(value):
    from trust_resolver import TrustResolution
    return TrustResolution.name(value) if value is not None else None


def check_trust(cwd: str, screen_text: str = "") -> bool:
    """Check if the current working directory is trusted."""
    if not _TRUST_RESOLVER_AVAILABLE:
        return True
    decision = _trust_resolver.resolve(cwd, None, screen_text)
    if decision.is_required:
        return decision.policy is not None and int(decision.policy) != 2  # Not DENY
    return True


def get_trust_events(cwd: str, screen_text: str = "") -> list:
    """Get trust events for a given path and screen text."""
    if not _TRUST_RESOLVER_AVAILABLE:
        return []
    decision = _trust_resolver.resolve(cwd, None, screen_text)
    return [
        {
            "type": event.event_type,
            "cwd": event.cwd,
            "policy": _policy_name(event.policy),
            "resolution": _resolution_name(event.resolution),
            "reason": event.reason,
        }
        for event in decision.events
    ]


# ---------------------------------------------------------------------------
# Model catalogue helper (for GET /api/models)
# ---------------------------------------------------------------------------
def list_model_catalogue() -> list:
    """Return MODEL_OPTIONS normalised to the API's {id, display_name, backend, description} shape."""
    catalogue = []
    for key, cfg in MODEL_OPTIONS.items():
        backend_raw = cfg.get("backend", "local_gguf")
        # API-facing backend is coarse: anything cloud/routed is "remote".
        api_backend = "local" if backend_raw == "local_gguf" else "remote"
        catalogue.append(
            {
                "id": key,
                "display_name": key,
                "backend": api_backend,
                "description": cfg.get("description", ""),
            }
        )
    return catalogue


# ---------------------------------------------------------------------------
# Conversation export (mirrored exactly from app.py's Export button)
# ---------------------------------------------------------------------------
def export_conversation_text(conversation: dict) -> str:
    """Return the plain-text export used by app.py's sidebar Export button.

    app.py joins messages as:
        f"{'User' if role=='user' else 'Assistant'}: {_clean_message_content(content)}"
    separated by "\\n\\n", then offers it as a `text/plain` download
    named `chat_history.txt`.
    """
    messages = conversation.get("messages", [])
    return "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: "
        f"{_clean_message_content(m.get('content', ''))}"
        for m in messages
    )


# Ensure the DB exists when this module is imported (mirrors app.py init_db()).
init_db()
# Create + backfill the normalized messages table for /api/stats.
migrate_messages_table()
