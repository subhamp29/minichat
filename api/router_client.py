"""Remote router client for MiniChat.

Streams OpenAI-compatible chat completions from a user-provided router endpoint.
All secrets come from environment variables; nothing is hardcoded in source.
"""

from __future__ import annotations

import json
import os
from typing import Generator, Optional

import requests


# ---------------------------------------------------------------------------
# Configuration (environment variables — never hardcode the key)
# ---------------------------------------------------------------------------
ROUTER_BASE_URL: str = os.environ.get("ROUTER_BASE_URL", "").rstrip("/")
ROUTER_API_KEY: str = os.environ.get("ROUTER_API_KEY", "")
ROUTER_MODELS_CSV: str = os.environ.get("ROUTER_MODELS", "")
ROUTER_TIMEOUT: int = int(os.environ.get("ROUTER_TIMEOUT", "60"))


def _get_available_models() -> list[str]:
    """Return router model slugs from the ROUTER_MODELS env var."""
    if ROUTER_MODELS_CSV:
        return [m.strip() for m in ROUTER_MODELS_CSV.split(",") if m.strip()]
    return []


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RouterConfigurationError(Exception):
    """Raised when the router is not configured (missing env vars)."""


class RouterError(Exception):
    """Raised when the router returns an error or is unreachable.

    The message is safe to display to users; it never contains the API key.
    """


# ---------------------------------------------------------------------------
# Streaming client
# ---------------------------------------------------------------------------
def stream_chat(
    messages: list[dict[str, str]],
    model_slug: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    timeout: int = ROUTER_TIMEOUT,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> Generator[str, None, None]:
    """Stream text chunks from the remote router.

    Args:
        messages: OpenAI-style messages list (system, user, assistant turns).
        model_slug: The router model identifier.
        temperature, top_p, max_tokens: Generation parameters.
        timeout: Request timeout in seconds.
        image_bytes: Optional raw image bytes to send as vision input.
        image_mime_type: MIME type of the image (e.g. ``image/jpeg``).

    Yields:
        Text chunks as they arrive from the server.

    Raises:
        RouterConfigurationError: If required env vars are missing.
        RouterError: On network errors, non-2xx responses, or malformed JSON.
    """
    if not ROUTER_BASE_URL:
        raise RouterConfigurationError(
            "Remote router is not configured. Set the ROUTER_BASE_URL environment variable."
        )
    if not ROUTER_API_KEY:
        raise RouterConfigurationError(
            "Remote router API key is missing. Set the ROUTER_API_KEY environment variable."
        )

    url = f"{ROUTER_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ROUTER_API_KEY}",
    }

    # --- Vision support: convert last user message to OpenAI vision format ---
    if image_bytes is not None and image_mime_type is not None:
        import base64 as _b64

        image_b64 = _b64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{image_mime_type};base64,{image_b64}"

        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = msg.get("content", "")
                msg["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
                break

    payload = {
        "model": model_slug,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    # --- HTTP request (errors raised here, before streaming starts) ---
    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
    except requests.ConnectionError as exc:
        raise RouterError(
            "Cannot reach the remote router. Check your network connection and the ROUTER_BASE_URL."
        ) from exc
    except requests.Timeout as exc:
        raise RouterError(
            f"Remote router timed out after {timeout}s. The model may be slow or the router is overloaded."
        ) from exc
    except requests.RequestException as exc:
        raise RouterError(f"Network error while contacting the router: {exc}") from exc

    if resp.status_code != 200:
        # Extract a short, user-friendly error message without leaking the API key.
        try:
            body = resp.json()
            detail = (
                body.get("error", {}).get("message")
                or body.get("message")
                or str(body)
            )
        except Exception:
            detail = resp.text[:200] if resp.text else "(empty body)"
        raise RouterError(
            f"Router returned HTTP {resp.status_code}: {detail}"
        )

    # --- SSE chunk parser ---
    content_type = resp.headers.get("Content-Type", "")
    is_sse = "text/event-stream" in content_type or "text/plain" in content_type

    if not is_sse:
        # Some routers return JSON directly even when stream=True.
        # Try to parse the whole body as a single completion.
        try:
            body = resp.json()
            text = body["choices"][0]["message"]["content"]
            if text:
                yield text
            return
        except Exception:
            pass  # fall through to line-by-line parsing

    buffer = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        line = str(line).strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        text = delta.get("content")
        if text:
            yield text


# ---------------------------------------------------------------------------
# Health-check helper (non-streaming)
# ---------------------------------------------------------------------------
def health_check(timeout: int = 10) -> dict:
    """Ping the router to verify connectivity and authentication.

    Returns:
        dict with keys: ok (bool), message (str), models (list[str]).
    """
    if not ROUTER_BASE_URL or not ROUTER_API_KEY:
        return {
            "ok": False,
            "message": "ROUTER_BASE_URL or ROUTER_API_KEY is not set.",
            "models": [],
        }

    # Try /models endpoint first
    url = f"{ROUTER_BASE_URL}/models"
    headers = {"Authorization": f"Bearer {ROUTER_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            return {"ok": True, "message": "Router reachable.", "models": models}
    except Exception:
        pass

    # Fallback: minimal chat completion
    url = f"{ROUTER_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ROUTER_API_KEY}",
    }
    slug = _get_available_models()[0] if _get_available_models() else "unknown"
    payload = {
        "model": slug,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return {
                "ok": True,
                "message": "Router reachable (chat completions).",
                "models": _get_available_models(),
            }
        return {"ok": False, "message": f"HTTP {resp.status_code}", "models": []}
    except requests.ConnectionError:
        return {"ok": False, "message": "Connection refused.", "models": []}
    except requests.Timeout:
        return {"ok": False, "message": "Connection timed out.", "models": []}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "models": []}
