"""Supabase & n8n orchestration integration client for MiniChat.

Handles:
1. Direct API interaction with Supabase REST endpoints (conversations & message history).
2. Dispatching user requests to the n8n Webhook orchestration pipeline.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests

# Set up logger
logger = logging.getLogger("n8n_supabase_client")

# Environment defaults
DEFAULT_N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/chat")
DEFAULT_SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
DEFAULT_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


class N8nOrchestrationError(Exception):
    """Raised when n8n webhook fails or returns an error response."""
    pass


class SupabaseError(Exception):
    """Raised when Supabase REST API calls fail."""
    pass


def get_n8n_config() -> dict[str, str]:
    """Return configured URLs and keys from environment variables."""
    return {
        "webhook_url": os.environ.get("N8N_WEBHOOK_URL", DEFAULT_N8N_WEBHOOK_URL),
        "supabase_url": os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL).rstrip("/"),
        "supabase_key": os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY),
    }


def send_to_n8n_webhook(
    message: str,
    conversation_id: str,
    model_slug: str = "claude-opus-free",
    backend: str = "n8n_orchestrated",
    webhook_url: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Send user message to n8n Webhook for full orchestration.

    Args:
        message: The user's input text.
        conversation_id: Unique conversation session ID.
        model_slug: Model slug identifier for the n8n workflow.
        backend: Backend descriptor (default: 'n8n_orchestrated').
        webhook_url: Optional explicit webhook URL override.
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict containing keys: status, conversation_id, response, model.

    Raises:
        N8nOrchestrationError: If n8n returns an error or is unreachable.
    """
    url = webhook_url or os.environ.get("N8N_WEBHOOK_URL", DEFAULT_N8N_WEBHOOK_URL)
    if not url:
        raise N8nOrchestrationError("N8N_WEBHOOK_URL is not configured.")

    payload = {
        "conversation_id": conversation_id,
        "message": message,
        "model": model_slug,
        "backend": backend,
        "title": message[:30] if message else "New Chat",
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.ConnectionError as exc:
        raise N8nOrchestrationError(
            f"Cannot connect to n8n Webhook at {url}. Ensure n8n is running and active."
        ) from exc
    except requests.Timeout as exc:
        raise N8nOrchestrationError(
            f"n8n Webhook timed out after {timeout} seconds."
        ) from exc
    except requests.RequestException as exc:
        raise N8nOrchestrationError(f"HTTP request error calling n8n: {exc}") from exc

    if resp.status_code not in (200, 201):
        raise N8nOrchestrationError(
            f"n8n Webhook returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        return data
    except json.JSONDecodeError as exc:
        raise N8nOrchestrationError(
            f"Invalid JSON returned from n8n Webhook: {resp.text[:200]}"
        ) from exc


def fetch_supabase_conversations(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch active conversations directly from Supabase REST API.

    Returns list of conversation dicts sorted by updated_at descending.
    """
    url = (supabase_url or os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL)).rstrip("/")
    key = supabase_key or os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)

    if not url or not key:
        return []

    endpoint = f"{url}/rest/v1/conversations?select=*&order=updated_at.desc&limit={limit}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get(endpoint, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch conversations from Supabase: %s", exc)

    return []


def fetch_supabase_messages(
    conversation_id: str,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch chat history for a conversation directly from Supabase REST API.

    Returns list of message dicts ordered by created_at ascending.
    """
    url = (supabase_url or os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL)).rstrip("/")
    key = supabase_key or os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)

    if not url or not key or not conversation_id:
        return []

    endpoint = (
        f"{url}/rest/v1/messages"
        f"?conversation_id=eq.{conversation_id}"
        f"&order=created_at.asc"
        f"&limit={limit}"
    )
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get(endpoint, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch messages from Supabase: %s", exc)

    return []


def health_check_n8n(webhook_url: Optional[str] = None, timeout: int = 5) -> dict[str, Any]:
    """Ping n8n webhook service host to verify connectivity."""
    url = webhook_url or os.environ.get("N8N_WEBHOOK_URL", DEFAULT_N8N_WEBHOOK_URL)
    if not url:
        return {"ok": False, "message": "N8N_WEBHOOK_URL is not set."}

    try:
        # OPTIONS or GET check on base webhook endpoint
        base_host = url.split("/webhook")[0] if "/webhook" in url else url
        resp = requests.get(base_host, timeout=timeout)
        return {
            "ok": resp.status_code < 500,
            "message": f"n8n server reachable (HTTP {resp.status_code}).",
            "url": url,
        }
    except requests.ConnectionError:
        return {"ok": False, "message": f"Connection refused to {url}.", "url": url}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "url": url}
