import os
import re
import sqlite3
import uuid
import json
from datetime import datetime
from html import escape
from pathlib import Path

from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from router_client import (
    RouterConfigurationError,
    RouterError,
    _get_available_models,
    stream_chat,
)

# n8n Supabase client is optional — only needed for the n8n_orchestrated backend.
try:
    from n8n_supabase_client import (
        N8nOrchestrationError,
        send_to_n8n_webhook,
    )
    _N8N_CLIENT_AVAILABLE = True
except ImportError:
    class N8nOrchestrationError(Exception):
        """Fallback when n8n_supabase_client is not installed."""
    _N8N_CLIENT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast": {
        "backend": "local_gguf",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "n_ctx": 4096,
        "template": "chatml",
        "description": "⚡ Ultra fast (~400MB RAM). Works smoothly on any device without memory errors.",
    },
    "Qwen2.5 1.5B (Q4_K_M) - Fast & Smart": {
        "backend": "local_gguf",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "n_ctx": 4096,
        "template": "chatml",
        "description": "🚀 Excellent balance of intelligence and low RAM usage (~1GB RAM).",
    },
    "Llama-3.2 1B Instruct (Q4_K_M)": {
        "backend": "local_gguf",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "n_ctx": 4096,
        "template": "llama3",
        "description": "🦙 Meta's compact 1B model. High quality instruction following (~850MB RAM).",
    },
    "Phi-3 Mini 4K (Q4)": {
        "backend": "local_gguf",
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "file": "Phi-3-mini-4k-instruct-q4.gguf",
        "n_ctx": 4096,
        "template": "phi3",
        "description": "Fast 3.8B model. Requires ~3.5GB+ RAM.",
    },
    "Llama-3.2-3B-Instruct (Q4)": {
        "backend": "local_gguf",
        "repo": "QuantFactory/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct.Q4_K_M.gguf",
        "n_ctx": 4096,
        "template": "llama3",
        "description": "Meta's compact 3B model. Great for instruction following (~3GB RAM).",
    },
    "Qwen2.5-3B-Instruct (Q4)": {
        "backend": "local_gguf",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file": "qwen2.5-3b-instruct-q4.gguf",
        "n_ctx": 4096,
        "template": "chatml",
        "description": "Alibaba's versatile 3B model. Strong multilingual support.",
    },
}

# ---------------------------------------------------------------------------
# Remote & n8n Orchestrated models
# ---------------------------------------------------------------------------
MODEL_OPTIONS["n8n Orchestrated (Supabase + Router)"] = {
    "backend": "n8n_orchestrated",
    "slug": "claude-opus-free",
    "n_ctx": 8192,
    "template": "chatml",
    "description": "🔄 Full n8n orchestration: Streamlit UI ➔ n8n Webhook ➔ Supabase DB ➔ Router",
}

MODEL_OPTIONS["Remote: claude-opus-free"] = {
    "backend": "remote",
    "slug": "claude-opus-free",
    "n_ctx": 8192,
    "template": "phi3",
    "description": "Remote combo model via router (claude-opus-free)",
}

_remote_slugs = _get_available_models()
for _slug in _remote_slugs:
    if _slug != "claude-opus-free":
        MODEL_OPTIONS[f"Remote: {_slug}"] = {
            "backend": "remote",
            "slug": _slug,
            "n_ctx": 8192,
            "template": "phi3",
            "description": f"Remote model via router ({_slug})",
        }

DEFAULT_MODEL_KEY = os.environ.get("DEFAULT_MODEL_KEY", "n8n Orchestrated (Supabase + Router)")

# Ensure the configured default model exists in MODEL_OPTIONS
if DEFAULT_MODEL_KEY not in MODEL_OPTIONS:
    DEFAULT_MODEL_KEY = "n8n Orchestrated (Supabase + Router)"

CACHE_DIR = Path(os.environ.get("HF_HOME", ".model_cache"))

N_THREADS = os.cpu_count() or 4
N_GPU_LAYERS = 0
N_BATCH = 128
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95
MAX_TOKENS = 512

# SQLite database for persistent chat memory
DB_PATH = Path("chat_history.db")

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


# ---------------------------------------------------------------------------
# Chat Memory / SQLite helpers
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
    except Exception as e:
        st.warning(f"Database initialization issue: {e}")


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
                "title": row[1],
                "created_at": row[2],
                "messages": json.loads(row[3]),
            }
            for row in rows
        ]
    except Exception:
        # If DB is corrupted, reset it gracefully
        try:
            DB_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        init_db()
        return []


def save_conversation(conversation_id, title, messages):
    """Upsert a conversation into SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO conversations (id, title, created_at, messages) VALUES (?, ?, ?, ?)",
            (
                conversation_id,
                title,
                datetime.utcnow().isoformat(),
                json.dumps(messages, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        st.warning(f"Failed to save conversation: {e}")


def delete_conversation(conversation_id):
    """Delete a conversation from SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        st.warning(f"Failed to delete conversation: {e}")


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
            return {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "messages": json.loads(row[3]),
            }
        return None
    except Exception:
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
            return {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "messages": json.loads(row[3]),
            }
        return None
    except Exception:
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
            return truncate_text(msg["content"].strip())
    return "New Chat"


def save_current_conversation():
    """Save the current session messages as a conversation."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return

    # Determine conversation ID
    conv_id = st.session_state.get("current_conversation_id")
    if not conv_id:
        conv_id = str(uuid.uuid4())
        st.session_state.current_conversation_id = conv_id

    title = generate_conversation_title(messages)
    save_conversation(conv_id, title, messages)


def load_conversation(conversation_id: str):
    """Load a conversation into the session state."""
    conv = get_conversation(conversation_id)
    if conv:
        st.session_state.current_conversation_id = conversation_id
        st.session_state.messages = conv["messages"]
        st.rerun()


def create_new_chat():
    """Save current chat and start a fresh empty conversation."""
    save_current_conversation()
    st.session_state.current_conversation_id = None
    st.session_state.messages = []
    st.rerun()


# Initialize DB and session state for conversation tracking
init_db()


def clean_response(text: str) -> str:
    """Remove leaked system metadata tags from model output."""
    # Step 1: Remove complete environment_details blocks (multiple patterns)
    text = re.sub(r"(?is)<\s*environment_details\b[^>]*>.*?<\s*/\s*environment_details\s*>", "", text)
    text = re.sub(r"(?is)<environment_details>.*?</environment_details>", "", text, flags=re.DOTALL)
    
    # Step 2: Line-by-line aggressive filtering
    lines = text.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        # Skip environment_details tags (opening, closing, self-closing)
        if stripped.startswith("<environment_details") or stripped.startswith("</environment_details"):
            continue
        if stripped == "<environment_details>" or stripped == "</environment_details>":
            continue
        # Skip lines that are clearly system metadata
        if any(keyword in lower for keyword in [
            "current time:",
            "working directory:",
            "workspace root folder:",
            "environment_details",
            "cwd:",
            "os:",
            "python version:",
            "cpu cores:",
        ]):
            continue
        filtered.append(line)
    text = "\n".join(filtered)
    
    # Step 3: Remove other system tags
    text = re.sub(r"(?is)<\s*system\b[^>]*>.*?<\s*/\s*system\s*>", "", text)
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_sep\|>", "", text)
    text = re.sub(r"(?is)<\s*/\s*im_\w+\s*>", "", text)
    text = re.sub(r"<\|im_[a-z_]+\|>", "", text)
    
    # Step 4: Remove any remaining known system special tokens/tags.
    # IMPORTANT: Do NOT use a blanket <[^>]+> regex here — it would also
    # strip legitimate text like math comparisons ("a < b") or code
    # containing "<" / ">" characters, producing wrong answers.
    # Only strip tokens that start with "<|" or match known system tag names.
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(
        r"</?\b(environment_details|system|start_header_id|end_header_id|eot_id|end_of_text)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    
    # Step 4b: Strip any verbatim leakage of the system prompt itself
    if SYSTEM_PROMPT.strip() and SYSTEM_PROMPT.strip() in text:
        text = text.replace(SYSTEM_PROMPT.strip(), "").strip()
    
    # Step 5: Remove any lines that still contain suspicious patterns
    lines = text.splitlines()
    final_lines = []
    for line in lines:
        lower = line.lower()
        if any(pattern in lower for pattern in [
            "environment_details",
            "current time:",
            "working directory:",
            "workspace root folder:",
            "cwd:",
            "<|im_",
        ]):
            continue
        final_lines.append(line)
    text = "\n".join(final_lines)
    
    # Step 6: Final validation - if any leaked metadata remains, use fallback
    lower_text = text.lower()
    if any(pattern in lower_text for pattern in [
        "<environment_details",
        "environment_details>",
        "current time:",
        "working directory:",
        "workspace root folder:",
        "<|im_",
    ]):
        return "I cannot provide that answer."
    
    # Step 6b: Normalize creator/owner pronoun consistency
    # Small models sometimes switch to second person or claim self-creation;
    # nudge them back to the intended first-person phrasing.
    name = "Bhavyam AI"
    creator = "Subham Mahapatra"
    origin = "Odisha, India"
    lower = lower_text
    # Wrong patterns: "your creator is...", "i am the creator", "i created myself"
    if (
        "your creator is" in lower
        or "i am the creator" in lower
        or "i created myself" in lower
        or ("i am " in lower and "ai" in lower and "created by" not in lower)
    ):
        text = f"I am {name}, created by {creator} from {origin}."
        return text
    # Also catch standalone wrong self-introductions
    if text.strip().lower().startswith("i am the creator"):
        text = f"I am {name}, created by {creator} from {origin}."
        return text
    
    # Step 7: Clean up whitespace
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


STOP_SEQUENCES = ["<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]

# Per-model prompt templates and stop sequences
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


# ---------------------------------------------------------------------------
# Trust Resolver Integration
# ---------------------------------------------------------------------------
try:
    from trust_resolver import TrustConfig, TrustResolver

    _trust_config = TrustConfig()
    _trust_config.add_allowlisted(os.getcwd())
    _trust_resolver = TrustResolver(_trust_config)
    _TRUST_RESOLVER_AVAILABLE = True
except ImportError:
    _TRUST_RESOLVER_AVAILABLE = False


def check_trust(cwd: str, screen_text: str = "") -> bool:
    """Check if the current working directory is trusted.
    
    Args:
        cwd: Current working directory to check
        screen_text: Optional screen text to analyze for trust cues
        
    Returns:
        True if trusted, False otherwise
    """
    if not _TRUST_RESOLVER_AVAILABLE:
        return True  # Allow if trust resolver not available
    
    decision = _trust_resolver.resolve(cwd, None, screen_text)
    if decision.is_required:
        return decision.policy is not None and int(decision.policy) != 2  # Not DENY
    return True


def get_trust_events(cwd: str, screen_text: str = "") -> list:
    """Get trust events for a given path and screen text.
    
    Args:
        cwd: Current working directory
        screen_text: Optional screen text to analyze
        
    Returns:
        List of trust event dictionaries
    """
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


def _policy_name(policy) -> Optional[str]:
    if policy is None:
        return None
    try:
        # Native Rust binding: PyTrustPolicy instance with .name property
        return policy.name
    except AttributeError:
        # Pure-Python fallback: integer value
        return TrustPolicy.name(int(policy))


def _resolution_name(resolution) -> Optional[str]:
    if resolution is None:
        return None
    try:
        # Native Rust binding: PyTrustResolution instance with .name property
        return resolution.name
    except AttributeError:
        # Pure-Python fallback: integer value
        return TrustResolution.name(int(resolution))


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------
def is_web_lookup_query(query: str) -> bool:
    """Decide whether Wikipedia web-search is appropriate for this query.

    Wikipedia search is great for real-world fact / entity / concept lookups
    ("who is X", "capital of France", "quantum entanglement"). For pure
    arithmetic, math reasoning, or coding tasks the search almost always
    returns an *unrelated* article (e.g. asking about "5 < 3" returns the
    album "My Aim Is True"), and that irrelevant summary gets force-injected
    into the prompt. The small local model is then told to "use" it, which
    reliably produces incorrect answers. For those cases we skip the search
    and let the model answer from its own knowledge.
    """
    if not query:
        return False
    q = query.lower().strip()
    if len(q) < 3:
        return False
    # Arithmetic / comparison expressions, e.g. "2 + 2", "5 < 3", "10 * 5"
    if re.search(r"\d+\s*[-+*/<>=!]=?\s*\d+", q):
        return False
    if re.search(r"\b\d+\s*(percent|%)\b", q):
        return False
    # Comparison / logic / true-or-false questions, e.g.
    # "Is 5 greater than 3?", "true or false", "which is bigger"
    if re.search(r"\b(greater|less|more|fewer|bigger|smaller|larger|smaller)\s+than\b", q):
        return False
    if re.search(r"\b(is|are)\s+(the\s+)?\d+\s+(greater|less|more|smaller|larger|bigger)\b", q):
        return False
    if "true or false" in q or "true/false" in q or re.search(r"\b(t/f|t or f)\b", q):
        return False
    # Explicit math / reasoning tasks
    if re.search(
        r"\b(solve\b|calculate|calculator|differentiat|integrat|derivative|integral|"
        r"probability|calculus|algebra|arithmetic|math problem|math question|"
        r"which (number|value) is (larger|smaller|greater|less)|compare)",
        q,
    ):
        return False
    # Coding implementation / debugging requests (not a factual term lookup)
    if re.search(
        r"\b(write\s+(a|some|me)\s+|implement|refactor|debug|stack trace|"
        r"fix\s+(the\s+)?(error|bug|code|issue)|syntax error|compile error|traceback)",
        q,
    ):
        return False
    return True


def web_search(query: str, max_results: int = 3) -> str:
    """Search Wikipedia for a summary of the topic — more reliable than DuckDuckGo IA for 'who is X' queries."""
    headers = {"User-Agent": "BhavyamAI/1.0 (https://github.com/subham-mahapatra/BhavyamAI)"}
    try:
        # Step 1: find the best matching page title
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            headers=headers,
            timeout=10,
        )
        search_data = search_resp.json()
        results = search_data.get("query", {}).get("search", [])
        if not results:
            return "NO_WEB_RESULTS_FOUND"

        page_title = results[0]["title"]

        # Step 2: get a plain-text summary of that page
        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(page_title)}",
            headers=headers,
            timeout=10,
        )
        if summary_resp.status_code != 200:
            return "NO_WEB_RESULTS_FOUND"

        summary_data = summary_resp.json()
        extract = summary_data.get("extract")
        if extract:
            return f"Answer: {extract}"
        return "NO_WEB_RESULTS_FOUND"
    except Exception as e:
        return f"SEARCH_ERROR: {e}"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bhavyam AI",
    page_icon="🤖",
    layout="wide",
)

def inject_custom_style():
    """Inject Bhavyam AI 3D Glassmorphic Neural OS theme."""
    style = """
    <!-- Fonts & Iconify CDN -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.cdnfonts.com/css/clash-grotesk" rel="stylesheet">
    <script src="https://code.iconify.design/iconify-icon/1.0.7/iconify-icon.min.js"></script>

    <style>
      /* ===== Design Tokens ===== */
      :root {
        --color-primary: #3B82F6;
        --color-secondary: #06B6D4;
        --color-accent: #A855F7;
        --color-magenta: #EC4899;
        --color-dark: #090D16;
        --color-black: #030712;
        --color-bg-glass: rgba(15, 23, 42, 0.65);
        --color-bg-glass-blue: rgba(59, 130, 246, 0.12);
        --color-border: rgba(168, 85, 247, 0.3);
        --color-border-hover: rgba(59, 130, 246, 0.6);
        --color-text: #FFFFFF;
        --color-text-muted: rgba(255, 255, 255, 0.65);
        --glow-shadow: 0 0 30px rgba(168, 85, 247, 0.35);
        --blur-amount: 35px;
        --radius-lg: 20px;
        --radius-md: 14px;
        --radius-sm: 10px;
      }

      * { box-sizing: border-box; }

      body, .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: var(--color-black) !important;
        color: var(--color-text) !important;
      }

      h1, h2, h3, h4, .app-title, .welcome-title, .font-display {
        font-family: 'Clash Grotesk', sans-serif !important;
      }

      code, pre, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
      }

      /* ===== Global Canvas & Ambient Glows ===== */
      [data-testid="stAppViewContainer"] {
        background: var(--color-black) !important;
        position: relative;
        overflow-x: hidden;
      }

      [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: -15%;
        left: -15%;
        width: 55%;
        height: 55%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.16) 0%, rgba(168, 85, 247, 0.1) 50%, transparent 70%);
        filter: blur(60px);
        pointer-events: none;
        z-index: 0;
        animation: floatDrift1 26s ease-in-out infinite alternate;
      }

      [data-testid="stAppViewContainer"]::after {
        content: '';
        position: fixed;
        bottom: -15%;
        right: -15%;
        width: 60%;
        height: 60%;
        background: radial-gradient(circle, rgba(236, 72, 153, 0.12) 0%, rgba(6, 182, 212, 0.08) 50%, transparent 70%);
        filter: blur(60px);
        pointer-events: none;
        z-index: 0;
        animation: floatDrift2 32s ease-in-out infinite alternate-reverse;
      }

      @keyframes floatDrift1 {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(8%, 12%) scale(1.2); }
      }
      @keyframes floatDrift2 {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(-10%, -8%) scale(1.25); }
      }

      /* ===== Header Bar ===== */
      .app-header {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(35px) !important;
        -webkit-backdrop-filter: blur(35px) !important;
        border-bottom: 1px solid rgba(59, 130, 246, 0.3) !important;
        padding: 16px 28px !important;
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
        border-radius: 0 0 20px 20px !important;
        margin-bottom: 20px !important;
      }

      .app-logo {
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        border-radius: 14px !important;
        background: linear-gradient(135deg, #3B82F6, #A855F7, #EC4899) !important;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.5) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 22px !important;
        animation: pulseGlow 4s infinite;
      }

      @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(168, 85, 247, 0.4); }
        50% { box-shadow: 0 0 35px rgba(236, 72, 153, 0.7); }
      }

      .app-title {
        background: linear-gradient(135deg, #FFFFFF 0%, #60A5FA 50%, #C084FC 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
      }

      .app-subtitle {
        color: #93C5FD !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        background: rgba(59, 130, 246, 0.15) !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        margin-left: auto !important;
      }

      /* ===== Status Indicator ===== */
      .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 10px #10B981;
        display: inline-block;
        margin-right: 6px;
        animation: statusPulse 2s infinite;
      }
      @keyframes statusPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
      }

      /* ===== Sidebar Styling ===== */
      .stSidebar, [data-testid="stSidebar"] {
        background: rgba(9, 13, 22, 0.85) !important;
        backdrop-filter: blur(35px) !important;
        -webkit-backdrop-filter: blur(35px) !important;
        border-right: 1px solid rgba(168, 85, 247, 0.25) !important;
        box-shadow: 10px 0 40px rgba(0, 0, 0, 0.8) !important;
      }

      .sidebar-section {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.6)) !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 18px !important;
        padding: 16px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
      }
      .sidebar-section:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.15) !important;
      }

      .sidebar-title {
        color: #C084FC !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin-bottom: 12px !important;
      }

      /* ===== Main Container & Chat Bubbles ===== */
      .main-content {
        max-width: 960px;
        margin: 0 auto;
        padding: 10px 16px 140px;
      }

      /* Chat Bubbles */
      [data-testid="stChatMessage"] {
        background: transparent !important;
        margin-bottom: 18px !important;
        border: none !important;
        animation: bubbleSlide 0.4s cubic-bezier(0.175, 0.885, 0.32, 1) both;
      }

      @keyframes bubbleSlide {
        from { opacity: 0; transform: translateY(16px) scale(0.96); }
        to { opacity: 1; transform: translateY(0) scale(1); }
      }

      /* User Message Bubble — pushed to the RIGHT */
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.28), rgba(236, 72, 153, 0.32)) !important;
        border: 3px solid rgba(236, 72, 153, 0.7) !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 10px 30px rgba(168, 85, 247, 0.2), 0 0 20px rgba(236, 72, 153, 0.15) !important;
        color: #FFFFFF !important;
        padding: 16px 22px !important;
        margin-left: auto !important;
        display: block !important;
        max-width: 75% !important;
      }

      /* Assistant Message Bubble — stays on the LEFT */
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.65)) !important;
        border: 3px solid rgba(59, 130, 246, 0.55) !important;
        border-radius: 20px 20px 20px 4px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), 0 0 20px rgba(59, 130, 246, 0.1) !important;
        color: #E2E8F0 !important;
        padding: 16px 22px !important;
        display: block !important;
        max-width: 82% !important;
      }

      /* ===== Buttons & Inputs ===== */
      .stButton > button, button[kind="primary"], button[kind="secondary"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(168, 85, 247, 0.15)) !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        border-radius: 14px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15) !important;
      }

      .stButton > button:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.35), rgba(168, 85, 247, 0.35)) !important;
        border-color: #3B82F6 !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35) !important;
      }

      .stButton > button[kind="primary"], button[type="primary"] {
        background: linear-gradient(135deg, #3B82F6, #A855F7) !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
      }

      /* Chat Input Bar */
      [data-testid="stChatInputContainer"], [data-testid="stChatInput"] {
        background: rgba(5, 9, 20, 0.85) !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7), 0 0 20px rgba(168, 85, 247, 0.2) !important;
        backdrop-filter: blur(30px) !important;
      }

      [data-testid="stChatInputContainer"]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.5) !important;
      }

      /* ===== Welcome Screen & Suggestion Cards ===== */
      .welcome-screen {
        text-align: center;
        padding: 50px 20px 30px;
        animation: bubbleSlide 0.5s ease-out both;
      }

      .welcome-icon {
        width: 80px;
        height: 80px;
        margin: 0 auto 20px;
        border-radius: 24px;
        background: linear-gradient(135deg, #3B82F6, #A855F7, #EC4899);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        box-shadow: 0 0 40px rgba(168, 85, 247, 0.5);
        animation: floatPulse 4s ease-in-out infinite;
      }

      @keyframes floatPulse {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
      }

      .welcome-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #60A5FA 50%, #C084FC 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 8px;
      }

      .welcome-subtitle {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.65);
        margin-bottom: 32px;
        line-height: 1.6;
      }

      .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        max-width: 680px;
        margin: 0 auto;
      }

      .suggestion-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.7)) !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
        border-radius: 18px !important;
        padding: 18px !important;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
      }

      .suggestion-card:hover {
        border-color: #3B82F6 !important;
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.25) !important;
      }

      .suggestion-card .icon { font-size: 22px; margin-bottom: 8px; }
      .suggestion-card .label { font-size: 12px; color: #93C5FD; margin-top: 4px; font-weight: 500; }

      /* ===== Sidebar Scrolling & Height Overrides ===== */
      [data-testid="stSidebar"], .stSidebar {
        overflow-y: auto !important;
        max-height: 100vh !important;
      }

      [data-testid="stSidebarUserContent"] {
        overflow-y: auto !important;
        height: auto !important;
        max-height: calc(100vh - 10px) !important;
        padding-bottom: 90px !important;
      }

      [data-testid="stSidebarNav"] {
        overflow-y: auto !important;
      }

      /* ===== Main scroll container — must NOT have overflow:visible or perspective ===== */
      .stMain, [data-testid="stMain"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        position: relative !important;
      }

      /* .main-content gets 3D tilt via JS mousemove — NO CSS perspective needed here */
      .main-content {
        transition: transform 0.12s cubic-bezier(0.1, 0.9, 0.2, 1);
        will-change: transform;
      }

      .sidebar-section, .suggestion-card, .metric-card-3d {
        transition: all 0.35s cubic-bezier(0.23, 1, 0.32, 1) !important;
      }

      .sidebar-section:hover, .suggestion-card:hover, .metric-card-3d:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 20px 45px rgba(59, 130, 246, 0.3), 0 0 35px rgba(168, 85, 247, 0.25) !important;
        border-color: rgba(59, 130, 246, 0.6) !important;
      }


      /* 3D Floating Metric Cards */
      .metric-card-3d {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8)) !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 20px !important;
        padding: 18px 22px !important;
        box-shadow: 0 12px 35px rgba(0,0,0,0.6) !important;
      }
      .metric-val-3d {
        font-family: 'Clash Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: #60A5FA;
        text-shadow: 0 0 18px rgba(59, 130, 246, 0.6);
      }
    </style>

    <!-- Background Particle Canvas & 3D Interactive Parallax Engine -->
    <script>
      (function() {
        // Always attach global mousemove listener for 3D tilt
        function attach3DTilt() {
          const glow = document.getElementById('cursor-glow-orb') || document.createElement('div');
          if (!glow.id) {
            glow.id = 'cursor-glow-orb';
            glow.style.cssText = 'position:fixed;width:550px;height:550px;background:radial-gradient(circle, rgba(168,85,247,0.16) 0%, transparent 70%);border-radius:50%;filter:blur(100px);pointer-events:none;z-index:1;will-change:transform;';
            document.body.appendChild(glow);
          }

          window.removeEventListener('mousemove', window.__bhavyamMouseMoveHandler);
          
          window.__bhavyamMouseMoveHandler = function(e) {
            const { clientX, clientY } = e;
            glow.style.transform = `translate3d(${clientX - 275}px, ${clientY - 275}px, 0)`;

            const centerX = window.innerWidth / 2;
            const centerY = window.innerHeight / 2;
            
            // Increased tilt ratio for responsive edge tracking
            const rotateX = ((centerY - clientY) / centerY) * 7.5;
            const rotateY = ((clientX - centerX) / centerX) * 9.5;

            const mainContent = document.querySelector('.main-content') || document.querySelector('[data-testid="stMain"]');
            const sidebar = document.querySelector('[data-testid="stSidebar"]');

            if (mainContent) {
              mainContent.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(30px)`;
            }
            if (sidebar) {
              sidebar.style.transform = `rotateX(${rotateX * 0.5}deg) rotateY(${rotateY * 0.5}deg) translateZ(20px)`;
            }
          };

          window.addEventListener('mousemove', window.__bhavyamMouseMoveHandler);
        }

        attach3DTilt();

        if (window.__bhavyamCanvasInitialized) return;
        window.__bhavyamCanvasInitialized = true;

        // Particle Canvas
        const canvas = document.createElement('canvas');
        canvas.id = 'bg-particles-canvas';
        canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;';
        document.body.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        let particles = [];

        function resize() {
          canvas.width = window.innerWidth;
          canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        for (let i = 0; i < 50; i++) {
          particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.45,
            vy: (Math.random() - 0.5) * 0.45,
            r: Math.random() * 2 + 0.8,
            color: Math.random() > 0.5 ? '168, 85, 247' : '59, 130, 246'
          });
        }

        function draw() {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          for (let i = 0; i < particles.length; i++) {
            let p = particles[i];
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${p.color}, 0.45)`;
            ctx.shadowBlur = 10;
            ctx.shadowColor = `rgba(${p.color}, 0.8)`;
            ctx.fill();
            ctx.shadowBlur = 0;

            for (let j = i + 1; j < particles.length; j++) {
              let p2 = particles[j];
              let dx = p.x - p2.x, dy = p.y - p2.y;
              let dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < 110) {
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.strokeStyle = `rgba(168, 85, 247, ${0.14 * (1 - dist / 110)})`;
                ctx.lineWidth = 0.8;
                ctx.stroke();
              }
            }
          }
          requestAnimationFrame(draw);
        }
        draw();

        // Web Audio Synthesizer SFX
        let audioCtx = null;
        function playBeep(freq = 520, type = 'sine', duration = 0.08) {
          try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === 'suspended') audioCtx.resume();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
          } catch(e){}
        }

        document.addEventListener('click', (e) => {
          if (e.target.closest('button, a, .suggestion-card, .stSelectbox')) {
            playBeep(680, 'sine', 0.06);
          }
        });
      })();
    </script>
    """
    st.markdown(style, unsafe_allow_html=True)


def inject_advanced_ui():
    """Advanced UI tokens."""
    st.markdown("""
    <style>
    /* Hide broken icon-font text on the collapsed-sidebar re-expand button */
    [data-testid="stSidebarCollapsedControl"] button > span:not(svg),
    [data-testid="stSidebarCollapsedControl"] button > div:not(svg) {
        font-size: 0 !important;
        color: transparent !important;
        text-indent: -9999px !important;
        overflow: hidden !important;
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        padding: 0 !important;
        margin: -1px !important;
        clip: rect(0, 0, 0, 0) !important;
        white-space: nowrap !important;
        border: 0 !important;
    }

    /* Draw our own visible arrow icon in its place */
    [data-testid="stSidebarCollapsedControl"] button {
        position: relative !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        width: 36px !important;
        height: 36px !important;
    }
    [data-testid="stSidebarCollapsedControl"] button::after {
        content: "»" !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        font-size: 18px !important;
         color: #4C9FFF !important;
    }

    /* ═══════════════════════════════════════════════════════════
       CHAT BUBBLES — iOS-inspired + glassmorphism
       ═══════════════════════════════════════════════════════════ */
    @keyframes msgEnter {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage { animation: msgEnter 0.35s cubic-bezier(0.16, 1, 0.3, 1) both; }

    /* USER BUBBLE — true liquid-glass */
    .chat-bubble.user .message-content {
        background: rgba(30, 70, 130, 0.35) !important;
        backdrop-filter: blur(36px) saturate(160%) brightness(1.05) !important;
        -webkit-backdrop-filter: blur(36px) saturate(160%) brightness(1.05) !important;
        border: 1px solid rgba(76,159,255,0.25) !important;
        border-radius: 20px 20px 4px 20px !important;
        position: relative !important;
    }
    .chat-bubble.user .message-content::before {
        content: '';
        position: absolute; top: 0; left: 16px; right: 16px; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(120,190,255,0.4), transparent);
    }

    /* ASSISTANT BUBBLE — true liquid-glass */
    .chat-bubble.assistant .message-content {
        background: rgba(20, 22, 28, 0.55) !important;
        backdrop-filter: blur(36px) saturate(150%) brightness(1.05) !important;
        -webkit-backdrop-filter: blur(36px) saturate(150%) brightness(1.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 20px 20px 20px 4px !important;
        position: relative !important;
    }
    .chat-bubble.assistant .message-content::before {
        content: '';
        position: absolute; top: 0; left: 16px; right: 16px; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent);
    }

    .chat-bubble {
        display: flex !important;
        width: 100% !important;
        gap: 12px !important;
        margin-bottom: 4px !important;
    }

    .chat-bubble.user {
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
    }

    .chat-bubble.assistant {
        flex-direction: row !important;
        justify-content: flex-start !important;
    }

    .message-content {
        display: inline-block !important;
        width: fit-content !important;
        max-width: 70% !important;
        border-width: 2px !important;
        border-style: solid !important;
        padding: 16px 24px !important;
    }

    .chat-bubble.user .message-content {
        margin-left: auto !important;
        margin-right: 0 !important;
        border-color: rgba(76,159,255,0.6) !important;
    }

    .chat-bubble.assistant .message-content {
        margin-right: auto !important;
        margin-left: 0 !important;
        border-color: rgba(255,255,255,0.18) !important;
    }

    /* Avatar styling */
    .stChatMessage [data-testid="chat-avatar-user"],
    .stChatMessage [data-testid="chat-avatar-assistant"] {
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* Code blocks inside messages */
    .stChatMessage code {
        background: rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
        font-size: 13px !important;
    }

    .stChatMessage pre {
        background: rgba(0,0,0,0.5) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ═══════════════════════════════════════════════════════════
       CHAT INPUT — floating dock with 3D perspective
       ═══════════════════════════════════════════════════════════ */
    .stChatInput {
        position: sticky !important;
        bottom: 16px !important;
        margin: 0 auto !important;
        max-width: 760px !important;
    }

    /* Input container glass surface */
    .stChatInput > div {
        background: rgba(15, 16, 20, 0.6) !important;
        backdrop-filter: blur(40px) saturate(160%) brightness(1.05) !important;
        -webkit-backdrop-filter: blur(40px) saturate(160%) brightness(1.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 28px !important;
    }

    /* Textarea inside input */
    .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        resize: none !important;
        padding: 10px 0 !important;
        outline: none !important;
        box-shadow: none !important;
    }

    .stChatInput textarea::placeholder {
        color: rgba(255,255,255,0.35) !important;
    }

    /* ═══════════════════════════════════════════════════════════
       SEND BUTTON — glowing interactive
       ═══════════════════════════════════════════════════════════ */
    .stChatInput button {
        background: linear-gradient(135deg, #4C9FFF 0%, #7C4DFF 100%) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        color: #FFFFFF !important;
        cursor: pointer !important;
        box-shadow:
            0 4px 16px rgba(76,159,255,0.35),
            0 0 24px rgba(76,159,255,0.15) !important;
        transition: all 0.25s var(--ease-m3-standard) !important;
        animation: glowPulse 2.8s ease-in-out infinite !important;
        will-change: transform, box-shadow !important;
        position: relative !important;
        overflow: hidden !important;
    }

    @keyframes glowPulse {
        0%, 100% {
            box-shadow:
                0 4px 16px rgba(76,159,255,0.35),
                0 0 24px rgba(76,159,255,0.15);
        }
        50% {
            box-shadow:
                0 4px 24px rgba(76,159,255,0.55),
                0 0 48px rgba(124,77,255,0.25);
        }
    }

    .stChatInput button:hover {
        transform: scale(1.08) !important;
        box-shadow:
            0 6px 24px rgba(76,159,255,0.5),
            0 0 40px rgba(124,77,255,0.3) !important;
        animation: none !important;
    }

    .stChatInput button:active {
        transform: scale(0.92) !important;
        transition: all 0.1s var(--ease-m3-standard) !important;
        box-shadow:
            0 2px 8px rgba(76,159,255,0.3) !important;
        background: linear-gradient(135deg, #3B8EE8 0%, #6B3DE8 100%) !important;
    }

    /* Ripple effect on button */
    .stChatInput button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s, opacity 0.6s;
    }

    .stChatInput button:active::after {
        width: 120px;
        height: 120px;
        opacity: 0;
        transition: 0s;
    }

    /* ═══════════════════════════════════════════════════════════
       CUSTOM SCROLLBARS
       ═══════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        transition: background 0.2s;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.2);
    }

    ::-webkit-scrollbar-corner {
        background: transparent;
    }

    /* ═══════════════════════════════════════════════════════════
       GENERAL BUTTON STYLING (not chat send)
       ═══════════════════════════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg,
            rgba(76,159,255,0.12) 0%,
            rgba(76,159,255,0.06) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(76,159,255,0.2) !important;
        border-radius: var(--radius-md) !important;
        color: var(--m3-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        transition: all 0.2s var(--ease-m3-standard) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg,
            rgba(76,159,255,0.2) 0%,
            rgba(76,159,255,0.1) 100%) !important;
        border-color: rgba(76,159,255,0.4) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(76,159,255,0.15) !important;
    }

    .stButton > button:active {
        transform: scale(0.97) !important;
        transition: all 0.1s !important;
    }

    /* ═══════════════════════════════════════════════════════════
       SPINNER / LOADING
       ═══════════════════════════════════════════════════════════ */
    .stSpinner > div {
        border-top-color: var(--m3-primary) !important;
        border-right-color: var(--m3-secondary) !important;
    }

    /* ═══════════════════════════════════════════════════════════
       EXPANDER (used for settings / info panels)
       ═══════════════════════════════════════════════════════════ */
    .stExpander {
        background: var(--glass-bg) !important;
        backdrop-filter: var(--glass-blur) !important;
        -webkit-backdrop-filter: var(--glass-blur) !important;
        border: var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
    }

    /* ═══════════════════════════════════════════════════════════
       SELECTBOX / INPUT WIDGETS IN MAIN AREA
       ═══════════════════════════════════════════════════════════ */
    .stSelectbox > div,
    .stTextInput > div {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: var(--radius-md) !important;
    }

    /* ═══════════════════════════════════════════════════════════
       ACCESSIBILITY
       ═══════════════════════════════════════════════════════════ */
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }

    /* ===== Mobile Optimization ===== */
    @media (max-width: 768px) {
        .app-header {
            padding: 10px 16px !important;
            gap: 10px !important;
            margin-bottom: 10px !important;
            border-radius: 0 0 14px 14px !important;
        }

        .app-logo {
            width: 32px !important;
            height: 32px !important;
            min-width: 32px !important;
            font-size: 18px !important;
        }

        .app-title {
            font-size: 16px !important;
            letter-spacing: -0.3px !important;
        }

        .app-subtitle {
            font-size: 10px !important;
            padding: 3px 8px !important;
        }

        .main-content {
            padding: 8px 12px 120px !important;
        }

        .input-container {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            z-index: 100 !important;
            background: rgba(9, 13, 22, 0.95) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-top: 1px solid rgba(168, 85, 247, 0.3) !important;
            padding: 10px 12px !important;
            padding-bottom: max(10px, env(safe-area-inset-bottom)) !important;
        }

        .input-wrapper {
            max-width: 100% !important;
        }

        [data-testid="stChatInputContainer"] {
            border-radius: 14px !important;
            font-size: 14px !important;
        }

        [data-testid="stChatMessageContent"] {
            max-width: 88% !important;
            padding: 12px 16px !important;
            font-size: 14px !important;
        }

        .chat-bubble.user .message-content,
        .chat-bubble.assistant .message-content {
            max-width: 88% !important;
            padding: 12px 16px !important;
            font-size: 14px !important;
        }

        .avatar {
            width: 28px !important;
            height: 28px !important;
            min-width: 28px !important;
            font-size: 14px !important;
        }

        .stSidebar {
            font-size: 14px !important;
        }

        .stButton > button {
            font-size: 13px !important;
            padding: 8px 12px !important;
        }
    }

    /* Focus visible outlines */
    *:focus-visible {
        outline: 2px solid var(--m3-primary) !important;
        outline-offset: 2px !important;
    }

    textarea:focus-visible {
        outline: none !important;
    }

    /* ═══════════════════════════════════════════════════════════
       BLOCKQUOTE STYLING (for model responses)
       ═══════════════════════════════════════════════════════════ */
    blockquote {
        border-left: 3px solid rgba(76,159,255,0.4) !important;
        padding-left: 16px !important;
        margin: 12px 0 !important;
        color: rgba(255,255,255,0.7) !important;
        font-style: italic !important;
    }

    /* ═══════════════════════════════════════════════════════════
       TOOLTIP
       ═══════════════════════════════════════════════════════════ */
    [data-testid="stTooltip"] {
        background: rgba(20,20,20,0.95) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
    }

    /* ═══════════════════════════════════════════════════════════
       STATUS (info/warning/error/success boxes)
       ═══════════════════════════════════════════════════════════ */
    .stAlert {
        background: var(--glass-bg) !important;
        backdrop-filter: var(--glass-blur) !important;
        border: var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
    }
    </style>

    <script>
    /* Auto-scroll to bottom when new messages appear */
    (function() {
        const observer = new MutationObserver(() => {
            const chatContainer = document.querySelector('[data-testid="stAppViewContainer"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });

        const target = document.querySelector('[data-testid="stAppViewContainer"]');
        if (target) {
            observer.observe(target, { childList: true, subtree: true, characterData: true });
        }

        /* Also scroll on initial load */
        window.addEventListener('load', () => {
            setTimeout(() => {
                const container = document.querySelector('[data-testid="stAppViewContainer"]');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            }, 800);
        });
    })();
    </script>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Trust Check
# ---------------------------------------------------------------------------
current_cwd = os.getcwd()
trust_events = get_trust_events(current_cwd)
if trust_events:
    for event in trust_events:
        if event["type"] == "trust_denied" or (
            event["type"] == "trust_resolved" and event["policy"] == "require_approval"
        ):
            st.error(
                f"⚠️ Trust issue detected for `{current_cwd}`: {event.get('reason', 'Manual approval required')}. "
                "Please ensure you are running from an allowed directory."
            )
            st.stop()

inject_custom_style()
inject_advanced_ui()

st.markdown("""
<script>
(function() {
    if (window.__cursorGlowActive) return;
    window.__cursorGlowActive = true;
    const doc = document;

    const glow = doc.createElement('div');
    glow.id = 'cursor-glow-real';
    glow.style.cssText = `
        position: fixed; width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(76,159,255,0.08) 0%, transparent 70%);
        border-radius: 50%; pointer-events: none; z-index: 99999;
        transition: transform 0.1s ease-out;
    `;
    doc.body.appendChild(glow);

    let ticking = false;
    doc.addEventListener('mousemove', function(e) {
        if (!ticking) {
            requestAnimationFrame(function() {
                const x = (e.clientX / window.innerWidth - 0.5) * 2;
                const y = (e.clientY / window.innerHeight - 0.5) * 2;

                const sidebar = doc.querySelector('[data-testid="stSidebar"]');
                const chatContainer = doc.querySelector('.chat-container');

                if (sidebar) {
                    sidebar.style.transform = `perspective(1200px) rotateY(${x * 1.5}deg) rotateX(${-y * 1}deg)`;
                }
                if (chatContainer) {
                    chatContainer.style.transform = `perspective(1200px) rotateY(${x * 0.8}deg) rotateX(${-y * 0.5}deg)`;
                }

                glow.style.left = (e.clientX - 250) + 'px';
                glow.style.top = (e.clientY - 250) + 'px';

                ticking = false;
            });
            ticking = true;
        }
    });
})();
</script>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model download & loading
# ---------------------------------------------------------------------------
def ensure_model(model_key: str):
    """Download the model if it is not present locally. Returns the path to the GGUF file."""
    model_config = MODEL_OPTIONS[model_key]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=model_config["repo"],
        filename=model_config["file"],
        cache_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    model_path = Path(model_path)

    # Windows: huggingface_hub may leave a 0-byte reparse point (symlink) in
    # snapshots/ when an existing cache entry is reused. llama-cpp-python
    # cannot open these. Replace with a real copy from blobs/ if needed.
    if model_path.exists() and model_path.stat().st_size == 0:
        blob_dir = model_path.parent.parent / "blobs"
        if blob_dir.exists():
            for blob_file in blob_dir.iterdir():
                if blob_file.is_file() and not blob_file.name.endswith(".incomplete"):
                    try:
                        import shutil
                        shutil.copy2(blob_file, model_path)
                        break
                    except OSError:
                        continue

    # Validate the model file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    file_size = model_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Model file is empty (0 bytes) at {model_path}")
    
    if file_size < 100_000_000:  # Less than 100MB is suspicious for a GGUF model
        raise ValueError(f"Model file seems too small ({file_size} bytes) at {model_path}")

    return str(model_path)

@st.cache_resource(show_spinner="Loading model into memory...")
def load_model(model_key: str):
    # Remote & n8n models don't need a local Llama instance.
    model_config = MODEL_OPTIONS.get(model_key, {})
    if model_config.get("backend") in ("remote", "n8n_orchestrated"):
        return None

    # Safety check: if model_key is invalid or missing, default
    if model_key not in MODEL_OPTIONS:
        model_key = DEFAULT_MODEL_KEY

    model_config = MODEL_OPTIONS[model_key]
    
    try:
        model_path = ensure_model(model_key)
    except Exception as download_err:
        if model_key != DEFAULT_MODEL_KEY:
            st.session_state.fallback_notice = (
                f"⚠️ Download issue with '{model_key}': {download_err}. "
                f"Automatically loaded '{DEFAULT_MODEL_KEY}' instead."
            )
            st.session_state.selected_model = DEFAULT_MODEL_KEY
            return load_model(DEFAULT_MODEL_KEY)
        raise download_err
    
    # Try context sizes and batch sizes to fit in available RAM
    fallback_ctxs = [
        model_config["n_ctx"],
        2048,
        1024,
        512,
        256,
    ]
    seen = set()
    ctx_candidates = []
    for ctx in fallback_ctxs:
        if ctx not in seen:
            seen.add(ctx)
            ctx_candidates.append(ctx)
    
    last_error = None
    for n_ctx in ctx_candidates:
        for batch_size in [N_BATCH, 256, 128]:
            try:
                return Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=N_THREADS,
                    n_gpu_layers=N_GPU_LAYERS,
                    n_batch=batch_size,
                    verbose=False,
                )
            except Exception as e:
                last_error = e
                continue
    
    # If loading failed for memory or buffer reason, attempt automatic fallback
    if model_key != DEFAULT_MODEL_KEY:
        st.session_state.fallback_notice = (
            f"⚠️ Model '{model_key}' requires more contiguous RAM than available on this machine. "
            f"Bhavyam AI automatically switched to '{DEFAULT_MODEL_KEY}' for instant responsiveness!"
        )
        st.session_state.selected_model = DEFAULT_MODEL_KEY
        return load_model(DEFAULT_MODEL_KEY)

    # Detailed error if default model also fails
    error_msg = f"Failed to load model '{model_key}' with any context size.\n\n"
    error_msg += f"Model path: {model_path}\n"
    error_msg += f"Last error: {last_error}\n\n"
    error_msg += "Possible solutions:\n"
    error_msg += "1. Close other applications to free up RAM\n"
    error_msg += "2. Restart the application\n"
    error_msg += "3. Check if the model file is corrupted"
    raise RuntimeError(error_msg)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    # On startup, try to load the most recent conversation
    recent = get_most_recent_conversation()
    if recent:
        st.session_state.current_conversation_id = recent["id"]
        st.session_state.messages = recent["messages"]
    else:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm Bhavyam AI, owned by Subham Mahapatra from Odisha. How can I help you today?",
            }
        ]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL_KEY

# Startup check: verify router env vars if default model is remote
if st.session_state.selected_model in MODEL_OPTIONS:
    model_cfg = MODEL_OPTIONS[st.session_state.selected_model]
    if model_cfg.get("backend") == "remote":
        missing = []
        if not os.environ.get("ROUTER_BASE_URL"):
            missing.append("ROUTER_BASE_URL")
        if not os.environ.get("ROUTER_API_KEY"):
            missing.append("ROUTER_API_KEY")
        if missing:
            st.error(
                f"⚠️ Remote model '{st.session_state.selected_model}' is selected as default, "
                f"but the following environment variables are not set: {', '.join(missing)}. "
                f"Please configure them in your deployment dashboard (e.g., Railway → Settings → Variables)."
            )
            st.stop()

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = None

if "retry_prompt" not in st.session_state:
    st.session_state.retry_prompt = None

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "message_order" not in st.session_state:
    st.session_state.message_order = "Oldest First ⬇️"

if "fallback_notice" not in st.session_state:
    st.session_state.fallback_notice = None

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(system: str, history: list, user_msg: str, template_key: str) -> str:
    t = CHAT_TEMPLATES[template_key]
    prompt = t["system"].format(c=system)
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        if role in ("user", "assistant"):
            prompt += t[role].format(c=content)
    prompt += t["user"].format(c=user_msg) + t["assistant_open"]
    return prompt

# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)

def trim_history(history: list, system: str, user_msg: str, template_key: str, max_history_tokens: int = 1024) -> list:
    """Keep last 10 exchanges and trim until prompt fits within max_history_tokens."""
    trimmed = history[:]
    # Keep last 10 exchanges (20 messages max)
    trimmed = trimmed[-20:]
    
    while trimmed and estimate_tokens(build_prompt(system, trimmed, user_msg, template_key)) > max_history_tokens:
        trimmed.pop(0)
    
    return trimmed

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
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
    except Exception as e:
        yield clean_response(f"\n\n*[Generation error: {e}]*")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-logo">🤖</div>
        <div>
            <div class="app-title">Bhavyam AI</div>
            <div style="display: flex; align-items: center; font-size: 11px; color: #10B981; font-weight: 600; margin-top: 2px;">
                <span class="status-dot"></span> Neural Link Active
            </div>
        </div>
        <div class="app-subtitle">Active Core: {escape(st.session_state.selected_model)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Branding
    st.markdown(
        """
        <div style="text-align: center; padding: 12px 0 18px;">
            <div style="width: 60px; height: 60px; margin: 0 auto 12px; border-radius: 18px; background: linear-gradient(135deg, #3B82F6, #A855F7, #EC4899); display: flex; align-items: center; justify-content: center; font-size: 30px; box-shadow: 0 0 25px rgba(168, 85, 247, 0.5);">🤖</div>
            <div style="font-size: 20px; font-weight: 800; font-family: 'Clash Grotesk', sans-serif; background: linear-gradient(135deg, #FFFFFF 0%, #60A5FA 50%, #C084FC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Bhavyam AI</div>
            <div style="font-size: 11px; color: #93C5FD; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">Neural Command OS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Chat Controls
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">💬 Chat & View</div>', unsafe_allow_html=True)

    order_choice = st.selectbox(
        "Message Order",
        options=["Oldest First ⬇️", "Newest First ⬆️"],
        index=0 if st.session_state.message_order == "Oldest First ⬇️" else 1,
        help="Reorder messages: Oldest First (chronological) or Newest First (latest messages at top)",
    )
    if order_choice != st.session_state.message_order:
        st.session_state.message_order = order_choice
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
    with col2:
        if st.button("💾 Export", use_container_width=True):
            if st.session_state.get("messages"):
                chat_text = "\n\n".join(
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in st.session_state.messages
                )
                st.download_button(
                    "Download",
                    data=chat_text,
                    file_name="chat_history.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🧠 Model</div>', unsafe_allow_html=True)

    # System RAM monitor
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        used_pct = mem.percent
        ram_color = "#22c55e" if avail_gb >= 2.0 else ("#eab308" if avail_gb >= 1.0 else "#ef4444")
        
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 8px 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; font-size: 11px;">
                    <span style="color: rgba(255,255,255,0.7); font-weight: 500;">💾 System RAM</span>
                    <span style="color: {ram_color}; font-weight: 600;">{avail_gb:.1f} GB Free</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 5px; overflow: hidden;">
                    <div style="background: {ram_color}; height: 100%; width: {used_pct}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

    selected_model = st.selectbox(
        "Choose Model",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in MODEL_OPTIONS else 0,
        help="Switch between available models. Changing model reloads the app.",
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.messages = []
        st.session_state.last_error = None
        st.session_state.last_user_input = None
        st.rerun()

    st.caption(
        f"<div style='color: #8b949e; font-size: 11px; line-height: 1.4;'>{MODEL_OPTIONS[selected_model]['description']}</div>",
        unsafe_allow_html=True,
    )

    # Custom model input
    st.divider()
    st.caption("<div style='color: #8b949e; font-size: 11px;'>Or load a custom GGUF from HuggingFace:</div>", unsafe_allow_html=True)
    
    custom_repo = st.text_input(
        "HuggingFace Repo ID",
        placeholder="e.g. microsoft/Phi-3-mini-4k-instruct-gguf",
        label_visibility="collapsed",
        help="Enter a HuggingFace repository ID that contains GGUF files",
    )
    custom_file = st.text_input(
        "GGUF Filename",
        placeholder="e.g. Phi-3-mini-4k-instruct-q4.gguf",
        label_visibility="collapsed",
        help="Enter the exact GGUF filename from the repo",
    )
    custom_n_ctx = st.number_input(
        "Context Window",
        min_value=512,
        max_value=131072,
        value=4096,
        step=512,
        label_visibility="collapsed",
        help="Context window size for the custom model",
    )
    
    if st.button("📥 Load Custom Model", use_container_width=True):
        if custom_repo.strip() and custom_file.strip():
            custom_key = f"Custom: {custom_repo}/{custom_file}"
            MODEL_OPTIONS[custom_key] = {
                "repo": custom_repo.strip(),
                "file": custom_file.strip(),
                "n_ctx": custom_n_ctx,
                "description": f"Custom model from {custom_repo}",
            }
            st.session_state.selected_model = custom_key
            st.session_state.messages = []
            st.session_state.last_error = None
            st.session_state.last_user_input = None
            st.rerun()
        else:
            st.warning("Please enter both HuggingFace Repo ID and GGUF Filename.")
    
    if st.button("🗑️ Clear Model Cache", use_container_width=True):
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        st.cache_resource.clear()
        st.session_state.messages = []
        st.session_state.last_error = None
        st.session_state.last_user_input = None
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Context Usage
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Context</div>', unsafe_allow_html=True)

    # Estimate current context usage
    context_messages = st.session_state.messages + [
        {"role": "user", "content": ""}  # placeholder for next user turn
    ]
    active_template_ctx = MODEL_OPTIONS[st.session_state.selected_model].get("template", "phi3")
    context_prompt = build_prompt(SYSTEM_PROMPT, context_messages[:-1], context_messages[-1]["content"], active_template_ctx)
    estimated_ctx = estimate_tokens(context_prompt)
    model_ctx = MODEL_OPTIONS[st.session_state.selected_model]["n_ctx"]
    ctx_pct = min(100.0, (estimated_ctx / model_ctx) * 100)

    st.progress(ctx_pct / 100.0, text=f"Context: ~{estimated_ctx} / {model_ctx} tokens ({ctx_pct:.1f}%)")
    st.caption(
        f"<div style='color: #8b949e; font-size: 11px;'>Messages: {len(st.session_state.messages)} | Estimated usage</div>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Generation Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚙️ Generation</div>', unsafe_allow_html=True)

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        help="Lower = more focused, Higher = more creative",
    )

    top_p = st.slider(
        "Top P",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_TOP_P,
        step=0.05,
        help="Nucleus sampling probability mass",
    )

    max_tokens_slider = st.slider(
        "Max Tokens",
        min_value=64,
        max_value=1024,
        value=MAX_TOKENS,
        step=64,
        help="Maximum length of generated response",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Features
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">✨ Features</div>', unsafe_allow_html=True)

    web_search_enabled = st.toggle(
        "🌐 Web Search",
        value=True,
        help="Search Wikipedia for real-time information",
    )

    stream_enabled = st.toggle(
        "⚡ Streaming",
        value=True,
        help="Stream responses token by token",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Previous Chats
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📂 Previous Chats</div>', unsafe_allow_html=True)

    # New Chat button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        create_new_chat()

    # List past conversations
    conversations = list_conversations()
    if conversations:
        for conv in conversations:
            col_title, col_delete = st.columns([4, 1])
            with col_title:
                if st.button(
                    f"{conv['title']}",
                    key=f"load_{conv['id']}",
                    use_container_width=True,
                    help=f"Created: {conv['created_at'][:10]}",
                ):
                    load_conversation(conv["id"])
            with col_delete:
                if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                    st.session_state[f"confirm_delete_{conv['id']}"] = True
                    st.rerun()
            
            # Confirm delete prompt
            if st.session_state.get(f"confirm_delete_{conv['id']}"):
                st.warning(f"Delete \"{conv['title']}\"?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes", key=f"yes_{conv['id']}", type="primary"):
                        delete_conversation(conv["id"])
                        # If deleted conversation was active, start fresh
                        if st.session_state.get("current_conversation_id") == conv["id"]:
                            st.session_state.current_conversation_id = None
                            st.session_state.messages = []
                        st.session_state.pop(f"confirm_delete_{conv['id']}", None)
                        st.rerun()
                with col_no:
                    if st.button("No", key=f"no_{conv['id']}"):
                        st.session_state.pop(f"confirm_delete_{conv['id']}", None)
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.divider()
    st.caption(
        "<div style='text-align: center; color: #8b949e; font-size: 11px;'>"
        "© 2026 Bhavyam AI<br>"
        "Owned by Subham Mahapatra<br>"
        "Odisha, India"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Display RAM / Fallback notice if present
if st.session_state.get("fallback_notice"):
    st.info(st.session_state.fallback_notice)
    if st.button("✖️ Dismiss Notice", key="dismiss_fallback"):
        st.session_state.fallback_notice = None
        st.rerun()


if not st.session_state.messages:
    # Welcome screen
    st.markdown(
        f"""
        <div class="welcome-screen">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">Hello! I'm Bhavyam AI</div>
            <div class="welcome-subtitle">
                Your intelligent local AI assistant powered by {escape(st.session_state.selected_model)}<br>
                Ask me anything — I'm here to help!
            </div>
            <div class="suggestion-grid">
                <div class="suggestion-card" onclick="document.querySelector('.stChatInput textarea').focus()">
                    <div class="icon">💡</div>
                    <div>Explain a concept</div>
                    <div class="label">I'll break it down simply</div>
                </div>
                <div class="suggestion-card" onclick="document.querySelector('.stChatInput textarea').focus()">
                    <div class="icon">✍️</div>
                    <div>Help me write</div>
                    <div class="label">Emails, essays, code & more</div>
                </div>
                <div class="suggestion-card" onclick="document.querySelector('.stChatInput textarea').focus()">
                    <div class="icon">💻</div>
                    <div>Write some code</div>
                    <div class="label">Python, JS, C++, and more</div>
                </div>
                <div class="suggestion-card" onclick="document.querySelector('.stChatInput textarea').focus()">
                    <div class="icon">🧮</div>
                    <div>Solve a math problem</div>
                    <div class="label">From arithmetic to calculus</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Chat history with Message Ordering support (Oldest First vs Newest First)
    messages_to_show = st.session_state.messages
    if st.session_state.get("message_order") == "Newest First ⬆️":
        messages_to_show = list(reversed(st.session_state.messages))

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for idx, msg in enumerate(messages_to_show):
        role = msg["role"]
        avatar_emoji = "👤" if role == "user" else "🤖"
        content_html = escape(msg["content"]).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="chat-bubble {role}">
                <div class="avatar {role}">{avatar_emoji}</div>
                <div class="message-content">{content_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input area
st.markdown(
    """
    <div class="input-container">
        <div class="input-wrapper">
    """,
    unsafe_allow_html=True,
)

# Load model with automatic fallback support
def _clean_error(text: str) -> str:
    """Remove leaked metadata from error messages."""
    text = re.sub(r"(?is)<\s*environment_details\b[^>]*>.*?<\s*/\s*environment_details\s*>", "", text)
    text = re.sub(r"(?is)<environment_details>.*?</environment_details>", "", text, flags=re.DOTALL)
    text = re.sub(r"(?is)<\s*environment_details\b[^>]*>.*", "", text, flags=re.DOTALL)
    
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        lower = line.strip().lower()
        if any(k in lower for k in ["current time:", "working directory:", "workspace root folder:", "<environment_details", "</environment_details>"]):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()

try:
    model = load_model(st.session_state.selected_model)
except Exception as e:
    st.error(f"Failed to load model: {_clean_error(str(e))}")
    st.stop()

st.chat_input("Type your message...", key="user_input")

# Determine the input to process: retry prompt takes precedence
user_input = st.session_state.get("retry_prompt") or st.session_state.get("user_input")

if user_input := user_input.strip() if isinstance(user_input, str) else None:
    if not user_input:
        st.warning("Please enter a non-empty message.")
        st.stop()

    # Check for image/file input attempts
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.webp']
    input_lower = user_input.lower()
    image_related_keywords = ['image input', 'upload image', 'send image', 'picture of', 'photo of', 'screenshot', 'ui3.png', 'ui3 png']
    is_image_attempt = (
        any(ext in input_lower for ext in image_extensions) 
        or any(kw in input_lower for kw in image_related_keywords)
        or 'uienhance' in input_lower
        or '.png' in input_lower
        or '.jpg' in input_lower
        or '.jpeg' in input_lower
    )
    if is_image_attempt:
        # Render as assistant chat bubble with enhanced styling
        with st.chat_message("assistant"):
            st.markdown(
                f"""
                <div class="chat-bubble assistant">
                    <div class="avatar assistant">🤖</div>
                    <div class="message-content" style="border-color: rgba(255,82,82,0.5) !important; background: rgba(255,82,82,0.1) !important;">
                        ⚠️ <strong>Image input is not supported</strong><br><br>
                        Bhavyam AI currently runs local text-only models and cannot read images. 
                        Please describe the image in text, or ask a text-based question.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.stop()

    # Clear retry prompt after using it
    if st.session_state.get("retry_prompt"):
        st.session_state.retry_prompt = None
        # Remove the duplicate user message from failed attempt if present
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Immediately show the user's bubble before the assistant starts streaming
    st.markdown(
        f"""
        <div class="chat-bubble user">
            <div class="avatar user">👤</div>
            <div class="message-content">{escape(user_input).replace(chr(10), '<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Web search injection — append to the user turn so the model treats it as current-question context.
    # Only search for genuine factual/entity/concept lookups; for math, code, and
    # reasoning questions a clean prompt (no Wikipedia injection) yields correct answers.
    search_context = ""
    augmented_user_msg = user_input
    if web_search_enabled and is_web_lookup_query(user_input):
        with st.spinner("🔍 Searching the web..."):
            search_results = web_search(user_input)
        
        if search_results == "NO_WEB_RESULTS_FOUND":
            search_context = (
                "\n\n<|web_search|>\n"
                "Web search returned no instant answers for this question.\n"
                "Answer from your own knowledge.\n"
                "</|web_search|>\n"
            )
        elif search_results.startswith("SEARCH_ERROR:"):
            search_context = (
                "\n\n<|web_search|>\n"
                f"Web search failed: {search_results}\n"
                "Answer from your own knowledge.\n"
                "</|web_search|>\n"
            )
        else:
            search_context = (
                "\n\n<|web_search|>\n"
                "IMPORTANT: Use the following web search results to answer the user's question accurately.\n"
                "Do NOT refuse to answer. Synthesize the information below into a helpful response.\n"
                f"{search_results}\n"
                "</|web_search|>\n"
            )
        if search_context:
            augmented_user_msg = f"{user_input}{search_context}"

    # Resolve active template for the currently loaded model
    active_template = MODEL_OPTIONS[st.session_state.selected_model].get("template", "phi3")
    stop_sequences = CHAT_TEMPLATES[active_template]["stop"]

    # Trim history if context would exceed limit
    history_for_prompt = trim_history(
        st.session_state.messages[:-1],  # exclude current user msg
        SYSTEM_PROMPT,
        user_input,
        active_template,
    )

    # Build prompt — keep system prompt clean; search context is inside the user turn
    prompt = build_prompt(SYSTEM_PROMPT, history_for_prompt, augmented_user_msg, active_template)

    # Stream assistant response token-by-token using the same UI mechanism
    prompt_tokens = estimate_tokens(prompt)

    # Determine backend for the selected model
    backend = MODEL_OPTIONS[st.session_state.selected_model].get("backend", "local_gguf")
    model_slug = MODEL_OPTIONS[st.session_state.selected_model].get("slug", "")

    # Use session_state to capture streaming state across generator calls
    st.session_state._stream_state = {
        "captured_tokens": [],
        "error_occurred": False,
        "error_message": "",
    }

    if backend == "remote":
        def streaming_generator():
            """Yield tokens from the remote router while capturing them."""
            state = st.session_state._stream_state
            try:
                remote_messages = (
                    [{"role": "system", "content": SYSTEM_PROMPT}]
                    + history_for_prompt
                    + [{"role": "user", "content": augmented_user_msg}]
                )
                for chunk in stream_chat(
                    messages=remote_messages,
                    model_slug=model_slug,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens_slider,
                ):
                    state["captured_tokens"].append(chunk)
                    display_token = re.sub(r"<\|[^|]*\|>", "", chunk)
                    if display_token:
                        yield display_token
            except (RouterConfigurationError, RouterError) as exc:
                state["error_occurred"] = True
                state["error_message"] = str(exc)
                yield str(exc)
    elif backend == "n8n_orchestrated":
        def streaming_generator():
            """Send message to n8n webhook and yield response."""
            if not _N8N_CLIENT_AVAILABLE:
                yield "n8n_orchestrated backend is not available: n8n_supabase_client module is missing. Deploy it alongside app.py or switch to a remote/local model."
                return
            state = st.session_state._stream_state
            try:
                n8n_res = send_to_n8n_webhook(
                    message=augmented_user_msg,
                    conversation_id=st.session_state.current_conversation_id,
                    model_slug=model_slug or "claude-opus-free",
                    backend="n8n_orchestrated",
                )
                answer = n8n_res.get("response", "")
                if not answer:
                    answer = "n8n completed workflow successfully but returned empty response."
                state["captured_tokens"].append(answer)
                yield answer
            except N8nOrchestrationError as exc:
                state["error_occurred"] = True
                state["error_message"] = str(exc)
                yield str(exc)
            except Exception as exc:
                state["error_occurred"] = True
                state["error_message"] = f"n8n Orchestration Error: {exc}"
                yield str(exc)
    else:
        def streaming_generator():
            """Yield tokens while capturing them for later use."""
            state = st.session_state._stream_state
            try:
                stream = model(
                    prompt,
                    max_tokens=max_tokens_slider,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.15,
                    stop=stop_sequences,
                    stream=True,
                )
                for chunk in stream:
                    token = chunk["choices"][0]["text"]
                    # Check for generation errors
                    if token.startswith("\n\n*[Generation error:"):
                        state["error_occurred"] = True
                        state["error_message"] = token
                        return
                    # Check for image input errors in streamed output
                    if "does not support image input" in token or "Cannot read" in token:
                        state["error_occurred"] = True
                        state["error_message"] = token
                        return
                    # Lightweight per-token cleanup only — don't strip whitespace-only tokens
                    state["captured_tokens"].append(token)  # Keep raw for history
                    display_token = re.sub(r"<\|[^|]*\|>", "", token)  # strip obvious special tokens only
                    if display_token:
                        yield display_token
            except Exception as e:
                state["error_occurred"] = True
                state["error_message"] = clean_response(f"\n\n*[Generation error: {e}]*")
                yield state["error_message"]

    # Display streaming response inside custom glass assistant bubble (prevents smart_toy icon flash)
    chat_placeholder = st.empty()
    accumulated_stream = ""
    for token in streaming_generator():
        accumulated_stream += token
        content_html = escape(accumulated_stream).replace("\n", "<br>")
        chat_placeholder.markdown(
            f"""
            <div class="chat-bubble assistant">
                <div class="avatar assistant">🤖</div>
                <div class="message-content">{content_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Retrieve captured streaming state
    state = st.session_state.pop("_stream_state", {})
    captured_tokens = state.get("captured_tokens", [])
    error_occurred = state.get("error_occurred", False)
    error_message = state.get("error_message", "")

    # After streaming completes, process the result
    if not error_occurred:
        raw_response = "".join(captured_tokens)
        final_response = clean_response(raw_response)
        response_tokens = estimate_tokens(final_response)
        st.session_state.last_error = None
        st.session_state.last_user_input = None

        # Token usage badge
        token_badge = (
            f"<div class='search-badge' style='background: rgba(102, 126, 234, 0.15); "
            f"color: #a78bfa; border: 1px solid rgba(102, 126, 234, 0.3);'>"
            f"🧮 Tokens: ~{prompt_tokens} in / ~{response_tokens} out</div>"
        )

        # Display badge below the streamed response
        if web_search_enabled and search_context and not search_context.startswith("Web search returned no instant answers") and not search_context.startswith("Web search failed"):
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; margin-top: 6px; margin-bottom: 8px;">
                    {token_badge}
                    <div class="search-badge">🌐 Web Search Enabled</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-top: 6px; margin-bottom: 8px;">
                    {token_badge}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Handle generation error with enhanced UI
        cleaned_token = clean_response(error_message)
        if backend == "remote":
            st.error(f"Remote router unavailable: {cleaned_token}")
            # Offer a one-click fallback to a local GGUF model for this turn.
            local_fallback_key = "Qwen2.5 0.5B (Q4_K_M) - Ultra Fast"
            if (
                local_fallback_key in MODEL_OPTIONS
                and MODEL_OPTIONS[local_fallback_key].get("backend") == "local_gguf"
                and st.session_state.selected_model != local_fallback_key
            ):
                if st.button(
                    f"Switch to {local_fallback_key} for this turn",
                    key="remote_fallback_btn",
                ):
                    st.session_state.retry_prompt = user_input
                    st.session_state.selected_model = local_fallback_key
                    st.session_state.fallback_notice = (
                        f"Switched to {local_fallback_key} due to remote router unavailability."
                    )
                    st.rerun()
        elif "does not support image input" in cleaned_token or "Cannot read" in cleaned_token:
            error_msg = (
                "⚠️ **Image input is not supported**\n\n"
                "Bhavyam AI currently runs local text-only models and cannot read images. "
                "Please describe the image in text, or ask a text-based question."
            )
            error_style = "border-color: rgba(255,82,82,0.5) !important; background: rgba(255,82,82,0.1) !important;"
            with st.chat_message("assistant"):
                st.markdown(
                    f"""
                    <div class="chat-bubble assistant">
                        <div class="avatar assistant">🤖</div>
                        <div class="message-content" style="{error_style}">
                            {escape(error_msg)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            error_msg = cleaned_token
            error_style = "border-color: rgba(255,82,82,0.35) !important;"
            with st.chat_message("assistant"):
                st.markdown(
                    f"""
                    <div class="chat-bubble assistant">
                        <div class="avatar assistant">🤖</div>
                        <div class="message-content" style="{error_style}">
                            {escape(error_msg)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.session_state.last_error = cleaned_token
        st.session_state.last_user_input = user_input

    # Save only clean, successful responses to history and DB
    if not error_occurred:
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        # Auto-save to database after each complete assistant response
        save_current_conversation()

# Close input container
st.markdown("</div></div>", unsafe_allow_html=True)

# Mobile keyboard fix: ensure input stays visible above keyboard
st.markdown(
    """
    <script>
    (function() {
        if (!/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)) return;
        
        const inputSelector = '.stChatInput textarea, .stTextInput textarea, textarea[aria-label="Type your message..."]';
        let lastHeight = window.innerHeight;
        
        function adjustForKeyboard() {
            const currentHeight = window.innerHeight;
            if (currentHeight < lastHeight - 100) {
                // Keyboard likely opened
                setTimeout(() => {
                    const input = document.querySelector(inputSelector);
                    if (input) {
                        input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }, 300);
            }
            lastHeight = currentHeight;
        }
        
        window.addEventListener('focusin', function(e) {
            if (e.target.tagName === 'TEXTAREA') {
                setTimeout(() => {
                    e.target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 300);
            }
        });
        
        window.addEventListener('resize', adjustForKeyboard);
        window.addEventListener('orientationchange', function() {
            setTimeout(adjustForKeyboard, 500);
        });
    })();

    (function() {
        function fixBrokenIconText() {
            const all = document.querySelectorAll('button, span, div');
            all.forEach(el => {
                if (el.children.length === 0) {
                    const text = el.textContent.trim();
                    if (/^[a-z_]+$/.test(text) && text.includes('_') && text.length > 5) {
                        el.textContent = '☰';
                        el.style.fontSize = '18px';
                        el.style.fontFamily = 'inherit';
                    }
                }
            });
        }
        fixBrokenIconText();
        const observer = new MutationObserver(fixBrokenIconText);
        observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)
