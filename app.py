import os
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_REPO = "TheBloke/phi-2-GGUF"
MODEL_FILE = "phi-2.Q4_K_M.gguf"
CACHE_DIR = Path(os.environ.get("HF_HOME", ".model_cache"))

N_CTX = 2048
N_THREADS = os.cpu_count() or 4
N_GPU_LAYERS = 0
N_BATCH = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "If you don't know the answer, say so."
)

STOP_SEQUENCES = ["<|user|>", "\nInstruct:", "<|endoftext|>"]

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MiniChat",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model download & loading
# ---------------------------------------------------------------------------
def ensure_model():
    """Download the model if it is not present locally. Returns the path to the GGUF file."""
    info_placeholder = st.info(f"📥 Downloading {MODEL_FILE} (~1.6 GB)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    info_placeholder.empty()
    st.success("✅ Model downloaded and cached!")
    return model_path

@st.cache_resource(show_spinner="Loading Phi-2 into memory...")
def load_model():
    model_path = ensure_model()
    return Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        verbose=False,
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(system: str, history: list, user_msg: str) -> str:
    prompt = f"Instruct: {system}\n\n"
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n"
    return prompt

# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)

def trim_history(history: list, system: str, user_msg: str, max_history_tokens: int = 1024) -> list:
    """Keep last 10 exchanges and trim until prompt fits within max_history_tokens."""
    trimmed = history[:]
    # Keep last 10 exchanges (20 messages max)
    trimmed = trimmed[-20:]
    
    while trimmed and estimate_tokens(build_prompt(system, trimmed, user_msg)) > max_history_tokens:
        trimmed.pop(0)
    
    return trimmed

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(model, prompt, temperature, top_p, max_tokens):
    """Yield tokens one by one from llama.cpp."""
    try:
        stream = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=STOP_SEQUENCES,
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            yield token
    except Exception as e:
        yield f"\n\n*[Generation error: {e}]*"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        help="Lower = more focused, Higher = more creative",
    )

    max_tokens_slider = st.slider(
        "Max Tokens",
        min_value=64,
        max_value=1024,
        value=MAX_TOKENS,
        step=64,
        help="Maximum length of generated response",
    )

    st.divider()
    st.caption(f"**Model:** Phi-2 Q4_K_M")
    st.caption(f"**Context:** {N_CTX} tokens")
    st.caption(f"**Threads:** {N_THREADS}")
    st.caption(f"**GPU Layers:** {N_GPU_LAYERS}")
    st.caption(f"**File:** {MODEL_FILE}")

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("💬 MiniChat")

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info(
        "If this is the first run, the model will be downloaded automatically. "
        "If download fails, check your internet connection and that you have "
        "~2 GB of free disk space."
    )
    st.stop()

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.text(msg["content"])
        else:
            st.markdown(msg["content"])

# Handle new user input
user_input = st.chat_input("Type your message...")

if user_input is not None:
    # Guard against empty / whitespace-only input
    if not user_input.strip():
        st.warning("Please enter a non-empty message.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Trim history if context would exceed limit
    history_for_prompt = trim_history(
        st.session_state.messages[:-1],  # exclude current user msg
        SYSTEM_PROMPT,
        user_input,
    )

    # Build prompt
    prompt = build_prompt(SYSTEM_PROMPT, history_for_prompt, user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        clean_response = ""
        error_occurred = False

        for token in generate(model, prompt, temperature, DEFAULT_TOP_P, max_tokens_slider):
            if token.startswith("\n\n*[Generation error:"):
                error_occurred = True
                st.error(token)
                break
            clean_response += token
            message_placeholder.markdown(clean_response + "▌")

        message_placeholder.markdown(clean_response)

    # Save only clean, successful responses to history
    if not error_occurred:
        st.session_state.messages.append({"role": "assistant", "content": clean_response})
