# 🤖 MiniChat

A fully **CPU‑only**, deployable ChatGPT‑style chatbot built with **Streamlit** + **llama‑cpp‑python** + **Phi‑2 (Q4_K_M)**.

## ✨ Features

- **100 % CPU inference** — no CUDA, no MPS, no dedicated GPU required.
- **Auto‑download model** — fetches `TheBloke/phi-2-GGUF` on first run and caches it locally.
- **Streaming responses** — token‑by‑token UI feedback.
- **Memory‑efficient history** — keeps last 10 exchanges, trims to ~1024 tokens.
- **Dark ChatGPT‑style UI** — `st.chat_message` bubbles with sidebar controls.
- **Production‑ready** — empty‑input guards, error handling, graceful degradation.

---

## 🛠 Tech Stack

| Component | Version |
|---|---|
| Python | 3.10+ |
| Streamlit | 1.28.0 |
| llama‑cpp‑python | 0.2.56 |
| Hugging Face Hub | 0.20.1 |
| Model | Microsoft Phi‑2 Q4_K_M (~1.6 GB) |

---

## 🚀 Local Setup

### 1. Clone / copy the project

```bash
git clone <your-repo-url>
cd MiniChat
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows users:** `llama-cpp-python==0.2.56` ships pre‑built wheels. No compiler is required.
> **Note for Linux/macOS users:** If a wheel is unavailable for your platform, install `cmake` and a C++ compiler first.

### 4. Run the app

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

### 5. First‑run model download

On the very first launch, the app will download `phi-2.Q4_K_M.gguf` (~1.6 GB) from Hugging Face Hub into the local `.model_cache/` folder. This only happens once.

---

## ☁️ Deploy to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **New Space**.
2. Name your space and select **Streamlit** as the SDK.
3. Upload the following files to the Space’s repo:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `README.md`
4. Wait a few minutes — the Space will build, install system dependencies (`cmake`, `build-essential`), install Python packages, and start automatically.
5. Open the app URL. It will download the model on first launch (~5 min) and cache it in `.model_cache/`.

> **Note:** On the Hugging Face Spaces free tier, the model will re‑download after the Space idle shutdown (storage is ephemeral). For persistent caching, upgrade to a paid Space with persistent storage, or use a Docker Space with a volume.

---

## 📁 Project Structure

```
MiniChat/
├── app.py            # Single-file Streamlit application
├── requirements.txt  # Python dependencies
├── packages.txt      # System dependencies for HF Spaces
└── README.md         # This file
```

---

## ⚙️ Configuration

Edit the constants at the top of `app.py` to tune behaviour:

| Variable | Default | Description |
|---|---|---|
| `N_CTX` | 2048 | Model context window (tokens) |
| `N_THREADS` | `os.cpu_count()` | CPU threads for inference |
| `N_GPU_LAYERS` | 0 | GPU layers (keep 0 for CPU‑only) |
| `N_BATCH` | 512 | Batch size for prompt processing |
| `DEFAULT_TEMPERATURE` | 0.7 | Sampling temperature |
| `DEFAULT_TOP_P` | 0.95 | Nucleus sampling cutoff |
| `MAX_TOKENS` | 512 | Maximum tokens per response |

---

## 🧠 Hardware Notes

| CPU | RAM | Expected performance |
|---|---|---|
| Intel i3‑1005G1 | 8 GB | ~8–15 tokens/sec (Q4_K_M) |
| Intel i5‑10400 | 16 GB | ~20–30 tokens/sec |
| AMD Ryzen 5 5600X | 16 GB | ~35–50 tokens/sec |

> **Tip:** If response speed is too slow, reduce `MAX_TOKENS` or switch to `Q3_K_M` quantization (smaller file, slightly lower quality).

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| **Model download hangs** | Check internet connectivity. The model is ~1.6 GB. Hugging Face may rate‑limit; retry after a few minutes. |
| **`pip install` fails on Linux** | Install build tools: `sudo apt install cmake g++ build-essential` then retry. |
| **Out of memory** | Close other applications. Phi‑2 Q4_K_M needs ~1.6 GB for the model + ~1 GB for inference buffers. 8 GB RAM is the minimum. |
| **Responses are cut off** | Increase `MAX_TOKENS` or check that `stop` sequences aren’t matching prematurely. |
| **Streamlit won’t start on HF Spaces** | Ensure `packages.txt` contains `cmake` and `build-essential`, and that `requirements.txt` is correct. Check the Space build logs for errors. |

---

## 📄 License

This project is provided as‑is for educational and personal use. The Phi‑2 model is subject to Microsoft’s license terms.
