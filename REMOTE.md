# Remote Router Backend

MiniChat can answer queries by calling your own multi-model router endpoint over HTTP, in addition to running inference locally via `llama_cpp`. The remote backend is the default when configured.

## Current Router Configuration

| Setting | Value |
|---------|-------|
| **Local endpoint** | `http://localhost:20128/v1` |
| **Tunnel endpoint** | `https://rv99zpu.abc-tunnel.us/v1` |
| **Default model slug** | `claude-opus-free` (a combo that internally routes across 20+ models) |
| **Auth header** | `Authorization: Bearer <ROUTER_API_KEY>` |

## Required Environment Variables

Set these before running `streamlit run app.py`. You can put them in a `.env` file (requires `python-dotenv`) or export them in your shell.

| Variable | Purpose | Example |
|----------|---------|---------|
| `ROUTER_BASE_URL` | Base URL of your router (no trailing slash) | `http://localhost:20128/v1` |
| `ROUTER_API_KEY` | Bearer token / API key for the router | `sk-991e47e54ac710e1-s9ip04-8c99955d` |
| `ROUTER_MODELS` | (Optional) comma-separated list of additional model slugs | `claude-opus-free` |
| `ROUTER_TIMEOUT` | (Optional) Request timeout in seconds. Default: `60` | `30` |

### Example `.env` file

```env
ROUTER_BASE_URL=http://localhost:20128/v1
ROUTER_API_KEY=sk-991e47e54ac710e1-s9ip04-8c99955d
ROUTER_MODELS=claude-opus-free
ROUTER_TIMEOUT=60
```

## How It Works

1. **Model picker** — The hardcoded combo `Remote: claude-opus-free` is always available as the default remote entry. Any additional slugs from `ROUTER_MODELS` are appended as extra `Remote: <slug>` options.

2. **Request format** — The app sends an OpenAI-compatible `POST /chat/completions` with `stream: true`:
   ```json
   {
     "model": "claude-opus-free",
     "messages": [
       {"role": "system", "content": "You are Bhavyam AI..."},
       {"role": "user", "content": "Hello!"}
     ],
     "stream": true,
     "temperature": 0.1,
     "top_p": 0.95,
     "max_tokens": 512
   }
   ```

3. **Streaming** — The router should return SSE (`text/event-stream`) chunks in OpenAI format:
   ```
   data: {"choices":[{"delta":{"content":"Hello"}}]}
   data: {"choices":[{"delta":{"content":" there"}}]}
   data: [DONE]
   ```

4. **Wikipedia lookup** — The existing Wikipedia web-search path works for both local and remote backends. Factual queries are auto-detected and searched before the model is called.

5. **Offline handling** — If the remote router is unreachable, the app shows a clear `st.error` and offers a one-click fallback to the local GGUF model (`Qwen2.5 0.5B`) for that turn.

## Switching Between Local and Tunnel

To switch from the local endpoint to the tunnel endpoint, change `ROUTER_BASE_URL`:

```env
# Local
ROUTER_BASE_URL=http://localhost:20128/v1

# Tunnel
ROUTER_BASE_URL=https://rv99zpu.abc-tunnel.us/v1
```

## Switching Backends

Use the **Choose Model** dropdown in the sidebar. Remote models are prefixed with `Remote:`. Local GGUF models keep their original names. Switching models reloads the chat.

## Troubleshooting

- **"Remote router is not configured"** — Check that `ROUTER_BASE_URL` and `ROUTER_API_KEY` are set.
- **"Cannot reach the remote router"** — Verify the URL is correct and the router is running. Check firewall / proxy settings.
- **"Router returned HTTP 401/403"** — The API key is missing or invalid.
- **Empty responses** — Ensure your router streams `choices[0].delta.content` chunks. If it uses a different schema, let the app developer know before editing `router_client.py`.
