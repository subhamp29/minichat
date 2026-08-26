# MiniChat (BhavyamAI) — Railway Deployment Guide

This service deploys the MiniChat Streamlit frontend on Railway, configured to use the built-in Groq/Gemini cloud models or local GGUF models.

## Prerequisites

- A Railway account and project.

## Railway Environment Variables

Set these in the Railway dashboard under **Settings → Variables**:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | No | Groq API key for cloud LLM inference. |
| `GEMINI_API_KEY` | No | Google Gemini API key (fallback if Groq fails). |
| `DEFAULT_MODEL_KEY` | No | The model key to select on first load. Must match a key in the model selector. Defaults to `Qwen2.5 0.5B (Q4_K_M) - Ultra Fast` if unset. |
| `N8N_WEBHOOK_URL` | No | n8n webhook URL if you want to use the n8n orchestration backend. |
| `SUPABASE_URL` | No | Supabase project URL for n8n orchestration. |
| `SUPABASE_KEY` | No | Supabase anon/service key for n8n orchestration. |

## Port Binding

Railway injects a `PORT` environment variable at runtime. The Docker entrypoint script binds Streamlit to `$PORT` automatically, with a fallback to `7860` for local/HuggingFace Spaces development.

No manual port configuration is needed in the Railway dashboard.

## Docker Health Check

The container exposes a Docker `HEALTHCHECK` that pings `http://localhost:${PORT}/_stcore/health`. Railway will use this (or its own port probe) to determine if the service is healthy.

## Local / HuggingFace Spaces Compatibility

The same Docker image works locally and on HuggingFace Spaces:

- **Railway**: `PORT` is set dynamically → entrypoint binds to `$PORT`.
- **HF Spaces / local**: `PORT` is unset → entrypoint falls back to `7860`.

No code changes are required to switch between deployments.

## Troubleshooting

- **"Model download errors on Railway"**: If you see model download errors, make sure the selected model backend is `n8n_orchestrated`, `groq_gemini`, or `local_gguf`.
