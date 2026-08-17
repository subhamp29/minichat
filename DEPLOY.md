# MiniChat (BhavyamAI) — Railway Deployment Guide

This service deploys the MiniChat Streamlit frontend on Railway, configured to call a separately-deployed 9Router backend.

## Prerequisites

- A Railway account and project.
- A running 9Router service (also on Railway or elsewhere) exposing an OpenAI-compatible `/v1` endpoint.

## Railway Environment Variables

Set these in the Railway dashboard under **Settings → Variables**:

| Variable | Required | Description |
|----------|----------|-------------|
| `ROUTER_BASE_URL` | **Yes** (for remote models) | Base URL of the 9Router service, e.g. `https://9Router-production.up.railway.app/v1` |
| `ROUTER_API_KEY` | **Yes** (for remote models) | API key for authenticating with the 9Router service. |
| `ROUTER_MODELS` | No | Comma-separated list of model slugs the router should expose, e.g. `claude-opus-free,gpt-4o-mini`. If omitted, remote models will not appear in the MiniChat model selector. |
| `DEFAULT_MODEL_KEY` | No | The model key to select on first load. Must match a key in the model selector. For remote models, use the `Remote: <slug>` format, e.g. `Remote: claude-opus-free`. Defaults to `n8n Orchestrated (Supabase + Router)` if unset. |
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

- **"Remote router is not configured"**: Ensure `ROUTER_BASE_URL` and `ROUTER_API_KEY` are set in the Railway dashboard. Redeploy after adding them.
- **No remote models in the selector**: Ensure `ROUTER_MODELS` is set to a comma-separated list of model slugs.
- **Model download errors on Railway**: The remote-router path does not download GGUF files. If you see model download errors, make sure the selected model backend is `remote` or `n8n_orchestrated`, not `local_gguf`.
