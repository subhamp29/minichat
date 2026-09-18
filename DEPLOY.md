# MiniChat (BhavyamAI) — Render Deployment Guide

This service deploys the MiniChat FastAPI backend on Render.

## Prerequisites

- A Render account.
- A GitHub repository connected to Render.

## Deploy the FastAPI backend (`api/`)

### Option A: Deploy via Render Dashboard (recommended)

1. In Render, create a new **Web Service**.
2. Connect your GitHub repo: `subhamp29/minichat`.
3. Set these values:
   - **Name:** `minichat-api`
   - **Root Directory:** `api`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables in **Environment**:
   | Variable | Required | Description |
   |----------|----------|-------------|
   | `GROQ_API_KEY` | No | Groq API key for cloud LLM inference. |
   | `GEMINI_API_KEY` | No | Google Gemini API key (fallback if Groq fails). |
   | `DEFAULT_MODEL_KEY` | No | The model key to select on first load. Must match a key in the model selector. Defaults to `Qwen2.5 0.5B (Q4_K_M) - Ultra Fast` if unset. |
   | `N8N_WEBHOOK_URL` | No | n8n webhook URL if you want to use the n8n orchestration backend. |
   | `SUPABASE_URL` | No | Supabase project URL for n8n orchestration. |
   | `SUPABASE_KEY` | No | Supabase anon/service key for n8n orchestration. |
5. Click **Create Web Service**.

Render will build and deploy automatically. Your backend URL will look like:
```
https://minichat-api.onrender.com
```

### Option B: Deploy via `render.yaml` blueprint

If you want to deploy from a blueprint file, create `api/render.yaml`:

```yaml
services:
  - type: web
    name: minichat-api
    env: python
    rootDir: api
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: DEFAULT_MODEL_KEY
        value: Qwen2.5 0.5B (Q4_K_M) - Ultra Fast
      - key: N8N_WEBHOOK_URL
        sync: false
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false
```

Then click **New** → **Blueprint** in the Render dashboard and paste your repo URL.

## Updating the frontend

After deployment, update `bhavyam-frontend/.env.production` with your Render backend URL:

```
NEXT_PUBLIC_API_BASE_URL=https://minichat-api.onrender.com
```

Then rebuild and redeploy the frontend:
```bash
cd D:\ai\bhavyam-frontend
npm run build
```

## CORS

The backend CORS middleware allows localhost origins by default. After deploying to Render, update the CORS regex in `api/main.py` to include your frontend origin, or use the broader regex already committed.

## Notes

- Render free tier services spin down after 15 minutes of inactivity. The first request after inactivity may take a few seconds to wake up.
- For always-on backend, upgrade to Render’s paid tier ($7/month).
- Render automatically sets the `PORT` environment variable; the start command uses `$PORT` so no code changes are needed.
