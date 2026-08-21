# 🚀 Full n8n + Supabase + Router Orchestration Setup Guide

This guide explains how to set up full orchestration for **MiniChat**, turning **Streamlit** into a lightweight frontend while **n8n** handles all prompt processing, chat history logging in **Supabase**, and LLM router calls.

---

## 🏗 Architecture Overview

```
 ┌──────────────────────┐   1. POST /webhook/chat (Message + Conv ID)
 │   Streamlit Frontend │ ───────────────────────────────────────────┐
 │       (app.py)       │                                           │
 └──────────┬───────────┘ ◄──────────────────────────────────────┐  │
            │               6. Returns { "response": "..." }     │  │
            │                                                    │  │
            │  (Optional) Sync history                          │  │
            ▼                                                    │  ▼
 ┌──────────────────────┐                               ┌──────────────────────┐
 │   Supabase Database  │ ◄─── 2. Insert User Message ──────│   n8n Orchestration  │
 │  (PostgreSQL / REST) │ ◄─── 3. Fetch History Context ────│       Workflow       │
 │   - conversations    │ ◄─── 5. Save AI Response ─────────│  (n8n_workflow.json) │
 │   - messages         │                               └──────────┬───────────┘
 └──────────────────────┘                                          │
                                                                   │ 4. POST /chat/completions
                                                                   ▼
                                                        ┌──────────────────────┐
                                                        │  Router / Local LLM  │
                                                        │ (OpenAI-compatible)  │
                                                        └──────────────────────┘
```

---

## 📋 Step 1: Initialize Supabase Database

1. Log into your [Supabase Dashboard](https://supabase.com) and open your project (or create a new free project).
2. Open the **SQL Editor** from the left navigation bar.
3. Copy and paste the contents of `supabase_schema.sql` into the SQL Editor and click **Run**.
4. Verify that two tables are created:
   - `conversations`
   - `messages`

---

## ⚡ Step 2: Import & Configure n8n Workflow

1. Open your **n8n instance** (e.g. `http://localhost:5678` or hosted n8n).
2. Click **Workflows** ➔ **Import from File**.
3. Select `n8n_workflow.json` from the `MiniChat` folder.
4. Set up environment variables in n8n (or configure node headers):
   - `SUPABASE_URL`: `https://your-project.supabase.co`
   - `SUPABASE_KEY`: Your Supabase `anon` or `service_role` API key.
   - `ROUTER_BASE_URL`: `http://localhost:20128/v1` (or your tunnel/remote router URL).
   - `ROUTER_API_KEY`: Your router Bearer API key.
5. Save and **Activate** the workflow in n8n.
6. Copy the **Production Webhook URL** (e.g., `http://localhost:5678/webhook/chat`).

---

## ⚙️ Step 3: Configure Environment Variables

Edit `.env` in the `MiniChat` root directory:

```env
# Router Credentials
ROUTER_BASE_URL=http://localhost:20128/v1
ROUTER_API_KEY=sk-991e47e54ac710e1-s9ip04-8c99955d
ROUTER_MODELS=claude-opus-free

# n8n & Supabase Orchestration
N8N_WEBHOOK_URL=http://localhost:5678/webhook/chat
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

---

## 🧪 Step 4: Run Automated Verification Test

Run the verification test script to test n8n webhook and Supabase connectivity:

```bash
python test_n8n_pipeline.py
```

---

## 🎈 Step 5: Launch Streamlit App

Run the Streamlit frontend:

```bash
streamlit run app.py
```

In the sidebar **Choose Model** dropdown:
- Select **`n8n Orchestrated (Supabase + Router)`**.
- Type a message and send.
- Watch n8n receive the webhook, log history to Supabase, query the Router LLM, save the assistant reply, and return the answer to Streamlit!
