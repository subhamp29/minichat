@echo off
title MiniChat Full Orchestration Stack (n8n + Supabase + Streamlit)
echo ============================================================
echo   Starting MiniChat Full Stack (n8n + Supabase + Streamlit)
echo ============================================================

echo [1/3] Starting Local Supabase REST API Server on http://localhost:8000 ...
start "Supabase REST Server" cmd /k "python local_supabase_server.py"

echo [2/3] Starting Local n8n Webhook Orchestration Server on http://localhost:5678 ...
start "n8n Webhook Server" cmd /k "python local_n8n_server.py"

echo [3/3] Launching Streamlit Frontend ...
start "Streamlit Frontend" cmd /k "streamlit run app.py"

echo.
echo ============================================================
echo   All 3 services are launching!
echo   Streamlit UI will open automatically in your browser.
echo ============================================================
pause
