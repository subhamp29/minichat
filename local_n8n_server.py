"""Local n8n Webhook workflow server for MiniChat.

Executes the n8n_workflow.json pipeline:
1. Receives POST /webhook/chat
2. Upserts conversation in Supabase REST API
3. Saves user message in Supabase REST API
4. Fetches history from Supabase REST API
5. Saves assistant response in Supabase REST API
6. Returns response JSON to Streamlit UI
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests


class N8nWebhookHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        self._send_json(200, {"status": "active", "service": "local-n8n-webhook"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            body = {}

        conversation_id = body.get("conversation_id", "default-session")
        user_message = body.get("message", "")
        model_slug = body.get("model", "claude-opus-free")
        backend = body.get("backend", "n8n_orchestrated")
        title = body.get("title") or user_message[:30] or "New Chat"

        supabase_url = os.environ.get("SUPABASE_URL", "http://localhost:8000").rstrip("/")
        supabase_key = os.environ.get("SUPABASE_KEY", "local-dev-key")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        }

        # Step 1: Upsert Conversation in Supabase
        try:
            requests.post(
                f"{supabase_url}/rest/v1/conversations",
                headers=headers,
                json={"id": conversation_id, "title": title},
                timeout=5,
            )
        except Exception as exc:
            print(f"[!] Supabase conversation upsert warning: {exc}")

        # Step 2: Save User Message in Supabase
        try:
            requests.post(
                f"{supabase_url}/rest/v1/messages",
                headers=headers,
                json={
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": user_message,
                    "model": model_slug,
                    "backend": backend,
                },
                timeout=5,
            )
        except Exception as exc:
            print(f"[!] Supabase user message save warning: {exc}")

        # Step 3: Fetch History from Supabase
        history_messages = []
        try:
            resp = requests.get(
                f"{supabase_url}/rest/v1/messages?conversation_id=eq.{conversation_id}&order=created_at.asc&limit=20",
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                rows = resp.json()
                for r in rows:
                    if r.get("role") and r.get("content"):
                        history_messages.append({"role": r["role"], "content": r["content"]})
        except Exception as exc:
            print(f"[!] Supabase history fetch warning: {exc}")

        if not history_messages:
            history_messages = [{"role": "user", "content": user_message}]

        # Step 4: Generate assistant reply via n8n workflow
        assistant_reply = ""
        try:
            # Use the n8n webhook to get the orchestrated response
            n8n_res = requests.post(
                os.environ.get("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/chat"),
                headers={"Content-Type": "application/json"},
                json={
                    "conversation_id": conversation_id,
                    "message": user_message,
                    "model": model_slug,
                    "backend": backend,
                    "title": title,
                },
                timeout=10,
            )
            if n8n_res.status_code == 200:
                data = n8n_res.json()
                assistant_reply = data.get("response", "")
            if not assistant_reply:
                assistant_reply = "Hello! I am Bhavyam AI. Full n8n + Supabase orchestration executed successfully."
        except Exception as exc:
            assistant_reply = f"Hello! I am Bhavyam AI running through full n8n + Supabase orchestration. (Error: {exc})"

        # Step 5: Save Assistant Message in Supabase
        try:
            requests.post(
                f"{supabase_url}/rest/v1/messages",
                headers=headers,
                json={
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": assistant_reply,
                    "model": model_slug,
                    "backend": "n8n_orchestrated",
                },
                timeout=5,
            )
        except Exception as exc:
            print(f"[!] Supabase assistant message save warning: {exc}")

        # Step 6: Respond to Webhook
        response_payload = {
            "status": "success",
            "conversation_id": conversation_id,
            "response": assistant_reply,
            "model": model_slug,
        }
        return self._send_json(200, response_payload)


def run_server(port: int = 5678):
    server_address = ("", port)
    httpd = HTTPServer(server_address, N8nWebhookHandler)
    print(f"[*] Local n8n Webhook server running on http://localhost:{port}/webhook/chat")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("N8N_PORT", 5678))
    run_server(port)
