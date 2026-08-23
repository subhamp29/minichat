"""Local Supabase REST API server emulator for MiniChat.

Emulates PostgREST / Supabase REST endpoints:
- POST /rest/v1/conversations
- GET  /rest/v1/conversations
- POST /rest/v1/messages
- GET  /rest/v1/messages
Backed by local SQLite database (supabase_local.db).
"""

from __future__ import annotations

import json
import os
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

DB_FILE = "supabase_local.db"


def init_db():
    """Create local SQLite tables mirroring Supabase schema."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT NOT NULL,
            backend TEXT NOT NULL,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            token_count INTEGER,
            response_ms INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    conn.close()


class SupabaseRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        if path.endswith("/conversations"):
            cursor.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            result = [
                {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
                for r in rows
            ]
            conn.close()
            return self._send_json(200, result)

        elif path.endswith("/messages"):
            conv_id = ""
            if "conversation_id" in query:
                # Expecting format conversation_id=eq.<id>
                raw_id = query["conversation_id"][0]
                conv_id = raw_id.replace("eq.", "")

            if conv_id:
                cursor.execute(
                    "SELECT id, conversation_id, role, content, model, backend, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                    (conv_id,),
                )
            else:
                cursor.execute("SELECT id, conversation_id, role, content, model, backend, created_at FROM messages ORDER BY created_at ASC")

            rows = cursor.fetchall()
            result = [
                {
                    "id": r[0],
                    "conversation_id": r[1],
                    "role": r[2],
                    "content": r[3],
                    "model": r[4],
                    "backend": r[5],
                    "created_at": r[6],
                }
                for r in rows
            ]
            conn.close()
            return self._send_json(200, result)

        conn.close()
        self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            payload = {}

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        if path.endswith("/conversations"):
            cid = payload.get("id", "default")
            title = payload.get("title", "New Chat")
            cursor.execute(
                "INSERT INTO conversations (id, title, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(id) DO UPDATE SET updated_at=CURRENT_TIMESTAMP",
                (cid, title),
            )
            conn.commit()
            conn.close()
            return self._send_json(201, [{"id": cid, "title": title}])

        elif path.endswith("/messages"):
            cid = payload.get("conversation_id", "default")
            role = payload.get("role", "user")
            content = payload.get("content", "")
            model = payload.get("model", "claude-opus-free")
            backend = payload.get("backend", "n8n_orchestrated")

            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, model, backend) VALUES (?, ?, ?, ?, ?)",
                (cid, role, content, model, backend),
            )
            msg_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return self._send_json(201, [{"id": msg_id, "conversation_id": cid, "role": role, "content": content}])

        conn.close()
        self._send_json(404, {"error": "Not Found"})


def run_server(port: int = 8000):
    init_db()
    server_address = ("", port)
    httpd = HTTPServer(server_address, SupabaseRequestHandler)
    print(f"[*] Local Supabase REST API server running on http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("SUPABASE_PORT", 8000))
    run_server(port)
