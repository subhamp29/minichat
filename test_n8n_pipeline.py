"""Diagnostic and verification script for n8n + Supabase orchestration."""

from __future__ import annotations

import sys
import uuid
import requests
from n8n_supabase_client import (
    get_n8n_config,
    health_check_n8n,
    fetch_supabase_conversations,
    fetch_supabase_messages,
    send_to_n8n_webhook,
    N8nOrchestrationError,
)


def main():
    print("=" * 60)
    print(" MiniChat n8n + Supabase Orchestration Diagnostics")
    print("=" * 60)

    config = get_n8n_config()
    print(f"\n1. Environment Configuration:")
    print(f"   - N8N_WEBHOOK_URL : {config['webhook_url'] or '(Not set)'}")
    print(f"   - SUPABASE_URL    : {config['supabase_url'] or '(Not set)'}")
    print(f"   - SUPABASE_KEY    : {'****' + config['supabase_key'][-4:] if config['supabase_key'] else '(Not set)'}")

    # 2. Supabase REST Check
    print(f"\n3. Testing Supabase REST API Connection...")
    if config["supabase_url"] and config["supabase_key"] and "your-project" not in config["supabase_url"]:
        try:
            convs = fetch_supabase_conversations(limit=5)
            print(f"   [+] Connected to Supabase. Found {len(convs)} existing conversation(s).")
        except Exception as exc:
            print(f"   [X] Supabase query failed: {exc}")
    else:
        print("   [!] SUPABASE_URL / SUPABASE_KEY not configured with active credentials.")
        print("       (Set active credentials in .env to test Supabase persistence)")

    # 4. n8n Webhook Check
    print(f"\n4. Testing n8n Webhook Endpoint...")
    n8n_health = health_check_n8n(timeout=5)
    if n8n_health["ok"]:
        print(f"   [+] n8n server host reachable: {n8n_health['message']}")
    else:
        print(f"   [!] n8n webhook warning: {n8n_health['message']}")

    # 5. Pipeline Test Prompt (Optional)
    print(f"\n5. Full Webhook Pipeline Simulation:")
    test_conv_id = f"test-{uuid.uuid4().hex[:8]}"
    print(f"   Sending test payload to n8n webhook with session ID: {test_conv_id}...")
    try:
        res = send_to_n8n_webhook(
            message="Hello! Reply with OK if orchestration is working.",
            conversation_id=test_conv_id,
            model_slug="claude-opus-free",
            timeout=10,
        )
        print(f"   [+] n8n Webhook Response received:")
        print(f"       - Status   : {res.get('status')}")
        print(f"       - Response : {res.get('response')}")
    except N8nOrchestrationError as exc:
        print(f"   [!] Pipeline test skipped or unsuccessful: {exc}")
        print("       (Ensure n8n workflow is active in n8n UI)")

    print("\n" + "=" * 60)
    print(" Diagnostic Check Completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
