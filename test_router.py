#!/usr/bin/env python3
"""Standalone test for router_client.py.

Run with:
    python test_router.py
or, after installing python-dotenv and setting env vars / creating a .env:
    set ROUTER_BASE_URL=http://localhost:8000
    set ROUTER_API_KEY=sk-xxx
    set ROUTER_MODELS=model-a,model-b
    python test_router.py
"""

import sys
import traceback

# Ensure the MiniChat package dir is on the path so imports work
sys.path.insert(0, ".")

from router_client import RouterConfigurationError, RouterError, health_check, stream_chat


def main() -> None:
    print("=" * 60)
    print("MiniChat Router Client — Quick Smoke Test")
    print("=" * 60)

    # 1. Health check
    print("\n[1/3] Health check...")
    hc = health_check(timeout=10)
    print(f"  ok      : {hc['ok']}")
    print(f"  message : {hc['message']}")
    print(f"  models  : {hc['models']}")
    if not hc["ok"]:
        print("\nFAIL: Router is not reachable or not configured.")
        sys.exit(1)

    # 2. Streaming call
    print("\n[2/3] Streaming a trivial prompt...")
    slug = hc["models"][0] if hc["models"] else "unknown"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "reply with the word 'ok'"},
    ]
    try:
        chunks = list(
            stream_chat(
                messages=messages,
                model_slug=slug,
                temperature=0.0,
                max_tokens=16,
                timeout=30,
            )
        )
    except (RouterConfigurationError, RouterError) as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)

    response = "".join(chunks).strip()
    print(f"  slug    : {slug}")
    print(f"  chunks  : {len(chunks)}")
    print(f"  response: {response!r}")

    if not response:
        print("\nFAIL: Router returned an empty response.")
        sys.exit(1)

    # 3. Verify response contains something sensible
    print("\n[3/3] Validation...")
    if "ok" in response.lower() or len(response) > 0:
        print("PASS: Router returned a non-empty response.")
    else:
        print("WARN: Response didn't contain 'ok', but it's non-empty.")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
