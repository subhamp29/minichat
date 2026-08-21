import time, requests

time.sleep(2)

KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFmZ2VzanlsemprbHBqcWp0ZWpxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODY5NjI1NzIsImV4cCI6MjEwMjUzODU3Mn0.uc_TXdbOKsmduFCCj_cilhALfRXb2PtXSUn5A7jvwUY"
SUPA = "https://afgesjylzjklpjqjtejq.supabase.co"
HEADERS = {"apikey": KEY, "Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
CONV_ID = "verify-end-to-end-001"

print("=" * 55)
print(" Full n8n + Supabase + Router Pipeline Verification")
print("=" * 55)

# Test 1: n8n webhook
print("\n[1] Sending message to n8n Webhook...")
try:
    r1 = requests.post(
        "http://localhost:5678/webhook/chat",
        json={"conversation_id": CONV_ID, "message": "Hello BhavyamAI! Are you working via n8n orchestration?", "model": "claude-opus-free"},
        timeout=20,
    )
    print("    HTTP Status:", r1.status_code)
    data = r1.json()
    print("    Status Field:", data.get("status"))
    print("    AI Response:", data.get("response", "")[:120])
    ok_n8n = r1.status_code == 200
except Exception as e:
    print("    ERROR:", e)
    ok_n8n = False

# Test 2: Check Supabase messages were stored
print("\n[2] Checking Supabase cloud DB for stored messages...")
try:
    r2 = requests.get(
        SUPA + "/rest/v1/messages?conversation_id=eq." + CONV_ID + "&order=created_at.asc",
        headers=HEADERS,
        timeout=10,
    )
    msgs = r2.json()
    print("    Messages in cloud DB:", len(msgs))
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")[:80]
        print(f"      [{role}]: {content}")
    ok_supa = len(msgs) >= 2
except Exception as e:
    print("    ERROR:", e)
    ok_supa = False

print("\n" + "=" * 55)
print(" Results:")
print("  n8n Webhook:     ", "[PASS]" if ok_n8n else "[FAIL]")
print("  Supabase Storage:", "[PASS]" if ok_supa else "[FAIL]")
if ok_n8n and ok_supa:
    print("\n  ALL SYSTEMS GO! Full orchestration is working.")
print("=" * 55)
