import sys
sys.stdout.reconfigure(encoding="utf-8")
import httpx

resp = httpx.get("http://localhost:8000/api/trending?limit=6", timeout=15)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    topics = data.get("topics", [])
    print(f"Topics: {len(topics)}")
    for i, t in enumerate(topics[:5]):
        label = t.get("label", "")
        traffic = t.get("traffic", "")
        print(f"  {i+1}. {label} ({traffic})")
else:
    print(f"Error: {resp.text[:500]}")
