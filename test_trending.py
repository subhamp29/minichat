import httpx
import urllib.request

url = "https://trends.google.com/trending/rss?geo=IN"

print("=== Testing new URL: /trending/rss?geo=IN ===")
try:
    resp = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type', 'unknown')}")
    print(f"Content-Length: {len(resp.content)} bytes")
    body = resp.text[:500]
    print(f"Body (first 500 chars):\n{body}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()
print("=== Testing old URL: /trends/trendingsearches/daily/rss?geo=IN ===")
old_url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=IN"
try:
    resp2 = httpx.get(old_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print(f"Status: {resp2.status_code}")
    print(f"Content-Type: {resp2.headers.get('content-type', 'unknown')}")
    print(f"Content-Length: {len(resp2.content)} bytes")
    body2 = resp2.text[:500]
    print(f"Body (first 500 chars):\n{body2}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
