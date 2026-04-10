import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from backend.api.server import app

client = TestClient(app)

def test_root():
    print("Testing root route / ...")
    response = client.get("/")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        if "<title>" in response.text or "<div id=\"root\">" in response.text:
            print("✅ SUCCESS: Root serves React index.html")
        else:
            print("❌ FAILURE: Root does not seem to serve index.html")
            print(f"Content snippet: {response.text[:100]}")
    else:
        print(f"❌ FAILURE: Status code {response.status_code}")
        print(f"Response: {response.text}")

def test_health():
    print("\nTesting health check /api/health ...")
    response = client.get("/api/health")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "online":
            print("✅ SUCCESS: /api/health returns correct JSON")
        else:
            print("❌ FAILURE: /api/health returned unexpected data")
    else:
        print(f"❌ FAILURE: Status code {response.status_code}")

if __name__ == "__main__":
    try:
        test_root()
        test_health()
    except Exception as e:
        print(f"Verification crashed: {e}")
