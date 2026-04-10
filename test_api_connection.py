import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_ping():
    print("Testing /api/ping...")
    try:
        response = requests.get(f"{BASE_URL}/api/ping", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Ping successful: {data}")
        return True
    except Exception as e:
        print(f"❌ Ping failed: {e}")
        return False

def test_market_data():
    print("\nTesting /api/market...")
    try:
        response = requests.get(f"{BASE_URL}/api/market?per_page=5", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "success" and len(data["data"]) > 0:
            print(f"✅ Market data retrieved: {len(data['data'])} items")
            return True
        else:
            print(f"❌ Market data format error: {data}")
            return False
    except Exception as e:
        print(f"❌ Market data request failed: {e}")
        return False

def test_market_summary():
    print("\nTesting /api/market/summary...")
    try:
        response = requests.get(f"{BASE_URL}/api/market/summary", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "success":
            print(f"✅ Market summary retrieved")
            return True
        else:
            print(f"❌ Market summary error: {data}")
            return False
    except Exception as e:
        print(f"❌ Market summary request failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting API Verification Tests")
    print("-" * 30)
    
    # We expect the server to be running as indicated by ADDITIONAL_METADATA
    results = [
        test_ping(),
        test_market_data(),
        test_market_summary()
    ]
    
    print("-" * 30)
    if all(results):
        print("🎉 All core API tests PASSED!")
    else:
        print("⚠️  Some tests FAILED. Check server logs.")
