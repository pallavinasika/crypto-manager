import requests
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_URL = "http://localhost:8001"

def test_ai_chat():
    print("\n--- Testing AI Assist Chat Endpoint ---")
    
    # Needs a token. We'll try to login as Paramesh (sample user)
    # Note: In a real test we'd hit the /api/login endpoint first
    # For now, let's assume we can hit it or that we have a valid mock flow.
    
    # Since we can't easily get a JWT without a running DB/User, 
    # let's just test if the endpoint exists and returns 401 (expected without token)
    # vs 500 (crash)
    
    queries = [
        "hi",
        "bitcoin price",
        "how is the market sentiment?",
        "what is my portfolio risk?",
        "top movers today"
    ]
    
    for query in queries:
        try:
            response = requests.post(
                f"{BASE_URL}/api/ai/chat", 
                json={"message": query},
                timeout=5
            )
            print(f"Query: '{query}' -> Status: {response.status_code}")
            if response.status_code == 500:
                print(f"FAIL: Crash detected: {response.text}")
            elif response.status_code == 401:
                print("Note: Expected 401 as we didn't provide a token.")
            else:
                print(f"Response: {response.json().get('response', 'N/A')}")
        except Exception as e:
            print(f"ERROR connecting: {e}")

if __name__ == "__main__":
    test_ai_chat()
