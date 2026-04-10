import requests
import time

BASE_URL = "http://127.0.0.1:8005"

def test_rate_limiting():
    print("\n--- Testing Rate Limiting on /api/auth/register ---")
    print("Limit is 5/minute. Hitting it 7 times quickly...")
    for i in range(7):
        try:
            # We don't even need valid data, just hitting the route
            response = requests.post(f"{BASE_URL}/api/auth/register", json={"email": "test@test.com"}, timeout=5)
            print(f"Request {i+1}: {response.status_code}")
            if response.status_code == 429:
                print("✅ Rate limit successfully triggered (429 Too Many Requests)")
                return True
        except Exception as e:
            print(f"Error: {e}")
    print("❌ Rate limit NOT triggered.")
    return False

def test_rbac_unauthorized():
    print("\n--- Testing RBAC on /api/ai/chat (Requires Premium) ---")
    try:
        # No token provided
        response = requests.post(f"{BASE_URL}/api/ai/chat", json={"message": "hello"}, timeout=5)
        print(f"Unauthorized Request: {response.status_code}")
        if response.status_code == 401:
            print("✅ Unauthorized access correctly blocked (401 Unauthorized)")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_rate_limiting()
    test_rbac_unauthorized()
