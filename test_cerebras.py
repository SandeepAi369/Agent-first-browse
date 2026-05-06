import os
import requests
import json
import time

API_KEY = "REDACTED_CEREBRAS_API_KEY"

# The user mentioned "Cerebral" API key. We will test against Cerebras.
BASE_URL = "https://api.cerebras.ai/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("=== API Key Diagnostics ===")

print("\n1. Fetching all available models...")
resp = requests.get(f"{BASE_URL}/models", headers=headers)

if resp.status_code != 200:
    print(f"❌ Failed to fetch models. Status: {resp.status_code}")
    print(f"Response: {resp.text}")
    exit(1)

data = resp.json()
models_info = data.get("data", [])
available_models = [m["id"] for m in models_info]
print(f"✅ Authentication Successful!")
print(f"Found {len(available_models)} models:")
for m in available_models:
    print(f" - {m}")

print("\n2. Scanning and testing ALL available models...")

success_models = []
failed_models = []

for model in available_models:
    print(f"\nTesting model: '{model}'...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Respond with exactly: 'API is working'"}],
        "max_tokens": 15
    }
    
    chat_resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    if chat_resp.status_code == 200:
        content = chat_resp.json()["choices"][0]["message"]["content"]
        print(f"  ✅ SUCCESS: {content.strip()}")
        success_models.append(model)
    else:
        print(f"  ❌ FAILED (Status {chat_resp.status_code})")
        try:
            error_msg = chat_resp.json().get("error", {}).get("message", chat_resp.text)
            print(f"     Error: {error_msg}")
        except:
            print(f"     Raw: {chat_resp.text}")
        failed_models.append(model)
    
    # Slight pause to avoid spamming the rate limiter on the loop
    time.sleep(1)

print("\n=== FINAL SCAN RESULTS ===")
print(f"Successful Models: {success_models}")
print(f"Failed Models: {failed_models}")

