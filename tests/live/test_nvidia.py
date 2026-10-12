import os
import requests
from dotenv import load_dotenv

load_dotenv()

nvidia_key = os.getenv("NVIDIA_NIM_API_KEY")

if not nvidia_key:
    print("NVIDIA_NIM_API_KEY not found in .env")
    exit(1)

base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

headers = {
    "Authorization": f"Bearer {nvidia_key}",
    "Content-Type": "application/json"
}

# Try to list models to see available models
print(f"Testing NVIDIA endpoint: {base_url}/models")
try:
    response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        models = response.json().get('data', [])
        print("Available Models:")
        for m in models:
            if "glm" in m.get('id', '').lower():
                print(f"  - {m.get('id')}")
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Try to ping chat completions with glm-5.1
print("\nTesting chat completions with glm-5.1...")
payload = {
    "model": "glm-5.1",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
}
try:
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Model is available.")
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
