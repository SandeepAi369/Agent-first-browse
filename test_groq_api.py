import requests

GROQ_API_KEY = "REDACTED_GROQ_API_KEY"

url = "https://api.groq.com/openai/v1/models"
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

print("Fetching models from Groq...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = response.json().get("data", [])
    print("Found models on Groq:")
    for m in models:
        print(f" - {m.get('id')}")
        
    test_model = None
    # Look for a 120b model or something matching user's request, else pick a prominent one
    target = "120b"
    for m in models:
        if target in m.get("id").lower():
            test_model = m.get("id")
            break
    
    if not test_model:
        if models:
            test_model = models[-1].get("id")
            print(f"\nCould not find '120b' model. Using {test_model} instead for testing.")
        else:
            print("No models returned.")
            exit(1)
            
    # Test generation
    chat_url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": test_model,
        "messages": [{"role": "user", "content": "Hello! Reply with exactly 5 words."}],
        "max_tokens": 100
    }
    print(f"\nTesting generation with {test_model}...")
    chat_resp = requests.post(chat_url, headers=headers, json=payload)
    if chat_resp.status_code == 200:
        reply = chat_resp.json()["choices"][0]["message"]["content"]
        print(f"\nSuccess! AI Response: {reply.strip()}")
    else:
        print(f"Error testing generation: {chat_resp.status_code}")
        print(chat_resp.text)
else:
    print(f"Error fetching models: {response.status_code}")
    print(response.text)
