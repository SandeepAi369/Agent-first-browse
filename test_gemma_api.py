import requests
import json

GEMINI_API_KEY = "AQ.Ab8RN6I2ZSFZVt_mT89was-DL7I5n2vge3C8F-jbj1Z0WIQ34w"

# 1. Fetch available models
print("Fetching models...")
response = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}")
if response.status_code != 200:
    print(f"Error fetching models: {response.status_code}")
    print(response.text)
else:
    models = response.json().get('models', [])
    gemma_models = [m['name'] for m in models if 'gemma' in m['name'].lower()]
    print("Found Gemma models in Gemini API:")
    for gm in gemma_models:
        print(f" - {gm}")
    
    # 2. Test asking a question
    test_model = "models/gemma-2-2bd-it" # Trying a known gemma model, let's use the 27b
    test_model = "models/gemma-2-27b-it" 
    if test_model not in gemma_models:
        if gemma_models:
            test_model = gemma_models[-1]
            print(f"\n{test_model} not in list, testing using {test_model} instead.")
        else:
            print("\nNo Gemma models found! Will try gemma-2-27b-it anyway.")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/{test_model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": "Hello, answer in 5 words."}]
        }]
    }
    
    print(f"\nTesting generation with {test_model}...")
    gen_resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    if gen_resp.status_code == 200:
        out = gen_resp.json()
        try:
            text = out['candidates'][0]['content']['parts'][0]['text']
            print("\nSuccess! AI Response: ", text.strip())
        except Exception as e:
            print("Failed to parse response:", out)
    else:
        print(f"Error generating content: {gen_resp.status_code}")
        print(gen_resp.text)
