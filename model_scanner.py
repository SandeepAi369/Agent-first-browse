import os
import urllib.request
import json
import ssl

def load_env_file(filepath=".env"):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'").strip('"')

def get_first_key(env_keys):
    for env_key in env_keys:
        raw = os.environ.get(env_key, "")
        if raw:
            # Handle comma separated keys
            first = [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]
            if first:
                return first[0]
    return None

def fetch_models(name, url, token, is_google=False):
    if not token:
        print(f"=== {name} (Key: NONE) ===")
        print("- Total Models Available: 0")
        print("- Error: No API key found in .env\n")
        return

    masked_token = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "***"
    print(f"=== {name} (Key: {masked_token}) ===")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if is_google:
        req_url = f"{url}?key={token}"
        req = urllib.request.Request(req_url)
    else:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            models = []
            if is_google:
                models = [m.get("name", "").replace("models/", "") for m in data.get("models", [])]
            else:
                models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            
            print(f"- Total Models Available: {len(models)}")
            if models:
                # Group models nicely
                print(f"- List: {', '.join(models)}\n")
            else:
                print("- List: (Empty)\n")

    except Exception as e:
        print(f"- Total Models Available: 0")
        print(f"- Error: {e}\n")

def main():
    load_env_file()

    providers = [
        {
            "name": "NVIDIA NIM",
            "keys": ["NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEYS"],
            "url": "https://integrate.api.nvidia.com/v1/models"
        },
        {
            "name": "GROQ",
            "keys": ["GROQ_API_KEY", "OPENAI_API_KEY", "WORKER_VLM_API_KEYS"],
            "url": "https://api.groq.com/openai/v1/models"
        },
        {
            "name": "DEEPSEEK",
            "keys": ["DEEPSEEK_API_KEY", "DEEPSEEK_API_KEYS"],
            "url": "https://api.deepseek.com/models"
        },
        {
            "name": "CEREBRAS",
            "keys": ["CEREBRAS_API_KEY", "CEREBRAS_API_KEYS"],
            "url": "https://api.cerebras.ai/v1/models"
        },
        {
            "name": "GITHUB MODELS / AZURE",
            "keys": ["GITHUB_TOKEN", "OPENAI_API_KEY"],
            "url": "https://models.inference.ai.azure.com/models"
        },
        {
            "name": "GOOGLE GEMINI",
            "keys": ["GOOGLE_API_KEY"],
            "url": "https://generativelanguage.googleapis.com/v1beta/models",
            "is_google": True
        }
    ]

    for p in providers:
        token = get_first_key(p["keys"])
        fetch_models(p["name"], p["url"], token, p.get("is_google", False))

if __name__ == "__main__":
    main()
