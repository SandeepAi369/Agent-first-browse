import os
import urllib.request
import urllib.error
import ssl
import json
from langchain_core.messages import HumanMessage

def load_env_file(filepath=".env"):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'").strip('"')

def main():
    load_env_file()
    
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("WORKER_VLM_API_KEYS") or os.environ.get("OPENAI_API_KEY")
    if not groq_key:
        print("ERROR: GROQ API Key not found in .env.")
        print("Please add GROQ_API_KEY=gsk_... to your .env file to verify the model.")
        return

    # Handle comma separated keys
    groq_key = [k.strip() for k in groq_key.replace(";", ",").split(",") if k.strip()][0]

    print("Verifying Groq endpoint with 'openai/gpt-oss-120b'...")
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "System diagnostic. Reply with exactly one word: PING_OK"}]
    }
    
    data = json.dumps(payload).encode('utf-8')
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, data=data)
    req.add_header("Authorization", f"Bearer {groq_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            status_code = response.getcode()
            body = response.read().decode('utf-8')
            print(f"Connection: Success ({status_code})")
            print("200 OK: Groq endpoint verified successfully.")
            
    except urllib.error.HTTPError as e:
        print(f"Connection: Fail ({e.code})")
        print(f"Raw Error Message: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"Connection: Fail")
        print(f"Raw Error Message: {str(e)}")

if __name__ == "__main__":
    main()
