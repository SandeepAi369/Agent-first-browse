import os
import urllib.request
import urllib.error
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

def get_keys(env_keys):
    for env_key in env_keys:
        raw = os.environ.get(env_key, "")
        if raw:
            first = [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]
            if first:
                return first[0]
    return None

def test_endpoint(name, keys_to_check, url):
    print(f"\n=== {name} DIAGNOSTIC ===")
    
    token = get_keys(keys_to_check)
    
    if not token:
        print(f"- Key Present: No (Checked {keys_to_check})")
        print("- Connection: N/A")
        print("- Raw Error Message: N/A")
        print("- Verdict: Key is missing from .env")
        return

    masked = f"{token[:3]}...{token[-3:]}" if len(token) > 6 else "***"
    print(f"- Key Present: Yes ({masked})")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            status_code = response.getcode()
            body = response.read().decode('utf-8')
            print(f"- Connection: Success ({status_code})")
            print(f"- Raw Error Message: None")
            print("- Verdict: Ready to use")
            
    except urllib.error.HTTPError as e:
        status_code = e.code
        try:
            body = e.read().decode('utf-8')
            try:
                # Try to parse json to print compactly
                json_body = json.loads(body)
                err_msg = json.dumps(json_body)
            except:
                err_msg = body.strip()
        except:
            err_msg = str(e)
            
        print(f"- Connection: Fail ({status_code})")
        print(f"- Raw Error Message: {err_msg}")
        
        if status_code == 401:
            print("- Verdict: Invalid API Key")
        elif status_code == 403:
            if "balance" in err_msg.lower() or "credit" in err_msg.lower():
                print("- Verdict: Account restricted (likely zero balance)")
            else:
                print("- Verdict: Forbidden (IP Blocked or Region Restricted)")
        elif status_code == 429:
            print("- Verdict: Rate limit exceeded or out of credits")
        else:
            print("- Verdict: API Error")
            
    except urllib.error.URLError as e:
        print(f"- Connection: Fail")
        print(f"- Raw Error Message: {e.reason}")
        print("- Verdict: Network/DNS Error")
    except Exception as e:
        print(f"- Connection: Fail")
        print(f"- Raw Error Message: {str(e)}")
        print("- Verdict: Unknown Error")

def main():
    load_env_file()
    print("Starting Dedicated Forensic Diagnostic...\n")
    
    test_endpoint("GROQ", ["GROQ_API_KEY", "OPENAI_API_KEY", "WORKER_VLM_API_KEYS"], "https://api.groq.com/openai/v1/models")
    test_endpoint("CEREBRAS", ["CEREBRAS_API_KEY", "CEREBRAS_API_KEYS"], "https://api.cerebras.ai/v1/models")

if __name__ == "__main__":
    main()
