"""Show which settings the project .env provides (secrets are never printed)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SECRETS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "VISION_GOOGLE_API_KEY", "VISION_GEMINI_API_KEY",
           "WORKER_VLM_API_KEY", "OPENAI_API_KEY")
SETTINGS = ("VISION_MODEL", "WORKER_VLM_MODEL", "SUPERVISOR_LLM_MODEL", "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_PROJECT", "AUTONOMOUS_CONTINUATION")

for name in SECRETS:
    value = os.getenv(name)
    print(f"{name}:", f"set ({len(value)} chars)" if value else "NOT SET")
for name in SETTINGS:
    print(f"{name}:", os.getenv(name, "NOT SET"))
