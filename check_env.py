import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
print("GEMINI_API_KEY:", (os.getenv("GEMINI_API_KEY") or "NOT SET")[:25])
print("GOOGLE_API_KEY:", (os.getenv("GOOGLE_API_KEY") or "NOT SET")[:25])
print("VISION_GOOGLE_API_KEY:", (os.getenv("VISION_GOOGLE_API_KEY") or "NOT SET")[:25])
print("VISION_GEMINI_API_KEY:", (os.getenv("VISION_GEMINI_API_KEY") or "NOT SET")[:25])
print("VISION_MODEL:", os.getenv("VISION_MODEL", "NOT SET"))
print("WORKER_VLM_API_KEY:", (os.getenv("WORKER_VLM_API_KEY") or "NOT SET")[:25])
print("OPENAI_API_KEY:", (os.getenv("OPENAI_API_KEY") or "NOT SET")[:25])
print("WORKER_VLM_MODEL:", os.getenv("WORKER_VLM_MODEL", "NOT SET"))
print("SUPERVISOR_LLM_MODEL:", os.getenv("SUPERVISOR_LLM_MODEL", "NOT SET"))
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2", "NOT SET"))
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT", "NOT SET"))
print("AUTONOMOUS_CONTINUATION:", os.getenv("AUTONOMOUS_CONTINUATION", "NOT SET"))
