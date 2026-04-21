import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent / "python-orchestrator"))

from app.browser_promoter.worker_planner import _build_worker_llm

llm, bootstrap_error = _build_worker_llm()
print(f"LLM type: {type(llm)}")
print(f"LLM: {llm}")
if bootstrap_error:
    print(f"BOOTSTRAP ERROR: {bootstrap_error}")
if llm is None:
    print("ERROR: _build_worker_llm returned no model client!")
else:
    print("SUCCESS: Worker LLM is ready!")
