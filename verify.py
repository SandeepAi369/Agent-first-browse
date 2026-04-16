#!/usr/bin/env python3
"""Quick import and structural verification script."""
import sys
from pathlib import Path

# Ensure python-orchestrator is on path
orchestrator_dir = Path(__file__).resolve().parent / "python-orchestrator"
sys.path.insert(0, str(orchestrator_dir))

checks = []

try:
    from app.browser_promoter.state import AgentState, BrowserConfig, CampaignContext
    checks.append(("state imports", True))
except Exception as e:
    checks.append(("state imports", f"FAIL: {e}"))

try:
    from app.browser_promoter.graph import build_graph
    checks.append(("graph imports", True))
except Exception as e:
    checks.append(("graph imports", f"FAIL: {e}"))

try:
    from app.browser_promoter.dashboard import print_terminal_dashboard, resolve_platform_name
    checks.append(("dashboard imports", True))
except Exception as e:
    checks.append(("dashboard imports", f"FAIL: {e}"))

try:
    from app.browser_promoter.database import initialize_persistence_database
    checks.append(("database imports", True))
except Exception as e:
    checks.append(("database imports", f"FAIL: {e}"))

try:
    from app.browser_promoter.worker_planner import WorkerVisionPlanner
    checks.append(("worker_planner imports", True))
except Exception as e:
    checks.append(("worker_planner imports", f"FAIL: {e}"))

try:
    from app.browser_promoter.browser_runtime import BrowserRuntime
    checks.append(("browser_runtime imports", True))
except Exception as e:
    checks.append(("browser_runtime imports", f"FAIL: {e}"))

try:
    from app.browser_promoter.supervisor_subgraph import build_supervisor_subgraph
    checks.append(("supervisor_subgraph imports", True))
except Exception as e:
    checks.append(("supervisor_subgraph imports", f"FAIL: {e}"))

try:
    from app.browser_promoter.nodes import supervisor_node, worker_node, browser_controller_node, router_function
    checks.append(("nodes imports", True))
except Exception as e:
    checks.append(("nodes imports", f"FAIL: {e}"))

try:
    from app.logger import get_logger
    checks.append(("logger imports", True))
except Exception as e:
    checks.append(("logger imports", f"FAIL: {e}"))

try:
    from app.config import OPENAI_API_KEY, WORKER_VLM_MODEL
    checks.append(("config imports", True))
except Exception as e:
    checks.append(("config imports", f"FAIL: {e}"))

try:
    from app import __version__
    checks.append((f"app version={__version__}", True))
except Exception as e:
    checks.append(("app version", f"FAIL: {e}"))

# Graph compilation
try:
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    checks.append(("graph compile", True))
    checks.append((f"  nodes: {sorted(node_names)}", True))
except Exception as e:
    checks.append(("graph compile", f"FAIL: {e}"))

try:
    subgraph = build_supervisor_subgraph()
    sub_nodes = set(subgraph.get_graph().nodes.keys())
    checks.append(("supervisor subgraph compile", True))
    checks.append((f"  nodes: {sorted(sub_nodes)}", True))
except Exception as e:
    checks.append(("supervisor subgraph compile", f"FAIL: {e}"))

print("=" * 60)
print("Agent First IDE — Import & Structure Verification")
print("=" * 60)
all_ok = True
for name, status in checks:
    if status is True:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}: {status}")
        all_ok = False

print("=" * 60)
if all_ok:
    print("ALL CHECKS PASSED ✓")
else:
    print("SOME CHECKS FAILED ✗")
    sys.exit(1)
