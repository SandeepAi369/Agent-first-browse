import sys
sys.path.insert(0, '.')

import brain_graph
import prm_critic
import cognition
import brain_state
import moe_router
from workers.base_worker import WorkerAction

print("=== V17 Verification ===")
print("All modules imported successfully!")

# Check WorkerAction has new fields
fields = list(WorkerAction.model_fields.keys())
print(f"WorkerAction fields: {fields}")
assert "missing_data" in fields, "missing_data field not found!"
assert "ask_user" in WorkerAction.model_fields["action_type"].description
print("  ask_user action type: OK")
print("  missing_data field: OK")

# Check ChecklistItem weighted scoring
from prm_critic import ChecklistItem
ci = ChecklistItem(id=0, description="test", weight=2.0)
print(f"ChecklistItem pending score (w=2.0): {ci.score}")
assert ci.score == 0.0

ci.status = "done"
print(f"ChecklistItem done score (w=2.0): {ci.score}")
assert ci.score == 2.0, f"Expected 2.0, got {ci.score}"

ci2 = ChecklistItem(id=1, description="nav step", weight=0.5)
ci2.status = "done"
print(f"ChecklistItem done score (w=0.5): {ci2.score}")
assert ci2.score == 0.5

# Check PRM audit frequency
print(f"PRM_AUDIT_EVERY: {cognition.PRM_AUDIT_EVERY}")
assert cognition.PRM_AUDIT_EVERY == 2, f"Expected 2, got {cognition.PRM_AUDIT_EVERY}"

# Check URL extraction fix
from app.browser_promoter.browser_warmup import extract_target_url_from_objective
url = extract_target_url_from_objective("Navigate to Amazon (https://www.amazon.in).")
print(f"URL extraction test: 'Navigate to Amazon (https://www.amazon.in).' -> '{url}'")
assert url == "https://www.amazon.in", f"Expected 'https://www.amazon.in', got '{url}'"

url2 = extract_target_url_from_objective("Go to https://github.com/user/repo.")
print(f"URL extraction test: 'Go to https://github.com/user/repo.' -> '{url2}'")
assert url2 == "https://github.com/user/repo", f"Expected clean URL, got '{url2}'"

print("\n=== ALL V17 VERIFICATION PASSED ===")
