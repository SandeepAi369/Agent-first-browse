"""V15.1 Patch A+C Unit Test: Base-name blacklist covers all model instances."""
import sys
sys.path.insert(0, '.')

from model_registry import ProviderHealthTracker

h = ProviderHealthTracker()

# ── Test 1: _base_model_name extraction ──
assert h._base_model_name("groq:openai/gpt-oss-120b:0") == "openai/gpt-oss-120b"
assert h._base_model_name("groq:openai/gpt-oss-120b:1") == "openai/gpt-oss-120b"
assert h._base_model_name("groq:openai/gpt-oss-120b:2") == "openai/gpt-oss-120b"
assert h._base_model_name("nvidia-text:z-ai/glm-5.1:0") == "z-ai/glm-5.1"
assert h._base_model_name("gemini-text:gemma-4-31b-it:2") == "gemma-4-31b-it"
assert h._base_model_name("simple-model") == "simple-model"
print("✅ Test 1: _base_model_name extraction — PASSED")

# ── Test 2: Blacklisting :0 blocks :1 and :2 ──
error_400 = "Error code: 400 - {'error': {'message': \"invalid JSON schema for response_format: 'ChecklistEvaluation': /properties/evaluations/items: `additionalProperties` is required\"}}"
h.record_failure("groq:openai/gpt-oss-120b:0", error_msg=error_400)

# :0 should be blacklisted
assert h.is_schema_blacklisted("groq:openai/gpt-oss-120b:0", "ChecklistEvaluation"), "FAIL: :0 not blacklisted"
# :1 and :2 should ALSO be blacklisted (same base model)
assert h.is_schema_blacklisted("groq:openai/gpt-oss-120b:1", "ChecklistEvaluation"), "FAIL: :1 not blacklisted"
assert h.is_schema_blacklisted("groq:openai/gpt-oss-120b:2", "ChecklistEvaluation"), "FAIL: :2 not blacklisted"
print("✅ Test 2: All key-variants blacklisted — PASSED")

# ── Test 3: Different model NOT blacklisted ──
assert not h.is_schema_blacklisted("nvidia-text:z-ai/glm-5.1:0", "ChecklistEvaluation"), "FAIL: nvidia wrongly blacklisted"
assert not h.is_schema_blacklisted("gemini-text:gemma-4-31b-it:0", "ChecklistEvaluation"), "FAIL: gemini wrongly blacklisted"
print("✅ Test 3: Different models NOT blacklisted — PASSED")

# ── Test 4: Per-schema — CandidateSet NOT blacklisted (only ChecklistEvaluation was) ──
# gpt-oss-120b was blacklisted specifically for ChecklistEvaluation
# It should NOT be blacklisted for CandidateSet unless that also fails
assert not h.is_schema_blacklisted("groq:openai/gpt-oss-120b:0", "CandidateSet"), "FAIL: CandidateSet wrongly blacklisted"
print("✅ Test 4: Per-schema isolation — PASSED")

# ── Test 5: Catch-all fallback ──
h2 = ProviderHealthTracker()
error_generic = "400 Bad Request: schema validation error for unknown schema"
h2.record_failure("test:model:0", error_msg=error_generic)
# Should be catch-all blacklisted
assert h2.is_schema_blacklisted("test:model:0"), "FAIL: catch-all not triggered"
assert h2.is_schema_blacklisted("test:model:1"), "FAIL: catch-all not covering other instances"
print("✅ Test 5: Catch-all fallback — PASSED")

print("\n" + "="*60)
print("🎯 ALL 5 PATCH A+C UNIT TESTS PASSED")
print("="*60)
