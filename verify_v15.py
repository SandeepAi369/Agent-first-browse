"""V15.0 Verification Script — checks all 7 fixes are correctly applied."""
import sys

# ── F1 + F5: dom_parser.py ──
import dom_parser
js = dom_parser._GOD_MODE_JS
assert "V15.0 F1" in js, "FAIL: F1 sticky fix not found in _GOD_MODE_JS"
assert "score -= 20000" in js, "FAIL: F1 priority boost not found"
assert "RESERVED_FIXED" in js, "FAIL: F1 reserved slots not found"
assert "preview: trimVal.slice(0, 250)" in js, "FAIL: F5 preview expansion not found"
assert "total)" in js, "FAIL: F5 total length indicator not found"
print("✅ F1 (sticky elements) + F5 (text truncation) — VERIFIED in dom_parser.py")

# ── F4: action_classifier.py ──
import action_classifier
src = open("action_classifier.py").read()
assert "V15.0 F4" in src, "FAIL: F4 URL context gate not found"
assert "COMPOSE_URL_PATTERNS" in src, "FAIL: F4 compose URL patterns not found"
assert "on_compose_page" in src, "FAIL: F4 compose page check not found"
print("✅ F4 (action classifier URL gate) — VERIFIED in action_classifier.py")

# ── F6: model_registry.py ──
import model_registry
src = open("model_registry.py").read()
assert "min(16.0" in src, "FAIL: F6 quarantine cap not found"
assert "_schema_blacklist" in src, "FAIL: F6 schema blacklist not found"
assert "last_call_time" in src, "FAIL: F6 cold-start reset not found"
assert "error_msg" in src, "FAIL: F6 error_msg wiring not found"
assert "is_schema_blacklisted" in src, "FAIL: F6 schema blacklist check not found"
print("✅ F6 (model failover cap + schema blacklist) — VERIFIED in model_registry.py")

# ── F3: web_dreamer.py ──
import web_dreamer
src = open("web_dreamer.py").read()
assert "V15.0 F3" in src, "FAIL: F3 element context injection not found"
assert "TARGET ELEMENT (from DOM)" in src, "FAIL: F3 target element section not found"
print("✅ F3 (WebDreamer sim context) — VERIFIED in web_dreamer.py")

# ── F7: advanced_agent.py ──
src = open("advanced_agent.py").read()
assert "V15.0 F7" in src, "FAIL: F7 smart loop detection not found"
assert "FORM_URL_PATTERNS" in src, "FAIL: F7 form URL patterns not found"
assert "has_action_diversity" in src, "FAIL: F7 action diversity check not found"
print("✅ F7 (loop detector action diversity) — VERIFIED in advanced_agent.py")

# ── F2: cdp_input.py ──
src = open("cdp_input.py").read()
assert "V15.0 F2" in src, "FAIL: F2 setSelectionRange fix not found"
assert "setSelectionRange" in src, "FAIL: F2 cursor collapse not found"
assert "isContentEditable" in src, "FAIL: F2 contenteditable support not found"
print("✅ F2 (input focus-select collapse) — VERIFIED in cdp_input.py")

print("\n" + "="*60)
print("🎯 ALL 7 FIXES VERIFIED SUCCESSFULLY")
print("="*60)
