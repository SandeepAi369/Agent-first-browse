"""V15.1 Patch A+C Unit Tests: Base-name schema blacklist covers all key variants."""

from __future__ import annotations

from model_registry import ProviderHealthTracker

_CHECKLIST_ERROR_400 = (
    "Error code: 400 - {'error': {'message': \"invalid JSON schema for response_format: "
    "'ChecklistEvaluation': /properties/evaluations/items: `additionalProperties` is required\"}}"
)


def test_base_model_name_extraction():
    tracker = ProviderHealthTracker()
    assert tracker._base_model_name("groq:openai/gpt-oss-120b:0") == "openai/gpt-oss-120b"
    assert tracker._base_model_name("groq:openai/gpt-oss-120b:1") == "openai/gpt-oss-120b"
    assert tracker._base_model_name("groq:openai/gpt-oss-120b:2") == "openai/gpt-oss-120b"
    assert tracker._base_model_name("nvidia-text:z-ai/glm-5.1:0") == "z-ai/glm-5.1"
    assert tracker._base_model_name("gemini-text:gemma-4-31b-it:2") == "gemma-4-31b-it"
    assert tracker._base_model_name("simple-model") == "simple-model"


def test_schema_blacklist_covers_all_key_variants():
    tracker = ProviderHealthTracker()
    tracker.record_failure("groq:openai/gpt-oss-120b:0", error_msg=_CHECKLIST_ERROR_400)

    assert tracker.is_schema_blacklisted("groq:openai/gpt-oss-120b:0", "ChecklistEvaluation")
    assert tracker.is_schema_blacklisted("groq:openai/gpt-oss-120b:1", "ChecklistEvaluation")
    assert tracker.is_schema_blacklisted("groq:openai/gpt-oss-120b:2", "ChecklistEvaluation")


def test_schema_blacklist_does_not_affect_other_models():
    tracker = ProviderHealthTracker()
    tracker.record_failure("groq:openai/gpt-oss-120b:0", error_msg=_CHECKLIST_ERROR_400)

    assert not tracker.is_schema_blacklisted("nvidia-text:z-ai/glm-5.1:0", "ChecklistEvaluation")
    assert not tracker.is_schema_blacklisted("gemini-text:gemma-4-31b-it:0", "ChecklistEvaluation")


def test_schema_blacklist_isolated_per_schema():
    tracker = ProviderHealthTracker()
    tracker.record_failure("groq:openai/gpt-oss-120b:0", error_msg=_CHECKLIST_ERROR_400)

    assert not tracker.is_schema_blacklisted("groq:openai/gpt-oss-120b:0", "CandidateSet")


def test_generic_schema_error_triggers_catch_all_blacklist():
    tracker = ProviderHealthTracker()
    tracker.record_failure(
        "test:model:0",
        error_msg="400 Bad Request: schema validation error for unknown schema",
    )

    assert tracker.is_schema_blacklisted("test:model:0")
    assert tracker.is_schema_blacklisted("test:model:1")
