"""CognitiveCore — PlanState & WorkingMemory for Agent First IDE v2.0.

Fixes:
  B-01  Purpose Amnesia — PlanState persists the mission + plan across all steps
  B-04  Raw String Buffer — WorkingMemory provides semantic/spatial/failure memory
  B-10  Dead Checkpoints — PlanState.from_checkpoints() wraps _decompose_objective output

Research Sources:
  - SWE-agent (arXiv 2405.15793): persistent plan + history compression
  - MemGPT (arXiv 2310.08560): tiered memory with eviction
  - Generative Agents (arXiv 2304.03442): episodic → reflective compression

Usage:
    plan = PlanState.from_checkpoints("Post a tweet", ["open x.com", "compose", ...])
    memory = WorkingMemory()

    # Every step:
    system_prompt = BASE + plan.render() + memory.render_facts()
    user_prompt = CONTEXT + memory.compress_history()

    # After action:
    plan.advance()
    memory.note("title_filled", "true")
    memory.record_failure("click(432,287)", "grounding: no element nearby")
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
#  PlanState — The agent's persistent, mutable mission plan
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanState:
    """Living plan object that persists across all steps.

    Injected into the system prompt every step so the LLM ALWAYS knows:
    - WHY it exists (mission)
    - WHAT it has done (completed steps)
    - WHAT it is doing now (current step)
    - WHAT remains (pending steps)

    SWE-agent pattern: plan is in the system prompt (stable anchor),
    compressed history is in the user turn.
    """

    mission: str
    steps: list[dict] = field(default_factory=list)
    current_step_id: int = 0
    reflections: list[str] = field(default_factory=list)
    checklist: list[Any] = field(default_factory=list)  # Phase 2: PRM ChecklistItems

    @property
    def checklist_progress(self) -> float:
        """Aggregate progress from PRM checklist (0.0 to 1.0)."""
        if not self.checklist:
            return 0.0
        done = sum(1 for item in self.checklist
                   if getattr(item, 'status', '') == 'done')
        return done / len(self.checklist)

    @classmethod
    def from_checkpoints(cls, mission: str, checkpoints: list[str]) -> PlanState:
        """Convert _decompose_objective output into a living plan.

        This is the fix for B-10: checkpoints are no longer thrown away.
        They become plan.steps and are tracked/updated throughout execution.
        """
        steps = []
        for i, desc in enumerate(checkpoints):
            steps.append({
                "id": i,
                "desc": desc.strip(),
                "status": "pending",  # pending | active | done | failed
            })
        if steps:
            steps[0]["status"] = "active"
        return cls(mission=mission, steps=steps, current_step_id=0)

    def _current(self) -> dict | None:
        """Return the current active step, or None if all done."""
        for s in self.steps:
            if s["status"] in ("active", "in_progress"):
                return s
        return None

    def advance(self) -> None:
        """Mark current step done and activate the next pending step."""
        cur = self._current()
        if cur:
            cur["status"] = "done"
        # Find next pending
        for s in self.steps:
            if s["status"] == "pending":
                s["status"] = "in_progress"  # v2.1: explicit in_progress (from report B)
                self.current_step_id = s["id"]
                return

    def mark_failed(self, reason: str = "") -> None:
        """Mark current step as failed."""
        cur = self._current()
        if cur:
            cur["status"] = "failed"
            if reason:
                cur["fail_reason"] = reason

    def replan(self, new_steps: list[str]) -> None:
        """Replace all remaining pending steps with new ones.

        Used when the critic triggers a replan after stagnation.
        Keeps completed/failed steps for history.
        """
        # Remove all pending steps
        self.steps = [s for s in self.steps if s["status"] in ("done", "failed")]
        # Add new steps
        base_id = max((s["id"] for s in self.steps), default=-1) + 1
        for i, desc in enumerate(new_steps):
            status = "active" if i == 0 else "pending"
            self.steps.append({
                "id": base_id + i,
                "desc": desc.strip(),
                "status": status,
            })
        if new_steps:
            self.current_step_id = base_id

    def add_reflection(self, reflection: str) -> None:
        """Add a Reflexion-style self-critique (kept to last 5)."""
        self.reflections.append(reflection)
        if len(self.reflections) > 5:
            self.reflections = self.reflections[-5:]

    @property
    def progress_pct(self) -> int:
        """Percentage of steps completed."""
        if not self.steps:
            return 0
        done = sum(1 for s in self.steps if s["status"] == "done")
        return int(done / len(self.steps) * 100)

    @property
    def is_complete(self) -> bool:
        """True if all steps are done."""
        return all(s["status"] == "done" for s in self.steps) if self.steps else False

    def render(self) -> str:
        """Render the plan for injection into the system prompt.

        Output is kept concise (<500 chars) to avoid flooding the context.
        """
        done = [s for s in self.steps if s["status"] == "done"]
        active = self._current()
        pending = [s for s in self.steps if s["status"] == "pending"]
        failed = [s for s in self.steps if s["status"] == "failed"]

        lines = [f"MISSION: {self.mission[:200]}"]
        lines.append(f"PROGRESS: {self.progress_pct}% ({len(done)}/{len(self.steps)} steps)")

        if done:
            done_str = "; ".join(s["desc"][:50] for s in done[-3:])  # Last 3 done
            if len(done) > 3:
                done_str = f"...({len(done)-3} earlier); " + done_str
            lines.append(f"DONE: {done_str}")

        if active:
            lines.append(f"CURRENT: ▶ {active['desc']}")
        elif not pending:
            lines.append("CURRENT: All steps completed — verify and finish")

        if pending:
            remaining = "; ".join(s["desc"][:40] for s in pending[:3])
            if len(pending) > 3:
                remaining += f"; ...+{len(pending)-3} more"
            lines.append(f"REMAINING: {remaining}")

        if failed:
            lines.append(f"FAILED: {'; '.join(s['desc'][:40] for s in failed[-2:])}")

        if self.reflections:
            lines.append(f"REFLECTION: {self.reflections[-1][:150]}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  WorkingMemory — Structured, queryable memory replacing raw string deque
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkingMemory:
    """Structured working memory with semantic facts, spatial data, and failure tracking.

    MemGPT pattern: tiered storage with eviction.
    - facts: semantic key-value store ("title_filled" → "true")
    - page_map: spatial memory of elements per URL
    - failures: structured failure tracking with counts
    - episodic: raw step log (last 15, then compressed)
    """

    facts: dict[str, str] = field(default_factory=dict)
    page_map: dict[str, list[dict]] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    episodic: deque = field(default_factory=lambda: deque(maxlen=15))
    _token_budget: int = 4000  # approximate token limit for memory in prompt

    def failure_probability(self, action: str) -> float:
        """Predict failure probability based on past failures (from report B).
        Returns 0.0 (no history) to 0.95 (almost certainly fails)."""
        for f in self.failures:
            if f["action"] == action:
                return min(f["count"] / (f["count"] + 2), 0.95)  # Bayesian prior
        return 0.0

    def evict_if_needed(self) -> None:
        """MemGPT-style eviction when memory approaches budget.
        70% = warning (summarize old episodic to facts).
        100% = flush (drop oldest facts, compress harder).
        """
        usage = self._estimate_usage()
        if usage > self._token_budget:
            # Flush: drop oldest episodic entries
            while len(self.episodic) > 5 and self._estimate_usage() > self._token_budget:
                old = self.episodic.popleft()
                self.note(f"compressed_{old['step']}", f"{old['action']} {old['outcome']}")
            # Drop old facts if still over
            while len(self.facts) > 15 and self._estimate_usage() > self._token_budget:
                oldest = next(iter(self.facts))
                del self.facts[oldest]
        elif usage > self._token_budget * 0.7:
            # Warning: compress old episodic to facts
            while len(self.episodic) > 8:
                old = self.episodic.popleft()
                self.note(f"compressed_{old['step']}", f"{old['action']} {old['outcome']}")

    def _estimate_usage(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        total = sum(len(k) + len(v) for k, v in self.facts.items())
        total += sum(len(str(e)) for e in self.episodic)
        total += sum(len(str(f)) for f in self.failures)
        return total // 4

    def note(self, key: str, value: str) -> None:
        """Remember a semantic fact. Overwrites existing value for same key."""
        self.facts[key] = value

    def recall(self, key: str) -> str | None:
        """Query a semantic fact by key."""
        return self.facts.get(key)

    def record_step(self, step_num: int, action: str, outcome: str,
                    screen_hint: str = "") -> None:
        """Record a step in episodic memory."""
        entry = {
            "step": step_num,
            "action": action,
            "outcome": outcome,
        }
        if screen_hint:
            entry["screen"] = screen_hint[:80]
        self.episodic.append(entry)

    def record_failure(self, action: str, why: str) -> None:
        """Track a failure with deduplication and counting.

        If the same action has failed before, increment count.
        This lets the agent learn: "clicking (432,287) failed 3 times — stop trying."
        """
        for f in self.failures:
            if f["action"] == action:
                f["count"] += 1
                f["last_why"] = why
                return
        self.failures.append({
            "action": action,
            "why": why,
            "last_why": why,
            "count": 1,
        })

    def update_page_map(self, url: str, elements: list[dict]) -> None:
        """Store spatial memory of the current page's interactive elements.

        Keeps only the last 3 pages to prevent memory bloat.
        """
        self.page_map[url] = [
            {"ref": el.get("ref", ""), "kind": el.get("kind", ""),
             "name": el.get("name", "")[:40], "x": el.get("x", 0), "y": el.get("y", 0)}
            for el in elements[:30]  # Cap at 30 elements per page
        ]
        # Evict old pages
        if len(self.page_map) > 3:
            oldest_key = next(iter(self.page_map))
            del self.page_map[oldest_key]

    def render_facts(self) -> str:
        """Render facts for injection into the system prompt."""
        if not self.facts:
            return ""
        return "\n".join(f"• {k}: {v}" for k, v in list(self.facts.items())[-10:])

    def render_failures(self) -> str:
        """Render failure summary for the prompt."""
        if not self.failures:
            return ""
        recent = [f for f in self.failures if f["count"] >= 2]
        if not recent:
            return ""
        return "⚠️ REPEATED FAILURES: " + "; ".join(
            f"{f['action']} failed {f['count']}x ({f['last_why'][:40]})"
            for f in recent[-3:]
        )

    def compress_history(self) -> str:
        """SWE-agent style: last 5 steps verbatim, older ones compressed.

        This keeps the context window manageable while preserving
        recent detail for accurate decision-making.
        """
        if not self.episodic:
            return "[]"

        entries = list(self.episodic)
        result_lines = []

        # Older entries: compressed to one line each
        if len(entries) > 5:
            for e in entries[:-5]:
                result_lines.append(
                    f"Step {e['step']}: {e['action']} {e['outcome']}"
                )

        # Recent 5: verbatim with screen context
        recent = entries[-5:]
        for e in recent:
            line = f"Step {e['step']}: {e['action']} {e['outcome']}"
            if e.get("screen"):
                line += f" | Screen: {e['screen']}"
            result_lines.append(line)

        # Append failure summary if any
        fail_summary = self.render_failures()
        if fail_summary:
            result_lines.append(fail_summary)

        return "\n".join(result_lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: Compute semantic hash for CriticV12 compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def compute_semantic_hash(elements: list[dict]) -> int:
    """Compute a deterministic hash of the page's element structure.

    Used by CriticV12 to detect whether the page semantically changed.
    Unlike screenshot hashes, this is immune to JPEG artifacts and
    sub-pixel rendering differences.
    """
    if not elements:
        return 0
    # Hash the structural fingerprint: role + name + approximate position
    fingerprint_parts = []
    for el in elements:
        fp = f"{el.get('role', '')}|{el.get('name', '')[:30]}|{el.get('kind', '')}"
        fingerprint_parts.append(fp)
    combined = "||".join(sorted(fingerprint_parts))
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


def dom_data_to_a11y_format(dom_data: dict) -> dict:
    """Bridge dom_parser.extract() output to CriticV12's expected a11y format.

    dom_parser returns: {"elements": [...], "element_count": N, "markdown": "..."}
    CriticV12 expects: {"elements": [...], "element_count": N, "semantic_hash": int}
    """
    elements = dom_data.get("elements", [])
    return {
        "elements": [
            {
                "ref": el.get("ref", el.get("id", f"e{i}")),
                "role": el.get("kind", "unknown"),
                "name": el.get("name", el.get("text", "")),
                "properties": {
                    "x": el.get("x", 0),
                    "y": el.get("y", 0),
                },
            }
            for i, el in enumerate(elements)
        ],
        "element_count": dom_data.get("element_count", len(elements)),
        "semantic_hash": compute_semantic_hash(elements),
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  V2.1 Infrastructure — AgentMetrics & Critic Result Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """PRE-ACT: Is this action valid to execute? (v2.1 B-03)"""
    valid: bool
    reason: str
    grounded_element: dict | None = None

@dataclass
class VerificationResult:
    """POST-ACT: Did this action achieve progress? (v2.1 B-03)"""
    progress: bool
    reason: str
    confidence: float = 0.0
    circuit_breaker: bool = False

@dataclass
class AgentMetrics:
    """Measurable quality metrics (v2.1 S-5)."""
    total_actions: int = 0
    grounding_rejects: int = 0
    critic_no_progress: int = 0
    critic_progress: int = 0
    reflexion_triggers: int = 0
    done_blocked: int = 0

    @property
    def hallucination_rate(self) -> float:
        """Fraction of actions rejected by grounding or critic."""
        if self.total_actions == 0:
            return 0.0
        return (self.grounding_rejects + self.critic_no_progress) / self.total_actions

    @property
    def keynode_completion_rate(self) -> float:
        """Fraction of progress verdicts vs total."""
        total = self.critic_progress + self.critic_no_progress
        return self.critic_progress / total if total > 0 else 0.0

