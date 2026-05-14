"""THE SPAWNER — Dynamic Agent Factory.

Takes a TaskNode from the DAG and instantiates the correct Skill
to execute it. This is the bridge between the CEO's abstract plan
and the concrete skill implementations.

Inspired by AutoGen's actor model where agents are independent
units that receive messages and produce results.
"""

from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import Page

from orchestrator.actions import TaskNode, ActionType
from orchestrator.event_log import EventLog
from skills.base import Skill, SkillResult
from skills.browser_navigate import NavigateSkill
from skills.browser_interact import InteractSkill
from skills.browser_extract import ExtractSkill

logger = logging.getLogger("orchestrator.spawner")


# ═══════════════════════════════════════════════════════════════════════════════
#  Skill Registry — Maps action types to Skill classes
# ═══════════════════════════════════════════════════════════════════════════════

_SKILL_MAP: dict[str, type[Skill]] = {
    "navigate":   NavigateSkill,
    "click":      InteractSkill,
    "type":       InteractSkill,
    "scroll":     InteractSkill,
    "extract":    ExtractSkill,
    "execute_js": ExtractSkill,     # JS eval is a mode of ExtractSkill
    "screenshot": ExtractSkill,     # Screenshot is a mode of ExtractSkill
    "wait":       None,             # Handled inline by the Executor
    "done":       None,             # Handled inline by the Executor
}


class Spawner:
    """Dynamic Agent Factory — Creates Skill instances for DAG nodes.

    The Spawner's job is simple but critical: given a TaskNode,
    it selects the right Skill class, instantiates it with the
    current Page context, and returns it ready for execution.
    """

    def __init__(self, page: Page) -> None:
        self._page = page
        self._skill_cache: dict[str, Skill] = {}

    def spawn(self, node: TaskNode) -> Skill | None:
        """Instantiate the correct Skill for a TaskNode.

        Returns None for actions that don't need a skill (wait, done).
        """
        skill_class = _SKILL_MAP.get(node.action)

        if skill_class is None:
            return None

        # Reuse instances of the same class to avoid re-initialization
        class_name = skill_class.__name__
        if class_name not in self._skill_cache:
            self._skill_cache[class_name] = skill_class(self._page)

        return self._skill_cache[class_name]

    def get_params(self, node: TaskNode) -> dict[str, Any]:
        """Prepare the parameters for skill execution.

        Some action types need parameter transformation:
        - click/type/scroll → need "interaction" key for InteractSkill
        - execute_js → needs "mode" key for ExtractSkill
        - screenshot → needs "mode" key for ExtractSkill
        """
        params = dict(node.params)

        if node.action in ("click", "type", "scroll"):
            params.setdefault("interaction", node.action)

            # Normalize "clear_and_type" pattern
            if node.action == "type" and params.get("clear_first", False):
                params["interaction"] = "clear_and_type"

        elif node.action == "execute_js":
            params["mode"] = "js_eval"

        elif node.action == "screenshot":
            params["mode"] = "screenshot"

        return params

    async def execute_node(self, node: TaskNode, event_log: EventLog) -> SkillResult:
        """Execute a single DAG node using the appropriate skill.

        This is the primary interface used by the Executor.

        Returns:
            SkillResult from the skill execution.
        """
        import asyncio

        # Handle special actions inline
        if node.action == "wait":
            duration_ms = node.params.get("duration_ms", 1000)
            await asyncio.sleep(duration_ms / 1000.0)
            result = SkillResult(
                success=True,
                summary=f"Waited {duration_ms}ms",
                data={"duration_ms": duration_ms},
            )
            event_log.action(node.id, f"Wait {duration_ms}ms")
            event_log.observe(node.id, result.summary)
            return result

        if node.action == "done":
            result = SkillResult(
                success=True,
                summary="Task marked as complete",
                data=node.params,
            )
            event_log.action(node.id, "Done")
            event_log.observe(node.id, "Task complete")
            return result

        # Spawn and execute the skill
        skill = self.spawn(node)
        if skill is None:
            return SkillResult(
                success=False,
                summary=f"No skill found for action: {node.action}",
                error=f"Unknown action: {node.action}",
            )

        params = self.get_params(node)

        # Log the action
        event_log.action(
            node.id,
            f"{node.action}: {node.description}",
            params=params,
        )

        # Execute
        logger.info(
            "⚡ Executing [%s] %s: %s",
            node.id, node.action, node.description[:80],
        )
        result = await skill.run(params)

        # Log the observation
        event_log.observe(
            node.id,
            result.summary,
            success=result.success,
            data=result.data,
            error=result.error,
        )

        return result
