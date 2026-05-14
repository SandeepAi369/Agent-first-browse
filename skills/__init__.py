"""Skills Library — Reusable action primitives.

Each Skill is a self-contained module that knows how to perform
one category of browser or system action. Skills are instantiated
by the Spawner and executed by the Executor.
"""

from skills.base import Skill
from skills.browser_navigate import NavigateSkill
from skills.browser_interact import InteractSkill
from skills.browser_extract import ExtractSkill

__all__ = ["Skill", "NavigateSkill", "InteractSkill", "ExtractSkill"]
