"""Adversarial Content Critic — Red-team review before publishing.

Agent A writes the content (from the prompt). Agent B (this module)
reviews, criticizes, and rewrites it to sound like a real human developer,
not an AI bot. Integrated into the agent loop before typing body content.
"""

from __future__ import annotations

import json
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.logger import get_logger

logger = get_logger("content_critic")

# ═══════════════════════════════════════════════════════════════════════════════
#  AI Jargon Blocklist — phrases that immediately flag content as AI-generated
# ═══════════════════════════════════════════════════════════════════════════════
JARGON_BLOCKLIST = [
    "leverage", "cutting-edge", "game-changer", "revolutionize",
    "unlock the power of", "harness the power", "delve into",
    "in today's fast-paced", "in the ever-evolving", "paradigm shift",
    "synergy", "disrupt", "ecosystem", "empower", "seamlessly",
    "comprehensive solution", "robust solution", "state-of-the-art",
    "transformative", "innovative solution", "next-generation",
    "groundbreaking", "unparalleled", "world-class", "best-in-class",
    "deep dive", "holistic approach", "at scale", "end-to-end",
    "it's worth noting", "it's important to note", "in conclusion",
    "without further ado", "let's dive in", "buckle up",
    "needless to say", "as we all know", "at the end of the day",
]


class CriticVerdict(BaseModel):
    """Structured output from the content critic."""
    approved: bool = Field(description="True if the content is ready to publish as-is")
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues found (e.g., 'Too many exclamation marks')"
    )
    rewritten_content: str = Field(
        default="",
        description="Cleaned/improved version of the content. Empty if approved as-is."
    )
    tone_score: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="0.0=robotic/corporate, 1.0=natural human developer voice"
    )


def _scan_jargon(content: str) -> list[str]:
    """Scan content for AI jargon blocklist matches."""
    found = []
    lower = content.lower()
    for phrase in JARGON_BLOCKLIST:
        if phrase in lower:
            found.append(phrase)
    return found


class ContentCritic:
    """Pre-publish content reviewer using the LLM failover chain.
    
    Usage:
        critic = ContentCritic()
        verdict = await critic.review(content, platform, failover_chain)
        if not verdict.approved:
            content = verdict.rewritten_content
    """

    def __init__(self):
        self._review_count = 0

    async def review(
        self,
        content: str,
        platform: str,
        failover_chain: list,
        invoke_fn=None,
    ) -> CriticVerdict:
        """Review content before publishing. Returns a CriticVerdict.
        
        Args:
            content: The article/post body to review
            platform: Target platform (e.g., "dev.to", "medium.com", "x.com")
            failover_chain: List of LLM clients for failover
            invoke_fn: The _invoke_with_failover function from advanced_agent
        """
        # Phase 1: Static analysis (instant, no LLM needed)
        jargon_hits = _scan_jargon(content)
        static_issues = []
        
        if jargon_hits:
            static_issues.append(
                f"AI jargon detected: {', '.join(jargon_hits[:5])}. "
                "Remove these — they immediately flag content as AI-generated."
            )

        exclamation_count = content.count("!")
        if exclamation_count > 3:
            static_issues.append(
                f"Too many exclamation marks ({exclamation_count}). "
                "Real developers rarely use more than 1-2 per post."
            )

        emoji_count = sum(1 for c in content if ord(c) > 0x1F600)
        if emoji_count > 5:
            static_issues.append(
                f"Excessive emojis ({emoji_count}). Keep it professional."
            )

        # Phase 2: LLM-based deep review (uses failover chain)
        if invoke_fn and failover_chain:
            try:
                verdict = await self._llm_review(
                    content, platform, static_issues, failover_chain, invoke_fn
                )
                self._review_count += 1
                logger.info(
                    "Critic review #%d: approved=%s, tone=%.2f, issues=%d",
                    self._review_count, verdict.approved,
                    verdict.tone_score, len(verdict.issues),
                )
                return verdict
            except Exception as e:
                logger.warning("LLM critic review failed: %s — using static analysis only", e)

        # Fallback: static analysis only
        if static_issues:
            # Do a simple cleanup — remove jargon phrases
            cleaned = content
            for phrase in jargon_hits:
                cleaned = cleaned.replace(phrase, "")
                cleaned = cleaned.replace(phrase.title(), "")

            return CriticVerdict(
                approved=False,
                issues=static_issues,
                rewritten_content=cleaned.strip(),
                tone_score=0.5,
            )

        return CriticVerdict(
            approved=True,
            issues=[],
            rewritten_content="",
            tone_score=0.8,
        )

    async def _llm_review(
        self,
        content: str,
        platform: str,
        static_issues: list[str],
        failover_chain: list,
        invoke_fn,
    ) -> CriticVerdict:
        """Use the LLM to deeply review content tone and authenticity."""
        static_context = ""
        if static_issues:
            static_context = (
                "\nSTATIC ANALYSIS FINDINGS (already detected):\n"
                + "\n".join(f"- {issue}" for issue in static_issues)
            )

        system_prompt = (
            "You are a ruthless content critic specializing in developer marketing.\n"
            "Your job is to ensure promotional posts sound like a REAL human developer, "
            "not an AI chatbot or corporate marketing team.\n\n"
            "RED FLAGS to check:\n"
            "- Corporate jargon (leverage, synergy, cutting-edge, game-changer, etc.)\n"
            "- Excessive enthusiasm or exclamation marks\n"
            "- Generic filler sentences that add no value\n"
            "- Overly perfect grammar (real devs make minor mistakes)\n"
            "- Claims without specifics (say '420 URLs in 10s' not 'blazing fast')\n"
            "- Robotic structure (formulaic intro → features → CTA pattern)\n\n"
            "GOOD SIGNALS:\n"
            "- First person voice ('I built', 'I noticed', 'Here is what I learned')\n"
            "- Specific numbers and technical details\n"
            "- Casual tone with contractions ('it is' → 'it's')\n"
            "- Honest limitations mentioned\n"
            "- Questions that invite discussion\n\n"
            f"TARGET PLATFORM: {platform}\n"
            "If you rewrite, keep the same key information but fix the tone.\n"
        )

        user_prompt = (
            f"Review this content for {platform}:\n\n"
            f"---\n{content}\n---\n"
            f"{static_context}\n\n"
            "Provide your verdict as a CriticVerdict."
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        verdict, model = await invoke_fn(failover_chain, messages, CriticVerdict)
        logger.info("Critic used model: %s", model)
        return verdict
