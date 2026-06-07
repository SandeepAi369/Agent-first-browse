"""Skill Memory — Cross-Session Workflow Learning & Retrieval.

Implements Agent Workflow Memory (AWM, Wang et al., arXiv:2409.07429):
  - After successful task completion: extract the workflow as a reusable template
  - Before starting a new task: retrieve relevant workflows and inject into context
  - Selective retrieval: only task-relevant skills, never the whole library

Storage: SQLite database in persistence/skills.db
Retrieval: TF-IDF keyword matching (fast, no embeddings needed)

Key design decisions:
  - Workflows are abstracted: concrete element IDs removed, patterns preserved
  - Success rate tracking: skills with declining success are retired
  - Domain-specific: workflows tagged by website domain
  - Selective injection: max 2 workflows per task to avoid context bloat
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from app.logger import get_logger
    logger = get_logger("skill_memory")
except ImportError:
    import logging
    logger = logging.getLogger("skill_memory")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Workflow:
    """A reusable workflow template extracted from a successful task execution."""
    id: str                     # Unique hash-based ID
    name: str                   # Human-readable name (e.g., "add_to_cart_amazon")
    domain: str                 # Website domain (e.g., "amazon.in")
    objective_pattern: str      # Abstracted objective (e.g., "Add {product} to cart on {site}")
    steps: list[dict]           # Abstracted action sequence
    keywords: list[str]         # Keywords for retrieval matching
    success_count: int = 1      # Number of successful uses
    failure_count: int = 0      # Number of failed uses
    avg_steps: float = 0.0      # Average steps to complete
    created_at: float = 0.0     # Unix timestamp
    last_used: float = 0.0      # Unix timestamp
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def is_reliable(self) -> bool:
        """A workflow is reliable if it has ≥60% success rate with ≥2 uses."""
        return self.success_rate >= 0.6 and (self.success_count + self.failure_count) >= 2
    
    def render_for_prompt(self) -> str:
        """Render this workflow for injection into the LLM prompt."""
        steps_text = "\n".join(
            f"  {i+1}. {s.get('action', '?')}: {s.get('description', '?')}"
            for i, s in enumerate(self.steps)
        )
        return (
            f"📌 LEARNED WORKFLOW: {self.name}\n"
            f"   Domain: {self.domain} | Success rate: {self.success_rate:.0%} "
            f"({self.success_count} successes)\n"
            f"   Pattern: {self.objective_pattern}\n"
            f"   Steps:\n{steps_text}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Workflow Extraction — Convert raw trajectories into abstract templates
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.hostname or ""
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def _abstract_step(step: dict) -> dict:
    """Abstract a concrete step into a reusable template.
    
    Removes concrete element IDs, coordinates, and specific text values.
    Preserves the action type, target description, and structural information.
    """
    abstracted = {
        "action": step.get("action", step.get("action_type", "?")),
        "description": "",
    }
    
    action = abstracted["action"]
    
    if action == "goto":
        url = step.get("url", "")
        domain = _extract_domain(url)
        abstracted["description"] = f"Navigate to {domain or url[:40]}"
    
    elif action == "click":
        target = step.get("target_name", step.get("screen", ""))[:40]
        abstracted["description"] = f"Click on '{target}'" if target else "Click target element"
    
    elif action == "type":
        # Don't store the actual text — just describe what was typed
        field_hint = step.get("target_name", "input field")[:30]
        text_len = len(step.get("text", ""))
        abstracted["description"] = f"Type {text_len} chars into {field_hint}"
    
    elif action == "scroll":
        abstracted["description"] = "Scroll to reveal more content"
    
    elif action == "press_enter":
        abstracted["description"] = "Press Enter to submit/search"
    
    elif action == "wait":
        abstracted["description"] = "Wait for page to load"
    
    else:
        abstracted["description"] = f"{action}: {step.get('outcome', '')[:40]}"
    
    return abstracted


def _extract_keywords(objective: str) -> list[str]:
    """Extract keywords from an objective for retrieval matching."""
    # Remove common stop words and normalize
    stop_words = {
        "the", "a", "an", "to", "on", "in", "for", "of", "and", "or",
        "is", "it", "this", "that", "with", "from", "at", "by", "as",
        "navigate", "go", "click", "find", "search", "your", "my",
        "then", "next", "step", "page", "please", "button",
    }
    
    # Tokenize and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', objective.lower())
    keywords = [w for w in words if w not in stop_words]
    
    # Keep unique keywords, preserve order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    
    return unique[:20]  # Cap at 20 keywords


def _compute_similarity(keywords_a: list[str], keywords_b: list[str]) -> float:
    """Compute keyword overlap similarity (Jaccard-like)."""
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = set(keywords_a)
    set_b = set(keywords_b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Skill Memory — SQLite-backed persistent workflow library
# ═══════════════════════════════════════════════════════════════════════════════

class SkillMemory:
    """Persistent cross-session skill library.
    
    Usage:
        skills = SkillMemory()
        
        # Before task: retrieve relevant workflows
        relevant = skills.retrieve_relevant("Add Ryzen to cart on Amazon", "amazon.in")
        prompt_addition = skills.inject_into_prompt(relevant)
        
        # After successful task: record the workflow
        skills.record_workflow(objective, steps, success=True, domain="amazon.in")
    """
    
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path(__file__).parent / "persistence" / "skills.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        try:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT '',
                    objective_pattern TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 1,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    avg_steps REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL
                )
            """)
            self._conn.commit()
            logger.info("SkillMemory: database initialized at %s", self._db_path)
        except Exception as e:
            logger.warning("SkillMemory: database init failed: %s", e)
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get or recreate the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
        return self._conn
    
    def record_workflow(
        self,
        objective: str,
        steps: list[dict],
        success: bool = True,
        domain: str = "",
        total_steps: int = 0,
    ) -> Workflow | None:
        """Record a workflow from a completed task execution.
        
        Args:
            objective: The task objective
            steps: List of step records from WorkingMemory.episodic
            success: Whether the task completed successfully
            domain: Website domain (auto-detected from steps if not provided)
            total_steps: Total steps taken
        
        Returns:
            The created/updated Workflow, or None on failure
        """
        if not steps:
            return None
        
        # Auto-detect domain from URLs in steps
        if not domain:
            for s in steps:
                url = s.get("url", "")
                if url:
                    domain = _extract_domain(url)
                    if domain:
                        break
        
        # Abstract the steps
        abstracted_steps = [_abstract_step(s) for s in steps if s.get("action") != "done"]
        
        # Remove consecutive duplicates (e.g., multiple waits)
        deduped = []
        for s in abstracted_steps:
            if not deduped or s["description"] != deduped[-1]["description"]:
                deduped.append(s)
        abstracted_steps = deduped[:15]  # Cap at 15 steps
        
        # Generate keywords
        keywords = _extract_keywords(objective)
        if domain:
            keywords.append(domain.split(".")[0])  # Add domain name as keyword
        
        # Generate workflow name
        name = re.sub(r'[^a-z0-9_]', '_', objective[:40].lower()).strip("_")
        if domain:
            name = f"{name}_{domain.split('.')[0]}"
        
        # Create workflow ID from objective hash
        wf_id = hashlib.md5(f"{objective}|{domain}".encode()).hexdigest()[:12]
        
        workflow = Workflow(
            id=wf_id,
            name=name,
            domain=domain,
            objective_pattern=objective[:200],
            steps=abstracted_steps,
            keywords=keywords,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            avg_steps=float(total_steps or len(steps)),
            created_at=time.time(),
            last_used=time.time(),
        )
        
        try:
            conn = self._get_conn()
            
            # Check if this workflow already exists
            existing = conn.execute(
                "SELECT success_count, failure_count, avg_steps FROM workflows WHERE id = ?",
                (wf_id,)
            ).fetchone()
            
            if existing:
                # Update existing workflow
                old_success, old_failure, old_avg = existing
                if success:
                    new_success = old_success + 1
                    new_failure = old_failure
                else:
                    new_success = old_success
                    new_failure = old_failure + 1
                new_avg = (old_avg * (old_success + old_failure) + total_steps) / (new_success + new_failure)
                
                conn.execute("""
                    UPDATE workflows 
                    SET success_count = ?, failure_count = ?, avg_steps = ?, last_used = ?
                    WHERE id = ?
                """, (new_success, new_failure, new_avg, time.time(), wf_id))
                conn.commit()
                
                workflow.success_count = new_success
                workflow.failure_count = new_failure
                workflow.avg_steps = new_avg
                
                logger.info(
                    "📝 SkillMemory: Updated workflow '%s' (success=%d, fail=%d, rate=%.0f%%)",
                    name, new_success, new_failure, workflow.success_rate * 100,
                )
            else:
                # Insert new workflow
                conn.execute("""
                    INSERT INTO workflows (id, name, domain, objective_pattern, steps_json, 
                        keywords_json, success_count, failure_count, avg_steps, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    wf_id, name, domain, workflow.objective_pattern,
                    json.dumps(abstracted_steps), json.dumps(keywords),
                    workflow.success_count, workflow.failure_count,
                    workflow.avg_steps, workflow.created_at, workflow.last_used,
                ))
                conn.commit()
                
                logger.info(
                    "📝 SkillMemory: Recorded new workflow '%s' (%d steps, domain=%s)",
                    name, len(abstracted_steps), domain,
                )
            
            return workflow
            
        except Exception as e:
            logger.warning("SkillMemory: record failed: %s", e)
            return None
    
    def retrieve_relevant(
        self,
        objective: str,
        domain: str = "",
        k: int = 2,
        min_similarity: float = 0.15,
    ) -> list[Workflow]:
        """Retrieve the most relevant workflows for a new task.
        
        Uses TF-IDF-like keyword matching. Domain match is a strong bonus.
        Only returns reliable workflows (≥60% success rate).
        
        Args:
            objective: The new task objective
            domain: Target website domain
            k: Maximum number of workflows to return
            min_similarity: Minimum keyword similarity threshold
        
        Returns:
            List of relevant Workflow objects, sorted by relevance
        """
        query_keywords = _extract_keywords(objective)
        if not query_keywords:
            return []
        
        try:
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT id, name, domain, objective_pattern, steps_json, keywords_json,
                       success_count, failure_count, avg_steps, created_at, last_used
                FROM workflows
                WHERE success_count > 0
                ORDER BY last_used DESC
                LIMIT 50
            """).fetchall()
            
            if not rows:
                return []
            
            scored_workflows: list[tuple[float, Workflow]] = []
            
            for row in rows:
                wf = Workflow(
                    id=row[0], name=row[1], domain=row[2],
                    objective_pattern=row[3],
                    steps=json.loads(row[4]),
                    keywords=json.loads(row[5]),
                    success_count=row[6], failure_count=row[7],
                    avg_steps=row[8], created_at=row[9], last_used=row[10],
                )
                
                # Skip unreliable workflows
                if not wf.is_reliable and (wf.success_count + wf.failure_count) >= 3:
                    continue
                
                # Calculate relevance score
                keyword_sim = _compute_similarity(query_keywords, wf.keywords)
                
                # Domain match bonus (strong signal)
                domain_bonus = 0.3 if domain and wf.domain == domain else 0.0
                
                # Recency bonus (recent workflows are more likely to still work)
                age_days = (time.time() - wf.last_used) / 86400
                recency_bonus = 0.1 * max(0, 1 - age_days / 30)  # Decay over 30 days
                
                # Success rate bonus
                success_bonus = 0.1 * wf.success_rate
                
                total_score = keyword_sim + domain_bonus + recency_bonus + success_bonus
                
                if total_score >= min_similarity:
                    scored_workflows.append((total_score, wf))
            
            # Sort by relevance score (descending)
            scored_workflows.sort(key=lambda x: x[0], reverse=True)
            
            results = [wf for _, wf in scored_workflows[:k]]
            
            if results:
                logger.info(
                    "🧠 SkillMemory: Retrieved %d relevant workflows for '%s': %s",
                    len(results), objective[:40],
                    ", ".join(f"'{w.name}' ({w.success_rate:.0%})" for w in results),
                )
            
            return results
            
        except Exception as e:
            logger.warning("SkillMemory: retrieval failed: %s", e)
            return []
    
    def inject_into_prompt(self, workflows: list[Workflow]) -> str:
        """Format retrieved workflows for injection into the LLM system prompt.
        
        Returns empty string if no workflows.
        Keeps the output concise (<500 chars per workflow).
        """
        if not workflows:
            return ""
        
        parts = ["═══ LEARNED WORKFLOWS (from past successful runs) ═══"]
        for wf in workflows[:2]:  # Max 2 workflows in prompt
            parts.append(wf.render_for_prompt())
        parts.append(
            "NOTE: These workflows are from past runs and may need adaptation. "
            "Use them as guidance, not rigid instructions."
        )
        
        return "\n".join(parts)
    
    def get_stats(self) -> dict:
        """Get summary statistics of the skill library."""
        try:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0]
            reliable = conn.execute(
                "SELECT COUNT(*) FROM workflows WHERE success_count >= 2 "
                "AND CAST(success_count AS REAL) / (success_count + failure_count) >= 0.6"
            ).fetchone()[0]
            domains = conn.execute("SELECT COUNT(DISTINCT domain) FROM workflows").fetchone()[0]
            return {
                "total_workflows": total,
                "reliable_workflows": reliable,
                "domains_covered": domains,
            }
        except Exception:
            return {"total_workflows": 0, "reliable_workflows": 0, "domains_covered": 0}
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
