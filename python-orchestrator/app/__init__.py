"""
Agent First IDE — Python Orchestrator.

Dual-model LangGraph browser automation orchestrator with:
- Supervisor subgraph (4 sub-agents: Context → Strategy → Copy + Risk → Merge)
- Worker Vision-Language Model for tactical browser action planning
- Persistent Playwright runtime with stealth and 3 connectivity modes
- SQLite persistence for campaign history and vector memory
"""

__version__ = "0.1.0"
