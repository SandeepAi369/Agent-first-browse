"""Project-relative locations, independent of the current working directory.

Runtime code lives in src/, but runtime data (.env, persistence/, uploads/)
stays at the project root. Anchor every such path here instead of on a
module's own __file__, so moving a module can never relocate user data.
"""

from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"
PERSISTENCE_DIR = PROJECT_ROOT / "persistence"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
