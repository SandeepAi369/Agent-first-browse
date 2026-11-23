#!/usr/bin/env python3
"""Thin executable shim for `eval_connector.cli` (also installed as `agent-eval`)."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_connector.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
