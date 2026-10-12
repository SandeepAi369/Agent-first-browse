#!/usr/bin/env bash
#
# agent.sh — one-click launcher for the Agent First IDE browser agent.
#
# You do NOT need to know anything about Python or virtual environments.
# Just give it a task:
#
#     ./agent.sh "search Flipkart for a water bottle under 300 rupees and add it to the cart"
#
# or run it with no task and it will ask you:
#
#     ./agent.sh
#
# It automatically uses the project's Python environment and launches the
# browser in stealth mode (a real, headed browser hidden inside a virtual
# display via Xvfb — far less likely to be flagged as a bot).
#
set -uo pipefail

# ── Always run from the folder this script lives in (works from anywhere) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "❌ Could not enter project folder."; exit 1; }

VENV_PY="$SCRIPT_DIR/.venv/bin/python"

# ── 1. Make sure the Python environment exists ──
if [[ ! -x "$VENV_PY" ]]; then
  echo "❌ Python environment not found at: .venv/"
  echo "   One-time setup is needed (create .venv and install requirements.txt)."
  exit 1
fi

# ── 2. Get the task: from the command line, or ask for it ──
if [[ $# -gt 0 ]]; then
  TASK="$*"
else
  echo "🤖 What should the agent do?"
  echo "   e.g.  search Flipkart for a water bottle under ₹300 and add it to the cart"
  printf "> "
  read -r TASK || true
fi

# Trim and validate
TASK="$(printf '%s' "${TASK:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [[ -z "$TASK" ]]; then
  echo "❌ No task given — nothing to do."
  exit 1
fi

# ── 3. Stealth-mode heads-up ──
if command -v Xvfb >/dev/null 2>&1; then
  echo "🖥️  Stealth mode ON: real headed browser inside a virtual display."
else
  echo "⚠️  Xvfb is not installed — using headless mode (more detectable)."
  echo "    For best results, install it once:  sudo apt-get install -y xvfb"
fi

echo "🚀 Starting agent…"
echo "   Task: $TASK"
echo

# ── 4. Run the agent. No --headless → it auto-picks the stealth headed display.
#     'exec' hands the terminal to Python so Ctrl-C shuts the browser down cleanly.
exec "$VENV_PY" src/run_v16.py run "$TASK"
