# Automated Release & Multi-Platform Sync Guide

`Agent-first-browse` uses a single-command + GitHub Actions OIDC release pipeline so every published destination stays automatically synchronized without manual steps or long-lived registry credentials.

---

## 1. Which Platforms Need Version Updates vs. Evergreen Directories

| Platform / Directory | Needs Update Per Release? | How It Stays in Sync |
|---|---|---|
| **In-repo version files** (`pyproject.toml`, `server.json`, `src/app/__init__.py`, `src/eval_connector/mcp_server.py`) | **Yes** | Updated atomically by `scripts/release.py`; enforced on every PR by CI (`python scripts/release.py --check`). |
| **PyPI** (`pypi.org/project/agent-first-browse/`) | **Yes** | Published automatically by `.github/workflows/release.yml` (`pypa/gh-action-pypi-publish@release/v1` via OIDC Trusted Publishing or optional `PYPI_API_TOKEN`). |
| **GitHub Releases** (`vX.Y.Z`) | **Yes** | Created automatically with generated release notes + attached `.whl`/`.tar.gz` via built-in `GITHUB_TOKEN`. |
| **Official MCP Registry** (`registry.modelcontextprotocol.io`) | **Yes** | Published automatically after PyPI confirms live propagation, using zero-secret `mcp-publisher login github-oidc`. |
| **Glama** (`glama.ai/mcp/servers/@SandeepAi369/Agent-first-browse`) | **No (Evergreen)** | Re-indexes GitHub `main`, `glama.json`, `README.md`, and PyPI automatically. |
| **Curated Awesome Lists** (`TensorBlock/awesome-mcp-servers`, `punkpeye/awesome-mcp-servers`, `steel-dev/awesome-web-agents`, `YuzeHao2023/Awesome-MCP-Servers`, `mcpservers.org`) | **No (Evergreen)** | None of the submitted entries hardcode version numbers — they link to the GitHub repo, Glama live SVG badge, `uvx --from agent-first-browse agent-eval-mcp`, and Official MCP Registry ID. |

---

## 2. One-Time Setup for Zero-Secret OIDC Publishing

- **Official MCP Registry (`github-oidc`) & GitHub Releases (`GITHUB_TOKEN`):** **Zero setup needed** — works out of the box on GitHub Actions.
- **PyPI Trusted Publishing (recommended — zero stored tokens, takes ~30 seconds once):**
  1. Go to https://pypi.org/manage/project/agent-first-browse/settings/publishing/
  2. Under **Add a new publisher** → **GitHub**, enter:
     - **Owner:** `SandeepAi369`
     - **Repository name:** `Agent-first-browse`
     - **Workflow name:** `release.yml`
     - **Environment name:** `pypi`
  3. *(Alternative)* If you prefer API-token auth instead of OIDC, add a repository secret named `PYPI_API_TOKEN` in GitHub Settings → Secrets → Actions. The workflow detects it automatically.

---

## 3. How to Cut Any Future Release (Two Zero-Touch Options)

### Option A: One-Click in GitHub Actions UI (Zero Local Commands)
1. Open **Actions → Release & Multi-Platform Sync → Run workflow** on GitHub.
2. Select `patch`, `minor`, or `major` (or enter an explicit version like `0.2.0`; toggle `dry_run: true` to test without publishing).
3. GitHub Actions automatically bumps all 4 version files, commits to `main`, creates tag `vX.Y.Z`, runs all tests & guardrails, builds & smoke-tests the wheel, publishes to **PyPI**, creates the **GitHub Release**, and publishes `server.json` to the **Official MCP Registry**.

### Option B: One Command Locally
```bash
.venv/bin/python scripts/release.py --bump patch --commit --tag --push
```
Pushing the `vX.Y.Z` tag automatically runs the entire `.github/workflows/release.yml` pipeline end-to-end.
