# Evaluation Connector Documentation Moved

The zero-API Browser MCP Server & CLI Connector guide has moved to **[`docs/MCP_SERVER.md`](MCP_SERVER.md)**.

## Backwards Compatibility Summary

Existing integrations continue to work without any breaking changes:

| Legacy Name | New Primary Name | Status |
|---|---|---|
| `agent-eval-mcp` | `agent-browse-mcp` | Fully supported alias (`pyproject.toml`) |
| `eval_start` | `browser_start` | Routed automatically in `tools/call` |
| `eval_perceive` | `browser_perceive` | Routed automatically in `tools/call` |
| `eval_act` | `browser_act` | Routed automatically in `tools/call` |
| `eval_screenshot` | `browser_screenshot` | Routed automatically in `tools/call` |
| `eval_stop` | `browser_stop` | Routed automatically in `tools/call` |
| `eval_compare` | `browser_compare` | Routed automatically in `tools/call` |
| `eval_metrics` | `browser_metrics` | Routed automatically in `tools/call` |
| `agent-eval` | `agent-eval` | Unchanged stateful CLI daemon |

See **[`docs/MCP_SERVER.md`](MCP_SERVER.md)** for full setup snippets, tool schemas, scenarios, and headed benchmarks.
