"""eval_connector.cli — Zero-API 'Agent-as-LLM' Stateful CLI Client.

Drives persistent live Chromium sessions in `--mode legacy` and `--mode v2`
through `eval_connector.daemon`, letting an AI agent (or human tester) inspect
real perception output, choose each action step-by-step, capture screenshots,
and compare both engines side-by-side with zero external LLM API keys.

Examples:
    # List built-in deep-test scenarios
    .venv/bin/python scripts/agent_eval.py scenarios

    # Start twin legacy + v2 sessions on a real URL or built-in scenario
    .venv/bin/python scripts/agent_eval.py start-pair modal --scenario modal_occlusion
    .venv/bin/python scripts/agent_eval.py start-pair hn --url https://news.ycombinator.com

    # Perceive, compare side-by-side, and act step-by-step
    .venv/bin/python scripts/agent_eval.py compare modal_legacy modal_v2
    .venv/bin/python scripts/agent_eval.py perceive modal_v2
    .venv/bin/python scripts/agent_eval.py act modal_v2 click e1 --auto-perceive
    .venv/bin/python scripts/agent_eval.py screenshot modal_v2
    .venv/bin/python scripts/agent_eval.py report --json-out eval_report.json
    .venv/bin/python scripts/agent_eval.py stop --daemon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval_connector.daemon import (
    default_socket_path,
    ensure_daemon_running,
    is_daemon_alive,
    send_daemon_request,
    stop_daemon,
)


def _parse_viewport(raw: str) -> list[int]:
    try:
        w_s, h_s = raw.lower().split("x", 1)
        w, h = int(w_s), int(h_s)
        if w < 200 or h < 200:
            raise ValueError
        return [w, h]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid viewport '{raw}'; expected WIDTHxHEIGHT (e.g. 1280x800)") from e


def _print_result(cmd: str, data: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if cmd == "scenarios":
        for sc in data.get("scenarios", []):
            print(f"- {sc['name']:<16} | {sc['title']}")
            print(f"  {sc['description']}")
        return

    if cmd == "sessions":
        rows = data.get("sessions", [])
        if not rows:
            print("(no active sessions)")
            return
        for r in rows:
            vis = "headless" if r["headless"] else "headed"
            print(f"- {r['session']:<16} [{r['mode']:<6} | {vis:<8}] steps={r['steps']} els={r['elements']} url={r['url']}")
        return

    if cmd == "status":
        if not data.get("running"):
            print(f"Daemon: STOPPED (socket={data['socket']})")
        else:
            print(
                f"Daemon: RUNNING pid={data['pid']} uptime={data['uptime_s']}s "
                f"sessions={data['session_count']} socket={data['socket']}"
            )
        return

    if cmd == "perceive":
        m = data.get("metrics", {})
        print(
            f"[{data['session']} | {data['mode'].upper()} | step {data['step']}] "
            f"url={data['url']} | {data['element_count']} els | "
            f"~{m.get('est_tokens', 0)} tok | {m.get('total_perceive_ms', 0)}ms "
            f"(settle={m.get('settle_ms', 0)}ms, tier={m.get('tier_used', 1)}, "
            f"kind={m.get('snapshot_kind', 'full')}, occluded-{m.get('occluded_filtered', 0)})"
        )
        print("-" * 80)
        print(data.get("markdown", ""))
        return

    if cmd == "act":
        status = "OK" if data.get("ok") else "FAIL"
        res = data.get("result", {})
        err = f" error={res.get('error')}" if res.get("error") else ""
        print(
            f"[{data['session']} | {data['mode'].upper()}] {data['verb']} → {status} "
            f"({data['duration_ms']}ms) url={data['url']}{err}"
        )
        if "perception" in data:
            print()
            _print_result("perceive", data["perception"], as_json=False)
        return

    if cmd == "compare":
        sc = data["step_comparison"]
        vd = data["visibility_diff"]
        print("=" * 84)
        print(f" SIDE-BY-SIDE: {data['legacy_session']} (LEGACY) vs {data['v2_session']} (V2)")
        print("=" * 84)
        print(
            f" Latency : legacy={sc['legacy_ms']}ms | v2={sc['v2_ms']}ms "
            f"({sc['speedup_x']}x speedup)"
        )
        print(
            f" Tokens  : legacy=~{sc['legacy_tokens']} | v2=~{sc['v2_tokens']} "
            f"({sc['token_savings_pct']}% saved)"
        )
        print(
            f" Elements: legacy={sc['legacy_elements']} | v2={sc['v2_elements']} "
            f"(occluded_pruned={sc['v2_occluded_filtered']}, tier={sc['v2_tier_used']}, "
            f"kind={sc['v2_snapshot_kind']})"
        )
        print(
            f" Diff    : shared={vd['shared_count']} | only_legacy={vd['only_in_legacy_count']} "
            f"| only_v2={vd['only_in_v2_count']}"
        )
        if vd["only_in_legacy_sample"]:
            print(f"   only_legacy sample: {vd['only_in_legacy_sample'][:6]}")
        if vd["only_in_v2_sample"]:
            print(f"   only_v2 sample    : {vd['only_in_v2_sample'][:6]}")
        print("=" * 84)
        return

    if cmd == "report":
        print(data.get("markdown_table", ""))
        return

    print(json.dumps(data, indent=2, ensure_ascii=False))


def _resolve_act_params(args: argparse.Namespace) -> dict[str, Any]:
    verb = args.verb.strip().lower()
    element_id = args.element_id
    text = args.text
    url = args.url
    key = args.key
    amount = args.amount
    wait_ms = args.wait_ms
    direction = args.direction

    pos = list(args.args or [])
    if verb == "goto":
        if not url and pos:
            url = pos[0]
    elif verb in ("click", "hover"):
        if not element_id and pos:
            element_id = pos[0]
    elif verb in ("type", "select_option"):
        if not element_id and pos:
            element_id = pos.pop(0)
        if text is None and pos:
            text = " ".join(pos)
    elif verb == "press_key":
        if not key and pos:
            key = pos[0]
    elif verb == "scroll":
        if pos and amount == 500:
            amount = int(pos[0])
    elif verb == "scroll_directional":
        if pos:
            direction = pos.pop(0)
        if pos:
            amount = int(pos[0])
    elif verb == "wait":
        if pos and wait_ms == 800:
            wait_ms = int(pos[0])

    return {
        "name": args.session,
        "verb": verb,
        "element_id": element_id,
        "text": text,
        "url": url,
        "x": args.x,
        "y": args.y,
        "key": key,
        "direction": direction,
        "amount": amount,
        "wait_ms": wait_ms,
        "auto_perceive": bool(args.auto_perceive),
        "objective": args.objective,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Zero-API Agent-as-LLM Evaluation CLI for Agent-first-browse (legacy vs v2).",
    )
    parser.add_argument("--socket", type=str, default=str(default_socket_path()), help="Daemon Unix socket path.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    parser.add_argument("--idle-timeout", type=float, default=1800.0, help="Daemon idle auto-shutdown seconds.")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Check eval daemon status.")
    sub.add_parser("scenarios", help="List built-in deep-test scenarios.")
    sub.add_parser("sessions", help="List active browser sessions.")

    p_start = sub.add_parser("start", help="Start a named browser session.")
    p_start.add_argument("name", help="Session name (e.g. leg, v2).")
    p_start.add_argument("--mode", choices=["legacy", "v2"], default="v2")
    p_start.add_argument("--url", type=str, default=None)
    p_start.add_argument("--scenario", type=str, default=None)
    p_start.add_argument("--headed", action="store_true", help="Launch visible/Xvfb headed browser.")
    p_start.add_argument("--viewport", type=_parse_viewport, default=[1280, 800])

    p_pair = sub.add_parser("start-pair", help="Start twin <prefix>_legacy and <prefix>_v2 sessions.")
    p_pair.add_argument("prefix", nargs="?", default="eval")
    p_pair.add_argument("--url", type=str, default=None)
    p_pair.add_argument("--scenario", type=str, default=None)
    p_pair.add_argument("--headed", action="store_true", help="Launch visible/Xvfb headed browsers.")
    p_pair.add_argument("--viewport", type=_parse_viewport, default=[1280, 800])

    p_load = sub.add_parser("load-scenario", help="Load a built-in scenario into an active session.")
    p_load.add_argument("session")
    p_load.add_argument("scenario")

    p_perc = sub.add_parser("perceive", help="Run legacy or v2 perception on an active session.")
    p_perc.add_argument("session")
    p_perc.add_argument("--objective", type=str, default="")
    p_perc.add_argument("--force-full", action="store_true")
    p_perc.add_argument("--no-settle", action="store_true")

    p_act = sub.add_parser("act", help="Execute a browser action inside an active session.")
    p_act.add_argument("session")
    p_act.add_argument(
        "verb",
        choices=[
            "goto", "click", "type", "hover", "select_option",
            "press_enter", "press_key", "scroll", "scroll_directional", "wait",
        ],
    )
    p_act.add_argument("args", nargs="*", help="Positional shorthand (e.g. URL, element_id, or element_id text).")
    p_act.add_argument("--id", dest="element_id", type=str, default=None)
    p_act.add_argument("--text", type=str, default=None)
    p_act.add_argument("--url", type=str, default=None)
    p_act.add_argument("--x", type=float, default=None)
    p_act.add_argument("--y", type=float, default=None)
    p_act.add_argument("--key", type=str, default=None)
    p_act.add_argument("--direction", type=str, default="down")
    p_act.add_argument("--amount", type=int, default=500)
    p_act.add_argument("--wait-ms", type=int, default=800)
    p_act.add_argument("--auto-perceive", action="store_true")
    p_act.add_argument("--objective", type=str, default="")

    p_cmp = sub.add_parser("compare", help="Compare an active legacy session and v2 session side-by-side.")
    p_cmp.add_argument("legacy_session")
    p_cmp.add_argument("v2_session")
    p_cmp.add_argument("--objective", type=str, default="")
    p_cmp.add_argument("--no-refresh", action="store_true")
    p_cmp.add_argument("--no-settle", action="store_true")

    p_shot = sub.add_parser("screenshot", help="Save a PNG screenshot of an active session.")
    p_shot.add_argument("session")
    p_shot.add_argument("--out", type=str, default=None)
    p_shot.add_argument("--full-page", action="store_true")

    p_rep = sub.add_parser("report", help="Print cumulative telemetry table and optionally save JSON report.")
    p_rep.add_argument("sessions", nargs="*")
    p_rep.add_argument("--json-out", type=str, default=None)

    p_stop = sub.add_parser("stop", help="Close a session, all sessions, or shut down the daemon.")
    p_stop.add_argument("session", nargs="?", default=None)
    p_stop.add_argument("--all", action="store_true", help="Close all sessions (keep daemon alive).")
    p_stop.add_argument("--daemon", action="store_true", help="Close all sessions and shut down the daemon.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    sock_path = Path(args.socket)
    cmd = args.command

    if cmd == "status":
        alive = is_daemon_alive(sock_path)
        payload = {"running": alive is not None, "socket": str(sock_path), **(alive or {})}
        _print_result("status", payload, args.json)
        return 0

    if cmd == "stop":
        if args.daemon:
            data = stop_daemon(sock_path)
            _print_result("stop", data, args.json)
            return 0
        if is_daemon_alive(sock_path) is None:
            _print_result("stop", {"ok": True, "already_stopped": True}, args.json)
            return 0
        if args.all or not args.session:
            res = send_daemon_request(sock_path, "close_all")
        else:
            res = send_daemon_request(sock_path, "close", {"name": args.session})
        if not res.get("ok"):
            print(f"Error: {res.get('error')}", file=sys.stderr)
            return 1
        _print_result("stop", res["data"], args.json)
        return 0

    # All remaining commands require a live daemon (auto-spawn if needed)
    ensure_daemon_running(sock_path, idle_timeout=args.idle_timeout)

    method_map: dict[str, tuple[str, dict[str, Any]]] = {
        "scenarios": ("scenarios", {}),
        "sessions": ("sessions", {}),
        "start": ("start", {
            "name": getattr(args, "name", None),
            "mode": getattr(args, "mode", "v2"),
            "url": getattr(args, "url", None),
            "scenario": getattr(args, "scenario", None),
            "headless": not getattr(args, "headed", False),
            "viewport": getattr(args, "viewport", [1280, 800]),
        }),
        "start-pair": ("start_pair", {
            "prefix": getattr(args, "prefix", "eval"),
            "url": getattr(args, "url", None),
            "scenario": getattr(args, "scenario", None),
            "headless": not getattr(args, "headed", False),
            "viewport": getattr(args, "viewport", [1280, 800]),
        }),
        "load-scenario": ("load_scenario", {
            "name": getattr(args, "session", None),
            "scenario": getattr(args, "scenario", None),
        }),
        "perceive": ("perceive", {
            "name": getattr(args, "session", None),
            "objective": getattr(args, "objective", ""),
            "force_full": getattr(args, "force_full", False),
            "settle": not getattr(args, "no_settle", False),
        }),
        "compare": ("compare", {
            "legacy_name": getattr(args, "legacy_session", None),
            "v2_name": getattr(args, "v2_session", None),
            "objective": getattr(args, "objective", ""),
            "refresh": not getattr(args, "no_refresh", False),
            "settle": not getattr(args, "no_settle", False),
        }),
        "screenshot": ("screenshot", {
            "name": getattr(args, "session", None),
            "out_path": getattr(args, "out", None),
            "full_page": getattr(args, "full_page", False),
        }),
        "report": ("report", {
            "names": getattr(args, "sessions", None) or None,
        }),
    }

    if cmd == "act":
        method, params = "act", _resolve_act_params(args)
    else:
        method, params = method_map[cmd]

    res = send_daemon_request(sock_path, method, params)
    if not res.get("ok"):
        print(f"Error: {res.get('error')}", file=sys.stderr)
        return 1

    data = res["data"]
    if cmd == "report" and getattr(args, "json_out", None):
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_result(cmd, data, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
