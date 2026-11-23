"""eval_connector.session — Dual-Mode Zero-API Stateful Browser Session Manager.

Enables any AI agent (via CLI daemon or MCP stdio server) to act directly as
the LLM brain for `Agent-first-browse` without requiring any external LLM API
keys. Runs identical live pages and multi-step workflows through the REAL
`legacy` and `v2` perception and action pipelines while keeping frozen legacy
modules 100% untouched.

Concurrency safety:
  `src/mcp_tools.py` stores `_PAGE` and `_SELECTOR_MAP` as module globals, and
  `v2_engine` reads `AGENT_ENGINE_MODE` from env. `DualModeSessionManager`
  serializes browser tool execution via an `asyncio.Lock` and `_bind_session()`
  so paired `legacy` + `v2` sessions run side-by-side inside one process with
  zero state bleed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import dom_parser
import mcp_tools
from eval_connector.scenarios import get_scenario
from perception_engine import _viewport_of, strict_viewport_filter
from v2_engine import (
    ENGINE_MODE_LEGACY,
    ENGINE_MODE_V2,
    get_engine_mode,
    normalize_engine_mode,
    set_engine_mode,
)
from v2_engine.actions import v2_click, v2_type
from v2_engine.graph_policy import detect_stealth_driver
from v2_engine.metrics import RunMetrics, StepMetric, estimate_tokens
from v2_engine.perception import V2SnapshotCompressor, perceive_v2

logger = logging.getLogger("eval_connector.session")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREENSHOT_DIR = ROOT / "logs" / "eval_screenshots"


@dataclass
class TranscriptEntry:
    seq: int
    kind: str  # "start" | "scenario" | "perceive" | "act" | "screenshot"
    mode: str
    url: str
    duration_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSession:
    name: str
    mode: str
    headless: bool
    context: Any
    page: Any
    viewport: tuple[int, int] = (1280, 800)
    compressor: V2SnapshotCompressor = field(default_factory=V2SnapshotCompressor)
    metrics: RunMetrics = field(init=False)
    elements: list[dict[str, Any]] = field(default_factory=list)
    selector_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_markdown: str = ""
    last_perceive: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.metrics = RunMetrics(mode=self.mode)

    def append_transcript(
        self,
        kind: str,
        url: str,
        duration_ms: float,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.transcript.append(
            TranscriptEntry(
                seq=len(self.transcript) + 1,
                kind=kind,
                mode=self.mode,
                url=url,
                duration_ms=round(duration_ms, 1),
                detail=detail or {},
            )
        )


async def _ensure_visual_cursor(page) -> None:
    """Inject the realistic visual arrow cursor in headed sessions (idempotent)."""
    try:
        from app.browser_promoter.cdp_stealth_launcher import VISUAL_CURSOR_INIT_SCRIPT
        from ghost_input import resync_visual_cursor
        await page.evaluate(VISUAL_CURSOR_INIT_SCRIPT)
        await resync_visual_cursor(page)
    except Exception:
        pass


def _norm_label(text: str) -> str:
    return " ".join((text or "").split()).strip()


class DualModeSessionManager:
    """Manages one or more isolated live Chromium sessions (`legacy` and/or `v2`)."""

    def __init__(self, screenshot_dir: Path | None = None) -> None:
        self._pw: Any = None
        self._browsers: dict[bool, Any] = {}  # keyed by headless: bool
        self._sessions: dict[str, EvalSession] = {}
        self._lock = asyncio.Lock()
        self._screenshot_dir = screenshot_dir or DEFAULT_SCREENSHOT_DIR
        self._xvfb_started = False

    async def _ensure_browser(self, headless: bool):
        if self._pw is None:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()

        effective_headless = headless
        if not headless:
            # Reuse existing virtual_display helper if running without $DISPLAY (e.g. headless WSL)
            import os
            if not os.environ.get("DISPLAY"):
                try:
                    from virtual_display import start_virtual_display
                    if start_virtual_display(1920, 1080):
                        self._xvfb_started = True
                    else:
                        logger.warning("No DISPLAY and Xvfb unavailable — falling back to headless")
                        effective_headless = True
                except Exception:
                    effective_headless = True

        if effective_headless not in self._browsers:
            self._browsers[effective_headless] = await self._pw.chromium.launch(
                headless=effective_headless,
            )
        return self._browsers[effective_headless], effective_headless

    @contextlib.contextmanager
    def _bind_session(self, session: EvalSession):
        """Temporarily bind global `mcp_tools` + `AGENT_ENGINE_MODE` to `session`."""
        prev_mode = get_engine_mode()
        prev_page = mcp_tools._PAGE
        prev_smap = mcp_tools._SELECTOR_MAP
        try:
            set_engine_mode(session.mode)
            mcp_tools.set_page(session.page)
            mcp_tools.set_selector_map(session.selector_map)
            yield
        finally:
            set_engine_mode(prev_mode)
            mcp_tools.set_page(prev_page)
            mcp_tools.set_selector_map(prev_smap)

    def _get_session(self, name: str) -> EvalSession:
        if name not in self._sessions:
            active = ", ".join(sorted(self._sessions)) or "(none)"
            raise KeyError(f"Session '{name}' not found. Active sessions: {active}")
        return self._sessions[name]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Session Lifecycle
    # ═══════════════════════════════════════════════════════════════════════════

    async def start_session(
        self,
        name: str,
        mode: str = ENGINE_MODE_V2,
        *,
        url: str | None = None,
        scenario: str | None = None,
        headless: bool = True,
        viewport: tuple[int, int] = (1280, 800),
        install_shadow_piercer: bool = True,
    ) -> dict[str, Any]:
        """Start a named isolated browser session in 'legacy' or 'v2' mode."""
        norm_mode = normalize_engine_mode(mode)
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Session name must be non-empty")

        async with self._lock:
            if clean_name in self._sessions:
                await self._close_session_unlocked(clean_name)

            t0 = time.perf_counter()
            browser, effective_headless = await self._ensure_browser(headless=headless)
            vw, vh = int(viewport[0]), int(viewport[1])
            context = await browser.new_context(viewport={"width": vw, "height": vh})
            if install_shadow_piercer:
                await dom_parser.install_shadow_piercer(context)
            if not effective_headless:
                try:
                    from app.browser_promoter.cdp_stealth_launcher import VISUAL_CURSOR_INIT_SCRIPT
                    await context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
                except Exception as e:
                    logger.debug("Visual cursor init script skipped: %s", e)
            page = await context.new_page()

            session = EvalSession(
                name=clean_name,
                mode=norm_mode,
                headless=effective_headless,
                context=context,
                page=page,
                viewport=(vw, vh),
            )
            self._sessions[clean_name] = session

            if scenario:
                spec = get_scenario(scenario)
                await page.set_content(spec.html, wait_until="domcontentloaded")
                if not effective_headless:
                    await _ensure_visual_cursor(page)
                session.append_transcript(
                    "scenario", page.url, (time.perf_counter() - t0) * 1000.0,
                    {"scenario": spec.name},
                )
            elif url:
                await page.goto(url, wait_until="domcontentloaded", timeout=25000)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            session.append_transcript(
                "start", page.url, elapsed_ms,
                {"headless": effective_headless, "url": url or "", "scenario": scenario or ""},
            )
            return {
                "ok": True,
                "session": clean_name,
                "mode": norm_mode,
                "headless": effective_headless,
                "url": page.url,
                "driver": detect_stealth_driver(),
                "startup_ms": elapsed_ms,
            }

    async def start_pair(
        self,
        prefix: str = "eval",
        *,
        url: str | None = None,
        scenario: str | None = None,
        headless: bool = True,
        viewport: tuple[int, int] = (1280, 800),
    ) -> dict[str, Any]:
        """Start twin `legacy` and `v2` sessions on identical starting state."""
        leg_name = f"{prefix}_legacy"
        v2_name = f"{prefix}_v2"
        leg = await self.start_session(
            leg_name, ENGINE_MODE_LEGACY,
            url=url, scenario=scenario, headless=headless, viewport=viewport,
        )
        v2 = await self.start_session(
            v2_name, ENGINE_MODE_V2,
            url=url, scenario=scenario, headless=headless, viewport=viewport,
        )
        return {"ok": True, "legacy": leg, "v2": v2}

    async def load_scenario(self, name: str, scenario_name: str) -> dict[str, Any]:
        """Load a built-in deep-test scenario into an existing session."""
        async with self._lock:
            session = self._get_session(name)
            spec = get_scenario(scenario_name)
            t0 = time.perf_counter()
            await session.page.set_content(spec.html, wait_until="domcontentloaded")
            if not session.headless:
                await _ensure_visual_cursor(session.page)
            session.compressor.invalidate()
            session.elements = []
            session.selector_map = {}
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            session.append_transcript("scenario", session.page.url, elapsed_ms, {"scenario": spec.name})
            return {
                "ok": True,
                "session": name,
                "mode": session.mode,
                "scenario": spec.summary(),
                "url": session.page.url,
            }

    async def list_sessions(self) -> list[dict[str, Any]]:
        async with self._lock:
            out = []
            for s in self._sessions.values():
                url = ""
                with contextlib.suppress(Exception):
                    url = s.page.url
                out.append({
                    "session": s.name,
                    "mode": s.mode,
                    "headless": s.headless,
                    "url": url,
                    "steps": s.step_count,
                    "elements": len(s.elements),
                })
            return out

    async def _close_session_unlocked(self, name: str) -> bool:
        session = self._sessions.pop(name, None)
        if session is None:
            return False
        try:
            await session.context.close()
        except Exception as e:
            logger.debug("Error closing context for %s: %s", name, e)
        return True

    async def close_session(self, name: str) -> dict[str, Any]:
        async with self._lock:
            closed = await self._close_session_unlocked(name)
            return {"ok": closed, "session": name}

    async def close_all(self) -> dict[str, Any]:
        async with self._lock:
            names = list(self._sessions.keys())
            for name in names:
                await self._close_session_unlocked(name)
            for browser in list(self._browsers.values()):
                with contextlib.suppress(Exception):
                    await browser.close()
            self._browsers.clear()
            if self._pw is not None:
                with contextlib.suppress(Exception):
                    await self._pw.stop()
                self._pw = None
            if self._xvfb_started:
                try:
                    from virtual_display import stop_virtual_display
                    stop_virtual_display()
                except Exception:
                    pass
                self._xvfb_started = False
            return {"ok": True, "closed": names}

    # ═══════════════════════════════════════════════════════════════════════════
    #  Perception (Exact Legacy vs Exact V2)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _perceive_unlocked(
        self,
        session: EvalSession,
        *,
        objective: str = "",
        force_full: bool = False,
        settle: bool = True,
    ) -> dict[str, Any]:
        page = session.page
        step_idx = session.step_count

        with self._bind_session(session):
            if session.mode == ENGINE_MODE_V2:
                pr = await perceive_v2(
                    page,
                    ctx={
                        "objective": objective,
                        "step_number": step_idx,
                        "force_full_snapshot": force_full,
                    },
                    compressor=session.compressor,
                    metrics=session.metrics,
                    do_settle=settle,
                    use_mcp_snapshot=False,
                )
                step_metric = session.metrics.steps[-1]
                session.elements = pr.elements
                session.selector_map = pr.selector_map
                session.last_markdown = pr.markdown
            else:
                t_total = time.perf_counter()
                settle_ms = 0.0
                if settle:
                    t_settle = time.perf_counter()
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=10000)
                        await page.wait_for_timeout(1500)
                    except Exception:
                        pass
                    settle_ms = round((time.perf_counter() - t_settle) * 1000.0, 1)

                t_ext = time.perf_counter()
                dom_data = await dom_parser.extract(page, timeout=5.0)
                raw_els = dom_data.get("elements", []) or []
                raw_smap = {e["id"]: e for e in raw_els if e.get("id")}
                vw, vh = _viewport_of(page)
                els, smap, md, off_dropped, _n_off = strict_viewport_filter(
                    raw_els, raw_smap, dom_data.get("markdown", ""), vw=vw, vh=vh,
                )
                extract_ms = round((time.perf_counter() - t_ext) * 1000.0, 1)
                total_ms = round((time.perf_counter() - t_total) * 1000.0, 1)

                step_metric = StepMetric(
                    step=step_idx + 1,
                    url=page.url or "",
                    mode=ENGINE_MODE_LEGACY,
                    settle_ms=settle_ms,
                    extract_ms=extract_ms,
                    total_perceive_ms=total_ms,
                    raw_elements=len(raw_els),
                    emitted_elements=len(els),
                    occluded_filtered=0,
                    offscreen_dropped=off_dropped,
                    markdown_chars=len(md),
                    est_tokens=estimate_tokens(md),
                    snapshot_kind="full",
                    tier_used=1,
                )
                session.metrics.record_step(step_metric)
                session.elements = els
                session.selector_map = smap
                session.last_markdown = md

        session.step_count += 1
        title = ""
        with contextlib.suppress(Exception):
            title = await page.title()

        page_url = str(page.url or "")
        payload = {
            "ok": True,
            "session": session.name,
            "mode": session.mode,
            "step": session.step_count,
            "url": page_url,
            "title": title,
            "markdown": session.last_markdown,
            "elements": session.elements,
            "element_count": len(session.elements),
            "metrics": asdict(step_metric),
        }
        session.last_perceive = payload
        session.append_transcript(
            "perceive",
            page_url,
            step_metric.total_perceive_ms,
            {
                "step": session.step_count,
                "element_count": payload["element_count"],
                "est_tokens": step_metric.est_tokens,
                "occluded_filtered": step_metric.occluded_filtered,
                "snapshot_kind": step_metric.snapshot_kind,
                "tier_used": step_metric.tier_used,
            },
        )
        return payload

    async def perceive(
        self,
        name: str,
        *,
        objective: str = "",
        force_full: bool = False,
        settle: bool = True,
    ) -> dict[str, Any]:
        """Perceive the active page in `name` using its configured engine mode."""
        async with self._lock:
            session = self._get_session(name)
            return await self._perceive_unlocked(
                session, objective=objective, force_full=force_full, settle=settle,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Action Dispatch (Real Legacy mcp_tools vs Real V2 Actions)
    # ═══════════════════════════════════════════════════════════════════════════

    async def act(
        self,
        name: str,
        verb: str,
        *,
        element_id: str | None = None,
        text: str | None = None,
        url: str | None = None,
        x: float | None = None,
        y: float | None = None,
        key: str | None = None,
        direction: str = "down",
        amount: int = 500,
        wait_ms: int = 800,
        auto_perceive: bool = False,
        objective: str = "",
    ) -> dict[str, Any]:
        """Execute an agent-chosen browser action inside session `name`."""
        async with self._lock:
            session = self._get_session(name)
            clean_verb = (verb or "").strip().lower()
            t0 = time.perf_counter()

            # Fill snapshot fallback coordinates from selector_map if caller gave element_id
            eff_x = float(x) if x is not None else 0.0
            eff_y = float(y) if y is not None else 0.0
            if element_id and (x is None or y is None):
                el_meta = session.selector_map.get(element_id) or {}
                if x is None and el_meta.get("x") is not None:
                    eff_x = float(el_meta["x"])
                if y is None and el_meta.get("y") is not None:
                    eff_y = float(el_meta["y"])

            if not session.headless:
                try:
                    await session.page.bring_to_front()
                    await _ensure_visual_cursor(session.page)
                except Exception:
                    pass
            with self._bind_session(session):
                result = await self._dispatch_verb(
                    session,
                    clean_verb,
                    element_id=element_id,
                    text=text,
                    url=url,
                    x=eff_x,
                    y=eff_y,
                    key=key,
                    direction=direction,
                    amount=amount,
                    wait_ms=wait_ms,
                )

            duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            cur_url = ""
            with contextlib.suppress(Exception):
                cur_url = session.page.url or ""

            session.append_transcript(
                "act",
                cur_url,
                duration_ms,
                {
                    "verb": clean_verb,
                    "element_id": element_id,
                    "success": bool(result.get("success")),
                    "error": result.get("error", ""),
                    "stale_ref": bool(result.get("stale_ref", False)),
                },
            )

            response: dict[str, Any] = {
                "ok": bool(result.get("success")),
                "session": session.name,
                "mode": session.mode,
                "verb": clean_verb,
                "url": cur_url,
                "duration_ms": duration_ms,
                "result": result,
            }
            if auto_perceive:
                response["perception"] = await self._perceive_unlocked(
                    session, objective=objective, settle=True,
                )
            return response

    async def _dispatch_verb(
        self,
        session: EvalSession,
        verb: str,
        *,
        element_id: str | None,
        text: str | None,
        url: str | None,
        x: float,
        y: float,
        key: str | None,
        direction: str,
        amount: int,
        wait_ms: int,
    ) -> dict[str, Any]:
        if verb == "goto":
            session.compressor.invalidate()
            return await mcp_tools.mcp_navigate(url or text or "")

        if verb == "click":
            if session.mode == ENGINE_MODE_V2:
                return await v2_click(
                    session.page,
                    x=x,
                    y=y,
                    element_id=element_id,
                    elements_list=session.elements,
                    metrics=session.metrics,
                )
            return await mcp_tools.mcp_click(x=x, y=y, element_id=element_id)

        if verb == "type":
            if session.mode == ENGINE_MODE_V2:
                return await v2_type(
                    session.page,
                    text=text or "",
                    x=x,
                    y=y,
                    element_id=element_id,
                    elements_list=session.elements,
                    metrics=session.metrics,
                )
            return await mcp_tools.mcp_type(text=text or "", x=x, y=y, element_id=element_id)

        if verb == "hover":
            return await mcp_tools.mcp_hover(element_id=element_id, x=x, y=y)

        if verb == "select_option":
            return await mcp_tools.mcp_select_option(element_id=element_id, value=text or "")

        if verb == "press_enter":
            return await mcp_tools.mcp_press_enter()

        if verb == "press_key":
            return await mcp_tools.mcp_press_key(key or text or "")

        if verb == "scroll":
            return await mcp_tools.mcp_scroll(int(amount))

        if verb == "scroll_directional":
            return await mcp_tools.mcp_scroll_directional(direction=direction, amount=int(amount))

        if verb == "wait":
            return await mcp_tools.mcp_wait(int(wait_ms))

        return {"success": False, "error": f"Unsupported action verb '{verb}'"}

    # ═══════════════════════════════════════════════════════════════════════════
    #  Side-by-Side Comparison, Screenshots & Reporting
    # ═══════════════════════════════════════════════════════════════════════════

    async def compare(
        self,
        legacy_name: str,
        v2_name: str,
        *,
        objective: str = "",
        refresh: bool = True,
        settle: bool = True,
    ) -> dict[str, Any]:
        """Compare a `legacy` session and `v2` session side-by-side."""
        async with self._lock:
            leg = self._get_session(legacy_name)
            v2 = self._get_session(v2_name)

            if refresh or not leg.last_perceive:
                leg_p = await self._perceive_unlocked(leg, objective=objective, settle=settle)
            else:
                leg_p = leg.last_perceive

            if refresh or not v2.last_perceive:
                v2_p = await self._perceive_unlocked(v2, objective=objective, settle=settle)
            else:
                v2_p = v2.last_perceive

        leg_m = leg_p["metrics"]
        v2_m = v2_p["metrics"]
        leg_ms = float(leg_m["total_perceive_ms"])
        v2_ms = float(v2_m["total_perceive_ms"])
        speedup_x = round(leg_ms / v2_ms, 2) if v2_ms > 0 else 0.0

        leg_tok = int(leg_m["est_tokens"])
        v2_tok = int(v2_m["est_tokens"])
        tok_saved_pct = round((1.0 - (v2_tok / leg_tok)) * 100.0, 1) if leg_tok > 0 else 0.0

        leg_labels = {_norm_label(e.get("text", "")) for e in leg_p["elements"] if _norm_label(e.get("text", ""))}
        v2_labels = {_norm_label(e.get("text", "")) for e in v2_p["elements"] if _norm_label(e.get("text", ""))}

        return {
            "ok": True,
            "legacy_session": legacy_name,
            "v2_session": v2_name,
            "urls": {"legacy": leg_p["url"], "v2": v2_p["url"]},
            "step_comparison": {
                "legacy_ms": leg_ms,
                "v2_ms": v2_ms,
                "speedup_x": speedup_x,
                "legacy_tokens": leg_tok,
                "v2_tokens": v2_tok,
                "token_savings_pct": tok_saved_pct,
                "legacy_elements": leg_p["element_count"],
                "v2_elements": v2_p["element_count"],
                "v2_occluded_filtered": v2_m["occluded_filtered"],
                "v2_tier_used": v2_m["tier_used"],
                "v2_snapshot_kind": v2_m["snapshot_kind"],
            },
            "visibility_diff": {
                "shared_count": len(leg_labels & v2_labels),
                "only_in_legacy_count": len(leg_labels - v2_labels),
                "only_in_v2_count": len(v2_labels - leg_labels),
                "only_in_legacy_sample": sorted(leg_labels - v2_labels)[:12],
                "only_in_v2_sample": sorted(v2_labels - leg_labels)[:12],
            },
            "cumulative": {
                "legacy": leg.metrics.summary(),
                "v2": v2.metrics.summary(),
            },
        }

    async def screenshot(
        self,
        name: str,
        *,
        out_path: str | None = None,
        full_page: bool = False,
    ) -> dict[str, Any]:
        """Capture a PNG screenshot of session `name` to disk."""
        async with self._lock:
            session = self._get_session(name)
            t0 = time.perf_counter()
            if out_path:
                target = Path(out_path)
            else:
                self._screenshot_dir.mkdir(parents=True, exist_ok=True)
                target = self._screenshot_dir / f"{session.name}_step{session.step_count}.png"
            target.parent.mkdir(parents=True, exist_ok=True)

            raw = await session.page.screenshot(path=str(target), full_page=full_page, type="png")
            vw, vh = _viewport_of(session.page)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            session.append_transcript("screenshot", session.page.url or "", elapsed_ms, {"path": str(target)})
            return {
                "ok": True,
                "session": session.name,
                "mode": session.mode,
                "path": str(target),
                "width": vw,
                "height": vh,
                "bytes": len(raw),
            }

    async def report(self, names: list[str] | None = None) -> dict[str, Any]:
        """Return structured telemetry + transcripts + Markdown summary table."""
        async with self._lock:
            selected = [self._get_session(n) for n in names] if names else list(self._sessions.values())
            sessions_data = {}
            lines = [
                "| Session | Mode | Steps | Avg ms | Avg Tok | Total Tok "
                "| Occluded Pruned | Delta | Tier-2 | Retries | Stale Rebinds |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for s in selected:
                summ = s.metrics.summary()
                sessions_data[s.name] = {
                    "mode": s.mode,
                    "headless": s.headless,
                    "summary": summ,
                    "metrics": s.metrics.to_dict(),
                    "transcript": [asdict(t) for t in s.transcript],
                }
                lines.append(
                    f"| `{s.name}` | {s.mode} | {summ['steps']} | {summ['avg_perceive_ms']} | "
                    f"{summ['avg_est_tokens']} | {summ['total_est_tokens']} | "
                    f"{summ['total_occluded_filtered']} | {summ['delta_snapshots']} | "
                    f"{summ['tier2_escalations']} | {summ['action_retries']} | "
                    f"{summ['stale_ref_recoveries']} |"
                )

            return {
                "ok": True,
                "driver": detect_stealth_driver(),
                "sessions": sessions_data,
                "markdown_table": "\n".join(lines),
            }
