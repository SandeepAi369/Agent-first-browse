#!/usr/bin/env python3
"""Record a clean, relatable 1280x720 README demo video of Agent-first-browse.

Demonstrates one clear, everyday end-to-end task that any viewer immediately
understands: signing into an online store, sorting products Low -> High,
adding the two cheapest items to the cart, filling out shipping details, and
completing checkout — powered by real `perceive_v2` perception, smooth Bézier
visible mouse cursor movement, click ripples, and character-by-character typing.

Guarantees:
  - Leaves `README.md` and all `src/` (including §6 frozen files) untouched.
  - Temporarily hides the overlay during `perceive_v2` so displayed latency
    and perception metrics are 100% genuine.
  - Saves the raw, unedited 1280x720 `.webm` video under `docs/assets/raw/`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from playwright.async_api import Page, async_playwright  # noqa: E402

import mcp_tools  # noqa: E402
from v2_engine import ENGINE_MODE_V2, set_engine_mode  # noqa: E402
from v2_engine.metrics import RunMetrics  # noqa: E402
from v2_engine.perception import V2SnapshotCompressor, perceive_v2  # noqa: E402

VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "assets" / "raw"
DEFAULT_VIDEO_NAME = "agent_first_browse_demo_1280x720.webm"
ALLOWED_OUTPUT_ROOTS: tuple[Path, ...] = (
    REPO_ROOT.resolve(),
    Path(tempfile.gettempdir()).resolve(),
)
STORE_URL = "https://www.saucedemo.com/"
USER_TASK = "Sort products Low → High, add the 2 cheapest items to cart, and complete checkout"


# ── Persistent Visible AI Mouse Cursor & Click Ripple ──────────────────────────

_CURSOR_INIT_JS = r"""
(() => {
    const ensureCursor = () => {
        if (!document.documentElement) return null;
        let root = document.getElementById('__afb_cursor_root__');
        if (root) return root;

        root = document.createElement('div');
        root.id = '__afb_cursor_root__';
        Object.assign(root.style, {
            position: 'fixed',
            inset: '0',
            zIndex: '2147483647',
            pointerEvents: 'none',
            overflow: 'hidden',
        });

        const style = document.createElement('style');
        style.textContent = `
            @keyframes afbClickRipple {
                0% { transform: translate(-50%, -50%) scale(0.35); opacity: 0.95; }
                100% { transform: translate(-50%, -50%) scale(2.2); opacity: 0; }
            }
            @keyframes afbPulseDot {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.45; transform: scale(0.85); }
            }
        `;
        root.appendChild(style);

        const ptr = document.createElement('div');
        ptr.id = '__afb_cursor_ptr__';
        Object.assign(ptr.style, {
            position: 'fixed',
            left: '640px',
            top: '360px',
            width: '30px',
            height: '30px',
            transform: 'translate(-2px, -2px)',
            filter: 'drop-shadow(0 4px 10px rgba(2, 6, 23, 0.55))',
            transition: 'opacity 120ms ease',
            willChange: 'left, top',
        });
        ptr.innerHTML = `
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 3L23 13.5L14.5 15.8L11 24L4 3Z"
                      fill="#0f172a" stroke="#22d3ee" stroke-width="2.2" stroke-linejoin="round"/>
                <circle cx="5.5" cy="4.5" r="2.2" fill="#34d399"/>
            </svg>
        `;
        root.appendChild(ptr);
        document.documentElement.appendChild(root);
        return root;
    };

    window.__afbCursorMove = (x, y) => {
        ensureCursor();
        const ptr = document.getElementById('__afb_cursor_ptr__');
        if (ptr) {
            ptr.style.left = `${x}px`;
            ptr.style.top = `${y}px`;
            ptr.style.opacity = '1';
        }
    };

    window.__afbCursorClick = (x, y) => {
        const root = ensureCursor();
        if (!root) return;
        const ripple = document.createElement('div');
        Object.assign(ripple.style, {
            position: 'fixed',
            left: `${x}px`,
            top: `${y}px`,
            width: '34px',
            height: '34px',
            borderRadius: '999px',
            border: '3px solid #22d3ee',
            background: 'rgba(34, 211, 238, 0.20)',
            animation: 'afbClickRipple 520ms cubic-bezier(0.22, 1, 0.36, 1) forwards',
        });
        root.appendChild(ripple);
        setTimeout(() => ripple.remove(), 560);
    };

    window.__afbSetOverlayVisible = (visible) => {
        const bar = document.getElementById('__afb_bar__');
        const hi = document.getElementById('__afb_highlight__');
        const cur = document.getElementById('__afb_cursor_root__');
        for (const node of [bar, hi, cur]) {
            if (node) node.style.display = visible ? '' : 'none';
        }
    };
})();
"""

_RENDER_BAR_JS = r"""
(payload) => {
    let bar = document.getElementById('__afb_bar__');
    if (bar) bar.remove();

    bar = document.createElement('div');
    bar.id = '__afb_bar__';
    Object.assign(bar.style, {
        position: 'fixed',
        left: '50%',
        bottom: '14px',
        transform: 'translateX(-50%)',
        zIndex: '2147483646',
        pointerEvents: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '16px',
        maxWidth: '1160px',
        width: 'calc(100% - 48px)',
        padding: '10px 16px',
        borderRadius: '14px',
        background: 'rgba(15, 23, 42, 0.92)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: payload.done
            ? '1.5px solid rgba(52, 211, 153, 0.85)'
            : '1px solid rgba(148, 163, 184, 0.30)',
        boxShadow: '0 14px 36px rgba(2, 6, 23, 0.45)',
        color: '#f8fafc',
        fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    });

    // Left: Agent badge + User Task prompt
    const left = document.createElement('div');
    Object.assign(left.style, {
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        minWidth: '0',
        flex: '1',
    });

    const dot = document.createElement('span');
    Object.assign(dot.style, {
        width: '10px',
        height: '10px',
        borderRadius: '999px',
        flexShrink: '0',
        background: payload.done ? '#34d399' : '#22d3ee',
        boxShadow: payload.done ? '0 0 10px #34d399' : '0 0 10px #22d3ee',
        animation: payload.done ? 'none' : 'afbPulseDot 1.3s ease-in-out infinite',
    });

    const brand = document.createElement('span');
    brand.textContent = 'Agent-first-browse';
    Object.assign(brand.style, {
        fontSize: '12.5px',
        fontWeight: '800',
        color: '#34d399',
        whiteSpace: 'nowrap',
        letterSpacing: '0.01em',
    });

    const sep = document.createElement('span');
    sep.textContent = '•';
    sep.style.color = '#475569';

    const task = document.createElement('span');
    task.textContent = payload.task;
    Object.assign(task.style, {
        fontSize: '12.5px',
        fontWeight: '500',
        color: '#cbd5e1',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
    });

    left.appendChild(dot);
    left.appendChild(brand);
    left.appendChild(sep);
    left.appendChild(task);

    // Right: Current plain-English action + clean perception speed pill
    const right = document.createElement('div');
    Object.assign(right.style, {
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        flexShrink: '0',
    });

    const actionPill = document.createElement('span');
    actionPill.textContent = payload.status;
    Object.assign(actionPill.style, {
        padding: '4px 11px',
        borderRadius: '999px',
        fontSize: '12.5px',
        fontWeight: '700',
        color: payload.done ? '#022c22' : '#f8fafc',
        background: payload.done
            ? 'linear-gradient(135deg, #34d399, #2dd4bf)'
            : 'rgba(14, 165, 233, 0.25)',
        border: payload.done ? 'none' : '1px solid rgba(56, 189, 248, 0.45)',
        whiteSpace: 'nowrap',
    });
    right.appendChild(actionPill);

    if (payload.latencyMs !== null && payload.latencyMs !== undefined) {
        const msPill = document.createElement('span');
        msPill.textContent = `${payload.latencyMs} ms`;
        Object.assign(msPill.style, {
            padding: '4px 9px',
            borderRadius: '8px',
            fontSize: '12px',
            fontWeight: '700',
            color: '#34d399',
            background: 'rgba(16, 185, 129, 0.14)',
            border: '1px solid rgba(52, 211, 153, 0.35)',
            whiteSpace: 'nowrap',
        });
        right.appendChild(msPill);
    }

    bar.appendChild(left);
    bar.appendChild(right);
    document.documentElement.appendChild(bar);
}
"""

_HIGHLIGHT_TARGET_JS = r"""
(payload) => {
    const old = document.getElementById('__afb_highlight__');
    if (old) old.remove();
    if (!payload) return;

    const box = document.createElement('div');
    box.id = '__afb_highlight__';
    Object.assign(box.style, {
        position: 'fixed',
        left: `${payload.x - 4}px`,
        top: `${payload.y - 4}px`,
        width: `${payload.w + 8}px`,
        height: `${payload.h + 8}px`,
        border: '2.5px solid #22d3ee',
        background: 'rgba(34, 211, 238, 0.12)',
        borderRadius: '8px',
        boxShadow: '0 0 0 4px rgba(34, 211, 238, 0.22)',
        zIndex: '2147483645',
        pointerEvents: 'none',
        boxSizing: 'border-box',
    });

    if (payload.label) {
        const tag = document.createElement('span');
        tag.textContent = payload.label;
        Object.assign(tag.style, {
            position: 'absolute',
            top: '-24px',
            left: '0',
            padding: '2px 8px',
            borderRadius: '5px',
            fontSize: '11px',
            fontWeight: '800',
            color: '#082f49',
            background: 'linear-gradient(135deg, #22d3ee, #34d399)',
            whiteSpace: 'nowrap',
            boxShadow: '0 4px 12px rgba(2, 6, 23, 0.35)',
            fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        });
        box.appendChild(tag);
    }

    document.documentElement.appendChild(box);
}
"""


class SmoothCursor:
    """Drives real Playwright mouse moves along smooth cubic Bézier curves."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.x: float = 640.0
        self.y: float = 360.0

    async def place(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)
        with contextlib.suppress(Exception):
            await self.page.evaluate("([x, y]) => window.__afbCursorMove && window.__afbCursorMove(x, y)", [x, y])

    async def glide_to(self, tx: float, ty: float, *, duration_s: float = 0.55) -> None:
        sx, sy = self.x, self.y
        dx, dy = tx - sx, ty - sy
        dist = (dx * dx + dy * dy) ** 0.5
        steps = max(16, min(40, int(dist / 22)))

        # Gentle arc control points so motion feels natural, not robotic linear teleport
        arc = max(-75.0, min(75.0, dist * 0.14))
        c1x = sx + dx * 0.32 - (dy / max(dist, 1.0)) * arc * 0.45
        c1y = sy + dy * 0.32 + (dx / max(dist, 1.0)) * arc * 0.45
        c2x = sx + dx * 0.72 + (dy / max(dist, 1.0)) * arc * 0.25
        c2y = sy + dy * 0.72 - (dx / max(dist, 1.0)) * arc * 0.25

        dt = duration_s / steps
        for i in range(1, steps + 1):
            t = i / steps
            # Ease-in-out cubic
            u = 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2
            inv = 1.0 - u
            px = (inv**3) * sx + 3 * (inv**2) * u * c1x + 3 * inv * (u**2) * c2x + (u**3) * tx
            py = (inv**3) * sy + 3 * (inv**2) * u * c1y + 3 * inv * (u**2) * c2y + (u**3) * ty
            await self.page.mouse.move(px, py)
            with contextlib.suppress(Exception):
                await self.page.evaluate(
                    "([x, y]) => window.__afbCursorMove && window.__afbCursorMove(x, y)",
                    [px, py],
                )
            await asyncio.sleep(dt)

        self.x, self.y = float(tx), float(ty)

    async def click_ripple(self) -> None:
        with contextlib.suppress(Exception):
            await self.page.evaluate(
                "([x, y]) => window.__afbCursorClick && window.__afbCursorClick(x, y)",
                [self.x, self.y],
            )


async def _set_bar(page: Page, status: str, *, latency_ms: float | None = None, done: bool = False) -> None:
    payload = {
        "task": USER_TASK,
        "status": status,
        "latencyMs": round(latency_ms, 1) if latency_ms is not None else None,
        "done": done,
    }
    with contextlib.suppress(Exception):
        await page.evaluate(_RENDER_BAR_JS, payload)


async def _highlight(page: Page, rect: dict[str, float] | None, label: str = "") -> None:
    payload = None if rect is None else {**rect, "label": label}
    with contextlib.suppress(Exception):
        await page.evaluate(_HIGHLIGHT_TARGET_JS, payload)


async def _perceive_step(
    page: Page,
    compressor: V2SnapshotCompressor,
    metrics: RunMetrics,
    step_log: list[dict[str, Any]],
    *,
    name: str,
    objective: str,
    step_num: int,
) -> tuple[Any, Any]:
    """Hide overlays briefly, run real `perceive_v2`, record metrics, restore overlays."""
    with contextlib.suppress(Exception):
        await page.evaluate("() => window.__afbSetOverlayVisible && window.__afbSetOverlayVisible(false)")

    pr = await perceive_v2(
        page,
        ctx={"objective": objective, "step_number": step_num, "force_full_snapshot": False},
        compressor=compressor,
        metrics=metrics,
        do_settle=True,
        use_mcp_snapshot=False,
    )
    mcp_tools.set_page(page)
    mcp_tools.set_selector_map(pr.selector_map)
    m = metrics.steps[-1]
    step_log.append({
        "step": step_num + 1,
        "name": name,
        "url": page.url,
        "ms": m.total_perceive_ms,
        "tokens": m.est_tokens,
        "elements": m.emitted_elements,
        "occluded_pruned": m.occluded_filtered,
        "kind": m.snapshot_kind,
    })

    with contextlib.suppress(Exception):
        await page.evaluate("() => window.__afbSetOverlayVisible && window.__afbSetOverlayVisible(true)")
    return pr, m


async def _box_of(page: Page, selector: str) -> dict[str, float]:
    loc = page.locator(selector).first
    await loc.scroll_into_view_if_needed(timeout=8000)
    box = await loc.bounding_box(timeout=8000)
    if not box:
        raise RuntimeError(f"Element has no bounding box: {selector}")
    return {"x": box["x"], "y": box["y"], "w": box["width"], "h": box["height"]}


async def _glide_click(
    page: Page,
    cursor: SmoothCursor,
    selector: str,
    *,
    status: str,
    label: str,
    latency_ms: float,
    glide_s: float = 0.55,
    pause_after_s: float = 0.55,
) -> None:
    await _set_bar(page, status, latency_ms=latency_ms)
    rect = await _box_of(page, selector)
    cx = rect["x"] + rect["w"] / 2.0
    cy = rect["y"] + rect["h"] / 2.0
    await _highlight(page, rect, label)
    await cursor.glide_to(cx, cy, duration_s=glide_s)
    await asyncio.sleep(0.18)
    await cursor.click_ripple()
    await page.mouse.click(cx, cy)
    await asyncio.sleep(0.22)
    await _highlight(page, None)
    await asyncio.sleep(pause_after_s)


async def _glide_type(
    page: Page,
    cursor: SmoothCursor,
    selector: str,
    text: str,
    *,
    status: str,
    label: str,
    latency_ms: float,
    char_delay_ms: int = 58,
) -> None:
    await _set_bar(page, status, latency_ms=latency_ms)
    rect = await _box_of(page, selector)
    cx = rect["x"] + min(rect["w"] * 0.35, 90.0)
    cy = rect["y"] + rect["h"] / 2.0
    await _highlight(page, rect, label)
    await cursor.glide_to(cx, cy, duration_s=0.50)
    await asyncio.sleep(0.15)
    await cursor.click_ripple()
    await page.mouse.click(cx, cy)
    await page.keyboard.type(text, delay=char_delay_ms)
    await asyncio.sleep(0.25)
    await _highlight(page, None)


async def _read_demo_store_credentials(page: Page) -> tuple[str, str]:
    """Read the public demo credentials printed directly on the storefront login page."""
    creds = await page.evaluate(
        r"""() => {
            const uBlock = document.querySelector('#login_credentials')?.innerText || '';
            const pBlock = document.querySelector('.login_password')?.innerText || '';
            const users = uBlock.split('\n').map(s => s.trim()).filter(s => s && !s.endsWith(':'));
            const passes = pBlock.split('\n').map(s => s.trim()).filter(s => s && !s.endsWith(':'));
            return { user: users[0] || '', pass: passes[0] || '' };
        }"""
    )
    if not creds.get("user") or not creds.get("pass"):
        raise RuntimeError("Could not read storefront public demo credentials from login page.")
    return str(creds["user"]), str(creds["pass"])


def _resolve_confined_out_dir(
    out_dir: Path,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
) -> Path:
    """Resolve `out_dir` and enforce confinement within the repository or system temp root."""
    roots = tuple(r.resolve() for r in (allowed_roots or ALLOWED_OUTPUT_ROOTS))
    resolved = out_dir.expanduser().resolve(strict=False)
    if not any(resolved.is_relative_to(root) for root in roots):
        allowed_str = ", ".join(str(r) for r in roots)
        raise ValueError(
            f"Refusing out_dir outside allowed workspace/temp roots: {resolved} (allowed: {allowed_str})"
        )
    return resolved


def _validate_video_filename(video_name: str) -> str:
    """Require a single safe filename segment without path separators or traversal."""
    cleaned = (video_name or "").strip()
    if (
        not cleaned
        or cleaned in (".", "..")
        or "/" in cleaned
        or "\\" in cleaned
        or Path(cleaned).name != cleaned
    ):
        raise ValueError(f"Invalid video filename (must be a single filename segment): {video_name!r}")
    return cleaned


def _safe_rmtree_confined(
    target: Path,
    *,
    expected_parent: Path,
    allowed_roots: tuple[Path, ...] | None = None,
) -> None:
    """Safely remove a temporary subdirectory only if strictly confined and non-symlink."""
    if target.is_symlink():
        raise ValueError(f"Refusing to remove symlinked temp directory: {target}")
    if not target.exists():
        return

    roots = tuple(r.resolve() for r in (allowed_roots or ALLOWED_OUTPUT_ROOTS))
    resolved_parent = expected_parent.resolve()
    resolved_target = target.resolve()

    if resolved_target == resolved_parent or not resolved_target.is_relative_to(resolved_parent):
        raise ValueError(
            f"Refusing to remove directory outside expected parent {resolved_parent}: {resolved_target}"
        )
    if not any(resolved_target.is_relative_to(root) for root in roots):
        raise ValueError(f"Refusing to remove directory outside allowed roots: {resolved_target}")
    shutil.rmtree(resolved_target)


async def record_demo(*, headless: bool, out_dir: Path, video_name: str) -> tuple[Path, Path]:
    """Run the relatable end-to-end e-commerce demo and save raw 1280x720 `.webm`."""
    confined_out_dir = _resolve_confined_out_dir(out_dir)
    safe_video_name = _validate_video_filename(video_name)
    confined_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_video_dir = confined_out_dir / "_recording_tmp"
    _safe_rmtree_confined(tmp_video_dir, expected_parent=confined_out_dir)
    tmp_video_dir.mkdir(parents=True, exist_ok=True)

    set_engine_mode(ENGINE_MODE_V2)
    compressor = V2SnapshotCompressor()
    metrics = RunMetrics(mode=ENGINE_MODE_V2)
    step_log: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--window-size=1280,720"],
        )
        context = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(tmp_video_dir),
            record_video_size=VIEWPORT,
        )
        await context.add_init_script(_CURSOR_INIT_JS)
        page = await context.new_page()
        video = page.video
        cursor = SmoothCursor(page)

        try:
            # ── 1. Open Storefront & Sign In ──────────────────────────────────
            print("[1/7] Opening store & signing in ...")
            await page.goto(STORE_URL, wait_until="domcontentloaded", timeout=30000)
            await cursor.place(640, 320)
            _, m1 = await _perceive_step(
                page, compressor, metrics, step_log,
                name="Login Screen",
                objective="Sign into the online store",
                step_num=0,
            )
            await _set_bar(page, "1/7 • Signing into store…", latency_ms=m1.total_perceive_ms)
            await asyncio.sleep(1.1)

            demo_user, demo_pass = await _read_demo_store_credentials(page)
            await _glide_type(
                page, cursor, "#user-name", demo_user,
                status="1/7 • Entering username",
                label="TYPE username",
                latency_ms=m1.total_perceive_ms,
                char_delay_ms=52,
            )
            await _glide_type(
                page, cursor, "#password", demo_pass,
                status="1/7 • Entering password",
                label="TYPE password",
                latency_ms=m1.total_perceive_ms,
                char_delay_ms=48,
            )
            await _glide_click(
                page, cursor, "#login-button",
                status="1/7 • Logging in",
                label="CLICK Login",
                latency_ms=m1.total_perceive_ms,
                pause_after_s=0.65,
            )

            # ── 2. Sort Catalog: Price Low -> High ────────────────────────────
            print("[2/7] Sorting products Low -> High ...")
            await page.wait_for_selector(".inventory_list", timeout=10000)
            await cursor.place(cursor.x, cursor.y)
            _, m2 = await _perceive_step(
                page, compressor, metrics, step_log,
                name="Store Catalog",
                objective="Sort products by price low to high",
                step_num=1,
            )
            await _set_bar(page, "2/7 • Sorting by Price (Low → High)", latency_ms=m2.total_perceive_ms)
            sort_rect = await _box_of(page, "[data-test='product-sort-container']")
            await _highlight(page, sort_rect, "SORT Low → High")
            await cursor.glide_to(sort_rect["x"] + sort_rect["w"] / 2, sort_rect["y"] + sort_rect["h"] / 2)
            await asyncio.sleep(0.25)
            await cursor.click_ripple()
            await page.select_option("[data-test='product-sort-container']", "lohi")
            await asyncio.sleep(0.45)
            await _highlight(page, None)
            await asyncio.sleep(0.75)

            # ── 3. Add Two Cheapest Items to Cart ─────────────────────────────
            print("[3/7] Adding 2 cheapest items ($7.99 & $9.99) to cart ...")
            _, m3 = await _perceive_step(
                page, compressor, metrics, step_log,
                name="Sorted Catalog (Low -> High)",
                objective="Add the two cheapest items to cart",
                step_num=2,
            )
            await _glide_click(
                page, cursor, "[data-test='add-to-cart-sauce-labs-onesie']",
                status="3/7 • Adding Sauce Labs Onesie ($7.99)",
                label="CLICK Add to cart ($7.99)",
                latency_ms=m3.total_perceive_ms,
                pause_after_s=0.6,
            )
            await _glide_click(
                page, cursor, "[data-test='add-to-cart-sauce-labs-bike-light']",
                status="3/7 • Adding Bike Light ($9.99)",
                label="CLICK Add to cart ($9.99)",
                latency_ms=m3.total_perceive_ms,
                pause_after_s=0.75,
            )

            # ── 4. Open Shopping Cart & Click Checkout ────────────────────────
            print("[4/7] Opening cart & proceeding to checkout ...")
            await _glide_click(
                page, cursor, ".shopping_cart_link",
                status="4/7 • Opening shopping cart (2 items)",
                label="CLICK Cart (2)",
                latency_ms=m3.total_perceive_ms,
                pause_after_s=0.65,
            )
            await page.wait_for_selector("#checkout", timeout=10000)
            await cursor.place(cursor.x, cursor.y)
            _, m4 = await _perceive_step(
                page, compressor, metrics, step_log,
                name="Shopping Cart",
                objective="Review items and proceed to checkout",
                step_num=3,
            )
            await _set_bar(page, "4/7 • Reviewing 2 items in cart", latency_ms=m4.total_perceive_ms)
            await asyncio.sleep(0.9)
            await _glide_click(
                page, cursor, "#checkout",
                status="4/7 • Proceeding to checkout",
                label="CLICK Checkout",
                latency_ms=m4.total_perceive_ms,
                pause_after_s=0.55,
            )

            # ── 5. Fill Shipping Form (Visible Typing) ────────────────────────
            print("[5/7] Filling shipping details ...")
            await page.wait_for_selector("#first-name", timeout=10000)
            await cursor.place(cursor.x, cursor.y)
            _, m5 = await _perceive_step(
                page, compressor, metrics, step_log,
                name="Checkout Shipping Form",
                objective="Fill customer shipping information",
                step_num=4,
            )
            await _glide_type(
                page, cursor, "#first-name", "Alex",
                status="5/7 • Typing First Name",
                label="TYPE Alex",
                latency_ms=m5.total_perceive_ms,
                char_delay_ms=65,
            )
            await _glide_type(
                page, cursor, "#last-name", "Rivera",
                status="5/7 • Typing Last Name",
                label="TYPE Rivera",
                latency_ms=m5.total_perceive_ms,
                char_delay_ms=65,
            )
            await _glide_type(
                page, cursor, "#postal-code", "94107",
                status="5/7 • Typing Postal Code",
                label="TYPE 94107",
                latency_ms=m5.total_perceive_ms,
                char_delay_ms=65,
            )
            await _glide_click(
                page, cursor, "#continue",
                status="5/7 • Continuing to order review",
                label="CLICK Continue",
                latency_ms=m5.total_perceive_ms,
                pause_after_s=0.65,
            )

            # ── 6. Review Order Total & Click Finish ──────────────────────────
            print("[6/7] Reviewing order summary & placing order ...")
            await page.wait_for_selector("#finish", timeout=10000)
            await cursor.place(cursor.x, cursor.y)
            _, m6 = await _perceive_step(
                page, compressor, metrics, step_log,
                name="Order Summary",
                objective="Review order total and place order",
                step_num=5,
            )
            await _set_bar(page, "6/7 • Verifying order total ($19.42)", latency_ms=m6.total_perceive_ms)
            await asyncio.sleep(0.55)
            await mcp_tools.mcp_scroll(260)
            await asyncio.sleep(0.55)
            await _glide_click(
                page, cursor, "#finish",
                status="6/7 • Placing order",
                label="CLICK Finish",
                latency_ms=m6.total_perceive_ms,
                pause_after_s=0.6,
            )

            # ── 7. Verify Confirmation Screen ─────────────────────────────────
            print("[7/7] Order confirmed! Holding final screen ...")
            await page.wait_for_selector(".complete-header", timeout=10000)
            await cursor.place(cursor.x, cursor.y)
            _, m7 = await _perceive_step(
                page, compressor, metrics, step_log,
                name="Order Confirmed",
                objective="Verify order confirmation page",
                step_num=6,
            )
            avg_ms = round(sum(s.total_perceive_ms for s in metrics.steps) / max(1, len(metrics.steps)), 1)
            await cursor.glide_to(640, 290, duration_s=0.55)
            await _set_bar(
                page,
                f"✓ Order Confirmed • 7 steps (avg {avg_ms} ms)",
                latency_ms=m7.total_perceive_ms,
                done=True,
            )
            await asyncio.sleep(3.0)
        finally:
            await context.close()
            raw_path = Path(await video.path()) if video is not None else None
            await browser.close()

    if raw_path is None or not raw_path.exists():
        raise RuntimeError("Playwright did not produce a recorded video file.")

    final_video = (confined_out_dir / safe_video_name).resolve(strict=False)
    if not final_video.is_relative_to(confined_out_dir):
        raise ValueError(f"Refusing to write video outside confined output dir: {final_video}")
    shutil.move(str(raw_path), str(final_video))
    with contextlib.suppress(OSError):
        _safe_rmtree_confined(tmp_video_dir, expected_parent=confined_out_dir)

    summary = {
        "video_path": str(final_video),
        "video_size_bytes": final_video.stat().st_size,
        "resolution": "1280x720",
        "task": USER_TASK,
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "avg_perceive_ms": round(sum(s.total_perceive_ms for s in metrics.steps) / max(1, len(metrics.steps)), 1),
        "avg_tokens": round(sum(s.est_tokens for s in metrics.steps) / max(1, len(metrics.steps)), 1),
        "steps": step_log,
    }
    metrics_path = confined_out_dir / "demo_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return final_video, metrics_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a clean, relatable 1280x720 README demo video of Agent-first-browse.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless instead of visible headed Chromium on DISPLAY.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to save raw recorded video (default: docs/assets/raw).",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_VIDEO_NAME,
        help=f"Output WebM filename (default: {DEFAULT_VIDEO_NAME}).",
    )
    args = parser.parse_args()

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_headless = args.headless or not has_display
    print(
        f"Recording 1280x720 raw demo ({'headless' if use_headless else 'headed on screen'}) "
        f"-> {args.out_dir / args.filename}"
    )
    video_path, metrics_path = asyncio.run(
        record_demo(headless=use_headless, out_dir=args.out_dir, video_name=args.filename)
    )
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"Saved raw video  : {video_path} ({size_mb:.2f} MB)")
    print(f"Saved metrics log: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
