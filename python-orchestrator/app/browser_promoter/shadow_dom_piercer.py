from __future__ import annotations

from dataclasses import dataclass

from playwright.async_api import Page


@dataclass(slots=True)
class TargetPoint:
    x: float
    y: float
    method: str
    selector_used: str = ""
    text_hint: str = ""


async def locate_target_point(
    page: Page,
    *,
    selector: str,
    text_hint: str = "",
) -> TargetPoint | None:
    """Locate a target in normal/light/shadow DOM with native then deep fallback.

    Flow:
      1) Native Playwright selector resolution (including >> shadow piercer path).
      2) Recursive JS traversal of open + captured closed shadow roots.
    """
    normalized = selector.strip()
    if normalized:
        point = await _locate_with_native_selector(page, normalized)
        if point is not None:
            return point

        if ">>>" in normalized:
            point = await _locate_with_native_selector(page, normalized.replace(">>>", ">>"))
            if point is not None:
                return point

    return await _locate_with_recursive_shadow_probe(page, selector=normalized, text_hint=text_hint.strip())


async def _locate_with_native_selector(page: Page, selector: str) -> TargetPoint | None:
    locator = page.locator(selector).first
    count = await locator.count()
    if count == 0:
        return None

    box = await locator.bounding_box()
    if box is None:
        return None

    return TargetPoint(
        x=box["x"] + box["width"] / 2.0,
        y=box["y"] + box["height"] / 2.0,
        method="native-shadow-piercer",
        selector_used=selector,
    )


async def _locate_with_recursive_shadow_probe(
    page: Page,
    *,
    selector: str,
    text_hint: str,
) -> TargetPoint | None:
    result = await page.evaluate(
        """
({ selector, textHint }) => {
  const normalize = (value) => String(value || '').trim().toLowerCase();
  const wantedText = normalize(textHint);

  const enqueueRoots = (root, queue, visited) => {
    if (!root || visited.has(root)) return;
    visited.add(root);
    queue.push(root);
  };

  const pickByText = (root) => {
    if (!wantedText || !root.querySelectorAll) return null;
    const all = root.querySelectorAll('*');
    for (const el of all) {
      const text = normalize(el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('title'));
      if (text && text.includes(wantedText)) return el;
    }
    return null;
  };

  const queue = [];
  const visited = new Set();
  enqueueRoots(document, queue, visited);

  while (queue.length > 0) {
    const root = queue.shift();
    let found = null;

    if (selector && root.querySelector) {
      try {
        found = root.querySelector(selector);
      } catch (_) {}
    }

    if (!found) {
      found = pickByText(root);
    }

    if (found) {
      const rect = found.getBoundingClientRect();
      if (rect && rect.width > 0 && rect.height > 0) {
        return {
          x: rect.left + rect.width / 2 + window.scrollX,
          y: rect.top + rect.height / 2 + window.scrollY,
          text: (found.innerText || found.textContent || '').slice(0, 120),
        };
      }
    }

    if (!root.querySelectorAll) continue;
    const nodes = root.querySelectorAll('*');
    for (const node of nodes) {
      if (node.shadowRoot) {
        enqueueRoots(node.shadowRoot, queue, visited);
      }
      if (node.__closedShadowRoot__) {
        enqueueRoots(node.__closedShadowRoot__, queue, visited);
      }
    }
  }

  return null;
}
""",
        {
            "selector": selector,
            "textHint": text_hint,
        },
    )

    if not result:
        return None

    return TargetPoint(
        x=float(result["x"]),
        y=float(result["y"]),
        method="recursive-shadow-probe",
        selector_used=selector,
        text_hint=text_hint,
    )
