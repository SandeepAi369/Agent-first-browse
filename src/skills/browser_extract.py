"""Extract Skill — Structured data extraction from the DOM.

Uses the God-Mode DOM parser to extract interactive elements,
semantic tree structure, and arbitrary data from any web page.
This is the perception layer of the orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from playwright.async_api import Page

from skills.base import Skill, SkillResult

logger = logging.getLogger("skills.extract")


class ExtractSkill(Skill):
    """Extract structured data from the current page DOM."""

    name = "extract"

    async def run(self, params: dict[str, Any]) -> SkillResult:
        """Extract data from the page.

        Params:
            mode (str): Extraction mode:
                - "dom_map": Get interactive elements + semantic tree (default)
                - "text_content": Get all visible text content
                - "query": Run a CSS selector and extract text/attributes
                - "js_eval": Evaluate arbitrary JS and return the result
                - "screenshot": Capture a screenshot
                - "form_fields": Detect all form fields

            selector (str): CSS selector for "query" mode
            attributes (list[str]): Element attributes to extract in "query" mode
            js_code (str): JavaScript code for "js_eval" mode
            target_hint (str): Semantic hint for dom_map (e.g., "title", "body", "submit")
        """
        mode = params.get("mode", "dom_map")

        if mode == "dom_map":
            return await self._extract_dom_map(params)
        elif mode == "text_content":
            return await self._extract_text(params)
        elif mode == "query":
            return await self._extract_query(params)
        elif mode == "js_eval":
            return await self._extract_js(params)
        elif mode == "screenshot":
            return await self._extract_screenshot(params)
        elif mode == "form_fields":
            return await self._extract_form_fields(params)
        else:
            return SkillResult(
                success=False,
                summary=f"Unknown extraction mode: {mode}",
                error=f"Unsupported mode: {mode}",
            )

    # ── DOM Map (God-Mode Parser) ─────────────────────────────────────────

    async def _extract_dom_map(self, params: dict) -> SkillResult:
        """Use dom_parser.extract() to get the full interactive element map + semantic tree."""
        import dom_parser

        target_hint = params.get("target_hint")
        timeout = params.get("timeout", 5.0)

        try:
            dom_data = await dom_parser.extract(
                self.page,
                target_hint=target_hint,
                timeout=timeout,
            )
            elements = dom_data.get("elements", [])
            dom_tree = dom_data.get("dom_tree", "")

            logger.info(
                "DOM extracted: %d elements, tree=%d chars",
                len(elements), len(dom_tree),
            )

            return SkillResult(
                success=True,
                summary=f"Extracted {len(elements)} interactive elements",
                data={
                    "elements": elements,
                    "dom_tree": dom_tree,
                    "image_size": dom_data.get("image_size", {}),
                },
            )
        except Exception as e:
            logger.warning("DOM extraction failed: %s", e)
            return SkillResult(
                success=False,
                summary="DOM extraction failed",
                error=str(e),
            )

    # ── Text Content ──────────────────────────────────────────────────────

    async def _extract_text(self, params: dict) -> SkillResult:
        """Extract all visible text from the page."""
        try:
            text = await asyncio.wait_for(
                self.page.evaluate("""
                    () => {
                        const walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_TEXT,
                            { acceptNode: (node) => {
                                const style = window.getComputedStyle(node.parentElement);
                                if (style.display === 'none' || style.visibility === 'hidden') {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
                            }}
                        );
                        const texts = [];
                        while (walker.nextNode()) {
                            texts.push(walker.currentNode.textContent.trim());
                        }
                        return texts.join('\\n').substring(0, 10000);
                    }
                """),
                timeout=5.0,
            )

            return SkillResult(
                success=True,
                summary=f"Extracted {len(text)} chars of visible text",
                data={"text": text, "length": len(text)},
            )
        except Exception as e:
            return SkillResult(
                success=False,
                summary="Text extraction failed",
                error=str(e),
            )

    # ── CSS Selector Query ────────────────────────────────────────────────

    async def _extract_query(self, params: dict) -> SkillResult:
        """Query DOM elements by CSS selector and extract their data."""
        selector = params.get("selector")
        if not selector:
            return SkillResult(
                success=False,
                summary="No selector provided",
                error="Missing param: selector",
            )

        attributes = params.get("attributes", ["textContent"])
        max_results = params.get("max_results", 50)

        try:
            results = await asyncio.wait_for(
                self.page.evaluate("""
                    ({selector, attributes, maxResults}) => {
                        const elements = document.querySelectorAll(selector);
                        const out = [];
                        for (let i = 0; i < Math.min(elements.length, maxResults); i++) {
                            const el = elements[i];
                            const item = {};
                            for (const attr of attributes) {
                                if (attr === 'textContent') {
                                    item[attr] = (el.textContent || '').trim().substring(0, 500);
                                } else if (attr === 'innerHTML') {
                                    item[attr] = (el.innerHTML || '').substring(0, 1000);
                                } else {
                                    item[attr] = el.getAttribute(attr) || '';
                                }
                            }
                            // Always include bounding box
                            const rect = el.getBoundingClientRect();
                            item._x = Math.round(rect.left + rect.width / 2);
                            item._y = Math.round(rect.top + rect.height / 2);
                            out.push(item);
                        }
                        return out;
                    }
                """, {"selector": selector, "attributes": attributes, "maxResults": max_results}),
                timeout=5.0,
            )

            logger.info("Query '%s': found %d elements", selector, len(results))
            return SkillResult(
                success=True,
                summary=f"Found {len(results)} elements matching '{selector}'",
                data={"results": results, "count": len(results)},
            )
        except Exception as e:
            return SkillResult(
                success=False,
                summary=f"Query '{selector}' failed",
                error=str(e),
            )

    # ── JavaScript Evaluation ─────────────────────────────────────────────

    async def _extract_js(self, params: dict) -> SkillResult:
        """Evaluate arbitrary JavaScript and return the result."""
        js_code = params.get("js_code")
        if not js_code:
            return SkillResult(
                success=False,
                summary="No JS code provided",
                error="Missing param: js_code",
            )

        try:
            result = await asyncio.wait_for(
                self.page.evaluate(js_code),
                timeout=10.0,
            )
            return SkillResult(
                success=True,
                summary="JS evaluated successfully",
                data={"result": result},
            )
        except Exception as e:
            return SkillResult(
                success=False,
                summary="JS evaluation failed",
                error=str(e),
            )

    # ── Screenshot ────────────────────────────────────────────────────────

    async def _extract_screenshot(self, params: dict) -> SkillResult:
        """Capture a screenshot of the current page."""
        try:
            fmt = params.get("format", "jpeg")
            quality = params.get("quality", 60)
            screenshot_bytes = await self.page.screenshot(type=fmt, quality=quality)
            return SkillResult(
                success=True,
                summary=f"Screenshot captured ({len(screenshot_bytes)} bytes)",
                data={"size_bytes": len(screenshot_bytes), "format": fmt},
            )
        except Exception as e:
            return SkillResult(
                success=False,
                summary="Screenshot failed",
                error=str(e),
            )

    # ── Form Fields Detection ─────────────────────────────────────────────

    async def _extract_form_fields(self, params: dict) -> SkillResult:
        """Detect all editable form fields on the page."""
        import dom_parser

        try:
            form_data = await dom_parser.detect_form_fields(self.page)
            fields = form_data.get("fields", [])
            return SkillResult(
                success=True,
                summary=f"Detected {len(fields)} form fields",
                data={"fields": fields},
            )
        except Exception as e:
            return SkillResult(
                success=False,
                summary="Form field detection failed",
                error=str(e),
            )
