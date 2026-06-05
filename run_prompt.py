import asyncio
from advanced_agent import run_agent

if __name__ == "__main__":
    prompt = """Go to https://issuetracker.google.com/issues/new and submit this bug report. If that page requires login or doesn't load a form, try https://support.google.com/gemini/community instead and create a new post there.

Title: Bug Report & Feedback: Gemini 3.1 Pro Search Instability & Antigravity UI Regressions

Overview
This report highlights two ongoing issues severely impacting the developer experience: persistent instability with Gemini 3.1 Pro's search capabilities and significant UI/UX regressions in the latest version of the Google Antigravity IDE.

1. Gemini 3.1 Pro Search Instability (API & Android)
Issue: The web search capability within Gemini 3.1 Pro frequently fails, hangs, or returns errors.
Environment: This is consistently encountered both on the Android application and programmatically via the Gemini API.
Impact: Despite multiple previous reports, this routing and search execution issue remains unresolved. The unreliability actively disrupts workflows that depend on real-time data retrieval and search-augmented generation. A permanent resolution is highly requested.

2. Google Antigravity UI Regressions & Window Management
Issue: Recent updates to Google Antigravity have introduced frustrating window management restrictions that clutter the workspace.
Missing Close Controls: When opening specific panes or agent windows, there is no dedicated button provided to close them. The interface quickly becomes unmanageable.
Derivative Interface: The current UI layout feels less like a polished Google product and more like a direct, unrefined copy of OpenAI's Codex/Cursor interfaces.
Loss of the Sandbox View: The previous version featured a highly effective Sandbox environment where developers could review code and execution plans in a single, unified view. This one-box approach felt professional, streamlined, and intentional. The new, fragmented windowing system is a major step backward in usability.

Expected Resolutions
1. Deploy a permanent fix for the Gemini 3.1 Pro API and Android search routing.
2. Restore standard window controls (close buttons) to all panels within the Antigravity IDE.
3. Reintroduce the unified Sandbox view for simultaneous code and plan review to restore professional workflow efficiency."""
    asyncio.run(run_agent(prompt))
