Reddit Login Failure Research Plan (Legitimate Access Only)

Goal
- Determine why manual login in automated browser profile reports wrong password or blocks auth despite correct credentials.
- Produce a reliable, policy-safe workflow for authenticated posting that does not bypass security controls.

Scope Guardrails
- No bypass techniques.
- No anti-bot evasion payloads.
- No credential stuffing or automation of password guessing.
- Focus on reproducible diagnostics, environment stability, and account-security flows.

Hypotheses To Validate
1. Account-state issue: password reset required, suspicious-login lock, 2FA mismatch, or temporary security hold.
2. Session profile corruption: stale cookies/local storage causing bad login state.
3. IP/network reputation or VPN/proxy mismatch triggering challenge loops.
4. Browser runtime mismatch: login succeeds in standard browser, fails in automation profile due profile/config differences.
5. Username format issue: username vs email login path mismatch.
6. Timing/rate-limit issue after repeated failed attempts.

Required Evidence
- Exact timestamped login attempts.
- Screenshot + HTML of login page after each failed attempt.
- Visible error text (verbatim).
- Cookie inventory before and after login attempt.
- Endpoint status for /api/v1/me after login attempt.
- Whether the same credentials succeed in normal browser session on same network.

Research Agent Prompt Template
Use this exact prompt with your research agent:

"Investigate legitimate root causes for Reddit login failures where a user enters correct credentials but receives wrong-password/security errors in a Playwright persistent profile.

Constraints:
- Do not provide bypass or anti-bot evasion guidance.
- Focus only on compliant troubleshooting and reliability practices.

Deliverables:
1) Ranked root-cause matrix with confidence levels.
2) Observable indicators for each cause (UI text, status codes, cookie patterns).
3) Safe remediation sequence ordered by speed and risk.
4) A minimal diagnostic checklist to run before each login attempt.
5) Recommendations for human-in-the-loop login handling with persistent sessions.
6) Rollback criteria: when to stop automation and require manual account recovery.

Context:
- Persistent profile path: /home/sandeep/agent first IDE/persistence/browser_sessions/reddit/reddit-main-session
- Current script: /home/sandeep/agent first IDE/scripts/direct_live_reddit_post.py
- Symptoms: repeated auth failures, possible challenge/captcha/rate-limit, uncertain UI state."

Execution Plan
Step 1: Baseline outside automation
- Login manually in a regular browser on same network.
- Confirm account health, 2FA requirements, and any security notices.

Step 2: Clean profile A/B test
- Test with fresh persistent profile directory.
- Compare with existing profile behavior.

Step 3: Challenge handling validation
- Confirm whether email/captcha/2FA prompts appear.
- Complete them manually and re-check authenticated state.

Step 4: Rate-limit and timing controls
- Space attempts to avoid lockouts.
- Capture whether error changes over cooldown period.

Step 5: Script-side diagnostics review
- Review saved artifacts under persistence/login_diagnostics.
- Map each artifact to root-cause matrix.

Step 6: Exit criteria
- Proceed to posting only after authenticated state is confirmed.
- If still failing after max attempts, stop and escalate to account recovery.

Expected Outputs
- Root-cause matrix table.
- Recommended standard operating procedure for login.
- Updated checklist for operators running live post scripts.
