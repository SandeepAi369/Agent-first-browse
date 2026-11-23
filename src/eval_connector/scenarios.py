"""eval_connector.scenarios — Reproducible Real-World Deep-Test Scenarios.

Each scenario targets a specific browser-agent failure mode for apples-to-apples
side-by-side testing between `--mode legacy` and `--mode v2`:

  1. `modal_occlusion` — Blocking cookie/consent dialog covering background cards.
  2. `spa_rerender`    — Dynamic DOM re-render that detaches & shifts buttons mid-task,
                         plus a toggle checkbox for testing post-click state verification.
  3. `sparse_aria`     — Custom ARIA elements (`treeitem`, `menuitemcheckbox`, `spinbutton`)
                         where Tier-1 DOM selectors return sparse results (<2) so V2
                         must escalate to Tier-2 Accessibility-Tree extraction.
  4. `multistep_form`  — Multi-step stateful form with secret password input, dropdown,
                         checkbox, and same-URL step transitions (delta compression +
                         V34 secret redaction).
  5. `a11y_visibility` — Mixed visible, CSS-hidden (`display:none`, `visibility:hidden`,
                         `opacity:0`), off-screen actionable, and duplicate-label rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    description: str
    html: str
    suggested_checks: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("html", None)
        return data


_MODAL_OCCLUSION_HTML = """<!doctype html><html><head><title>Modal Occlusion Scenario</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; }
  nav { display: flex; gap: 16px; margin-bottom: 16px; }
  .bg-grid { display: grid; grid-template-columns: repeat(4, 200px); gap: 12px; }
  .card { border: 1px solid #d0d7de; padding: 10px; border-radius: 6px; }
  .backdrop {
    position: fixed; inset: 0; background: rgba(0, 0, 0, 0.65); z-index: 100;
    display: flex; align-items: center; justify-content: center;
  }
  .modal {
    background: #fff; padding: 24px; width: 400px; border-radius: 10px; z-index: 101;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
  }
  .hidden { display: none; }
</style></head><body>
  <header><nav>
    <a href="#home">Home</a>
    <a href="#catalog">Catalog</a>
    <a href="#pricing">Pricing</a>
    <a href="#docs">Docs</a>
  </nav></header>
  <main class="bg-grid">
    __BG_CARDS__
  </main>
  <div id="banner-status" style="margin-top:16px;font-weight:bold;">Dialog Open</div>
  <div id="consent-backdrop" class="backdrop">
    <div class="modal" role="dialog" aria-modal="true" aria-label="Privacy Preferences">
      <h2>Privacy Preferences</h2>
      <p>Select your cookie preference to continue to the catalog.</p>
      <button id="accept-essential" type="button">Accept Essential Cookies</button>
      <button id="manage-prefs" type="button">Manage Settings</button>
    </div>
  </div>
  <script>
    document.getElementById('accept-essential').addEventListener('click', () => {
      document.getElementById('consent-backdrop').classList.add('hidden');
      document.getElementById('banner-status').textContent = 'Dialog Dismissed';
    });
  </script>
</body></html>
""".replace(
    "__BG_CARDS__",
    "".join(
        f'<div class="card"><a href="#item-{i}">Catalog Item {i}</a> '
        f'<button type="button">Inspect Item {i}</button></div>'
        for i in range(20)
    ),
)

_SPA_RERENDER_HTML = """<!doctype html><html><head><title>SPA Dynamic Re-render Scenario</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 24px; }
  #status { margin-top: 320px; font-weight: bold; }
</style></head><body>
  <main id="app">
    <div id="mount">
      <button id="deploy-btn" type="button"
              style="position:absolute;left:40px;top:40px;padding:12px 20px;">
        Confirm Deployment
      </button>
    </div>
    <button id="rerender-btn" type="button"
            style="position:absolute;left:40px;top:120px;padding:8px 14px;">
      Simulate SPA Re-render
    </button>
    <label style="position:absolute;left:40px;top:200px;">
      <input id="canary-cb" type="checkbox" /> Enable Canary Rollout
    </label>
    <div id="status">Idle</div>
  </main>
  <script>
    window.__deployClicks = 0;
    window.__rerendered = false;
    function bindDeploy(btn) {
      btn.addEventListener('click', () => {
        window.__deployClicks += 1;
        btn.setAttribute('aria-pressed', 'true');
        const badge = document.createElement('p');
        badge.textContent =
          'Deployed #' + window.__deployClicks + (window.__rerendered ? ' (after rerender)' : '');
        document.getElementById('status').appendChild(badge);
      });
    }
    bindDeploy(document.getElementById('deploy-btn'));
    window.__triggerRerender = function() {
      window.__rerendered = true;
      const mount = document.getElementById('mount');
      mount.innerHTML =
        '<button id="deploy-btn-v2" type="button" '
        + 'style="position:absolute;left:380px;top:160px;padding:12px 20px;">'
        + 'Confirm Deployment</button>';
      bindDeploy(document.getElementById('deploy-btn-v2'));
      return true;
    };
    document.getElementById('rerender-btn').addEventListener('click', window.__triggerRerender);
  </script>
</body></html>
"""

_SPARSE_ARIA_HTML = """<!doctype html><html><head><title>Sparse Custom ARIA Scenario</title>
<style>
  body { font-family: system-ui, sans-serif; padding: 24px; }
  x-tree, x-item, x-spin, x-check { display: block; padding: 8px 12px; margin: 6px 0; border: 1px solid #ccc; }
</style></head><body>
  <h1>Custom Web-Component Workspace</h1>
  <x-tree role="tree" aria-label="Workspace Navigation">
    <x-item role="treeitem" aria-label="Release Pipeline">Release Pipeline</x-item>
    <x-item role="treeitem" aria-label="Staging Metrics">Staging Metrics</x-item>
    <x-check role="menuitemcheckbox" aria-checked="false" aria-label="Auto-Scale Workers">Auto-Scale Workers</x-check>
    <x-spin role="spinbutton" aria-valuenow="4" aria-label="Replica Count">4</x-spin>
  </x-tree>
</body></html>
"""

_MULTISTEP_FORM_HTML = """<!doctype html><html><head><title>Multi-Step Workflow Form</title>
<style>
  body { font-family: system-ui, sans-serif; padding: 20px; }
  nav { display: flex; gap: 14px; margin-bottom: 16px; }
  form { display: flex; flex-direction: column; gap: 10px; max-width: 420px; }
  .hidden { display: none; }
  #docs-links { margin-top: 20px; display: flex; flex-wrap: wrap; gap: 8px; }
</style></head><body>
  <header><nav>
    <a href="#dash">Dashboard</a>
    <a href="#settings">Settings</a>
    <a href="#billing">Billing</a>
    <a href="#team">Team</a>
    <a href="#audit">Audit Log</a>
  </nav></header>
  <main>
    <form id="onboarding-form">
      <label for="f-name">Full Name</label>
      <input id="f-name" name="full_name" type="text" placeholder="Full Name" />

      <label for="f-email">Work Email</label>
      <input id="f-email" name="work_email" type="email" placeholder="Work Email" />

      <label for="f-pass">Account Password</label>
      <input id="f-pass" name="password" type="password" autocomplete="current-password" value="TopSecret#987" />

      <label for="f-plan">Plan Tier</label>
      <select id="f-plan" name="plan" aria-label="Plan Tier">
        <option value="starter">Starter</option>
        <option value="pro">Pro Team</option>
        <option value="enterprise">Enterprise</option>
      </select>

      <label><input id="f-terms" type="checkbox" /> Accept Service Terms</label>
      <button id="continue-btn" type="button">Continue to Review</button>
    </form>

    <section id="review-panel" class="hidden" aria-label="Review Submission">
      <h2>Review Details</h2>
      <p id="review-summary"></p>
      <button id="final-submit" type="button">Submit Application</button>
    </section>

    <div id="done-banner" class="hidden">Application Submitted</div>

    <section id="docs-links">
      __DOC_LINKS__
    </section>
  </main>
  <script>
    document.getElementById('continue-btn').addEventListener('click', () => {
      const name = document.getElementById('f-name').value || '(empty)';
      const plan = document.getElementById('f-plan').value;
      const terms = document.getElementById('f-terms').checked;
      document.getElementById('review-summary').textContent =
        'Applicant: ' + name + ' | Plan: ' + plan + ' | Terms: ' + terms;
      document.getElementById('review-panel').classList.remove('hidden');
    });
    document.getElementById('final-submit').addEventListener('click', () => {
      document.getElementById('done-banner').classList.remove('hidden');
    });
  </script>
</body></html>
""".replace(
    "__DOC_LINKS__",
    "".join(f'<a href="#guide-{i}">Reference Guide {i}</a>' for i in range(18)),
)

_A11Y_VISIBILITY_HTML = """<!doctype html><html><head><title>Visibility & A11y Parity Scenario</title>
<style>
  body { font-family: system-ui, sans-serif; padding: 20px; }
  .d-none { display: none; }
  .v-hidden { visibility: hidden; }
  .op-zero { opacity: 0; }
  .spacer { height: 700px; }
</style></head><body>
  <main>
    <h1>Visibility Matrix</h1>
    <button id="vis-btn">Visible Primary Action</button>
    <button id="none-btn" class="d-none">Display None Ghost</button>
    <button id="hid-btn" class="v-hidden">Visibility Hidden Ghost</button>
    <button id="op-btn" class="op-zero">Zero Opacity Ghost</button>
    <table>
      <tr><td>Alpha Widget</td><td><a href="/open?id=alpha">Open</a></td></tr>
      <tr><td>Beta Widget</td><td><a href="/open?id=beta">Open</a></td></tr>
      <tr><td>Gamma Widget</td><td><a href="/open?id=gamma">Open</a></td></tr>
    </table>
    <div class="spacer">Long content section</div>
    <button id="far-btn">Save Final Report</button>
    <a href="/privacy-footer">Privacy Policy Footer</a>
  </main>
</body></html>
"""

SCENARIOS: dict[str, ScenarioSpec] = {
    "modal_occlusion": ScenarioSpec(
        name="modal_occlusion",
        title="Modal & Overlay Occlusion",
        description=(
            "Blocking consent dialog covering 40+ background links/buttons. "
            "Tests paint-order hit-test pruning, active modal header, and prompt token reduction."
        ),
        html=_MODAL_OCCLUSION_HTML,
        suggested_checks=(
            "Compare occluded background elements pruned by V2 vs exposed by Legacy",
            "Dismiss modal via 'Accept Essential Cookies' and re-perceive to confirm background unlocks",
        ),
    ),
    "spa_rerender": ScenarioSpec(
        name="spa_rerender",
        title="SPA Dynamic Re-render & Stale-Ref Recovery",
        description=(
            "Detaches and shifts 'Confirm Deployment' button after perception to test "
            "V2 stale-ref semantic rebinding vs Legacy stale-coordinate fallback."
        ),
        html=_SPA_RERENDER_HTML,
        suggested_checks=(
            "Perceive initial DOM, trigger re-render, then click original 'Confirm Deployment' ref",
            "Toggle 'Enable Canary Rollout' checkbox to verify live window.__aid state detection",
        ),
    ),
    "sparse_aria": ScenarioSpec(
        name="sparse_aria",
        title="Sparse Custom ARIA Components (Tier-2 Escalation)",
        description=(
            "Custom elements using ARIA roles (treeitem, menuitemcheckbox, spinbutton) "
            "where Tier-1 DOM extraction is sparse (<2) so V2 escalates to Tier-2 A11y."
        ),
        html=_SPARSE_ARIA_HTML,
        suggested_checks=(
            "Compare legacy element_count (0) vs V2 Tier-2 A11y escalation (4 elements)",
        ),
    ),
    "multistep_form": ScenarioSpec(
        name="multistep_form",
        title="Multi-Step Stateful Form & Secret Redaction",
        description=(
            "Consecutive same-URL form steps with password input, dropdown, checkbox, "
            "and dynamic review panel reveal. Tests delta compression and V34 secret masking."
        ),
        html=_MULTISTEP_FORM_HTML,
        suggested_checks=(
            "Fill Full Name, Work Email, select Pro Team, check Terms, click Continue to Review",
            "Verify Step 2+ triggers V2 delta compression and never exposes 'TopSecret#987'",
        ),
    ),
    "a11y_visibility": ScenarioSpec(
        name="a11y_visibility",
        title="Visibility & Accessibility Tree Parity",
        description=(
            "Mix of visible, display:none, visibility:hidden, opacity:0 controls, "
            "duplicate-label rows, and off-screen actionables."
        ),
        html=_A11Y_VISIBILITY_HTML,
        suggested_checks=(
            "Verify CSS-hidden ghosts are excluded in both modes",
            "Verify duplicate 'Open' links carry distinct disambiguation hints and off-screen button is recalled",
        ),
    ),
}


def list_scenarios() -> list[dict[str, Any]]:
    """Return metadata summaries for all built-in deep-test scenarios."""
    return [spec.summary() for spec in SCENARIOS.values()]


def get_scenario(name: str) -> ScenarioSpec:
    """Lookup a built-in scenario by name (case-insensitive)."""
    key = (name or "").strip().lower()
    if key not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario '{name}'. Available scenarios: {valid}")
    return SCENARIOS[key]
