"""Sidebar "N live" badge: a row with live delegate children renders the badge.

`_live_subagent_count` is a point-in-time fact fetched from
`GET /api/subagents/live` (state.db: `source='subagent' AND ended_at IS NULL`),
re-applied on every sidebar render by `_applyLiveSubagentCounts`. The renderer
must treat it as data — never infer liveness from a title, a timestamp or silence.

The repo intentionally avoids a node/jsdom dependency (see AGENTS.md), so this
file follows the WEBUI-SUB convention: precise structural assertions on
`static/sessions.js` / `static/style.css` / `static/i18n.js`, plus real
execution of the extracted render fragment and of `_applyLiveSubagentCounts`
under node with a minimal fake DOM. That way dropping the guard, painting a
zero, or letting a finished run keep its badge reddens a test instead of
silently rendering nothing.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_JS = (REPO_ROOT / "static" / "sessions.js").read_text(encoding="utf-8")
STYLE_CSS = (REPO_ROOT / "static" / "style.css").read_text(encoding="utf-8")
I18N_JS = (REPO_ROOT / "static" / "i18n.js").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node required")


LIVE_PREDICATE = "typeof s._live_subagent_count==='number'&&s._live_subagent_count>0"


def _live_badge_block() -> str:
    """The `if(<live predicate>){...}` fragment from the row renderer."""
    pattern = r"^    if\(" + re.escape(LIVE_PREDICATE) + r"\)\{.*?^    \}\n"
    m = re.search(pattern, SESSIONS_JS, re.S | re.M)
    assert m, "live subagent badge block not found in static/sessions.js"
    return m.group(0)


def _run_render(rows):
    """Execute the extracted fragment against a minimal fake DOM."""
    block = _live_badge_block()
    harness = f"""
class FakeEl {{
  constructor(tag) {{
    this.tagName = tag.toUpperCase();
    this.className = '';
    this.textContent = '';
    this.title = '';
    this.children = [];
  }}
  appendChild(child) {{ this.children.push(child); return child; }}
}}
const document = {{ createElement: (tag) => new FakeEl(tag) }};
function t(key, ...args) {{
  const table = {{
    session_live_subagents: (n) => `${{n}} live`,
    session_live_subagents_tooltip: 'Subagents running now',
  }};
  const value = table[key];
  if (value === undefined) return key;
  if (typeof value === 'function') return value(...args);
  return value.replace('{{0}}', String(args[0] === undefined ? '' : args[0]));
}}

const out = [];
for (const s of {json.dumps(rows)}) {{
  const titleRow = new FakeEl('span');
  {block}
  out.push(titleRow.children.map(c => ({{
    className: c.className,
    textContent: c.textContent,
    title: c.title,
  }})));
}}
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _run_apply(rows, counts):
    """Execute the real `_applyLiveSubagentCounts` against a fake live map."""
    m = re.search(
        r"^function _applyLiveSubagentCounts\(rows\)\{.*?^\}\n", SESSIONS_JS, re.S | re.M
    )
    assert m, "_applyLiveSubagentCounts not found in static/sessions.js"
    harness = f"""
let _liveSubagentCounts = {json.dumps(counts)};
{m.group(0)}
const rows = {json.dumps(rows)};
console.log(JSON.stringify(_applyLiveSubagentCounts(rows)));
"""
    proc = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# ── renderer ────────────────────────────────────────────────────────────────


def test_row_with_live_children_renders_the_badge():
    out = _run_render([{"session_id": "a", "_live_subagent_count": 3}])
    assert len(out[0]) == 1, out
    badge = out[0][0]
    assert badge["className"] == "session-live-subagents-badge"
    assert badge["textContent"] == "3 live"
    assert badge["title"] == "Subagents running now"


def test_row_without_live_children_renders_nothing():
    out = _run_render(
        [{"session_id": "b"}, {"session_id": "c", "_live_subagent_count": 0}]
    )
    assert out == [[], []], out


def test_predicate_rejects_non_numeric_counts():
    """A string or null from a malformed payload must not paint a badge."""
    out = _run_render(
        [
            {"session_id": "d", "_live_subagent_count": "2"},
            {"session_id": "e", "_live_subagent_count": None},
        ]
    )
    assert out == [[], []], out


# ── decoration loop ─────────────────────────────────────────────────────────


def test_apply_sets_the_count_from_the_live_map():
    rows = _run_apply([{"session_id": "a"}, {"session_id": "z"}], {"a": 2})
    assert rows[0]["_live_subagent_count"] == 2
    assert "_live_subagent_count" not in rows[1], rows


def test_apply_clears_a_stale_count():
    """A finished run must lose its badge on the next render, not on reload."""
    rows = _run_apply([{"session_id": "a", "_live_subagent_count": 5}], {})
    assert "_live_subagent_count" not in rows[0], rows


def test_apply_ignores_empty_and_malformed_maps():
    for counts in ({}, [], None):
        rows = _run_apply([{"session_id": "a", "_live_subagent_count": 1}], counts)
        assert "_live_subagent_count" not in rows[0], counts


def test_render_pipeline_applies_live_counts():
    """The decorator must run inside the render path, or the badge never paints."""
    m = re.search(
        r"^function _renderSidebarRowsFromRawSessions\(.*?^\}\n", SESSIONS_JS, re.S | re.M
    )
    assert m, "_renderSidebarRowsFromRawSessions not found"
    assert "_applyLiveSubagentCounts(rows)" in m.group(0), (
        "the sidebar render path must decorate rows with live subagent counts"
    )
    # Guarded on purpose: browser tests extract this function alone and eval it
    # without the decorator, so the call must be tolerated when undefined.
    assert "typeof _applyLiveSubagentCounts==='function'" in m.group(0), (
        "the decorator call must stay guarded for the extracted-function harnesses"
    )


# ── fetch wiring ────────────────────────────────────────────────────────────


def test_client_fetches_the_live_endpoint():
    m = re.search(
        r"^async function refreshLiveSubagentCounts\(\)\{.*?^\}\n", SESSIONS_JS, re.S | re.M
    )
    assert m, "refreshLiveSubagentCounts not found"
    body = m.group(0)
    assert "/api/subagents/live" in body
    # The badge is decoration: a failing poll must not toast or redirect.
    assert "timeoutToast:false" in body
    assert "redirect401:false" in body


def test_poll_refreshes_counts_before_rendering():
    """Counts must be current when the list repaints, not one tick behind."""
    m = re.search(r"^function startStreamingPoll\(\)\{.*?\n  \}\n", SESSIONS_JS, re.S | re.M)
    assert m, "startStreamingPoll not found"
    body = m.group(0)
    assert body.count("refreshLiveSubagentCounts()") >= 2, (
        "both the interval tick and the visibilitychange handler must refresh"
    )
    tick = body.index("refreshLiveSubagentCounts()")
    render = body.index("renderSessionList({deferWhileInteracting:true})")
    assert tick < render, "refresh must precede the repaint it feeds"


# ── style ───────────────────────────────────────────────────────────────────


def test_badge_style_is_defined_once():
    assert "\n.session-live-subagents-badge{" in STYLE_CSS
    assert STYLE_CSS.count("\n.session-live-subagents-badge{") == 1
    assert ".session-item.active .session-live-subagents-badge{" in STYLE_CSS


def test_badge_respects_reduced_motion():
    assert "prefers-reduced-motion:reduce" in STYLE_CSS
    assert ".session-live-subagents-badge{animation:none;}" in STYLE_CSS


# ── i18n ────────────────────────────────────────────────────────────────────


def _locale_block(loc: str) -> str:
    m = re.search(rf"^  (?:'{loc}'|{loc}): \{{\s*$", I18N_JS, re.M)
    assert m, f"locale {loc} not found"
    start = m.start()
    nxt = re.search(r"^  (?:'[A-Za-z-]+'|[a-z]{2}): \{\s*$", I18N_JS[start + 5 :], re.M)
    end = start + 5 + nxt.start() if nxt else len(I18N_JS)
    return I18N_JS[start:end]


def test_every_locale_defines_both_live_keys():
    # Locale keys are bare (`en: {`) or quoted when they carry a hyphen
    # (`'zh-Hant': {`). An unquoted-only pattern silently skips zh-Hant and the
    # "every locale" assertion becomes a 14-of-15 claim that still passes.
    locales = re.findall(r"^  (?:'([A-Za-z-]+)'|([a-z]{2})): \{\s*$", I18N_JS, re.M)
    names = [a or b for a, b in locales]
    assert names, "no locale blocks found"
    assert len(names) >= 15, f"only {len(names)} locales found: {names}"
    missing = [
        loc
        for loc in names
        if "session_live_subagents:" not in _locale_block(loc)
        or "session_live_subagents_tooltip:" not in _locale_block(loc)
    ]
    assert not missing, f"locales missing live-subagent keys: {missing}"


# ── endpoint ────────────────────────────────────────────────────────────────


class _FakeHandler:
    def __init__(self):
        self.status = None
        self.headers = {}
        self.wfile = io.BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass

    def json_body(self):
        return json.loads(self.wfile.getvalue().decode("utf-8"))


def _routes_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import api.routes as routes

    return routes


def _reset_cache(routes):
    routes._live_subagents_cache["at"] = 0.0
    routes._live_subagents_cache["payload"] = None


def test_endpoint_returns_the_parent_to_count_map(monkeypatch):
    routes = _routes_module()
    monkeypatch.setattr(
        routes, "_live_subagent_counts", lambda now=None: {"parent1": 3, "parent2": 1}
    )
    _reset_cache(routes)
    handler = _FakeHandler()
    routes._handle_live_subagents(handler, None)
    assert handler.status == 200
    body = handler.json_body()
    assert body["counts"] == {"parent1": 3, "parent2": 1}
    assert body["total"] == 4
    assert body["read_only"] is True


def test_endpoint_degrades_to_an_empty_map_on_failure(monkeypatch):
    """The badge is decoration: a broken probe must not 500 the sidebar."""
    routes = _routes_module()

    def boom(now=None):
        raise RuntimeError("state.db unavailable")

    monkeypatch.setattr(routes, "_live_subagent_counts", boom)
    _reset_cache(routes)
    handler = _FakeHandler()
    routes._handle_live_subagents(handler, None)
    assert handler.status == 200
    assert handler.json_body()["counts"] == {}


def test_endpoint_serves_from_cache_inside_the_ttl(monkeypatch):
    routes = _routes_module()
    calls = {"n": 0}

    def counting(now=None):
        calls["n"] += 1
        return {"p": calls["n"]}

    monkeypatch.setattr(routes, "_live_subagent_counts", counting)
    _reset_cache(routes)
    first, second = _FakeHandler(), _FakeHandler()
    routes._handle_live_subagents(first, None)
    routes._handle_live_subagents(second, None)
    assert calls["n"] == 1, "the second call inside the TTL must not re-probe"
    assert first.json_body()["generated_at"] == second.json_body()["generated_at"]


def test_endpoint_is_registered_on_get():
    assert 'parsed.path == "/api/subagents/live"' in (
        REPO_ROOT / "api" / "routes.py"
    ).read_text(encoding="utf-8")


def test_liveness_predicate_is_the_runner_lifecycle():
    """`ended_at IS NULL` is the delegate run lifecycle — verified against a real
    fan-out. Losing it would count finished children forever."""
    routes_src = (REPO_ROOT / "api" / "routes.py").read_text(encoding="utf-8")
    m = re.search(r"def _live_subagent_counts\(.*?\n\n\n", routes_src, re.S)
    assert m, "_live_subagent_counts not found"
    body = m.group(0)
    assert "s.source = 'subagent'" in body
    assert "s.ended_at IS NULL" in body
    assert "s.parent_session_id" in body
