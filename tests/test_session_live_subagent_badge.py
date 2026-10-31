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
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_JS = (REPO_ROOT / "static" / "sessions.js").read_text(encoding="utf-8")
STYLE_CSS = (REPO_ROOT / "static" / "style.css").read_text(encoding="utf-8")
I18N_JS = (REPO_ROOT / "static" / "i18n.js").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node required")


LIVE_PREDICATE = (
    "typeof s._live_subagent_count==='number'&&s._live_subagent_count>0&&childCount===0"
)


def _live_badge_block() -> str:
    """The `if(<live predicate>){...}` fragment from the row renderer."""
    pattern = r"^    if\(" + re.escape(LIVE_PREDICATE) + r"\)\{.*?^    \}\n"
    m = re.search(pattern, SESSIONS_JS, re.S | re.M)
    assert m, "live subagent badge block not found in static/sessions.js"
    return m.group(0)


def _child_pill_block() -> str:
    """The row's child pill, live-number included, from `const childCount=` on."""
    start = SESSIONS_JS.index("const childCount=typeof s._child_session_count")
    end = SESSIONS_JS.index("titleRow.appendChild(childCountEl);", start) + len(
        "titleRow.appendChild(childCountEl);\n    }"
    )
    block = SESSIONS_JS[start:end]
    assert "session-child-count" in block
    return block


# The pill reads `t()`, the tooltip helper and the toggle plumbing; only the
# last three are stubbed — the class, the label and the tooltip under test are
# the real lines.
CHILD_PILL_STUBS = """
const _expandedChildSessionKeys = new Set();
function _retireProjectPickerForExplicitRepaint(){}
function _sidebarLineageKeyForRow(s){ return s.session_id; }
function renderSessionListFromCache(){}
function _sessionChildBadgeTooltip(label){ return label + ' — click to show nested sessions'; }
function _formatRelativeSessionTime(){ return 'now'; }
function _isMessagingSession(){ return false; }
function _getChannelLabel(){ return null; }
function _sourceKeyForSession(){ return null; }
"""


_PRELUDE_JS = """
const fs = require('fs');
const vm = require('vm');
const i18nSrc = fs.readFileSync(__I18N_PATH__, 'utf8');
const sandbox = {
  navigator: { language: 'en' },
  localStorage: { getItem: () => null, setItem: () => {} },
  document: { documentElement: {}, addEventListener() {} },
};
sandbox.window = sandbox;
const ctx = vm.createContext(sandbox);
vm.runInContext(i18nSrc + ';globalThis.__LOCALES = LOCALES;', ctx);
const LOCALES = sandbox.__LOCALES;
const t = (key, ...args) => {
  const val = LOCALES.en[key];
  if (val === undefined) return key;
  if (typeof val === 'function') return val(...args);
  if (args.length) return String(val).replace(/\\{(\\d+)\\}/g, (m, i) => (i === '0' ? String(args[0]) : m));
  return val;
};

class FakeEl {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.className = '';
    this.textContent = '';
    this.title = '';
    this.children = [];
  }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener() {}
}
const document = { createElement: (tag) => new FakeEl(tag) };
""".replace("__I18N_PATH__", json.dumps(str(REPO_ROOT / "static" / "i18n.js")))

_COLLECT_JS = """
  out.push(titleRow.children.map(c => ({
    className: c.className,
    textContent: c.textContent,
    title: c.title,
  })));
"""


def _run_render(rows):
    """Execute the extracted fragment against a minimal fake DOM.

    The i18n table is the REAL `static/i18n.js`, loaded in the same node process:
    a hand-rolled fake table is how the shipped "undefined subagents live"
    tooltip (function-valued entry called with no argument) stayed green here
    while being broken in every locale. Loading the real module means the
    harness asserts what production actually renders.
    """
    block = _live_badge_block()
    harness = _PRELUDE_JS + f"""
const out = [];
for (const s of {json.dumps(rows)}) {{
  const titleRow = new FakeEl('span');
  const childCount=typeof s._child_session_count==='number'?s._child_session_count:(Array.isArray(s._child_sessions)?s._child_sessions.length:0);
  {block}
{_COLLECT_JS}}}
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _run_child_pill(rows):
    """Execute the REAL child-pill fragment (live number included).

    This is the surface the card is about: the number the sidebar paints must
    be the children running now, not the historical total.
    """
    block = _child_pill_block()
    harness = _PRELUDE_JS + CHILD_PILL_STUBS + f"""
const out = [];
for (const s of {json.dumps(rows)}) {{
  const titleRow = new FakeEl('span');
  {block}
{_COLLECT_JS}}}
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
    # The count must appear in the label — that is the whole point of the badge.
    assert badge["textContent"] == "3 live"
    # And the tooltip must be real text, never a rendered `undefined` (the
    # shipped bug: a function-valued locale entry called with no argument).
    assert badge["title"] == "Subagents running now"
    assert "undefined" not in badge["title"]
    assert "{" not in badge["textContent"] and "{" not in badge["title"]


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


# ── child pill: the visible number is live ──────────────────────────────────


def _three_children(**extra):
    row = {
        "session_id": "p",
        "_child_session_count": 3,
        "_child_sessions": [
            {"session_id": "k1"},
            {"session_id": "k2"},
            {"session_id": "k3"},
        ],
    }
    row.update(extra)
    return row


def test_three_finished_children_paint_no_number():
    """The card's case: subagents that finished days ago must not read "3".

    The pill survives as the expander for the nested children — it just carries
    no number, so the row never claims work that is not running. The historical
    total is only in the tooltip.
    """
    out = _run_child_pill([_three_children()])
    assert len(out[0]) == 1, out
    pill = out[0][0]
    assert pill["className"] == "session-child-count", pill
    assert "is-live" not in pill["className"], pill
    assert not any(ch.isdigit() for ch in pill["textContent"]), pill
    assert "3" not in pill["textContent"], pill
    assert pill["textContent"] == "›", pill
    # The count is not lost — it moved to the tooltip, which costs nothing.
    assert "3" in pill["title"], pill


def test_two_finished_one_running_reads_one_live():
    out = _run_child_pill([_three_children(_live_subagent_count=1)])
    assert len(out[0]) == 1, out
    pill = out[0][0]
    assert pill["className"] == "session-child-count is-live", pill
    assert pill["textContent"] == "1 live", pill
    # Historical total stays available but must not be the painted number.
    assert "3" not in pill["textContent"], pill
    assert "3" in pill["title"], pill
    assert "Subagents running now" in pill["title"], pill


def test_pill_never_paints_zero():
    """No live child means no badge at all — never "0 live"."""
    for extra in ({"_live_subagent_count": 0}, {}):
        out = _run_child_pill([_three_children(**extra)])
        assert "0" not in out[0][0]["textContent"], (extra, out)
        assert "live" not in out[0][0]["textContent"], (extra, out)


def test_pill_number_follows_the_live_map_between_renders():
    """A child that goes idle past the threshold drops out of the number.

    `_applyLiveSubagentCounts` replaces the count wholesale, so the same row
    reads "2 live" while the map holds it and loses the number once the child
    stops being live — no page reload, no stale count.
    """
    row = _three_children()
    live = _run_apply([dict(row)], {"p": 2})[0]
    assert live["_live_subagent_count"] == 2
    assert _run_child_pill([live])[0][0]["textContent"] == "2 live"

    idle = _run_apply([dict(live)], {})[0]
    assert "_live_subagent_count" not in idle
    pill = _run_child_pill([idle])[0][0]
    assert pill["textContent"] == "›", pill
    assert "is-live" not in pill["className"], pill


def test_standalone_live_badge_is_not_doubled_on_a_child_row():
    """Both markers carry the same number; a row must paint it once."""
    out = _run_child_pill([_three_children(_live_subagent_count=2)])
    live_markers = [
        c for c in out[0]
        if "session-live-subagents-badge" in c["className"]
        or "is-live" in c["className"]
    ]
    assert len(live_markers) == 1, out


def test_pill_is_not_drawn_when_there_are_no_children():
    assert _run_child_pill([{"session_id": "solo"}]) == [[]]


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


def test_apply_rejects_coercible_non_numbers():
    """A string `"3"`, `true`, or a fraction must not become a badge count.

    `Number()` coercion would turn the first two into a paintable number, and a
    fraction would render "1.5 live" — the counts are child counts, so integers only.
    """
    for bogus in ("3", True, "abc", float("nan"), 1.5, 0):
        rows = _run_apply([{"session_id": "a"}], {"a": bogus})
        assert "_live_subagent_count" not in rows[0], bogus


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


def test_poll_repaints_first_and_badges_after():
    """The sidebar repaint must never be gated on the badge probe.

    `_refreshSidebarForBackgroundChurn` is what defers behind interaction (the
    Firefox scroll-stability contract pins its direct call in the poll), so it
    fires first and unconditionally; the counts fetch follows and triggers a
    second repaint only when the map actually changed.
    """
    poll = re.search(r"^function startStreamingPoll\(\)\{.*?\n\}\n", SESSIONS_JS, re.S | re.M)
    assert poll, "startStreamingPoll not found"
    body = poll.group(0)
    repaint = body.index("_refreshSidebarForBackgroundChurn()")
    fetch = body.index("refreshLiveSubagentCounts()")
    assert repaint < fetch, "the sidebar repaint must not be gated on the badge fetch"
    # A second repaint after the fetch is what actually shows the badge.
    assert body.count("_refreshSidebarForBackgroundChurn()") >= 2, (
        "the sidebar must repaint again once the counts land"
    )
    # Interval tick + visibilitychange + one immediate first paint: every entry
    # point refreshes the counts (the last one via the compound helper).
    assert body.count("refreshLiveSubagentCounts()") + body.count("_refreshSidebarThenLiveCounts()") >= 3


def test_badge_fetch_has_its_own_short_timeout():
    """The probe must not inherit api()'s 30s default: it is decoration."""
    m = re.search(
        r"^async function refreshLiveSubagentCounts\(\)\{.*?^\}\n", SESSIONS_JS, re.S | re.M
    )
    assert m, "refreshLiveSubagentCounts not found"
    assert "timeoutMs:3000" in m.group(0), (
        "the badge fetch needs an explicit short timeout, not the 30s default"
    )
    # It must report whether it changed the map, so the caller repaints only then.
    assert "return false" in m.group(0) and "return true" in m.group(0)


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


def test_every_locale_label_interpolates_the_count():
    """The badge label must be a `{0}` string in every locale.

    A bare label ('Live', '実行中') silently drops the count, so the badge stops
    saying *how many* children are live — its entire purpose.
    """
    locales = re.findall(r"^  (?:'([A-Za-z-]+)'|([a-z]{2})): \{\s*$", I18N_JS, re.M)
    names = [a or b for a, b in locales]
    bad = []
    for loc in names:
        m = re.search(r"session_live_subagents: ('[^']*')", _locale_block(loc))
        assert m, f"locale {loc} has no plain-string label"
        if "{0}" not in m.group(1):
            bad.append((loc, m.group(1)))
    assert not bad, f"labels without a {{0}} count placeholder: {bad}"


def test_every_locale_tooltip_is_a_plain_string():
    """`t()` is called with no argument for the tooltip.

    A function-valued entry would be invoked with `undefined` and render
    literally "undefined subagents live" — the bug this test exists to catch.
    """
    locales = re.findall(r"^  (?:'([A-Za-z-]+)'|([a-z]{2})): \{\s*$", I18N_JS, re.M)
    names = [a or b for a, b in locales]
    bad = []
    for loc in names:
        m = re.search(r"session_live_subagents_tooltip: (.*),\n", _locale_block(loc))
        assert m, f"locale {loc} has no tooltip entry"
        value = m.group(1).strip()
        if not (value.startswith("'") and value.endswith("'")):
            bad.append((loc, value))
    assert not bad, f"tooltips must be plain strings, found: {bad}"


def test_the_rendered_tooltip_never_contains_undefined():
    """End-to-end guard on the real i18n module for the language we can assert."""
    out = _run_render([{"session_id": "a", "_live_subagent_count": 2}])
    assert "undefined" not in out[0][0]["textContent"]
    assert "undefined" not in out[0][0]["title"]


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
    routes._live_subagents_cache.clear()


def test_endpoint_returns_the_parent_to_count_map(monkeypatch):
    routes = _routes_module()
    monkeypatch.setattr(
        routes, "_live_subagent_counts", lambda now=None, db_path=None: {"parent1": 3, "parent2": 1}
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

    def boom(now=None, db_path=None):
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

    def counting(now=None, db_path=None):
        calls["n"] += 1
        return {"p": calls["n"]}

    monkeypatch.setattr(routes, "_live_subagent_counts", counting)
    routes._live_subagents_cache.clear()
    first, second = _FakeHandler(), _FakeHandler()
    routes._handle_live_subagents(first, None)
    routes._handle_live_subagents(second, None)
    assert calls["n"] == 1, "the second call inside the TTL must not re-probe"
    assert first.json_body()["generated_at"] == second.json_body()["generated_at"]


def test_cache_is_keyed_by_state_db_so_profiles_never_share_counts(monkeypatch, tmp_path):
    """The state.db path follows the active profile, so a single cache slot would
    serve profile A's counts (and its parent session ids) to a profile B request
    inside the TTL — wrong data plus cross-profile exposure."""
    routes = _routes_module()
    db_a, db_b = tmp_path / "a" / "state.db", tmp_path / "b" / "state.db"
    db_a.parent.mkdir()
    db_b.parent.mkdir()
    monkeypatch.setattr(
        routes,
        "_live_subagent_counts",
        lambda now=None, db_path=None: {"session-of-" + str(db_path.parent.name): 1},
    )
    routes._live_subagents_cache.clear()

    monkeypatch.setattr(routes, "_active_state_db_path", lambda: db_a)
    routes._handle_live_subagents(h := _FakeHandler(), None)
    assert h.json_body()["counts"] == {"session-of-a": 1}

    monkeypatch.setattr(routes, "_active_state_db_path", lambda: db_b)
    routes._handle_live_subagents(h2 := _FakeHandler(), None)
    assert h2.json_body()["counts"] == {"session-of-b": 1}, (
        "a profile-B request inside the TTL must not be served profile A's counts"
    )


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


# ── the movement clock behind the number ────────────────────────────────────

_STATE_DB_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT,
    content TEXT,
    timestamp REAL NOT NULL
);
"""


def test_a_child_idle_past_the_threshold_leaves_the_count(tmp_path, monkeypatch):
    """Liveness is movement, not the row's existence.

    A subagent row the runner never closed (`ended_at IS NULL`, a crash) must
    stop reading "live" once it has been silent past the grace window, and must
    come back if it moves again — that is the motion the badge mirrors.
    """
    routes = _routes_module()
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.executescript(_STATE_DB_SCHEMA)
    now = time.time()
    con.execute(
        "INSERT INTO sessions (id, source, parent_session_id, started_at, ended_at)"
        " VALUES (?,?,?,?,?)",
        ("child-1", "subagent", "parent-q", now - 90 * 60, None),
    )
    con.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        ("child-1", "assistant", "step", now - 5 * 60),
    )
    con.commit()
    assert routes._live_subagent_counts(now, db) == {"parent-q": 1}

    # Silent for longer than the window: no longer live.
    stale = now - (routes._LIVE_SUBAGENTS_GRACE_SECONDS + 60)
    con.execute("UPDATE messages SET timestamp=? WHERE session_id='child-1'", (stale,))
    con.commit()
    con.close()
    assert routes._live_subagent_counts(now, db) == {}


def test_the_badge_clock_covers_a_real_model_call():
    """The window must clear a long tool call, not just a fast one.

    A 2.5-minute movement clock (the HUD chip's) paints a running child as idle
    during one slow model call; this badge is the sidebar's own long-run tail,
    so its window stays in minutes-to-half-hour territory.
    """
    routes = _routes_module()
    assert routes._LIVE_SUBAGENTS_GRACE_SECONDS >= 20 * 60
