"""Session context menu: the Mark as done / Mark as not done toggle.

`done_at` is the structural "this conversation is finished" flag. The sidebar
renders it (see `test_session_done_badge.py`); the close/checkmark scripts write
it through POST /api/session/mark-done. This file pins the third writer — the
right-click / three-dot menu — so the menu and the scripts cannot drift into two
dialects of the same fact.

The repo avoids a node/jsdom dependency, so the behaviour test extracts
`_setSessionDone` from `static/sessions.js` and executes it under node against
stubs. Structural assertions cover the menu wiring itself (the item must live
inside `_openSessionActionMenu`, after Archive, and reuse the same predicate the
row renderer uses). i18n: every locale must carry the five keys, or the item
renders raw key names.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_JS = (REPO_ROOT / "static" / "sessions.js").read_text(encoding="utf-8")
I18N_JS = (REPO_ROOT / "static" / "i18n.js").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node required")

DONE_PREDICATE = "s.done_at!==null&&s.done_at!==undefined&&s.done_at!==''"
DONE_KEYS = (
    "session_mark_done",
    "session_mark_undone",
    "session_marked_done",
    "session_marked_undone",
    "session_mark_done_failed",
)


def _set_session_done_source() -> str:
    pattern = r"^async function _setSessionDone\(session, done=true\)\{.*?^}\n"
    m = re.search(pattern, SESSIONS_JS, re.S | re.M)
    assert m, "_setSessionDone not found in static/sessions.js"
    return m.group(0)


def _run(scenarios):
    """Execute the extracted handler once per scenario against stubs.

    Each scenario is a dict: {done, readonly, fail, done_at}. The handler's
    return value, the state it left on the session object, the request it sent,
    and the toasts it raised all come back so the test can assert on behaviour
    rather than on source text.
    """
    harness = """
const scenarios = __SCENARIOS__;
const out = [];
function t(key){ return key; }
(async () => {
  for (const sc of scenarios) {
    const calls = [];
    const toasts = [];
    let repaints = 0;
    let renders = 0;
    let _allSessions = [{ session_id: 'a', done_at: null }];
    const S = { session: { session_id: 'a', done_at: null } };
    function _isReadOnlySession(){ return !!sc.readonly; }
    function showToast(msg){ toasts.push(msg); }
    function _repaintSidebarAfterSessionCompletion(){ repaints++; }
    function renderSessionList(){ renders++; }
    function api(path, opts){
      calls.push({ path, body: opts && opts.body ? JSON.parse(opts.body) : null });
      if (sc.fail) return Promise.reject(new Error('boom'));
      return Promise.resolve({ session: { done_at: sc.done_at === undefined ? 111.5 : sc.done_at } });
    }
    __FN__
    const session = { session_id: 'a', done_at: sc.done ? null : 999 };
    const result = await _setSessionDone(session, sc.done);
    out.push({
      result,
      sessionDoneAt: session.done_at === undefined ? '__undef__' : session.done_at,
      cachedDoneAt: _allSessions[0].done_at === undefined ? '__undef__' : _allSessions[0].done_at,
      activeDoneAt: S.session.done_at === undefined ? '__undef__' : S.session.done_at,
      calls,
      toasts,
      repaints,
      renders,
    });
  }
  console.log(JSON.stringify(out));
})();
"""
    harness = harness.replace("__SCENARIOS__", json.dumps(scenarios))
    harness = harness.replace("__FN__", _set_session_done_source())
    proc = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_marks_a_session_done_and_stamps_the_server_time():
    out = _run([{"done": True, "done_at": 111.5}])[0]
    assert out["result"] is True
    assert out["calls"] == [
        {"path": "/api/session/mark-done", "body": {"session_id": "a", "done": True}}
    ]
    # The server owns the clock: its stamp wins over the client's.
    assert out["sessionDoneAt"] == 111.5
    assert out["cachedDoneAt"] == 111.5
    assert out["activeDoneAt"] == 111.5
    assert out["toasts"] == ["session_marked_done"]
    assert out["repaints"] == 1 and out["renders"] == 1


def test_reopens_a_session_by_clearing_done_at():
    out = _run([{"done": False, "done_at": None}])[0]
    assert out["result"] is True
    assert out["calls"] == [
        {"path": "/api/session/mark-done", "body": {"session_id": "a", "done": False}}
    ]
    # Cleared everywhere: a stale timestamp would keep the check on the row.
    assert out["sessionDoneAt"] is None
    assert out["cachedDoneAt"] is None
    assert out["activeDoneAt"] is None
    assert out["toasts"] == ["session_marked_undone"]


def test_falls_back_to_now_when_the_response_omits_the_stamp():
    out = _run([{"done": True, "done_at": None}])[0]
    assert out["result"] is True
    assert isinstance(out["sessionDoneAt"], (int, float))
    assert out["sessionDoneAt"] > 1700000000


def test_failure_toasts_the_error_and_leaves_state_untouched():
    out = _run([{"done": True, "fail": True}])[0]
    assert out["result"] is False
    assert out["sessionDoneAt"] is None  # untouched
    assert out["toasts"] == ["session_mark_done_failedboom"]
    # No repaint on failure: the row must not flicker into a state the server
    # never accepted.
    assert out["repaints"] == 0


def test_read_only_session_never_reaches_the_endpoint():
    out = _run([{"done": True, "readonly": True}])[0]
    assert out["result"] is False
    assert out["calls"] == []


def test_menu_item_lives_inside_the_session_action_menu():
    m = re.search(
        r"^function _openSessionActionMenu\(.*?^\}\n", SESSIONS_JS, re.S | re.M
    )
    assert m, "_openSessionActionMenu not found"
    body = m.group(0)
    assert "t('session_mark_done')" in body
    assert "t('session_mark_undone')" in body
    assert "_setSessionDone(session," in body
    # Same predicate the row renderer uses (DONE_PREDICATE), only under this
    # scope's variable name — presence, not truthiness, so a stored 0 still
    # reads as done.
    assert "const _isDone=" + DONE_PREDICATE.replace("s.done_at", "session.done_at") + ";" in body
    # Ordered right after Archive: archive hides the row, done keeps it.
    assert body.index("t('session_archive')") < body.index("t('session_mark_done')")


def test_right_click_reaches_the_menu():
    """The row's contextmenu handler is what makes this a right-click item."""
    m = re.search(r"el\.oncontextmenu=\(e\)=>\{.*?^\s*\};\n", SESSIONS_JS, re.S | re.M)
    assert m, "row oncontextmenu handler not found"
    assert "_openSessionActionMenu(s, actions||el)" in m.group(0)


def test_menu_item_has_its_own_icons():
    assert "markdone:'<svg" in SESSIONS_JS
    assert "markundone:'<svg" in SESSIONS_JS


def test_every_locale_defines_the_mark_done_keys():
    locales = re.findall(r"^  (?:'([A-Za-z-]+)'|([a-z]{2})): \{\s*$", I18N_JS, re.M)
    names = [a or b for a, b in locales]
    assert names, "no locale blocks found"
    assert len(names) >= 15, f"only {len(names)} locales found: {names}"
    missing = {
        loc: [k for k in DONE_KEYS if f"{k}:" not in _locale_block(loc)]
        for loc in names
    }
    missing = {loc: keys for loc, keys in missing.items() if keys}
    assert not missing, f"locales missing mark-done keys: {missing}"


def _locale_block(loc: str) -> str:
    m = re.search(rf"^  (?:'{loc}'|{loc}): \{{\s*$", I18N_JS, re.M)
    assert m, f"locale {loc} not found"
    start = m.start()
    nxt = re.search(r"^  (?:'[A-Za-z-]+'|[a-z]{2}): \{\s*$", I18N_JS[start + 5 :], re.M)
    end = start + 5 + nxt.start() if nxt else len(I18N_JS)
    return I18N_JS[start:end]
