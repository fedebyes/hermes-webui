"""Sidebar `done` marker: a row with `done_at` renders the resting check, and the
check outranks the unread dot.

`done_at` is the structural "this conversation is finished" flag (written by the
close/checkmark scripts through POST /api/session/mark-done). The sidebar must
render it as data — never by inferring from a title prefix or from silence.

The repo intentionally avoids a node/jsdom dependency (see AGENTS.md), so this
file follows the WEBUI-SUB convention: precise structural assertions on
`static/sessions.js` + `static/style.css`, plus one real execution of the
extracted marker-decision fragment under node with a minimal fake DOM. That way
removing the `done_at` presence guard, dropping the `.done` row class, letting
the unread dot displace the check again, or inlining the wrong i18n key reddens
the test instead of silently rendering nothing.

i18n: every locale block must carry both keys, or the badge falls back to the
raw key name in that language.
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
STYLE_CSS = (REPO_ROOT / "static" / "style.css").read_text(encoding="utf-8")
I18N_JS = (REPO_ROOT / "static" / "i18n.js").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node required")


DONE_PREDICATE = "s.done_at!==null&&s.done_at!==undefined&&s.done_at!==''"
DECISION_END = "    ts.textContent=hasAttentionState?'':_formatRelativeSessionTime(tsMs);\n"

# The marker decision, as the ordered source lines the renderer executes. Each
# entry is a line PREFIX, so the harness follows the code when the blocks move
# (and reports the one line that is missing when a priority rule is removed)
# instead of pinning a character offset that any refactor invalidates.
DECISION_LINES = (
    "    const _doneHere=",
    "    const _unreadDot=",
    "    const hasAttentionState=",
    "    ts.className=",
    "    state.className=",
)

REQUIRED_REASON = {
    "    const _unreadDot=": (
        "the unread dot is not gated by `done`: a finished conversation with "
        "unread replies loses its resting check to the neutral dot"
    ),
}


def _source_line(prefix: str) -> str:
    m = re.search(r"^" + re.escape(prefix) + r".*?;$", SESSIONS_JS, re.M)
    if m:
        return m.group(0) + "\n"
    assert prefix not in REQUIRED_REASON, REQUIRED_REASON.get(prefix, "")
    assert False, f"marker decision line not found in static/sessions.js: {prefix!r}"


def _marker_decision_block() -> str:
    """The marker-priority lines from the row renderer, in source order.

    Together they decide which single marker the right-hand slot carries
    (`_doneHere` -> the resting check, `_unreadDot` -> the neutral dot) and
    whether the timestamp yields to it. Executed as a whole, so the test reads
    the real decision instead of restating it.
    """
    return "".join(_source_line(p) for p in DECISION_LINES)


def _done_check_block() -> str:
    """The `if(_doneHere){...}` block that appends the resting check.

    The check is appended either to the title row directly or to the
    right-aligned group that also holds the timestamp; the harness aliases the
    two so one extraction covers both shapes.
    """
    start = SESSIONS_JS.find("    if(_doneHere){")
    assert start != -1, "done check block not found in static/sessions.js"
    prop = SESSIONS_JS.find("appendChild(doneEl);", start)
    assert prop != -1, "done marker not found inside the block"
    closing = SESSIONS_JS.find("\n    }\n", prop)
    assert closing != -1, "done marker block not terminated"
    return SESSIONS_JS[start : closing + len("\n    }\n")]


def _run_render(rows):
    """Execute the extracted fragments against a minimal fake DOM.

    Each input row carries `done_at` plus the three flags the renderer derives
    from the live session (`isStreaming`, `hasUnread`, `attention`).
    """
    block = _marker_decision_block()
    check = _done_check_block()
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
  setAttribute(name, value) {{ this[name] = value; }}
}}
const document = {{ createElement: (tag) => new FakeEl(tag) }};
function t(key, ...args) {{
  const table = {{
    session_done_badge: 'Done',
    session_done_tooltip: 'Marked done {{0}}',
    session_time_unknown: 'unknown',
  }};
  const value = table[key];
  if (value === undefined) return key;
  return value.replace('{{0}}', String(args[0] === undefined ? '' : args[0]));
}}
function _formatRelativeSessionTime(ms) {{ return ms ? '2h ago' : 'unknown'; }}

const out = [];
for (const row of {json.dumps(rows)}) {{
  const s = {{ session_id: row.session_id, done_at: row.done_at }};
  const isStreaming = Boolean(row.isStreaming);
  const hasUnread = Boolean(row.hasUnread);
  const attention = row.attention || null;
  const attentionDotClass = attention
    ? (attention.kind === 'approval' ? ' is-attention-approval'
      : (attention.kind === 'clarify' ? ' is-attention-clarify' : ' is-attention-generic'))
    : '';
  const titleRow = new FakeEl('div');
  // The check lands in the title row (branch shape) or in the right-aligned
  // group beside the timestamp (live-tree shape); alias both onto one list.
  const _rightGroup = titleRow;
  const tsMs = 1700000000000;
  const ts = document.createElement('span');
  const state = {{ className: '' }};
{block}{check}  out.push({{
    doneHere: _doneHere,
    unreadDot: _unreadDot,
    hasAttentionState,
    timestampHidden: ts.className.includes('is-hidden'),
    stateClass: state.className,
    children: titleRow.children.map(c => ({{
      className: c.className,
      textContent: c.textContent,
      title: c.title,
    }})),
  }});
}}
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_row_with_done_at_renders_the_check():
    out = _run_render([{"session_id": "a", "done_at": 1700000000.0}])
    assert len(out[0]["children"]) == 1, out
    check = out[0]["children"][0]
    # A resting check beside the timestamp, not a text pill beside the title:
    # the pill competed with the live-subagent badge for the same row.
    assert check["className"] == "session-done-check"
    assert check["textContent"] == ""
    assert check["title"] == "Marked done 2h ago"
    assert out[0]["doneHere"] is True
    assert out[0]["timestampHidden"] is False


def test_done_beats_unread_so_the_check_keeps_the_slot():
    """A closed conversation with unread replies keeps its check.

    Regression: `hasAttentionState` folded `hasUnread` in unconditionally, so the
    neutral dot took the slot the check belongs to and a finished session read as
    "needs attention" instead of "settled".
    """
    out = _run_render([{"session_id": "a", "done_at": 1700000000.0, "hasUnread": True}])
    row = out[0]
    assert row["doneHere"] is True, row
    assert row["unreadDot"] is False, row
    assert " is-unread" not in row["stateClass"], row
    assert [c["className"] for c in row["children"]] == ["session-done-check"], row
    # The dot is gone, so nothing claims the timestamp's slot: "✓ 2h" stays whole.
    assert row["timestampHidden"] is False, row


def test_unread_without_done_keeps_the_dot():
    out = _run_render([{"session_id": "b", "hasUnread": True}])
    row = out[0]
    assert row["doneHere"] is False
    assert row["unreadDot"] is True
    assert " is-unread" in row["stateClass"], row
    assert row["children"] == [], row
    assert row["timestampHidden"] is True, row


def test_streaming_still_wins_over_done():
    """A running turn is never a closed row — the spinner must not be replaced."""
    out = _run_render([{"session_id": "c", "done_at": 1700000000.0, "isStreaming": True}])
    row = out[0]
    assert row["doneHere"] is False, row
    assert " is-streaming" in row["stateClass"], row
    assert row["children"] == [], row


def test_pending_attention_still_wins_over_done():
    """A pending clarify/approval means the agent is blocked, not finished."""
    for kind in ("clarify", "approval"):
        out = _run_render(
            [{"session_id": "d", "done_at": 1700000000.0, "attention": {"kind": kind}}]
        )
        row = out[0]
        assert row["doneHere"] is False, row
        assert f" is-attention-{kind}" in row["stateClass"], row
        assert row["children"] == [], row


def test_row_without_done_at_renders_nothing():
    out = _run_render([{"session_id": "e"}, {"session_id": "f", "done_at": None}])
    for row in out:
        assert row["doneHere"] is False
        assert row["children"] == []


def test_done_at_zero_still_reads_as_done():
    """0 is a valid timestamp: presence, not truthiness, decides (JS `if(0)` is false)."""
    out = _run_render([{"session_id": "g", "done_at": 0}])
    assert out[0]["doneHere"] is True
    assert len(out[0]["children"]) == 1, out


def test_done_row_gets_the_done_class_and_archived_stays_independent():
    """`done` is workflow state, `archived` is storage: both classes can coexist."""
    # The class is appended next to `archived`, never as a replacement for it.
    m = re.search(r"el\.className='session-item'.*?;(?=\n)", SESSIONS_JS)
    assert m, "session-item className line not found"
    line = m.group(0)
    assert "(s.archived?' archived':'')" in line
    assert f"(({DONE_PREDICATE})?' done':'')" in line
    assert "s.archived?s.done_at" not in line and "s.done_at?s.archived" not in line


def test_badge_style_is_defined_once():
    # Base rule exactly once; the `.active` variant is a separate selector and
    # contains the base string as a substring, so count the full selector line.
    assert "\n.session-done-badge{" in STYLE_CSS
    assert STYLE_CSS.count("\n.session-done-badge{") == 1
    assert ".session-item.active .session-done-badge{" in STYLE_CSS


def test_done_row_class_has_a_style_consumer():
    """A class the renderer emits and a test asserts must actually be styled —
    otherwise it is dead weight that reads as a feature."""
    assert ".session-item.done .session-title{" in STYLE_CSS


def test_done_check_has_a_style_and_yields_on_hover():
    assert "\n  .session-done-check{" in STYLE_CSS
    assert ".session-item:hover .session-done-check" in STYLE_CSS


def test_every_locale_defines_both_done_keys():
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
        if "session_done_badge:" not in _locale_block(loc)
        or "session_done_tooltip:" not in _locale_block(loc)
    ]
    assert not missing, f"locales missing done keys: {missing}"


def _locale_block(loc: str) -> str:
    m = re.search(rf"^  (?:'{loc}'|{loc}): \{{\s*$", I18N_JS, re.M)
    assert m, f"locale {loc} not found"
    start = m.start()
    nxt = re.search(r"^  (?:'[A-Za-z-]+'|[a-z]{2}): \{\s*$", I18N_JS[start + 5 :], re.M)
    end = start + 5 + nxt.start() if nxt else len(I18N_JS)
    return I18N_JS[start:end]
