"""Silence auto-stop checkbox must not fork from the value in force (PR #8564).

`_micAutoStopEnabled()` reads `hermes-mic-auto-stop` from localStorage at capture
time, so that key IS the effective preference: a refused `setItem` (quota,
private mode) used to leave the checkbox on the new value while the meter kept
the old one, and aborted the handler before the autosave was scheduled.

These run the shipped `onchange` handler from `static/panels.js` under node,
together with the shipped `_micAutoStopEnabled()` predicate, against a
controlled localStorage that can start refusing writes.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PANELS_JS = (ROOT / "static" / "panels.js").read_text(encoding="utf-8")
BOOT_JS = (ROOT / "static" / "boot.js").read_text(encoding="utf-8")
NODE = shutil.which("node")


def _extract_function(src, name):
    """Return the full source of a top-level `function name(...)`."""
    marker = "function %s(" % name
    start = src.index(marker)
    brace = src.index("{", start)
    depth = 0
    for pos in range(brace, len(src)):
        if src[pos] == "{":
            depth += 1
        elif src[pos] == "}":
            depth -= 1
            if depth == 0:
                return src[start:pos + 1]
    raise AssertionError("unbalanced braces around %s" % name)


def _extract_mic_autostop_block(src):
    """The `if(micAutoStopCb){...}` block that wires the checkbox, verbatim."""
    # The name also appears in `_preferencesPayloadFromUi`; the wiring block is the
    # last one, followed by the voice-silence setup.
    start = src.rindex("const micAutoStopCb=$('settingsMicAutoStop');")
    end = src.index("const voiceSilence=parseInt(", start)
    block = src[start:end].rstrip()
    if not block.endswith("}"):
        raise AssertionError("mic auto-stop block did not end with a closing brace")
    return block


_HARNESS = r"""
const _MIC_AUTOSTOP_KEY = 'hermes-mic-auto-stop';
const KEY = 'hermes-mic-auto-stop';
const store = {};
let failWrites = false;
globalThis.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { if (failWrites) throw Object.assign(new Error('QuotaExceededError'), {name: 'QuotaExceededError'}); store[k] = String(v); },
  removeItem: (k) => { delete store[k]; },
};
const els = {};
function mk(id){ if(!els[id]) els[id] = {id: id, checked: false, hidden: true}; return els[id]; }
globalThis.$ = (id) => (id === 'settingsMicAutoStop' || id === 'settingsMicAutoStopNote') ? mk(id) : null;
let saves = 0;
const changed = [];
globalThis._speechBool = () => false;
globalThis._speechPreferenceIsOwned = () => true;
globalThis._syncSpeechPreferenceCache = (k, v) => { localStorage.setItem(KEY, String(v)); };
globalThis._markSpeechPreferenceChanged = (k) => { changed.push(k); };
globalThis._schedulePreferencesAutosave = () => { saves += 1; };

__MIC_PREDICATE__

__MIC_BLOCK__

const cb = mk('settingsMicAutoStop');
const note = mk('settingsMicAutoStopNote');
function click(next){
  cb.checked = next;
  cb.onchange.call(cb);
}
function snap(){
  return {
    checkbox: cb.checked,
    effective: _micAutoStopEnabled(),
    note: note.hidden === false,
    saves: saves,
    stored: (KEY in store) ? store[KEY] : null,
    changed: changed.length,
  };
}
const out = {};
out.defaultOff = snap();
click(true);  out.onWritable = snap();
click(false); out.offWritable = snap();
click(true);  const savesBeforeRefusal = saves;
failWrites = true;
click(false); out.offRefused = snap();
out.savesBeforeRefusal = savesBeforeRefusal;
failWrites = false;
click(true);  out.onAgain = snap();
failWrites = true;
click(false); out.offRefusedTwice = snap();
console.log(JSON.stringify(out));
"""


def _run():
    js = (
        _HARNESS
        .replace("__MIC_PREDICATE__", _extract_function(BOOT_JS, "_micAutoStopEnabled"))
        .replace("__MIC_BLOCK__", _extract_mic_autostop_block(PANELS_JS))
    )
    proc = subprocess.run(
        [NODE, "-e", js], capture_output=True, text=True, cwd=ROOT, timeout=60
    )
    assert proc.returncode == 0, "node harness failed:\n%s" % proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    if NODE is None:
        pytest.skip("node not on PATH")
    return _run()


def test_absent_preference_is_off_and_unsaved(runs):
    # Init seeds the runtime cache from the (absent) server value, so the key
    # exists as "false" and the predicate agrees: nothing enabled, nothing saved.
    assert runs["defaultOff"] == {
        "checkbox": False,
        "effective": False,
        "note": False,
        "saves": 0,
        "stored": "false",
        "changed": 0,
    }


def test_writable_toggle_moves_both_checkbox_and_effective_value(runs):
    on = runs["onWritable"]
    assert on["checkbox"] is True
    assert on["effective"] is True
    assert on["stored"] == "true"
    assert on["saves"] == 1
    off = runs["offWritable"]
    assert (off["checkbox"], off["effective"], off["stored"]) == (False, False, "false")
    assert off["saves"] == 2


def test_refused_write_keeps_displayed_and_effective_in_agreement(runs):
    refused = runs["offRefused"]
    # The box rolls back to the value that is actually in force instead of
    # showing a preference the meter will not honour.
    assert refused["checkbox"] is True
    assert refused["effective"] is True
    assert refused["stored"] == "true"


def test_refused_write_is_visible_and_does_not_schedule_a_save(runs):
    refused = runs["offRefused"]
    assert refused["note"] is True, "a refused write must be reported, not swallowed"
    assert refused["saves"] == runs["savesBeforeRefusal"], "nothing changed, nothing to persist"


def test_repeated_refusal_never_forks_the_control(runs):
    again = runs["onAgain"]
    assert (again["checkbox"], again["effective"]) == (True, True)
    assert again["note"] is False
    twice = runs["offRefusedTwice"]
    assert twice["checkbox"] is twice["effective"] is True
    assert twice["stored"] == "true"
    assert twice["note"] is True
