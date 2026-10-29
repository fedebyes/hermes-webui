"""#7685 Scripts subtab: the Jobs wording must not leak into the Scripts tab.

The maintainer's 2026-10-29 review left two open items on the branch:

* "the main column still says 'Select a scheduled job'"
* "the '+' button still reads 'New job'"

Both are Jobs-tab wording. On the Scripts tab they are wrong twice over:

* the "+" button opens the cron create form -- a *job* action. The Scripts
  tab is a read-only browse of the profile's scripts directory, so there is
  no create action there to offer, and relabelling it "New script" would
  promise a panel that cannot create a script either. It is hidden.
* the main column's empty state names a scheduled job and points at the job
  detail. A script row is not selectable (there is no script detail view),
  so the empty state has to describe the list the user is actually looking
  at.

These tests drive the real ``switchTasksSubtab`` in node against a DOM
double, so they fail if the behaviour regresses -- not just if the source
text changes.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
PANELS_JS = (REPO_ROOT / "static" / "panels.js").read_text(encoding="utf-8")
INDEX_HTML = (REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
I18N_JS = (REPO_ROOT / "static" / "i18n.js").read_text(encoding="utf-8")

NODE = shutil.which("node")


def _extract(marker: str) -> str:
    """Return the source of the function that starts at ``marker``."""
    start = PANELS_JS.find(marker)
    assert start >= 0, marker
    i = PANELS_JS.find("{", start)
    depth = 0
    while True:
        ch = PANELS_JS[i]
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return PANELS_JS[start : i + 1]
        i += 1


HARNESS = """
const els = {};
function mk(id){ const a={}; const cls=new Set();
  return els[id]={ id, dataset:{}, hidden:false, textContent:'',
    setAttribute:(k,v)=>{a[k]=String(v);}, getAttribute:(k)=>a[k],
    classList:{ toggle:(c,on)=>{ on?cls.add(c):cls.delete(c); } } }; }
['panelTasks','tasksPanelTitle','tasksTabJobs','tasksTabScripts','cronList',
 'scriptsList','cronRefreshBtn','cronNewJobBtn','taskDetailEmptyTitle',
 'taskDetailEmptySub'].forEach(mk);
function $(id){ return els[id]; }
// The real t() falls back to the key itself when a locale lacks the entry.
const STRINGS = {
  tasks_empty_title: 'Select a scheduled job',
  tasks_empty_sub: 'Pick a job from the sidebar to view its details and runs, or create a new one.',
  scripts_detail_empty_title: 'Scripts',
  scripts_detail_empty_sub: 'This panel lists the scripts in this profile. Open a script file on the server to run it.',
};
function t(k){ return STRINGS[k] !== undefined ? STRINGS[k] : k; }
let _currentTasksSubtab='jobs';
let scriptsLoads=0; function loadScriptsList(){ scriptsLoads++; }
"""


def _run(body: str) -> str:
    """Run ``body`` against the real switchTasksSubtab and return stdout."""
    harness = HARNESS + _extract("function switchTasksSubtab(") + "\n" + body + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
        path = fh.name
    try:
        proc = subprocess.run(
            [NODE, path], capture_output=True, text=True, timeout=60, check=False
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_scripts_tab_hides_the_new_job_button() -> None:
    out = _run(
        """
switchTasksSubtab('scripts');
console.log('scripts:' + els.cronNewJobBtn.hidden);
switchTasksSubtab('jobs');
console.log('jobs:' + els.cronNewJobBtn.hidden);
"""
    )
    assert out.splitlines() == ["scripts:true", "jobs:false"], out


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_scripts_tab_uses_scripts_wording_in_the_main_column() -> None:
    out = _run(
        """
switchTasksSubtab('scripts');
console.log(els.taskDetailEmptyTitle.textContent);
console.log(els.taskDetailEmptySub.textContent);
console.log(els.taskDetailEmptyTitle.getAttribute('data-i18n'));
console.log(els.taskDetailEmptySub.getAttribute('data-i18n'));
"""
    )
    lines = out.splitlines()
    assert lines[0] == "Scripts", out
    assert "scripts" in lines[1].lower(), out
    assert lines[2] == "scripts_detail_empty_title", out
    assert lines[3] == "scripts_detail_empty_sub", out


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_jobs_tab_restores_the_job_wording() -> None:
    out = _run(
        """
switchTasksSubtab('scripts');
switchTasksSubtab('jobs');
console.log(els.taskDetailEmptyTitle.textContent);
console.log(els.taskDetailEmptySub.textContent);
console.log(els.taskDetailEmptyTitle.getAttribute('data-i18n'));
console.log(els.taskDetailEmptySub.getAttribute('data-i18n'));
"""
    )
    lines = out.splitlines()
    assert lines[0] == "Select a scheduled job", out
    assert lines[1].startswith("Pick a job"), out
    assert lines[2] == "tasks_empty_title", out
    assert lines[3] == "tasks_empty_sub", out


def test_the_new_job_button_carries_an_id() -> None:
    # The tab switch hides it by id, so the id has to exist in the markup.
    assert re.search(r'id="cronNewJobBtn"', INDEX_HTML)


def test_the_empty_state_carries_ids() -> None:
    assert re.search(r'id="taskDetailEmptyTitle"', INDEX_HTML)
    assert re.search(r'id="taskDetailEmptySub"', INDEX_HTML)


def test_every_locale_carries_the_scripts_empty_strings() -> None:
    # A locale missing the key would fall back to the raw key name in the UI.
    starts = {}
    for m in re.finditer(r"^  ('?[a-zA-Z-]+'?): \{$", I18N_JS, re.MULTILINE):
        starts[m.start()] = m.group(1).strip("'")
    assert len(starts) == 15, sorted(starts.values())

    offsets = sorted(starts)
    for idx, start in enumerate(offsets):
        end = offsets[idx + 1] if idx + 1 < len(offsets) else len(I18N_JS)
        block = I18N_JS[start:end]
        locale = starts[start]
        assert "scripts_detail_empty_title:" in block, locale
        assert "scripts_detail_empty_sub:" in block, locale
