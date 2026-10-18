"""Regressions for the maintainer's review 5565050910 on #8162.

1. A *dirty* ordinary primary checkout (modified / staged / untracked-child
   under a committed directory) must still be recovered as the main root for a
   linked-worktree input. The previous head required a clean `git status`.
2. A failed *optional* `ls-files` query must not be read as ownership: it must
   fall back to the caller's own checkout, never select the storage dir as the
   project root.
3. An optional `status` OSError/timeout must not escape `create_worktree_for_workspace`.
"""

import subprocess
from pathlib import Path

import pytest


def _git(*args, cwd=None, check=True):
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


@pytest.fixture()
def git_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.com")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig-none"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return tmp_path


def _import():
    from api.worktrees import find_main_worktree_root

    return find_main_worktree_root


def _stub_agent_setup(monkeypatch, created_root):
    import api.worktrees as worktrees

    seen = {}

    def fake(repo_root):
        seen["repo_root"] = repo_root
        tree = Path(repo_root) / ".worktrees" / "hermes-test"
        tree.mkdir(parents=True, exist_ok=True)
        created_root.append(tree)
        return {"path": str(tree), "branch": "hermes-test", "repo_root": str(repo_root)}

    monkeypatch.setattr(worktrees, "_setup_agent_worktree", fake)
    return seen


def _ordinary_primary(base):
    """Ordinary primary with a committed file plus a committed `src/` dir."""
    primary = base / "primary"
    primary.mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=primary)
    (primary / "tracked.txt").write_text("committed\n", encoding="utf-8")
    (primary / "src").mkdir()
    (primary / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "tracked.txt", "src/mod.py", cwd=primary)
    _git("commit", "-q", "-m", "init", cwd=primary)
    linked = base / "linked"
    _git("worktree", "add", "-q", "-b", "side", str(linked), cwd=primary)
    nested = linked / "nested"
    nested.mkdir()
    return primary, linked, nested


def _dirty(primary, kind):
    if kind == "modified":
        (primary / "tracked.txt").write_text("modified content\n", encoding="utf-8")
    elif kind == "staged":
        (primary / "tracked.txt").write_text("staged content\n", encoding="utf-8")
        _git("add", "tracked.txt", cwd=primary)
    elif kind == "untracked-in-dir":
        (primary / "src" / "new_untracked.py").write_text("y = 2\n", encoding="utf-8")
    else:  # pragma: no cover
        raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["modified", "staged", "untracked-in-dir"])
def test_dirty_ordinary_primary_still_unnests_to_primary(git_env, monkeypatch, kind):
    """A working tree owns its repository whether or not it is clean."""
    primary, linked, nested = _ordinary_primary(git_env / f"dirty-{kind}")
    _dirty(primary, kind)

    find = _import()
    for workspace in (linked, nested):
        assert find(workspace) == primary.resolve(), kind

    created = []
    seen = _stub_agent_setup(monkeypatch, created)
    from api.worktrees import create_worktree_for_workspace

    info = create_worktree_for_workspace(nested)
    assert Path(seen["repo_root"]).resolve() == primary.resolve(), kind
    assert Path(info["path"]).resolve().is_relative_to(primary.resolve())
    assert not Path(info["path"]).resolve().is_relative_to(linked.resolve())


def test_ownership_rule_dirty_checkout_versus_storage_collision(git_env):
    """Pin the rule itself: partial divergence is a checkout, total is a copy.

    A storage dir whose notes collide with *every* tracked name differs on all
    of them; a dirty checkout differs only on the paths the operator touched.
    """
    from api.worktrees import _tracks_paths_in_parent

    # Multi-entry dirty checkout: one modified, one staged -> still the tree.
    dirty = git_env / "rule-dirty" / "checkout"
    dirty.mkdir(parents=True)
    storage = dirty.parent / "storage"
    storage.mkdir()
    _git("init", "-q", "-b", "main", f"--separate-git-dir={storage / '.git'}", cwd=dirty)
    for name in ("a.txt", "b.txt"):
        (dirty / name).write_text(f"committed {name}\n", encoding="utf-8")
        _git("add", name, cwd=dirty)
    _git("commit", "-q", "-m", "init", cwd=dirty)
    (dirty / "a.txt").write_text("modified\n", encoding="utf-8")
    (dirty / "b.txt").write_text("staged\n", encoding="utf-8")
    _git("add", "b.txt", cwd=dirty)
    assert _tracks_paths_in_parent((storage / ".git").resolve(), dirty.resolve()) is True

    # Same repository, storage holding notes that collide with every name.
    for name in ("a.txt", "b.txt"):
        (storage / name).write_text(f"unrelated notes about {name}\n", encoding="utf-8")
    assert _tracks_paths_in_parent((storage / ".git").resolve(), storage.resolve()) is False


def _distant_separate(base):
    """Distant primary with a gitfile, separate storage, linked input nested."""
    checkout = base.joinpath("elsewhere", "deep", "more", "further", "actual-checkout")
    storage = base / "storage-only"
    checkout.mkdir(parents=True)
    storage.mkdir(parents=True)
    _git("init", "-q", "-b", "main", f"--separate-git-dir={storage / '.git'}", cwd=checkout)
    (checkout / "sentinel.txt").write_text("keep me\n", encoding="utf-8")
    _git("add", "sentinel.txt", cwd=checkout)
    _git("commit", "-q", "-m", "init", cwd=checkout)
    linked = base / "linked-wt"
    _git("worktree", "add", "-q", "-b", "side", str(linked), cwd=checkout)
    nested = linked / "nested"
    nested.mkdir()
    return checkout, storage, linked, nested


@pytest.mark.parametrize("failure", ["oserror", "returncode", "timeout"])
def test_failed_ls_files_query_never_selects_storage(git_env, monkeypatch, failure):
    """A failed optional `ls-files` is unknown, not ownership: keep our checkout."""
    import api.worktrees as worktrees

    checkout, storage, linked, nested = _distant_separate(git_env / f"lsfail-{failure}")
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and "ls-files" in cmd:
            if failure == "oserror":
                raise OSError("injected: index unreadable")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(cmd="git", timeout=5)
            return subprocess.CompletedProcess(args=list(cmd), returncode=128, stdout="", stderr="boom")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(worktrees.subprocess, "run", fake_run)

    created = []
    seen = _stub_agent_setup(monkeypatch, created)
    from api.worktrees import create_worktree_for_workspace

    info = create_worktree_for_workspace(nested)
    root = Path(seen["repo_root"]).resolve()
    assert root != storage.resolve(), f"{failure}: storage was handed to the Agent"
    # A failed optional query is unknown, never a claim on storage. What remains
    # is decided by recorded facts: here the primary's gitfile still claims the
    # metadata, so the real checkout is recovered — otherwise the caller keeps
    # its own checkout. Either answer is acceptable; storage is not.
    assert root in (checkout.resolve(), linked.resolve()), failure
    assert Path(info["path"]).resolve().is_relative_to(root)
    assert not Path(info["path"]).resolve().is_relative_to(storage.resolve())


def test_optional_status_failure_does_not_abort_creation(git_env, monkeypatch):
    """No `status` query may raise out of create_worktree_for_workspace."""
    import api.worktrees as worktrees

    primary, linked, nested = _ordinary_primary(git_env / "statusfail")
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and "status" in cmd:
            raise OSError("injected: status unavailable")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(worktrees.subprocess, "run", fake_run)

    created = []
    seen = _stub_agent_setup(monkeypatch, created)
    from api.worktrees import create_worktree_for_workspace

    info = create_worktree_for_workspace(nested)  # must not raise
    assert Path(seen["repo_root"]).resolve() == primary.resolve()
    assert Path(info["path"]).resolve().is_relative_to(primary.resolve())
