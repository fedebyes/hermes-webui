"""Real-Git regressions for find_main_worktree_root() (review P2 on #8162).

The reviewer's finding: the previous implementation treated any git *metadata*
directory as a checkout. `--git-common-dir` is not always `<main>/.git`, so
`git init --separate-git-dir=…/metadata` and a submodule (`super/.git/modules/child`)
both returned the metadata path instead of the working tree, and session trees were
created beneath it.

These tests build real repositories on disk — no mocks of git — and assert the
resolved root is the actual checkout, not the metadata dir.
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
    """Isolated git identity so commits work on a bare CI runner."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.com")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig-none"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return tmp_path


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=path)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    _git("add", "README.md", cwd=path)
    _git("commit", "-q", "-m", "init", cwd=path)
    return path


def _import():
    from api.worktrees import find_main_worktree_root

    return find_main_worktree_root


def test_regular_repo_returns_its_root(git_env):
    repo = _init_repo(git_env / "regular")
    find = _import()
    assert find(repo) == repo.resolve()


def test_regular_linked_worktree_unnests_to_main(git_env):
    repo = _init_repo(git_env / "regular-linked")
    linked = git_env / "regular-linked-wt"
    _git("worktree", "add", "-q", "-b", "side", str(linked), cwd=repo)
    find = _import()
    assert find(linked) == repo.resolve()


def test_separate_git_dir_returns_checkout_not_metadata(git_env):
    """The reviewer's primary case: metadata dir must not be mistaken for a checkout."""
    checkout = git_env / "sep-checkout"
    metadata = git_env / "sep-metadata"
    checkout.mkdir()
    _git("init", "-q", "-b", "main", f"--separate-git-dir={metadata}", cwd=checkout)
    (checkout / "README.md").write_text("x\n", encoding="utf-8")
    _git("add", "README.md", cwd=checkout)
    _git("commit", "-q", "-m", "init", cwd=checkout)

    find = _import()
    root = find(checkout)
    assert root is not None
    assert root == checkout.resolve(), f"expected the checkout, got {root}"
    assert root != metadata.resolve()


def test_separate_git_dir_linked_worktree(git_env):
    checkout = git_env / "sep2-checkout"
    metadata = git_env / "sep2-metadata"
    checkout.mkdir()
    _git("init", "-q", "-b", "main", f"--separate-git-dir={metadata}", cwd=checkout)
    (checkout / "README.md").write_text("x\n", encoding="utf-8")
    _git("add", "README.md", cwd=checkout)
    _git("commit", "-q", "-m", "init", cwd=checkout)

    linked = git_env / "sep2-linked"
    _git("worktree", "add", "-q", "-b", "side", str(linked), cwd=checkout)

    find = _import()
    assert find(linked) == checkout.resolve()
    assert find(checkout) == checkout.resolve()


def test_submodule_returns_submodule_checkout_not_modules_dir(git_env):
    super_repo = _init_repo(git_env / "super")
    child = _init_repo(git_env / "child-src")

    _git(
        "-c", "protocol.file.allow=always",
        "submodule", "add", "-q", str(child), "child",
        cwd=super_repo,
    )
    _git("commit", "-q", "-m", "add submodule", cwd=super_repo)

    sub_checkout = super_repo / "child"
    modules_dir = super_repo / ".git" / "modules" / "child"
    assert modules_dir.is_dir(), "fixture: submodule metadata dir missing"

    find = _import()
    root = find(sub_checkout)
    assert root is not None
    assert root == sub_checkout.resolve(), f"expected the submodule checkout, got {root}"
    assert root != modules_dir.resolve()


def test_submodule_linked_worktree(git_env):
    super_repo = _init_repo(git_env / "super2")
    child = _init_repo(git_env / "child2-src")
    _git(
        "-c", "protocol.file.allow=always",
        "submodule", "add", "-q", str(child), "child",
        cwd=super_repo,
    )
    _git("commit", "-q", "-m", "add submodule", cwd=super_repo)

    sub_checkout = super_repo / "child"
    linked = git_env / "child2-linked"
    _git("worktree", "add", "-q", "-b", "subside", str(linked), cwd=sub_checkout)

    find = _import()
    assert find(linked) == sub_checkout.resolve()


def test_nested_workspace_inside_regular_repo(git_env):
    repo = _init_repo(git_env / "nested-repo")
    nested = repo / "a" / "b" / "c"
    nested.mkdir(parents=True)
    find = _import()
    assert find(nested) == repo.resolve()


def test_non_git_directory_returns_none(git_env):
    plain = git_env / "not-a-repo"
    plain.mkdir()
    find = _import()
    assert find(plain) is None


def test_separate_git_dir_inside_unrelated_repo_is_not_mistaken_for_it(git_env):
    """P1 on #8162: the metadata's parent must not be taken for the checkout.

    A checkout can keep ``--separate-git-dir`` inside another repository's tree
    while the checkout itself lives elsewhere. The parent is a valid checkout,
    so a parent-only check returns it and starts the session in the wrong
    project. The checkout owns its own metadata, so the correct answer is the
    checkout — never ``host``.
    """
    host = _init_repo(git_env / "host-repo")
    checkout = git_env / "elsewhere" / "checkout"
    checkout.mkdir(parents=True)
    metadata = host / "nested-metadata"
    _git("init", "-q", "-b", "main", f"--separate-git-dir={metadata}", cwd=checkout)

    find = _import()
    root = find(checkout)
    assert root == checkout.resolve(), f"expected the checkout, got {root}"
    assert root != host.resolve()


# --- Reviewer follow-up: separate metadata literally named ".git" -------------
#
# ``git init --separate-git-dir=<storage>/.git`` makes the common dir's *basename*
# ``.git`` while its parent is storage, not a checkout. A basename-only ownership
# assumption returned ``storage`` and created session trees beneath it. These
# cases assert the real allocation: which root reaches the Agent helper and where
# the tree lands.


def _separate_dotgit_fixture(base):
    """A checkout whose metadata is ``<base>/storage-only/.git``."""
    checkout = base / "actual-checkout"
    storage = base / "storage-only"
    checkout.mkdir(parents=True, exist_ok=True)
    storage.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", f"--separate-git-dir={storage / '.git'}", cwd=checkout)
    (checkout / "README.md").write_text("x\n", encoding="utf-8")
    _git("add", "README.md", cwd=checkout)
    _git("commit", "-q", "-m", "init", cwd=checkout)
    return checkout, storage


def _stub_agent_setup(monkeypatch, created_root):
    """Replace the Agent helper with one that records its argument.

    The real helper is exercised by the maintainer's replay; here the contract
    under test is *which root WebUI passes down* and where the tree is placed.
    """
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


def test_separate_git_dir_named_dotgit_direct_allocation(git_env, monkeypatch):
    """Direct input: the Agent must get the checkout, never the storage dir."""
    checkout, storage = _separate_dotgit_fixture(git_env)
    created = []
    seen = _stub_agent_setup(monkeypatch, created)

    from api.worktrees import create_worktree_for_workspace

    info = create_worktree_for_workspace(checkout)

    assert Path(seen["repo_root"]).resolve() == checkout.resolve()
    assert Path(info["repo_root"]).resolve() == checkout.resolve()
    assert Path(info["path"]).resolve().is_relative_to(checkout.resolve())
    assert not Path(info["path"]).resolve().is_relative_to(storage.resolve())
    assert (checkout / "README.md").exists()


def test_separate_git_dir_named_dotgit_linked_allocation(git_env, monkeypatch):
    """Linked input: still the checkout that owns the metadata."""
    checkout, storage = _separate_dotgit_fixture(git_env)
    linked = git_env / "linked-wt"
    _git("worktree", "add", "-q", "-b", "side", str(linked), cwd=checkout)

    created = []
    seen = _stub_agent_setup(monkeypatch, created)

    from api.worktrees import create_worktree_for_workspace

    info = create_worktree_for_workspace(linked)

    assert Path(seen["repo_root"]).resolve() == checkout.resolve()
    assert Path(info["repo_root"]).resolve() == checkout.resolve()
    assert not Path(info["path"]).resolve().is_relative_to(storage.resolve())


def test_separate_git_dir_named_dotgit_resolver_direct(git_env):
    """The resolver alone answers the checkout for both inputs."""
    checkout, storage = _separate_dotgit_fixture(git_env)
    linked = git_env / "linked-wt2"
    _git("worktree", "add", "-q", "-b", "side2", str(linked), cwd=checkout)

    find = _import()
    assert find(checkout) == checkout.resolve()
    assert find(linked) == checkout.resolve()
    assert find(checkout) != storage.resolve()


def test_distant_primary_with_linked_input_never_returns_storage(git_env, monkeypatch):
    """Reviewer's residual layout: the primary checkout is outside the search.

    ``base/elsewhere/deep/actual-checkout`` holds the gitfile, the metadata is
    ``base/storage-only/.git``, and the session enters ``base/linked-wt/nested``.
    Scanning the metadata's children cannot reach the checkout, so the resolver
    must not fall back to ``storage-only`` (git reports it as a working tree
    because of the basename). The contract: either the real checkout, or the
    explicit conservative ``None`` that lets the caller keep the current
    checkout — never storage.
    """
    base = git_env / "distant"
    checkout = base / "elsewhere" / "deep" / "actual-checkout"
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

    find = _import()
    for workspace in (checkout, linked, nested):
        root = find(workspace)
        assert root != storage.resolve(), (
            f"storage must never be returned as a project root (input {workspace})"
        )
        assert root is None or root == checkout.resolve(), (
            f"expected the checkout or the conservative None, got {root}"
        )


def test_distant_primary_allocation_never_lands_in_storage(git_env, monkeypatch):
    """The allocation control: whatever the resolver answers, not storage."""
    base = git_env / "distant-alloc"
    checkout = base / "elsewhere" / "deep" / "checkout-two"
    storage = base / "storage-two"
    checkout.mkdir(parents=True)
    storage.mkdir(parents=True)
    _git("init", "-q", "-b", "main", f"--separate-git-dir={storage / '.git'}", cwd=checkout)
    (checkout / "tracked.txt").write_text("x\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=checkout)
    _git("commit", "-q", "-m", "init", cwd=checkout)
    linked = base / "linked-two"
    _git("worktree", "add", "-q", "-b", "side2", str(linked), cwd=checkout)

    created = []
    seen = _stub_agent_setup(monkeypatch, created)

    from api.worktrees import create_worktree_for_workspace

    info = create_worktree_for_workspace(linked)

    root = Path(seen["repo_root"]).resolve()
    assert root != storage.resolve(), "the Agent must never be handed storage"
    assert not Path(info["path"]).resolve().is_relative_to(storage.resolve())
