"""Helpers for WebUI-managed Hermes Agent git worktrees."""

from __future__ import annotations

import subprocess
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import logging

from api.subprocess_utils import windows_hide_flags

logger = logging.getLogger(__name__)


def _run_git(args: list[str], cwd: str | Path, timeout: float = 2) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        creationflags=windows_hide_flags(),
    )


def _resolve_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    try:
        return Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return Path(path).expanduser()


def _worktree_list_cwd(worktree_path: Path, repo_root: str | Path | None) -> Path | None:
    repo = _resolve_path(repo_root)
    if repo and repo.is_dir():
        return repo
    if worktree_path.is_dir():
        return worktree_path
    return None


def _parse_worktree_list_porcelain(output: str) -> set[str]:
    paths: set[str] = set()
    for line in str(output or "").splitlines():
        if not line.startswith("worktree "):
            continue
        path = line[len("worktree "):].strip()
        if not path:
            continue
        resolved = _resolve_path(path)
        paths.add(str(resolved or Path(path).expanduser()))
    return paths


def _worktree_listed(worktree_path: Path, repo_root: str | Path | None) -> bool:
    """Return whether git currently lists the worktree.

    False is a safe fallback for probe failures, not definitive orphan proof.
    Future cleanup UI must combine this with the rest of the status payload.
    """
    cwd = _worktree_list_cwd(worktree_path, repo_root)
    if cwd is None:
        return False
    try:
        result = _run_git(["worktree", "list", "--porcelain"], cwd)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return str(worktree_path) in _parse_worktree_list_porcelain(result.stdout)


def _status_porcelain(worktree_path: Path) -> tuple[bool, int]:
    try:
        result = _run_git(
            ["status", "--porcelain", "--untracked-files=normal"],
            worktree_path,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, 0
    if result.returncode != 0:
        return False, 0
    lines = [line for line in result.stdout.splitlines() if line]
    return bool(lines), sum(1 for line in lines if line.startswith("??"))


def _ahead_behind(worktree_path: Path) -> dict:
    payload = {
        "ahead": 0,
        "behind": 0,
        "available": False,
        "upstream": None,
    }
    try:
        upstream = _run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            worktree_path,
        )
    except (OSError, subprocess.TimeoutExpired):
        return payload
    if upstream.returncode != 0:
        return payload
    upstream_ref = upstream.stdout.strip()
    if not upstream_ref:
        return payload
    payload["upstream"] = upstream_ref
    try:
        counts = _run_git(
            ["rev-list", "--left-right", "--count", "HEAD...@{u}"],
            worktree_path,
        )
    except (OSError, subprocess.TimeoutExpired):
        return payload
    if counts.returncode != 0:
        return payload
    parts = counts.stdout.strip().split()
    if len(parts) != 2:
        return payload
    try:
        payload["ahead"] = max(0, int(parts[0]))
        payload["behind"] = max(0, int(parts[1]))
        payload["available"] = True
    except ValueError:
        pass
    return payload


def _locked_by_stream(session) -> bool:
    stream_id = getattr(session, "active_stream_id", None)
    if not stream_id:
        return False
    try:
        from api.config import STREAMS, STREAMS_LOCK

        with STREAMS_LOCK:
            return stream_id in STREAMS
    except Exception:
        return False


def _locked_by_terminal(session_id: str, worktree_path: Path) -> bool:
    try:
        from api.terminal import get_terminal

        term = get_terminal(session_id)
    except Exception:
        return False
    if not term:
        return False
    try:
        if not term.is_alive():
            return False
        terminal_workspace = _resolve_path(getattr(term, "workspace", None))
        return terminal_workspace == worktree_path
    except Exception:
        return False


def worktree_status_for_session(session) -> dict:
    """Return a read-only worktree status snapshot for a WebUI session."""
    raw_path = getattr(session, "worktree_path", None)
    if not raw_path:
        raise ValueError("Session is not worktree-backed")

    worktree_path = _resolve_path(raw_path)
    if worktree_path is None:
        raise ValueError("Session is not worktree-backed")

    exists = worktree_path.is_dir()
    status = {
        "path": str(worktree_path),
        "exists": bool(exists),
        "dirty": False,
        "untracked_count": 0,
        "ahead_behind": {
            "ahead": 0,
            "behind": 0,
            "available": False,
            "upstream": None,
        },
        "locked_by_stream": _locked_by_stream(session),
        "locked_by_terminal": _locked_by_terminal(
            getattr(session, "session_id", ""),
            worktree_path,
        ),
        "listed": _worktree_listed(
            worktree_path,
            getattr(session, "worktree_repo_root", None),
        ),
    }
    if not exists:
        return status

    dirty, untracked_count = _status_porcelain(worktree_path)
    status["dirty"] = dirty
    status["untracked_count"] = untracked_count
    status["ahead_behind"] = _ahead_behind(worktree_path)
    return status


def remove_worktree_for_session(session, *, force: bool = False) -> dict:
    """Remove a session's git worktree from disk.

    Returns status dict with keys: ok, removed_path, warnings.
    Raises ValueError for terminal blockers (locked by stream/terminal,
    dirty with force=False).
    """
    raw_path = getattr(session, "worktree_path", None)
    if not raw_path:
        raise ValueError("Session is not worktree-backed")

    worktree_path = _resolve_path(raw_path)
    if worktree_path is None:
        raise ValueError("Session is not worktree-backed")

    # Read current status before removal
    status = worktree_status_for_session(session)

    if not status["exists"]:
        return {
            "ok": True,
            "removed_path": str(worktree_path),
            "warnings": ["Worktree directory no longer exists on disk."],
        }

    warnings = []

    # Guard: locked by stream
    if status["locked_by_stream"]:
        raise ValueError("Worktree is locked by an active streaming session")

    # Guard: locked by terminal
    if status["locked_by_terminal"]:
        raise ValueError("Worktree is locked by an active terminal session")

    # Guard: local changes and unpushed commits without explicit force.
    if status["dirty"] and not force:
        raise ValueError(
            "Worktree has uncommitted changes. Use force=true to override."
        )
    if status["untracked_count"] > 0:
        if force:
            warnings.append(
                f"{status['untracked_count']} untracked file(s) will be removed."
            )
        else:
            raise ValueError(
                f"Worktree has {status['untracked_count']} untracked file(s). "
                "Use force=true to override."
            )
    ahead = int((status.get("ahead_behind") or {}).get("ahead") or 0)
    if ahead > 0:
        if force:
            warnings.append(f"{ahead} unpushed commit(s) will be removed.")
        else:
            raise ValueError(
                f"Worktree has {ahead} unpushed commit(s). "
                "Use force=true to override."
            )

    # Remove the worktree — must run from the repo root, not the worktree dir
    repo_root = getattr(session, "worktree_repo_root", None)
    if not repo_root:
        raise ValueError("Session missing worktree_repo_root")

    # Unlock the creation-time bookkeeping lock before removing (fail-soft).
    # The agent locks every worktree it creates (cli._setup_worktree), and git
    # refuses to remove a locked worktree with a single --force, so without
    # this every WebUI-created worktree is unremovable.  The dirty/untracked/
    # unpushed/stream/terminal guards above are the real safety layer, not
    # git's lock — mirrors the agent's own _cleanup_worktree ordering.
    try:
        _run_git(["worktree", "unlock", str(worktree_path)], str(repo_root), timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass  # already unlocked / never locked — non-fatal

    try:
        remove_args = ["worktree", "remove"]
        if force:
            remove_args.append("--force")
        remove_args.append(str(worktree_path))
        result = _run_git(remove_args, str(repo_root), timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Failed to remove worktree: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().split("\n")[-1]
        raise ValueError(
            f"git worktree remove failed: {stderr or result.stdout.strip()}"
        )

    # Prune in case the worktree dir was already gone
    try:
        _run_git(
            ["worktree", "prune"],
            str(repo_root),
            timeout=5,
        )
    except Exception:
        pass

    return {
        "ok": True,
        "removed_path": str(worktree_path),
        "warnings": warnings or None,
    }


def find_git_repo_root(workspace: str | Path) -> Path:
    """Return the enclosing git repo root for *workspace*.

    Use git itself instead of checking ``workspace/.git`` so nested workspaces
    and linked git worktrees are both handled correctly.
    """
    ws = Path(workspace).expanduser().resolve()
    if not ws.is_dir():
        raise ValueError("Workspace path does not exist or is not a directory")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ws,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Workspace is not inside a git repository") from exc
    if result.returncode != 0:
        raise ValueError("Workspace is not inside a git repository")
    root = result.stdout.strip()
    if not root:
        raise ValueError("Workspace is not inside a git repository")
    return Path(root).expanduser().resolve()


def _git_toplevel(cwd: str | Path) -> Path | None:
    """``git rev-parse --show-toplevel`` for *cwd*, or None when it is not a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--show-toplevel"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    if not out:
        return None
    path = Path(out).expanduser()
    return path.resolve() if path.is_dir() else None


def _common_git_dir(cwd: str | Path) -> Path | None:
    """``git rev-parse --git-common-dir`` as an absolute path, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return Path(out).expanduser() if out else None


def find_main_worktree_root(workspace: str | Path) -> Path | None:
    """Return the *main* working-tree root, resolving linked worktrees.

    ``find_git_repo_root`` returns ``git rev-parse --show-toplevel``, which for
    a session started inside a linked worktree (e.g. ``<repo>/.worktrees/x``)
    is that worktree itself.  Passing it to the agent's ``_setup_worktree``
    nests the new tree *inside* the current one, and no pruner ever reclaims a
    nest.

    The main tree is resolved from the **common git dir**, but that dir is not
    always ``<main>/.git``: ``git init --separate-git-dir=X`` puts the metadata
    in ``X``, and a submodule's common dir is
    ``super/.git/modules/<name>``.  Treating either as a checkout created the
    session trees *inside the metadata* and reported that path as ``repo_root``
    (the reviewer's P2).  So: take the common dir, ask git for the working tree
    that owns it, and only fall back to the parent-of-``.git`` shape when the
    checkout cannot be recovered.  Returns ``None`` when the workspace is not in
    a git repo, letting the caller use ``find_git_repo_root``.
    """
    ws = Path(workspace).expanduser().resolve()
    if not ws.is_dir():
        return None

    common_path = _common_git_dir(ws)
    if common_path is None:
        return None

    # 1. The workspace's own gitfile is the strongest evidence available: a
    #    ``<checkout>/.git`` holding ``gitdir: <this common dir>`` names the
    #    checkout that owns the metadata, whatever the metadata directory is
    #    called. This is checked before any ``name == '.git'`` shape test,
    #    because ``--separate-git-dir=<storage>/.git`` produces a common dir
    #    whose *basename* is ``.git`` while its parent is storage, not a
    #    checkout (the reviewer's remaining blocker on #8162).
    #    A *linked* worktree carries a gitfile too, but it owns an admin dir
    #    under the common dir, not the common dir itself. Its own gitfile names
    #    the primary checkout's git dir, and ``git worktree list`` reports that
    #    primary path — but git also reports a ``.git``-named storage dir as a
    #    working tree, so that output is not ownership proof on its own. Use it
    #    only as a *search start*: the candidate must still carry a gitfile
    #    pointing at this common dir.
    if _is_linked_worktree(ws, common_path):
        near = _linked_primary_candidate(ws)
    else:
        near = ws
    owning = _checkout_owning_gitdir(common_path, near=near)
    if owning is not None:
        return owning

    # 2. A common dir shaped ``<main>/.git`` names its checkout directly — but
    #    only when the parent really is that checkout. Two traps to exclude:
    #    a separate-git-dir storage dir named ``.git`` (its parent holds
    #    metadata, not a checkout) and that same storage dir itself, which git
    #    happily reports as a working tree because of the basename.
    #    The discriminator: in a real checkout, ``<root>/.git`` IS a path that
    #    resolves to this common dir (a gitfile pointing here, or the common
    #    dir itself). In the storage trap, ``<root>/.git`` resolves to the
    #    common dir too — so that alone proves nothing. What separates them is
    #    that storage *is* the common dir's own parent and contains no other
    #    checkout evidence. Anything not proven falls through.
    if common_path.name == ".git":
        root = common_path.parent
        if root.is_dir():
            git_entry = root / ".git"
            try:
                entry_resolves_to_common = git_entry.exists() and (
                    git_entry.resolve() == common_path.resolve()
                )
            except OSError:
                entry_resolves_to_common = False
            # A plain repository: ``<root>/.git`` IS the common dir. The storage
            # trap looks structurally identical from here (``<storage>/.git`` is
            # the common dir, and git reports ``storage`` as a working tree
            # because of the basename), so a plain-repo answer additionally
            # requires the candidate to hold working-tree content: a real
            # checkout has files besides ``.git``, storage holds only ``.git``.
            if git_entry.is_dir() and entry_resolves_to_common:
                toplevel = _git_toplevel(root)
                if (
                    toplevel is not None
                    and toplevel == root.resolve()
                    and _looks_like_working_tree(root)
                ):
                    return toplevel
            # A gitfile here means this is a separate-git-dir *checkout* whose
            # metadata lives elsewhere: never the storage trap, resolve it.
            elif git_entry.is_file():
                toplevel = _git_toplevel(root)
                if toplevel is not None and toplevel == root.resolve():
                    return toplevel
            elif git_entry.is_dir() and not entry_resolves_to_common:
                toplevel = _git_toplevel(root)
                if toplevel is not None and toplevel == root.resolve():
                    owned = _common_git_dir(root)
                    try:
                        if owned is not None and owned.resolve() == common_path.resolve():
                            return toplevel
                    except OSError:
                        pass

    # 3. separate-git-dir / submodule: the metadata dir is NOT a checkout, and
    #    ``--show-toplevel`` run *in* the metadata answers with the cwd's repo
    #    (a separate-git-dir checkout sets no core.worktree), so it cannot be
    #    trusted. The reliable source is the gitfile that points at this
    #    metadata: ``<checkout>/.git`` containing ``gitdir: <common dir>``.
    found = _checkout_owning_gitdir(common_path)
    if found is not None:
        return found

    # 4. Resolve a separate-git-dir checkout through ``core.worktree`` when set
    #    (git uses it for the same purpose in other layouts).
    try:
        result = subprocess.run(
            ["git", "--git-dir", str(common_path), "config", "--get", "core.worktree"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=windows_hide_flags(),
        )
        raw = result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        raw = ""
    if raw:
        wt = Path(raw).expanduser()
        if not wt.is_absolute():
            wt = common_path / wt
        wt = wt.resolve()
        if wt.is_dir() and _git_toplevel(wt) is not None:
            return wt

    # 5. Conservative fallback: unnest to the parent of the metadata dir only
    #    when that parent is itself a real checkout *that owns this common dir*.
    #    Being a checkout is not enough: a repository can keep its separate Git
    #    directory inside an unrelated repo's tree while the checkout lives
    #    elsewhere, and returning that parent would start the session in the
    #    wrong project. A storage dir literally named ``.git`` is excluded here
    #    by requiring the parent to carry its own ``.git`` entry: storage holds
    #    the metadata *as* ``.git`` and has no checkout of its own. Otherwise
    #    None -> the caller keeps the workspace's own toplevel (no worse than
    #    today).
    parent = common_path.parent
    if parent.is_dir() and common_path.name == ".git":
        git_entry = parent / ".git"
        if git_entry.is_file() or (
            git_entry.is_dir() and git_entry.resolve() != common_path.resolve()
        ):
            toplevel = _git_toplevel(parent)
            if toplevel is not None and toplevel == parent.resolve():
                owned = _common_git_dir(parent)
                if owned is not None:
                    try:
                        if owned.resolve() == common_path.resolve():
                            return toplevel
                    except OSError:
                        pass
    return None


def _looks_like_working_tree(root: Path) -> bool:
    """True when *root* holds working content, not just its ``.git`` metadata.

    A checkout has at least one entry besides ``.git`` (a tracked file, a
    subdirectory, even an untracked artefact). A ``--separate-git-dir`` storage
    directory holds the metadata as its only child, and git still reports it as
    a working tree because of the ``.git`` basename — which is exactly the shape
    that must never be returned as a project root.
    """
    try:
        for entry in root.iterdir():
            if entry.name != ".git":
                return True
    except OSError:
        return False
    return False


def _linked_primary_candidate(workspace: Path) -> Path | None:
    """Best-effort primary-checkout path for a linked worktree, or None.

    Reads the linked worktree's own ``gitdir`` file: it points at
    ``<common>/worktrees/<name>``, whose sibling ``gitdir`` file records the
    primary checkout's ``.git`` path. That gives the primary checkout directly,
    even when it is nowhere near the metadata directory.

    This is a *search start*, never an answer: git also treats a storage dir
    literally named ``.git`` as a working tree, so the caller still has to prove
    ownership by finding a gitfile that points at the common dir.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-dir"],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    admin_dir = Path(result.stdout.strip()).expanduser()
    if not admin_dir.is_dir():
        return None
    # ``<common>/worktrees/<name>/gitdir`` holds the linked worktree's own gitfile.
    # Sibling worktrees share the same primary, so any entry works.
    try:
        entries = sorted((admin_dir.parent).iterdir()) if admin_dir.parent.is_dir() else []
    except OSError:
        entries = []
    for entry in entries:
        candidate_file = entry / "gitdir"
        if not candidate_file.is_file():
            continue
        try:
            raw = candidate_file.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = entry / path
        # A worktree's gitfile lives at ``<checkout>/.git``.
        if path.name != ".git":
            continue
        checkout = path.parent
        if checkout.is_dir():
            return checkout
    return None


def _is_linked_worktree(workspace: Path, common_path: Path) -> bool:
    """True when *workspace* is a linked worktree of the repo at *common_path*.

    Decided by git itself: the per-worktree admin dir of a linked worktree sits
    under ``<common>/worktrees/<name>`` and ``--git-dir`` reports that path,
    while a primary checkout reports the common dir (or ``<checkout>/.git``,
    which resolves to it). Comparing the two paths avoids guessing from the
    ``.git`` shape — the whole point of the #8162 fix.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-dir"],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    out = result.stdout.strip()
    if not out:
        return False
    try:
        git_dir = Path(out).expanduser().resolve()
        return git_dir != common_path.resolve()
    except OSError:
        return False


def _gitfile_points_at(pointed: Path, common_target: Path) -> bool:
    """True when the gitdir *pointed* belongs to the repo at *common_target*.

    A checkout's gitfile points either straight at the common dir (plain
    ``--separate-git-dir``) or at a per-worktree admin dir under it
    (``<common>/worktrees/<name>``, which is how a *linked* worktree's gitfile
    is written). The per-worktree case is confirmed through its own
    ``commondir`` file, so a linked worktree of a separate-git-dir checkout
    still resolves to the checkout that owns the metadata instead of storage.
    """
    try:
        if pointed.resolve() == common_target:
            return True
    except OSError:
        return False

    commondir = pointed / "commondir"
    if not commondir.is_file():
        return False
    try:
        raw = commondir.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return False
    if not raw:
        return False
    resolved = Path(raw)
    if not resolved.is_absolute():
        resolved = pointed / resolved
    try:
        return resolved.resolve() == common_target
    except OSError:
        return False


def _checkout_owning_gitdir(common_path: Path, near: Path | None = None) -> Path | None:
    """Find a checkout whose ``.git`` gitfile points at *common_path*.

    ``git init --separate-git-dir=X`` leaves ``<checkout>/.git`` holding a single
    line ``gitdir: X``. The most reliable candidate is the workspace itself and
    its ancestors: a session started anywhere inside the real checkout carries
    that gitfile on the path up to the root, so *near* is searched first.
    Failing that, the metadata dir's siblings are scanned (the usual layout:
    ``<parent>/checkout`` + ``<parent>/metadata``), which is what recovers a
    *linked* worktree, whose gitfile lives outside the checkout. Every candidate
    is validated by reading its gitfile, so storage directories (including one
    literally named ``.git``) are never mistaken for checkouts. Submodule
    metadata (``<super>/.git/modules/<name>``) is matched the same way when the
    submodule is checked out under ``<super>/<name>``.
    """
    target = common_path.resolve()
    parent = common_path.parent
    candidates = []

    # The workspace ancestry first: for a direct separate-git-dir checkout this
    # is the checkout itself, so the answer does not depend on how the metadata
    # directory is named or where its siblings sit.
    if near is not None:
        try:
            for ancestor in [near, *near.parents]:
                if ancestor.is_dir():
                    candidates.append(ancestor)
        except OSError:
            pass

    # Siblings of the metadata dir (separate-git-dir), then its grandparent's
    # children — a storage dir literally named ``.git`` puts the checkout one
    # level further up (``<base>/checkout`` next to ``<base>/storage/.git``).
    # Then the superproject's own working tree (submodule:
    # ``<super>/.git/modules/<name>`` -> ``<super>/<name>``).
    if parent.is_dir():
        try:
            candidates.extend(sorted(p for p in parent.iterdir() if p.is_dir()))
        except OSError:
            pass
    grandparent = parent.parent
    if common_path.name == ".git" and grandparent.is_dir():
        try:
            candidates.extend(sorted(p for p in grandparent.iterdir() if p.is_dir()))
        except OSError:
            pass
    if parent.name == "modules":
        super_git = parent.parent
        if super_git.name == ".git" and super_git.parent.is_dir():
            candidates.extend(sorted(p for p in super_git.parent.iterdir() if p.is_dir()))

    for candidate in candidates:
        gitfile = candidate / ".git"
        if not gitfile.is_file():
            continue
        try:
            text = gitfile.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text.startswith("gitdir:"):
            continue
        raw = text.split(":", 1)[1].strip()
        pointed = Path(raw)
        if not pointed.is_absolute():
            pointed = (candidate / pointed)
        if not _gitfile_points_at(pointed, target):
            continue
        # A gitfile that resolves to the common dir through a per-worktree
        # admin dir (``<common>/worktrees/<name>``) belongs to a *linked*
        # worktree, not to the primary checkout. Returning it would report the
        # linked worktree as the main tree and defeat the unnesting this
        # function exists for, so only a gitfile pointing straight at the
        # common dir may answer.
        try:
            direct = pointed.resolve() == target
        except OSError:
            direct = False
        if not direct:
            continue
        toplevel = _git_toplevel(candidate)
        if toplevel is None:
            continue
        if toplevel != candidate.resolve():
            continue
        return toplevel
    return None



def _setup_agent_worktree(repo_root: str) -> dict:
    try:
        import importlib.util
        from pathlib import Path

        from api.config import _AGENT_DIR  # Hermes Agent source root

        # Use importlib to load cli.py from its absolute path instead of a bare
        # ``from cli import _setup_worktree``.  The bare import is vulnerable to
        # namespace-package shadowing — any third-party package (e.g. the ``cli``
        # namespace shipped by stringzilla) that creates a cli/ directory in
        # site-packages will preempt the real hermes-agent/cli.py module.
        cli_path = str(Path(_AGENT_DIR) / "cli.py")
        spec = importlib.util.spec_from_file_location(
            "hermes_cli_worktree", cli_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not locate hermes-agent worktree helper at {cli_path}"
            )
        cli_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_mod)
        _setup_worktree = cli_mod._setup_worktree
    except Exception as exc:
        raise RuntimeError("Hermes Agent worktree helper is unavailable") from exc
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        info = _setup_worktree(repo_root)
    emitted = output.getvalue().strip()
    if emitted:
        logger.debug("Hermes Agent worktree helper output: %s", emitted)
    if not info:
        raise RuntimeError("Hermes Agent failed to create a git worktree")
    return info


def create_worktree_for_workspace(workspace: str | Path) -> dict:
    # Place the tree under the *main* working tree, never under the current
    # linked worktree: a workspace that is itself a worktree (the common case
    # once webui_state/last_workspace.txt carries a session's tree) would
    # otherwise nest the new tree inside it, out of reach of every pruner.
    repo_root = find_main_worktree_root(workspace) or find_git_repo_root(workspace)
    info = _setup_agent_worktree(str(repo_root))
    path = info.get("path")
    branch = info.get("branch")
    if not path or not branch:
        raise RuntimeError("Hermes Agent returned incomplete worktree metadata")
    return {
        "path": str(Path(path).expanduser().resolve()),
        "branch": str(branch),
        "repo_root": str(Path(info.get("repo_root") or repo_root).expanduser().resolve()),
        "created_at": time.time(),
    }
