"""Helpers for WebUI-managed Hermes Agent git worktrees."""

from __future__ import annotations

import subprocess
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import logging
import os

from api.subprocess_utils import windows_hide_flags

logger = logging.getLogger(__name__)

# Upper bound on ancestor steps taken while looking for the gitfile that owns a
# common git dir. A real checkout sits a handful of levels below the metadata
# parent; the bound only stops a pathological path, it is not a search radius.
_CHECKOUT_WALK_MAX_DEPTH = 256

# How many ancestors above a metadata parent are searched when looking for the
# gitfile that claims it. Each ancestor is one bounded descent (see
# ``_CHECKOUT_LISTING_DEPTH``), never an unbounded walk.
_CHECKOUT_OWNING_GITDIR_MAX_UP = 4

# How many directory levels below each root the gitfile claim search looks.
# Every level is listed once and never walked twice; descent is decided before a
# directory is entered, so a large sibling subtree is never walked to completion
# (the ``rglob`` failure mode). Depth is not what grants ownership: when the
# search is *truncated* — by this bound or by the listing budget below — the
# ownership is unknown and the caller keeps its own checkout. Truncation is
# therefore strictly conservative, and raising the bound only makes the answer
# better (the real checkout instead of ``None``); it can never crown a
# ``--separate-git-dir`` storage dir.
_CHECKOUT_LISTING_DEPTH = 16

# Hard cap on directories visited by one claim search. A plain repository has no
# claim to find, so the search runs to completion over its tree; the budget keeps
# that finite for operator layouts with very many directories. Exhausting the
# budget is *truncation*, not "no claim": it must never be read as ownership.
_CHECKOUT_SEARCH_MAX_DIRS = 8192

# Directory names the claim search does not descend into. They hold dependency or
# tooling trees, not a repository checkout; skipping them keeps the search cheap
# on ordinary working trees and is a deliberate stop, not an unfinished search.
# A checkout of this metadata living inside one would go unfound, which costs the
# plain-layout answer at worst — it can never promote a candidate.
_CHECKOUT_SKIP_DESCENT = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        "venv",
    }
)


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
    """Return the *main* working-tree root that owns the workspace's repo.

    ``find_git_repo_root`` returns ``git rev-parse --show-toplevel``, which for
    a session started inside a linked worktree (e.g. ``<repo>/.worktrees/x``) is
    that worktree itself. Passing it to the agent's ``_setup_worktree`` nests
    the new tree *inside* the current one, and no pruner ever reclaims a nest.

    The answer is derived from the **common git dir**, which is not always
    ``<main>/.git``: ``--separate-git-dir=X`` and submodules put the metadata
    elsewhere. One rule decides, and it never looks at a directory's name, at
    stray files, at the index or at committed content — all of those describe a
    ``--separate-git-dir`` *storage* dir exactly as well as a real checkout,
    since git even reports storage as a working tree (its basename is ``.git``):

        the working tree is the directory whose ``.git`` **gitfile** claims
        this metadata — the workspace's own ancestry first, then the metadata's
        parent and its ancestors, each listed with bounded descent (a plain
        repository's parent dir holds the metadata itself as ``.git``).

    A gitfile naming ``<common>/worktrees/<name>`` belongs to a linked worktree
    and is followed through that admin dir's ``commondir``; ``core.worktree`` is
    accepted when git records it.

    When no claim exists the layout is genuinely ambiguous — a
    ``--separate-git-dir`` storage dir is shaped exactly like a plain checkout —
    and the answer is ``None``: the caller then keeps the workspace's own
    toplevel, which nests at worst. It never starts a session inside git
    bookkeeping. The plain layout is trusted only when the claim search finished
    without being truncated; a truncated search (depth cap, listing budget or an
    unreadable level) is unknown ownership, never a promotion of metadata shape
    into ownership.
    """
    ws = Path(workspace).expanduser().resolve()
    if not ws.is_dir():
        return None

    common_path = _common_git_dir(ws)
    if common_path is None:
        return None
    try:
        common_resolved = common_path.resolve()
    except OSError:
        return None

    # A linked worktree is never the main tree: unnest to whatever owns the
    # metadata, never to the workspace's own toplevel.
    if not _is_linked_worktree(ws, common_path):
        owner = _checkout_from_ancestry(ws, common_resolved)
        if owner is not None:
            return owner

    owner, search_complete = _claiming_checkout(common_path, common_resolved)
    if owner is not None:
        return owner

    # No gitfile claimed the metadata. Decide from recorded topology only, and
    # only when the claim search actually finished: a truncated search says
    # nothing about who owns the metadata, so "found nothing" stays unknown.
    candidate = _topology_candidate(
        common_path, common_resolved, claim_search_complete=search_complete
    )
    if candidate is not None:
        return candidate

    recorded = _core_worktree(common_path)
    if recorded is not None:
        try:
            recorded_resolved = recorded.resolve()
        except OSError:
            recorded_resolved = None
        if (
            recorded_resolved is not None
            and recorded.is_dir()
            and _git_toplevel(recorded) == recorded_resolved
        ):
            return recorded_resolved

    return None


def _core_worktree(common_path: Path) -> Path | None:
    """``core.worktree`` for the repository at *common_path*, as an absolute path."""
    try:
        result = subprocess.run(
            ["git", "--git-dir", str(common_path), "config", "--get", "core.worktree"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    raw = result.stdout.strip() if result.returncode == 0 else ""
    if not raw:
        return None
    wt = Path(raw).expanduser()
    if not wt.is_absolute():
        wt = common_path / wt
    return wt


def _checkout_from_ancestry(start: Path, common_resolved: Path) -> Path | None:
    """The nearest ancestor-or-self of *start* whose gitfile claims the metadata.

    Only a gitfile pointing **straight** at the common dir counts: a linked
    worktree's gitfile names ``<common>/worktrees/<name>`` and belongs to a
    different working tree. Depth is bounded only against a symlink loop.
    """
    try:
        current = start.resolve()
    except OSError:
        return None
    for _ in range(_CHECKOUT_WALK_MAX_DEPTH):
        owner = _gitfile_owner(current, common_resolved)
        if owner is not None:
            return owner
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return None


def _gitfile_owner(directory: Path, common_resolved: Path) -> Path | None:
    """*directory*, when its ``.git`` gitfile points straight at the metadata."""
    gitfile = directory / ".git"
    try:
        if not gitfile.is_file():
            return None
        text = gitfile.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    pointed = Path(text.split(":", 1)[1].strip())
    if not pointed.is_absolute():
        pointed = directory / pointed
    try:
        if pointed.resolve() != common_resolved:
            return None
    except OSError:
        return None
    toplevel = _git_toplevel(directory)
    if toplevel is None:
        return None
    try:
        return toplevel if toplevel == directory.resolve() else None
    except OSError:
        return None


def _repo_effective_root(common_resolved: Path) -> Path | None:
    """The topmost directory a checkout of this repository can live under.

    ``<root>/.git`` (plain repo, or a separate-git-dir storage dir named
    ``.git``) bounds where its checkout can be: below ``root``. Metadata with
    any other name gives no such bound.

    The bound must be the *topmost* one available, or a checkout that needs a
    deeper search level is pruned before it can be found: for a metadata
    ``<base>/storage/git`` the bound is ``<base>``, the level the search starts
    from, so the search may continue upward.
    """
    if common_resolved.name != ".git":
        return None
    # ``<base>/storage/.git`` and ``<base>/<checkout>/.git`` both bound the
    # repository inside ``<base>``: the checkout cannot live above the level
    # that holds the metadata directory.
    root = common_resolved.parent.parent
    try:
        return root if root.is_dir() else None
    except OSError:
        return None


def _claiming_checkout(
    common_path: Path, common_resolved: Path
) -> tuple[Path | None, bool]:
    """A checkout whose gitfile claims the metadata, and whether we searched all.

    Returns ``(owner, complete)``. ``owner`` is a checkout whose ``.git`` gitfile
    points at this metadata (directly, or at its per-worktree admin dir); that
    claim is the only authoritative evidence of who owns separate metadata, and
    the first one found wins.

    ``complete`` says whether the walk exhausted every directory it was allowed
    to enter. ``False`` means truncated — by ``_CHECKOUT_LISTING_DEPTH``, by the
    listing budget or by an unreadable level — so "no claim found" is *unknown
    ownership*, not evidence that the metadata's parent owns it. The caller must
    then keep the workspace's own checkout: nesting at worst, never a session
    inside a ``--separate-git-dir`` storage dir.

    The metadata's parent is tried first — for a plain repository the parent *is*
    the working tree, but its ``.git`` is a directory rather than a gitfile, so
    nothing is claimed here — then the ancestors of the metadata parent, each
    listed with the same bounded descent so a checkout sitting beside the
    metadata (``<base>/storage/.git`` with the checkout in ``<base>/elsewhere/…``)
    is found.

    A directory that is itself a working-tree root (it holds a ``.git`` entry) is
    never descended into: it is a checkout, not a container of one. That prunes
    session trees under ``.worktrees/`` and nested repositories in one step.
    """
    parent = common_path.parent
    seen: set[str] = set()

    # When the local search finds nothing, the *ancestor* levels are listed too:
    # the primary can sit above the metadata's parent (``<base>/storage/.git``
    # with the checkout at ``<base>/elsewhere/...``). Those levels are finite,
    # since ``$HOME`` and ``/`` are excluded.
    roots = [parent]
    stop_at = _repo_effective_root(common_resolved)
    ancestor = parent.parent
    for _ in range(_CHECKOUT_OWNING_GITDIR_MAX_UP):
        if not ancestor.is_dir() or ancestor == ancestor.parent or ancestor == Path("/"):
            break
        if stop_at is not None and ancestor == stop_at:
            roots.append(ancestor)
            break
        roots.append(ancestor)
        ancestor = ancestor.parent
    if parent.name == "modules" and parent.parent.name == ".git":
        # ``<super>/.git/modules/<name>``: the submodule lives in ``<super>/<name>``.
        roots.append(parent.parent.parent)
    # A git metadata directory is never a container of checkouts (the submodule
    # or storage checkout lives beside it, not inside it), so walking it would
    # only spend the budget on ``objects/``.
    roots = [root for root in roots if root.name != ".git"]

    budget = _CHECKOUT_SEARCH_MAX_DIRS
    truncated = False
    for root in roots:
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = sorted(os.scandir(current), key=lambda entry: entry.name)
            except OSError:
                # Unreadable level: we could not look, so the answer is unknown.
                return None, False
            children: list[tuple[Path, int]] = []
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if entry.name == ".git":
                    continue
                child = Path(entry.path)
                key = str(child)
                if key in seen:
                    continue
                seen.add(key)
                if budget <= 0:
                    return None, False
                budget -= 1
                owner = _gitfile_owner(child, common_resolved)
                if owner is not None:
                    return owner, True
                # Descend only past the dependency trees and outside other
                # checkouts; stopping there is a deliberate policy, not an
                # unfinished search. Stopping at the cap is different: a claim
                # may sit below it, so that is truncation, not "no claim".
                if entry.name in _CHECKOUT_SKIP_DESCENT:
                    continue
                try:
                    if os.path.lexists(child / ".git"):
                        # A checkout of some repository: not a container of one.
                        continue
                except OSError:
                    continue
                if depth + 1 >= _CHECKOUT_LISTING_DEPTH:
                    truncated = True
                    continue
                children.append((child, depth + 1))
            stack.extend(reversed(children))

    return None, not truncated


def _topology_candidate(
    common_path: Path, common_resolved: Path, *, claim_search_complete: bool
) -> Path | None:
    """The working tree git's own records tie to this metadata, or ``None``.

    Reached only after no gitfile claimed the metadata. Two recorded facts can
    still name a working tree, in this order:

    * ``core.worktree``, which names it outright (a ``--separate-git-dir``
      checkout may carry it);
    * ``<tree>/.git`` *is* this metadata (a real directory, not a gitfile) — the
      plain layout, where the parent is the working tree.

    Both are *layout* facts, never content or presence facts. A working tree owns
    its repository whether its files are clean, modified, staged, deleted or
    entirely rewritten, so no index query, ``git status`` or per-path existence
    test takes part in this decision: an ordinary primary that deleted a tracked
    leaf still owns its repository. An empty index is not a special case either —
    a fresh ``init`` tracks nothing and still owns its parent.

    A linked admin dir (``commondir``) is never a primary. The plain layout is
    only trusted when the gitfile claim search actually finished
    (``claim_search_complete``): a ``--separate-git-dir`` storage dir named
    ``.git`` has exactly the same shape as a plain checkout, so its ownership is
    decided by the claim its real checkout makes elsewhere — and when that claim
    was not found because the search was truncated, the answer is ``None``.
    Unknown ownership stays unknown, and the caller keeps the checkout it has:
    nesting at worst, never a session inside git's bookkeeping.
    """
    recorded = _core_worktree(common_path)
    if recorded is not None:
        resolved = _as_dir(recorded)
        if resolved is not None:
            return resolved

    try:
        if (common_resolved / "commondir").is_file():
            # A per-worktree admin dir: never a primary checkout.
            return None
    except OSError:
        return None

    parent = _as_dir(common_resolved.parent)
    if parent is None:
        return None
    # The plain layout: the parent's own ``.git`` *is* this metadata directory.
    # Shape alone cannot tell the parent of a ``--separate-git-dir`` storage dir
    # apart from a plain checkout — both hold a ``.git`` directory, both are
    # reported as the toplevel by git, and ``git worktree list`` names both as the
    # primary worktree — so the claim search above is what refuses storage. Here
    # we only accept the layout once that search provably found no claim.
    if (
        claim_search_complete
        and common_resolved.name == ".git"
        and _plain_git_dir_owner(parent, common_resolved)
    ):
        return parent
    return None


def _plain_git_dir_owner(parent: Path, common_resolved: Path) -> bool:
    """True when *parent*'s own ``.git`` is this metadata directory.

    The plain layout, decided without asking git to interpret a gitfile: the
    parent holds a ``.git`` *directory* that is the metadata itself. This is a
    layout fact only — a storage dir's parent holds the very same ``.git``
    directory (git's bookkeeping lives there too), which is why it is not
    ownership on its own.
    """
    try:
        parent_git = parent / ".git"
        if not parent_git.is_dir():
            return False
        return parent_git.samefile(common_resolved)
    except OSError:
        return False


def _as_dir(raw: Path) -> Path | None:
    """*raw* resolved, when it is a directory."""
    try:
        resolved = raw.resolve()
    except OSError:
        return None
    try:
        return resolved if resolved.is_dir() else None
    except OSError:
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
