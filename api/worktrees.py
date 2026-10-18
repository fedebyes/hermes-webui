"""Helpers for WebUI-managed Hermes Agent git worktrees."""

from __future__ import annotations

import subprocess
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Iterator
from pathlib import Path

import logging
import os

from api.subprocess_utils import windows_hide_flags

logger = logging.getLogger(__name__)

# Upper bound on ancestor steps taken while looking for the gitfile that owns a
# common git dir. A real checkout sits a handful of levels below the metadata
# parent; the bound only stops a pathological path, it is not a search radius.
_CHECKOUT_WALK_MAX_DEPTH = 256

# How many ancestors above a metadata parent are listed when looking for the
# gitfile that claims it. Each step is one flat ``scandir`` of that level plus
# one pruned level, never a recursive walk.
_CHECKOUT_OWNING_GITDIR_MAX_UP = 4

# How many directory levels below each ancestor of a metadata parent are
# listed when looking for the gitfile that claims it. Every level is listed
# once and never walked twice; this is what keeps the search bounded. The bound
# must cover the layouts operators actually build: the maintainer's regression
# fixture puts the checkout at ``<base>/elsewhere/d0/d1/d2/d3/checkout``, six
# levels below ``<base>``, and a checkout the search cannot reach leaves only
# the weaker topology fallback to decide ownership — which is how a
# ``--separate-git-dir`` storage dir got crowned.
_CHECKOUT_LISTING_DEPTH = 6


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
    elsewhere. One rule decides, and it never looks at a directory's name or at
    stray files — both describe a ``--separate-git-dir`` *storage* dir exactly
    as well as a real checkout, since git even reports storage as a working
    tree (its basename is ``.git``):

        the working tree is the directory whose ``.git`` **gitfile** claims
        this metadata — the workspace's own ancestry first, then one flat
        listing level around the metadata parent (a plain repository's parent
        dir holds the metadata itself as ``.git``).

    Only a gitfile pointing *straight* at the common dir counts; one naming
    ``<common>/worktrees/<name>`` belongs to a linked worktree and says nothing
    about the primary. ``core.worktree`` is accepted when git records it.

    When no claim exists the layout is genuinely ambiguous — a
    ``--separate-git-dir`` checkout outside the listed levels leaves its
    metadata as the only ``.git`` in its parent — and the answer is ``None``:
    the caller then keeps the workspace's own toplevel, which nests at worst.
    It never starts a session inside a Git metadata directory, and a bounded
    search that found no claim never turns metadata shape, a Git-inferred
    toplevel, an empty index or a colliding filename into positive ownership.
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

    owner = _checkout_claiming_common(common_path, common_resolved)
    if owner is not None:
        return owner

    # No gitfile claimed the metadata: decide from recorded topology only.
    candidate = _topology_candidate(common_path, common_resolved)
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


def _iter_dirs_bounded(start: Path, max_depth: int) -> Iterator[Path]:
    """Every directory under *start*, descending at most *max_depth* levels.

    Descent is bounded before a directory is entered, so an unrelated deep
    subtree is never walked to completion just to be filtered out afterwards —
    the failure mode of ``rglob``. ``.git`` is never entered: git metadata holds
    no checkouts. ``start`` itself is not yielded.

    Depth is counted in *levels below* ``start``, so ``max_depth=4`` yields the
    four levels ``a``, ``a/b``, ``a/b/c`` and ``a/b/c/d``. The child is added to
    the walk while ``depth + 1 < max_depth`` — a child at the last level is
    yielded but not descended into, which is exactly the cap.
    """
    try:
        if not start.is_dir():
            return
    except OSError:
        return
    stack: list[tuple[Path, int]] = [(start, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError:
            continue
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
            yield child
            # Descend from a child below the last permitted level. Levels are
            # counted from ``start``: ``depth`` is the level of ``current``, so a
            # child sits at ``depth + 1`` and may be entered only while that is
            # still below ``max_depth``.
            if depth + 1 < max_depth:
                children.append((child, depth + 1))
        stack.extend(reversed(children))


def _looks_plain(common_resolved: Path) -> bool:
    """The local shape a plain repository's ``.git`` (or its storage twin) has.

    Decidable without a claim, so it gates the claim search: metadata that is a
    ``.git`` *directory* by basename, carries no ``commondir`` (that marker means
    a linked worktree's admin dir), and lives nowhere under a submodule's
    ``.git/modules``. Everything else is a shape the search cannot resolve by
    gitfile, and continuing would only spend directory listings.
    """
    try:
        if common_resolved.name != ".git":
            return False
        if (common_resolved / "commondir").is_file():
            return False
    except OSError:
        return False
    return not (
        common_resolved.parent.name == "modules"
        and common_resolved.parent.parent.name == ".git"
    )


def _checkout_claiming_common(common_path: Path, common_resolved: Path) -> Path | None:
    """A checkout whose gitfile claims the metadata, found by directory listing.

    The metadata's parent is tried first — for a plain repository it *is* the
    working tree (its ``.git`` is the metadata dir, not a gitfile) — then the
    ancestors of the metadata parent, each listed with bounded descent so a
    checkout sitting beside the metadata (``<base>/storage/.git`` with the
    checkout in ``<base>`` or a couple of levels below it) is still found.

    Only a *claim* decides: a gitfile pointing straight at this metadata, or a
    ``.git`` directory that **is** this metadata *and* whose recorded topology
    says the candidate is its working tree. Without a claim the ownership is
    unknown and the answer is ``None``: a search that finds nothing must not
    promote metadata shape, a Git-inferred toplevel, an empty index or a
    colliding filename into positive evidence (see ``_topology_candidate``).
    """
    parent = common_path.parent
    seen: set[str] = set()

    # Every descendant of the metadata parent is listed once, with descent
    # capped at ``_CHECKOUT_LISTING_DEPTH`` *from that level*: a checkout can sit
    # beside the metadata, or a few directory levels down in the operator's own
    # layout (``<base>/elsewhere/deep/<checkout>``). Descent is bounded before a
    # directory is entered, so a large sibling subtree is never walked to
    # completion after a failed match — the ``rglob`` failure mode.
    #
    # When the local search finds nothing, the *ancestor* levels are listed too:
    # the primary can sit above the metadata's parent (``<base>/storage/.git``
    # with the checkout at ``<base>/elsewhere/deep/...``). Those levels are
    # finite, since ``$HOME`` and ``/`` are excluded.
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

    for root in roots:
        level: list[Path] = []
        for candidate in _iter_dirs_bounded(root, _CHECKOUT_LISTING_DEPTH):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            level.append(candidate)
        for candidate in level:
            owner = _gitfile_owner(candidate, common_resolved)
            if owner is not None:
                return owner
    return None


def _topology_candidate(common_path: Path, common_resolved: Path) -> Path | None:
    """The working tree git's own records tie to this metadata, or ``None``.

    Reached only after no gitfile claimed the metadata. Three recorded facts can
    still name a working tree, in this order:

    * ``core.worktree``, which names it outright (a ``--separate-git-dir``
      checkout may carry it);
    * ``<tree>/.git`` *is* this metadata and the parent carries the repository's
      tracked tree — the plain layout, where the parent is the working tree.

    A linked admin dir (``commondir``) is never a primary. Nothing else is a
    candidate: basename alone, a Git-inferred toplevel, ``git worktree list``,
    an empty index beside a ``worktrees/`` admin area, or a single file sharing
    a tracked name are all true of a ``--separate-git-dir`` storage dir too, so
    they are refused here — unknown ownership stays unknown and the caller keeps
    the checkout it has. That is the deliberate failure mode: nesting at worst,
    never creating a session inside git's bookkeeping.
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
    # A plain ``<tree>/.git`` claims its parent when the parent carries the
    # repository's tracked tree.
    #
    # Shape alone proves nothing: a ``--separate-git-dir`` storage dir has the
    # identical ``.git``-beside-parent layout, ``git rev-parse --show-toplevel``
    # answers the storage dir itself, and ``git worktree list`` names it as the
    # primary worktree. All three are the false signal this resolver exists to
    # reject, so none of them is used.
    #
    # Nor is *content*: a work tree owns its repository whether or not its files
    # match the index, so nothing is compared against the committed bytes. A
    # ``diff-files``-style test is wrong in both directions — it rejects an
    # ordinary primary whose every tracked file was edited, and (because it is
    # stat-dependent and never refreshes the index) it can misread a checkout
    # whose files merely got a new mtime. What a storage dir cannot reproduce is
    # the *tracked tree itself*: a checkout holds every tracked path, while
    # storage holds git's bookkeeping plus at most an operator note that reuses a
    # tracked name.
    if common_resolved.name == ".git" and _plain_git_dir_owner(parent, common_resolved):
        if _tracks_paths_in_parent(common_resolved, parent):
            return parent
        # No tracked set at all: a fresh ``init`` or an empty commit, whose
        # checkout legitimately holds nothing but ``.git``. Resolving the parent
        # keeps that ordinary positive working (a repository that simply tracks
        # nothing yet), and it is the pre-existing behaviour for this layout.
        # A *failed* index query is unknown rather than empty and claims nothing.
        tops = _tracked_tops(common_resolved)
        if tops is not None and not tops:
            return parent
        return None
    return None


def _tracks_paths_in_parent(common_resolved: Path, parent: Path) -> bool:
    """True when the parent's tree is the repository's work tree, not a copy.

    Every tracked path must exist under the parent. That single test covers both
    layouts without consulting, or trusting, the committed content:

    * a **checkout** — clean, dirty, staged, or with every tracked file edited —
      holds its whole tracked tree;
    * a **``--separate-git-dir`` storage dir** holds git's bookkeeping, plus at
      most an operator note that happens to reuse a tracked basename. It cannot
      hold the repository's tree, because git never puts one there.

    Full paths, not top-level names: a note colliding with a top-level
    ``tracked.txt`` must not speak for a tracked ``src/tracked.txt``, and a
    partial copy of the tree is a copy.

    Deliberately *not* used, both because a work tree owns its repository
    regardless of uncommitted state:

    * ``git status`` — a clean requirement rejects every dirty primary;
    * ``git diff-files`` — a divergence ratio rejects a primary whose every
      tracked file was edited, and, being stat-driven, also depends on whether
      the index happened to be refreshed (a same-content rewrite or a bare
      ``touch`` either counts as a difference or does not).

    ``False`` for an empty index, an unreadable index or a failed query: an
    *unknown* answer is never ownership.
    """
    paths = _tracked_paths(common_resolved)
    if not paths:
        return False
    for rel in paths:
        try:
            if not (parent / rel).exists():
                return False
        except OSError:
            return False
    return True


def _plain_git_dir_owner(parent: Path, common_resolved: Path) -> bool:
    """True when *parent*'s own ``.git`` is this metadata directory.

    The plain layout, decided without asking git to interpret a gitfile: the
    parent holds a ``.git`` *directory* that is the metadata itself. A storage
    dir's parent holds no ``.git`` at all.
    """
    try:
        parent_git = parent / ".git"
        if not parent_git.is_dir():
            return False
        return parent_git.samefile(common_resolved)
    except OSError:
        return False


def _tracked_paths(common_resolved: Path) -> set[str] | None:
    """Every tracked path in the index; ``None`` when the query failed.

    ``None`` and ``set()`` are deliberately distinct: a *successful* empty index
    says "this repository tracks nothing", while a failed, timed-out or
    unreadable query says nothing at all. Neither is ownership, but only the
    failure is a reason to distrust an otherwise-empty answer.

    Full paths, not top-level entries: a subtree rule ("every top-level name is
    present somewhere below") would accept a storage dir holding one colliding
    note for a repository whose only colliding name is top-level, which is
    exactly the partial-copy layout this resolver must refuse.
    """
    try:
        result = subprocess.run(
            ["git", "--git-dir", str(common_resolved), "ls-files", "-z"],
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
    return {p for p in result.stdout.split("\0") if p}


def _tracked_tops(common_resolved: Path) -> set[str] | None:
    """The first path component of every tracked entry; ``None`` when unknown."""
    paths = _tracked_paths(common_resolved)
    if paths is None:
        return None
    return {p.split("/", 1)[0] for p in paths}


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
