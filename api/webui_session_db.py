"""Dormant JSON-backed SessionDB-shaped adapter for WebUI sessions.

This module intentionally does not replace existing WebUI runtime call sites.
It provides a small compatibility surface over the current JSON sidecars so the
unified SessionDB contract can be tested without changing persistence behavior.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

import api.models as models


_METADATA_FIELDS = frozenset(
    {
        "title",
        "workspace",
        "model",
        "model_provider",
        "created_at",
        "updated_at",
        "pinned",
        "archived",
        "project_id",
        "profile",
        "input_tokens",
        "output_tokens",
        "estimated_cost",
        "cache_read_tokens",
        "cache_write_tokens",
        "personality",
        "active_stream_id",
        "pending_user_message",
        "pending_attachments",
        "pending_started_at",
        "compression_anchor_visible_idx",
        "compression_anchor_message_key",
        "compression_anchor_summary",
        "pre_compression_snapshot",
        "context_engine",
        "compression_anchor_engine",
        "compression_anchor_mode",
        "compression_anchor_details",
        "context_engine_state",
        "context_length",
        "threshold_tokens",
        "last_prompt_tokens",
        "compression_recovery",
        "recommended_recovery_action",
        "compression_recovery_source_session_id",
        "compression_recovery_action",
        "truncation_watermark",
        "truncation_boundary",
        "transcript_generation",
        "transcript_generation_baseline",
        "gateway_routing",
        "gateway_routing_history",
        "llm_title_generated",
        "manual_title",
        "parent_session_id",
        "done_at",
        "worktree_path",
        "worktree_branch",
        "worktree_repo_root",
        "worktree_created_at",
        "is_cli_session",
        "source_tag",
        "raw_source",
        "session_source",
        "source_label",
        "read_only",
        "enabled_toolsets",
        "composer_draft",
    }
)

_UNSAFE_FIELDS = frozenset({"session_id", "messages", "tool_calls", "message_count"})


class WebUIJsonSessionDB:
    """Small SessionDB-like facade over existing WebUI session JSON files."""

    def __init__(self, session_dir: Path | str | None = None):
        self._session_dir = Path(session_dir).expanduser().resolve() if session_dir else None

    @property
    def session_dir(self) -> Path:
        return self._session_dir or models.SESSION_DIR

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return compact metadata for persisted WebUI JSON sessions.

        Reads are direct JSON loads and never call ``Session.load()``, because
        that path may self-heal and write repaired transcripts.
        """
        rows: list[dict[str, Any]] = []
        if not self.session_dir.exists():
            return rows
        for path in self.session_dir.glob("*.json"):
            if path.name.startswith("_"):
                continue
            data = self._read_metadata_row_source(path)
            if not isinstance(data, dict):
                continue
            sid = str(data.get("session_id") or path.stem)
            if not models.is_safe_session_id(sid):
                continue
            rows.append(self._metadata_row(sid, data))
        rows.sort(key=lambda row: (bool(row.get("pinned")), self._sort_timestamp(row)), reverse=True)
        return rows

    def read_session(self, sid: str) -> dict[str, Any] | None:
        """Return the full JSON session payload for ``sid`` without mutation."""
        path = self._path_for_sid(sid)
        if path is None or not path.exists():
            return None
        data = self._read_path(path)
        if not isinstance(data, dict):
            return None
        return copy.deepcopy(data)

    def update_metadata(self, sid: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Persist allowlisted metadata fields while preserving messages.

        This dormant adapter method is for migration experiments and tests only.
        Runtime wiring must add Session lock/cache/index parity before using it
        from live WebUI routes.
        """
        if not isinstance(fields, dict):
            raise TypeError("fields must be a dict")
        unsafe = sorted((set(fields) & _UNSAFE_FIELDS) | (set(fields) - _METADATA_FIELDS))
        if unsafe:
            raise ValueError(f"Unsafe session metadata fields: {', '.join(unsafe)}")

        path = self._existing_path_for_sid(sid)
        data = self._read_writable_session(path)
        data.update(copy.deepcopy(fields))
        data["message_count"] = len(data["messages"])
        self._atomic_write(path, data)
        return self._metadata_row(str(data.get("session_id") or sid), data)

    def archive(self, sid: str, archived: bool = True) -> dict[str, Any]:
        """Set the archived metadata flag without touching transcript messages."""
        return self.update_metadata(sid, {"archived": bool(archived)})

    def write_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Write a full session payload for tests and migration experiments."""
        if not isinstance(session, dict):
            raise TypeError("session must be a dict")
        sid = session.get("session_id")
        path = self._path_for_sid(sid)
        if path is None:
            raise ValueError(f"Unsafe session_id {sid!r}")
        messages = session.get("messages")
        if not isinstance(messages, list):
            raise ValueError("session payload must include a messages list")
        payload = copy.deepcopy(session)
        payload["message_count"] = len(messages)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, payload)
        return copy.deepcopy(payload)

    def _path_for_sid(self, sid: str) -> Path | None:
        if not models.is_safe_session_id(sid):
            return None
        return self.session_dir / f"{sid}.json"

    def _existing_path_for_sid(self, sid: str) -> Path:
        path = self._path_for_sid(sid)
        if path is None:
            raise ValueError(f"Unsafe session_id {sid!r}")
        if not path.exists():
            raise KeyError(sid)
        return path

    def _read_writable_session(self, path: Path) -> dict[str, Any]:
        data = self._read_path(path)
        if not isinstance(data, dict):
            raise ValueError(f"Malformed session JSON: {path.name}")
        sid = data.get("session_id")
        if not models.is_safe_session_id(sid):
            raise ValueError(f"Unsafe session_id {sid!r}")
        if not isinstance(data.get("messages"), list):
            raise ValueError(f"Refusing to write metadata-only session stub: {sid!r}")
        return data

    @staticmethod
    def _read_path(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    #: Bytes scanned from the head of a sidecar to recover the metadata prefix.
    #: Metadata keys are serialized before ``messages``, so they all live in the
    #: first few KB: measured over 637 real sidecars the top-level ``"messages"``
    #: key started at p50 2.6 KB, p99 32 KB, max 43 KB.
    _METADATA_HEAD_BYTES = 64 * 1024

    @classmethod
    def _read_metadata_prefix(cls, path: Path) -> dict[str, Any] | None:
        """Parse only the metadata that precedes ``messages`` in a sidecar.

        ``list_sessions()`` needs ~20 scalar fields out of each sidecar, but a
        full ``json.loads`` also decodes every message: on a store of 639 files
        (1.18 GB) that costs ~16 s per refresh, spent entirely on bytes the
        sidebar never reads. Metadata is serialized before ``messages``, so the
        head is cut at that key and closed with ``}`` -- a few KB instead of
        megabytes.

        Returns ``None`` when the cut cannot be found (key beyond the window, or
        a non-dict document), so the caller falls back to a full parse.
        """
        try:
            with open(path, "rb") as fh:
                head = fh.read(cls._METADATA_HEAD_BYTES)
        except OSError:
            return None
        cut = head.find(b'\n  "messages":')
        if cut < 0:
            cut = head.find(b'"messages":')
        if cut < 0:
            return None
        prefix = head[:cut].rstrip()
        # Trim a trailing separator so the object closes cleanly.
        while prefix and prefix[-1:] in (b",", b"\n", b" ", b"\t", b"\r"):
            prefix = prefix[:-1]
        prefix += b"\n}"
        try:
            data = json.loads(prefix.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @classmethod
    def _read_metadata_row_source(cls, path: Path) -> dict[str, Any] | None:
        """Metadata-only read used by ``list_sessions``. See ``_read_metadata_prefix``."""
        prefix = cls._read_metadata_prefix(path)
        if prefix is not None:
            # ``_metadata_row`` derives ``message_count`` from ``len(messages)``
            # when the field is absent. The head read never sees ``messages``,
            # so a sidecar that omits ``message_count`` must be parsed whole --
            # otherwise the sidebar would report 0 messages for it.
            if isinstance(prefix.get("message_count"), int):
                return prefix
        return cls._read_path(path)

    @staticmethod
    def _metadata_row(sid: str, data: dict[str, Any]) -> dict[str, Any]:
        messages = data.get("messages")
        message_count = data.get("message_count")
        if not isinstance(message_count, int):
            message_count = len(messages) if isinstance(messages, list) else 0
        row = {field: copy.deepcopy(data.get(field)) for field in _METADATA_FIELDS if field in data}
        row["session_id"] = sid
        row["message_count"] = message_count
        row["last_message_at"] = data.get("last_message_at") or data.get("updated_at") or data.get("created_at")
        return row

    @staticmethod
    def _sort_timestamp(row: dict[str, Any]) -> float:
        for key in ("last_message_at", "updated_at", "created_at"):
            value = row.get(key)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _atomic_write(path: Path, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.current_thread().ident}")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def list_sessions() -> list[dict[str, Any]]:
    return WebUIJsonSessionDB().list_sessions()


def read_session(sid: str) -> dict[str, Any] | None:
    return WebUIJsonSessionDB().read_session(sid)


def update_metadata(sid: str, fields: dict[str, Any]) -> dict[str, Any]:
    return WebUIJsonSessionDB().update_metadata(sid, fields)


def archive(sid: str, archived: bool = True) -> dict[str, Any]:
    return WebUIJsonSessionDB().archive(sid, archived)


def write_session(session: dict[str, Any]) -> dict[str, Any]:
    return WebUIJsonSessionDB().write_session(session)
