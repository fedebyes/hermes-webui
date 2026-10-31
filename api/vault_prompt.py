"""Masked vault prompts for the WebUI's in-process chat runtime.

Hermes Agent never asks for a vault secret itself: a tool (``browser_vault``'s
unlock / save_login / enter_code) calls one of three per-thread callbacks that the
*surface* installs through ``agent.vault_backends.unlock``.  The TUI installs them in
``cli.py`` / ``hermes_cli.cli_chat_turn_mixin``, the Desktop app in
``tui_gateway/agent_callbacks.py::_wire_callbacks`` (server request
``vault.unlock_prompt`` / ``vault.save_login`` / ``vault.code``).  The WebUI's
in-process runtime installed none, so ``can_prompt_here()`` was False and the tool
answered ``unlock_unavailable`` / ``prompt_unavailable``: a password or 2FA code could
only be typed from the TUI or Telegram.

This module is the WebUI surface for that contract.  A request is published on the
chat stream as a ``vault_prompt`` event carrying *metadata only* (backend name, site,
the hint the site shows — never a secret), the tool thread parks on ``await_answer``,
and the browser POSTs the value back to ``/api/vault/prompt/respond``.  The value goes
straight to the waiting callback: it is never journalled, never put on the stream,
never written to the transcript and never passed to a logger.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

UNLOCK_TIMEOUT_SECONDS = 120
CODE_TIMEOUT_SECONDS = 180
SAVE_LOGIN_TIMEOUT_SECONDS = 180

#: Prompt kinds and the payload fields each one may carry to the browser.  A kind not
#: listed here is never published (fail closed).
_KIND_FIELDS = {
    "unlock": ("backend", "display_name"),
    "code": ("site", "hint"),
    "save_login": ("origin", "site"),
}

_lock = threading.Lock()
_pending: dict[str, "VaultPromptEntry"] = {}


class VaultPromptEntry:
    """One outstanding secret request for a session."""

    __slots__ = ("prompt_id", "kind", "data", "event", "result", "resolved", "cancelled")

    def __init__(self, kind: str, data: dict):
        self.prompt_id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.data = data
        self.event = threading.Event()
        self.result: object = None
        self.resolved = False
        self.cancelled = False


def _public(entry: VaultPromptEntry) -> dict:
    """The browser-visible projection: the allowlisted metadata plus lifecycle fields."""
    out = {"prompt_id": entry.prompt_id, "kind": entry.kind, "requested_at": entry.data.get("requested_at"),
           "timeout_seconds": entry.data.get("timeout_seconds")}
    for field in _KIND_FIELDS.get(entry.kind, ()):
        if field in entry.data:
            out[field] = entry.data[field]
    return out


def submit_pending(session_id: str, kind: str, data: dict) -> Optional[VaultPromptEntry]:
    """Register a request and return the entry the tool thread should wait on.

    Only one prompt per session is live: a second request for the same session while
    a prompt is open is refused (the tool reports a decline) rather than queued, so a
    confused agent cannot stack secret prompts the UI has to multiplex.
    """
    if kind not in _KIND_FIELDS:
        return None
    entry = VaultPromptEntry(kind, data)
    with _lock:
        if session_id in _pending:
            return None
        _pending[session_id] = entry
    return entry


def pending(session_id: str) -> Optional[dict]:
    """Metadata for the session's open request (used to re-attach after a reload)."""
    with _lock:
        entry = _pending.get(session_id)
    return _public(entry) if entry is not None else None


def pending_count(session_id: str) -> int:
    with _lock:
        return 1 if session_id in _pending else 0


def await_answer(entry: VaultPromptEntry, timeout: float,
                 cancel_event: Optional[threading.Event] = None) -> object:
    """Park until the browser answers, the turn is cancelled, or the timeout expires.

    Returns the answer (``str`` for unlock/code, ``dict`` for save_login) or ``None``
    when declined/expired.  The pending slot is always released.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    try:
        while True:
            if entry.event.wait(0.25):
                break
            if cancel_event is not None and cancel_event.is_set():
                entry.cancelled = True
                break
            if time.monotonic() >= deadline:
                entry.cancelled = True
                break
    finally:
        with _lock:
            current = _pending.get(entry.data.get("session_id", ""))
            if current is entry:
                _pending.pop(entry.data.get("session_id", ""), None)
    if not entry.resolved:
        return None
    value = entry.result
    entry.result = None
    return value


def resolve(session_id: str, prompt_id: str, value: object) -> bool:
    """Settle the session's open request.  ``value`` is consumed by the tool thread."""
    with _lock:
        entry = _pending.get(session_id)
        if entry is None or entry.resolved or entry.cancelled:
            return False
        if prompt_id and prompt_id != entry.prompt_id:
            return False
        entry.result = value
        entry.resolved = True
        _pending.pop(session_id, None)
    entry.event.set()
    return True


def clear_pending(session_id: str) -> int:
    """Cancel the session's open request (turn ended, client gone, session closed)."""
    with _lock:
        entry = _pending.pop(session_id, None)
    if entry is None:
        return 0
    entry.cancelled = True
    entry.event.set()
    return 1


def install_callbacks(session_id: str, put_event: Callable[[str, dict], None],
                      cancel_event: Optional[threading.Event] = None) -> Callable[[], None]:
    """Install this turn's vault prompts on the current (tool-dispatching) thread.

    Mirrors ``tui_gateway.agent_callbacks._wire_callbacks``: the same three
    ``set_*_prompt_callback`` slots and ``set_current_session_id``, so an unlock made
    on this turn is released when the WebUI session ends.  Returns an uninstaller that
    restores the previous (empty) slots and drops any request still parked.

    ``put_event`` receives only ``w._public(entry)``: the answer never travels back
    through it.
    """
    try:
        from agent.vault_backends.unlock import (
            can_prompt_here,  # noqa: F401  (imported for the contract, not called here)
            set_code_prompt_callback,
            set_current_session_id,
            set_save_login_prompt_callback,
            set_unlock_prompt_callback,
        )
    except ImportError:
        logger.debug("Vault prompt callbacks unavailable in this Agent build")
        return lambda: None

    def _ask(kind: str, fields: dict, timeout: float):
        data = dict(fields)
        data["session_id"] = session_id
        data["requested_at"] = time.time()
        data["timeout_seconds"] = timeout
        entry = submit_pending(session_id, kind, data)
        if entry is None:
            return None
        try:
            put_event("vault_prompt", _public(entry))
        except Exception:
            logger.debug("Failed to publish vault_prompt for session %s", session_id)
        return await_answer(entry, timeout, cancel_event)

    def _unlock(backend: str, display_name: str) -> str:
        answer = _ask("unlock", {"backend": backend, "display_name": display_name},
                      UNLOCK_TIMEOUT_SECONDS)
        return answer if isinstance(answer, str) else ""

    def _code(site: str, hint: str) -> str:
        answer = _ask("code", {"site": site, "hint": hint}, CODE_TIMEOUT_SECONDS)
        return answer if isinstance(answer, str) else ""

    def _save_login(origin: str, site: str) -> Optional[dict]:
        answer = _ask("save_login", {"origin": origin, "site": site}, SAVE_LOGIN_TIMEOUT_SECONDS)
        if not isinstance(answer, dict):
            return None
        if not answer.get("password") or not answer.get("identifier"):
            return None
        return {"identifier": str(answer.get("identifier")), "password": str(answer.get("password"))}

    set_unlock_prompt_callback(_unlock)
    set_code_prompt_callback(_code)
    set_save_login_prompt_callback(_save_login)
    try:
        set_current_session_id(session_id)
    except Exception:
        logger.debug("Could not bind the vault session owner for %s", session_id)

    def _uninstall() -> None:
        set_unlock_prompt_callback(None)
        set_code_prompt_callback(None)
        set_save_login_prompt_callback(None)
        clear_pending(session_id)

    return _uninstall
