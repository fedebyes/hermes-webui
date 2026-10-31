"""Vault secret prompts in the WebUI's in-process runtime (card #14058).

Hermes Agent never asks for a vault secret itself: ``browser_vault``'s unlock /
save_login / enter_code calls a per-thread callback that the *surface* installs
through ``agent.vault_backends.unlock``. The TUI installs them in
``hermes_cli.cli_chat_turn_mixin``, the Desktop app in
``tui_gateway/agent_callbacks.py::_wire_callbacks``; the WebUI installed none, so
``can_prompt_here()`` was False and the tool answered ``unlock_unavailable`` /
``prompt_unavailable``. ``api/vault_prompt.py`` is that missing surface.

These tests pin the three things that make the wiring safe:

  1. the request the browser sees carries metadata only (backend / site / hint),
  2. the answer reaches the waiting tool callback and nowhere else — not the SSE
     stream, not the journal, not the logs, not the HTTP response,
  3. a cancelled turn or an expired prompt declines instead of parking forever.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types

import pytest

from api import vault_prompt


# ── stub of the Agent-side callback module ───────────────────────────────────
class _StubUnlockModule(types.ModuleType):
    """Stands in for ``agent.vault_backends.unlock`` (not importable in tests)."""

    def __init__(self):
        super().__init__("agent.vault_backends.unlock")
        self.unlock = None
        self.code = None
        self.save_login = None
        self.session_id = None

    def set_unlock_prompt_callback(self, cb):
        self.unlock = cb

    def get_unlock_prompt_callback(self):
        return self.unlock

    def set_code_prompt_callback(self, cb):
        self.code = cb

    def get_code_prompt_callback(self):
        return self.code

    def set_save_login_prompt_callback(self, cb):
        self.save_login = cb

    def get_save_login_prompt_callback(self):
        return self.save_login

    def set_current_session_id(self, sid):
        self.session_id = sid

    def can_prompt_here(self):
        return self.unlock is not None


@pytest.fixture()
def unlock_mod(monkeypatch):
    stub = _StubUnlockModule()
    for name in ("agent", "agent.vault_backends"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "agent.vault_backends.unlock", stub)
    vault_prompt._pending.clear()
    yield stub
    vault_prompt._pending.clear()


@pytest.fixture()
def events():
    return []


def _install(unlock_mod, events, sid="sess-1", cancel_event=None):
    return vault_prompt.install_callbacks(sid, lambda name, data: events.append((name, data)), cancel_event)


def _ask_in_thread(cb, *args):
    box = {}

    def _run():
        box["value"] = cb(*args)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, box


def _wait_for(events, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in list(events):
            if predicate(event):
                return event
        time.sleep(0.01)
    return None


# ── 1. the request carries metadata only ────────────────────────────────────
def test_unlock_prompt_publishes_metadata_only(unlock_mod, events):
    _install(unlock_mod, events)
    thread, box = _ask_in_thread(unlock_mod.unlock, "onepassword", "1Password")
    published = _wait_for(events, lambda e: e[0] == "vault_prompt")
    assert published is not None, "the unlock request was never published to the client"
    _, payload = published
    assert payload["kind"] == "unlock"
    assert payload["backend"] == "onepassword"
    assert payload["display_name"] == "1Password"
    assert set(payload) == {"prompt_id", "kind", "requested_at", "timeout_seconds", "backend", "display_name"}

    assert vault_prompt.resolve("sess-1", payload["prompt_id"], "master-password")
    thread.join(2)
    assert box["value"] == "master-password"


def test_code_prompt_and_save_login_shapes(unlock_mod, events):
    _install(unlock_mod, events)

    thread, box = _ask_in_thread(unlock_mod.code, "github.com", "sent to +31 6 ****")
    published = _wait_for(events, lambda e: e[0] == "vault_prompt")
    assert published[1]["kind"] == "code"
    assert published[1]["hint"] == "sent to +31 6 ****"
    vault_prompt.resolve("sess-1", published[1]["prompt_id"], "123456")
    thread.join(2)
    assert box["value"] == "123456"

    thread, box = _ask_in_thread(unlock_mod.save_login, "https://example.com", "example.com")
    published = _wait_for(events, lambda e: e[0] == "vault_prompt" and e[1]["kind"] == "save_login")
    assert published[1]["origin"] == "https://example.com"
    vault_prompt.resolve("sess-1", published[1]["prompt_id"],
                         {"identifier": "me@example.com", "password": "pw"})
    thread.join(2)
    assert box["value"] == {"identifier": "me@example.com", "password": "pw"}


def test_save_login_declines_without_both_fields(unlock_mod, events):
    _install(unlock_mod, events)
    thread, box = _ask_in_thread(unlock_mod.save_login, "https://example.com", "example.com")
    published = _wait_for(events, lambda e: e[0] == "vault_prompt")
    vault_prompt.resolve("sess-1", published[1]["prompt_id"], {"identifier": "me@example.com"})
    thread.join(2)
    assert box["value"] is None


# ── 2. the answer never leaks ───────────────────────────────────────────────
def test_secret_is_not_emitted_or_logged(unlock_mod, events, caplog):
    caplog.set_level(logging.DEBUG)
    _install(unlock_mod, events)
    secret = "correct-horse-battery-staple"
    thread, box = _ask_in_thread(unlock_mod.unlock, "bitwarden", "Bitwarden")
    published = _wait_for(events, lambda e: e[0] == "vault_prompt")
    assert vault_prompt.resolve("sess-1", published[1]["prompt_id"], secret)
    thread.join(2)
    assert box["value"] == secret

    # Nothing the client (or a log reader) can see may contain the value.
    assert secret not in repr(events)
    assert secret not in caplog.text
    # And settling the prompt emits no further event of its own.
    assert [name for name, _ in events] == ["vault_prompt"]


def test_secret_is_not_in_the_journal_payload(unlock_mod, events):
    """The event is what ``put()`` journals; it must stay value-free."""
    _install(unlock_mod, events)
    thread, _ = _ask_in_thread(unlock_mod.unlock, "bitwarden", "Bitwarden")
    published = _wait_for(events, lambda e: e[0] == "vault_prompt")
    payload = published[1]
    assert "value" not in payload and "password" not in payload
    assert not any(isinstance(v, str) and v.startswith("correct") for v in payload.values())
    vault_prompt.clear_pending("sess-1")
    thread.join(2)


def test_respond_route_never_echoes_the_secret(unlock_mod, events):
    from api import routes

    _install(unlock_mod, events)
    thread, box = _ask_in_thread(unlock_mod.unlock, "bitwarden", "Bitwarden")
    published = _wait_for(events, lambda e: e[0] == "vault_prompt")

    captured = []
    original_j, original_bad = routes.j, routes.bad
    routes.j = lambda handler, payload, status=200, *a, **k: captured.append((status, payload))
    routes.bad = lambda handler, message, status=400: captured.append((status, {"error": message}))
    try:
        routes._handle_vault_prompt_respond(None, {
            "session_id": "sess-1", "prompt_id": published[1]["prompt_id"], "value": "s3cret-value",
        })
    finally:
        routes.j, routes.bad = original_j, original_bad

    thread.join(2)
    assert box["value"] == "s3cret-value"
    assert captured == [(200, {"ok": True})]
    assert "s3cret-value" not in repr(captured)


def test_respond_route_rejects_unknown_or_stale_prompt():
    from api import routes

    vault_prompt._pending.clear()
    captured = []
    original_j, original_bad = routes.j, routes.bad
    routes.j = lambda handler, payload, status=200, *a, **k: captured.append((status, payload))
    routes.bad = lambda handler, message, status=400: captured.append((status, {"error": message}))
    try:
        routes._handle_vault_prompt_respond(None, {"session_id": "nobody", "value": "x"})
        routes._handle_vault_prompt_respond(None, {"session_id": "", "value": "x"})
    finally:
        routes.j, routes.bad = original_j, original_bad
    assert captured[0] == (409, {"ok": False, "error": "no_pending_prompt"})
    assert captured[1][0] == 400


# ── 3. declining is always possible ─────────────────────────────────────────
def test_cancelled_turn_declines_instead_of_hanging(unlock_mod, events):
    cancel_event = threading.Event()
    _install(unlock_mod, events, cancel_event=cancel_event)
    thread, box = _ask_in_thread(unlock_mod.unlock, "bitwarden", "Bitwarden")
    _wait_for(events, lambda e: e[0] == "vault_prompt")
    cancel_event.set()
    thread.join(3)
    assert not thread.is_alive()
    assert box["value"] == ""
    assert vault_prompt.pending("sess-1") is None


def test_expired_prompt_declines_and_releases_the_slot(monkeypatch):
    monkeypatch.setattr(vault_prompt, "UNLOCK_TIMEOUT_SECONDS", 0.05)
    stub = _StubUnlockModule()
    for name in ("agent", "agent.vault_backends"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "agent.vault_backends.unlock", stub)
    vault_prompt.install_callbacks("sess-exp", lambda *a: None)
    started = time.monotonic()
    assert stub.unlock("bitwarden", "Bitwarden") == ""
    assert time.monotonic() - started < 3
    assert vault_prompt.pending("sess-exp") is None


def test_second_prompt_is_refused_while_one_is_open(unlock_mod, events):
    _install(unlock_mod, events)
    thread, box = _ask_in_thread(unlock_mod.unlock, "bitwarden", "Bitwarden")
    _wait_for(events, lambda e: e[0] == "vault_prompt")
    assert unlock_mod.code("github.com", "sms") == ""
    vault_prompt.clear_pending("sess-1")
    thread.join(2)


def test_install_binds_session_and_uninstall_restores_empty_slots(unlock_mod, events):
    uninstall = _install(unlock_mod, events, sid="sess-9")
    assert unlock_mod.session_id == "sess-9"
    assert unlock_mod.unlock is not None and unlock_mod.code is not None and unlock_mod.save_login is not None
    uninstall()
    assert unlock_mod.unlock is None and unlock_mod.code is None and unlock_mod.save_login is None
    assert vault_prompt.pending("sess-9") is None


def test_missing_agent_module_is_a_noop(monkeypatch):
    """An older Agent build without the vault callbacks must not break a turn."""
    monkeypatch.setitem(sys.modules, "agent.vault_backends.unlock", None)
    uninstall = vault_prompt.install_callbacks("sess-x", lambda *a: None)
    assert callable(uninstall)
    uninstall()


# ── 4. replay never resurrects a settled secret card ────────────────────────
def test_vault_prompt_row_is_settled_by_the_run_terminal_row():
    from api import run_journal

    prompt = {"event": "vault_prompt", "seq": 4, "run_id": "run-1"}
    done = {"event": "done", "seq": 9, "run_id": "run-1", "terminal": True}
    assert run_journal.journal_replay_prompt_visible(prompt, set()) is True, (
        "a reconnecting tab must still be able to answer a live secret prompt"
    )
    settled = run_journal.terminal_settled_prompt_events([prompt, done])
    assert run_journal.journal_replay_prompt_visible(prompt, set(), settled) is False
