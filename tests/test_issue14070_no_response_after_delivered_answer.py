"""Regression tests for card #14070: a delivered answer must not collect a
"No response from provider" card.

Production shape (four recorded turns on 2026-10-03, sessions ``c21c340635c4``
and ``b41294f0336b``): the provider streamed the complete answer to the client
through ``stream_delta_callback``, the agent then returned a ``partial`` result
whose ``messages`` carried only the pre-turn history, and the settlement
appended an error row that told the user "the provider returned no content and
no error" directly under the answer they had just read.

The tests below pin both halves of the contract:

* a turn that streamed its answer settles as a normal assistant reply — no
  ``apperror``, no ``_error`` row, and the streamed text is what the saved
  transcript keeps;
* a turn with no delivered text still reports ``no_response``, because that is
  exactly the silent-failure shape the card exists for.
"""

from __future__ import annotations

import queue
import sys
import types
from unittest import mock

import pytest

import api.config as config
import api.models as models
import api.streaming as streaming
from api.models import Session

ANSWER = (
    "Il 404 è la firma del DB pacman stale: la versione nel DB locale non "
    "esiste più sui mirror.\n\n"
    "Il DB si è aggiornato: ora `extra` dice **158.0b3-1**, e quel file c'è.\n\n"
    "Quindi basta rilanciare, senza sync aggiuntivo:\n"
    "```\nsudo bash /tmp/asus-install-firefox-devedition.sh\n```\n\n"
    "`yay` non serve (è in `extra`, non AUR)."
)


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    index_file = session_dir / "_index.json"
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index_file)
    models.SESSIONS.clear()
    yield
    models.SESSIONS.clear()


@pytest.fixture(autouse=True)
def _isolate_stream_state():
    config.STREAMS.clear()
    config.CANCEL_FLAGS.clear()
    config.AGENT_INSTANCES.clear()
    config.STREAM_PARTIAL_TEXT.clear()
    if hasattr(config, "STREAM_REASONING_TEXT"):
        config.STREAM_REASONING_TEXT.clear()
    if hasattr(config, "STREAM_LIVE_TOOL_CALLS"):
        config.STREAM_LIVE_TOOL_CALLS.clear()
    yield
    config.STREAMS.clear()
    config.CANCEL_FLAGS.clear()
    config.AGENT_INSTANCES.clear()
    config.STREAM_PARTIAL_TEXT.clear()
    if hasattr(config, "STREAM_REASONING_TEXT"):
        config.STREAM_REASONING_TEXT.clear()
    if hasattr(config, "STREAM_LIVE_TOOL_CALLS"):
        config.STREAM_LIVE_TOOL_CALLS.clear()


@pytest.fixture(autouse=True)
def _isolate_agent_locks():
    config.SESSION_AGENT_LOCKS.clear()
    yield
    config.SESSION_AGENT_LOCKS.clear()


@pytest.fixture(autouse=True)
def _neutralize_credential_self_heal(monkeypatch):
    """Keep the 401 self-heal path a no-op: these turns never touch auth."""
    monkeypatch.setattr(streaming, "_attempt_credential_self_heal", lambda *a, **k: None)
    yield


@pytest.fixture(autouse=True)
def _mock_hermes_modules():
    fake_runtime_module = types.ModuleType("hermes_cli.runtime_provider")
    fake_runtime_module.resolve_runtime_provider = lambda requested=None, **_kw: {
        "provider": requested or "test-provider",
        "api_key": "synthetic-key",
        "base_url": None,
    }
    fake_hermes_cli = types.ModuleType("hermes_cli")
    fake_hermes_cli.runtime_provider = fake_runtime_module
    fake_hermes_state = types.ModuleType("hermes_state")
    fake_hermes_state.SessionDB = mock.Mock(return_value=None)

    injected = {
        "hermes_cli": fake_hermes_cli,
        "hermes_cli.runtime_provider": fake_runtime_module,
        "hermes_state": fake_hermes_state,
    }
    missing = object()
    saved = {k: sys.modules.get(k, missing) for k in injected}
    sys.modules.update(injected)
    yield
    for name, prev in saved.items():
        if prev is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


class MockAgent:
    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.reasoning_callback = kwargs.get("reasoning_callback")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.context_compressor = None
        self._last_error = None
        self.ephemeral_system_prompt = None

    def run_conversation(self, **kwargs):
        raise NotImplementedError

    def interrupt(self, _message):
        pass


def _prepare_session(session_id: str, stream_id: str, *, pending_user_message: str):
    session = Session(session_id=session_id, title="Test Session")
    session.messages = []
    session.context_messages = []
    session.pending_user_message = pending_user_message
    session.pending_attachments = ["attachment.txt"]
    session.pending_started_at = 1234567890.0
    session.pending_user_source = "cli"
    session.active_stream_id = stream_id
    session.save()
    models.SESSIONS[session_id] = session
    return session


def _seed_prior_turn(session, *, prior_user: str, prior_assistant: str):
    session.messages = [
        {"role": "user", "content": prior_user, "timestamp": 1},
        {"role": "assistant", "content": prior_assistant, "timestamp": 2},
    ]
    session.context_messages = [
        {"role": "user", "content": prior_user},
        {"role": "assistant", "content": prior_assistant},
    ]
    session.save()


def _run_stream(monkeypatch, session, stream_id, agent_cls, *, workspace):
    fake_queue = queue.Queue()
    streaming.STREAMS[stream_id] = fake_queue
    config.STREAM_PARTIAL_TEXT[stream_id] = ""

    with mock.patch.object(streaming, "get_session", return_value=session), \
         mock.patch.object(streaming, "_get_ai_agent", return_value=agent_cls), \
         mock.patch.object(streaming, "resolve_model_provider", return_value=("test-model", "test-provider", None)), \
         mock.patch("api.config.get_config", return_value={}), \
         mock.patch("api.config._resolve_cli_toolsets", return_value=[]):
        streaming._run_agent_streaming(
            session_id=session.session_id,
            msg_text=session.pending_user_message,
            model="test-model",
            workspace=workspace,
            stream_id=stream_id,
        )

    return fake_queue


def _events(fake_queue):
    return [(item[0], item[1]) for item in list(fake_queue.queue)]


PROMPT = "Dammi il comando per installare firefox-developer-edition su Asus"


def test_partial_result_after_streamed_answer_settles_as_answer(tmp_path, monkeypatch):
    """The #14070 shape: full answer streamed, result payload lacks the row."""

    session = _prepare_session("delivered_answer_partial", "stream_delivered_answer_partial", pending_user_message=PROMPT)

    class DeliveredThenPartialAgent(MockAgent):
        def run_conversation(self, **kwargs):
            if self.stream_delta_callback is not None:
                self.stream_delta_callback(ANSWER)
            return {
                "status": "partial",
                "partial": True,
                "messages": list(kwargs.get("conversation_history") or []),
                "error": "",
            }

    fake_queue = _run_stream(
        monkeypatch,
        session,
        "stream_delivered_answer_partial",
        DeliveredThenPartialAgent,
        workspace=str(tmp_path),
    )
    saved = Session.load("delivered_answer_partial")
    assert saved is not None

    events = _events(fake_queue)
    apperrors = [data for event, data in events if event == "apperror"]
    assert not apperrors, (
        "a turn that delivered its answer must not report a provider failure: "
        f"{apperrors}"
    )
    assert any(event == "done" for event, _ in events)
    assert not any(msg.get("_error") for msg in saved.messages)

    answers = [msg for msg in saved.messages if msg.get("role") == "assistant"]
    assert answers, "the streamed answer must be saved as an assistant row"
    assert answers[-1]["content"] == ANSWER
    assert not answers[-1].get("_partial")

    # The settlement must not re-materialise the prompt as a second user bubble.
    assert [msg["content"] for msg in saved.messages if msg.get("role") == "user"] == [PROMPT]

    assert saved.active_stream_id is None
    assert saved.pending_user_message is None


def test_partial_result_after_streamed_answer_keeps_prior_turns_intact(tmp_path, monkeypatch):
    """Seeded multi-turn session: the new answer joins the history, no card."""

    session = _prepare_session("delivered_answer_seeded", "stream_delivered_answer_seeded", pending_user_message=PROMPT)
    _seed_prior_turn(session, prior_user="Earlier question", prior_assistant="Earlier answer")

    class DeliveredThenPartialAgent(MockAgent):
        def run_conversation(self, **kwargs):
            if self.stream_delta_callback is not None:
                self.stream_delta_callback(ANSWER)
            return {
                "status": "partial",
                "partial": True,
                "messages": list(kwargs.get("conversation_history") or []),
                "error": "",
            }

    fake_queue = _run_stream(
        monkeypatch,
        session,
        "stream_delivered_answer_seeded",
        DeliveredThenPartialAgent,
        workspace=str(tmp_path),
    )
    saved = Session.load("delivered_answer_seeded")
    assert saved is not None

    events = _events(fake_queue)
    assert not [data for event, data in events if event == "apperror"]
    assert any(event == "done" for event, _ in events)
    assert any(msg.get("role") == "assistant" and msg.get("content") == "Earlier answer" for msg in saved.messages)
    assert not any(msg.get("_error") for msg in saved.messages)
    assert [msg["content"] for msg in saved.messages if msg.get("role") == "assistant"][-1] == ANSWER


def test_partial_result_without_streamed_text_still_reports_no_response(tmp_path, monkeypatch):
    """Zero delivered tokens is a genuine silent failure: the card must stay."""

    session = _prepare_session("silent_partial", "stream_silent_partial", pending_user_message="Please respond")

    class SilentPartialAgent(MockAgent):
        def run_conversation(self, **kwargs):
            return {
                "status": "partial",
                "partial": True,
                "messages": list(kwargs.get("conversation_history") or []),
                "error": "",
            }

    fake_queue = _run_stream(monkeypatch, session, "stream_silent_partial", SilentPartialAgent, workspace=str(tmp_path))
    saved = Session.load("silent_partial")
    assert saved is not None

    events = _events(fake_queue)
    apperrors = [data for event, data in events if event == "apperror"]
    assert apperrors, "expected apperror when no assistant text was delivered"
    assert apperrors[-1]["type"] == "no_response"
    assert not any(event == "done" for event, _ in events)
    assert saved.messages[-1]["_error"] is True


def test_failed_result_after_streamed_answer_still_reports_no_response(tmp_path, monkeypatch):
    """A hard `failed` result keeps its error card even when text was streamed."""

    session = _prepare_session("delivered_answer_failed", "stream_delivered_answer_failed", pending_user_message="Please finish")

    class DeliveredThenFailedAgent(MockAgent):
        def run_conversation(self, **kwargs):
            if self.stream_delta_callback is not None:
                self.stream_delta_callback("Half an answer")
            return {
                "status": "failed",
                "messages": list(kwargs.get("conversation_history") or []),
                "error": "",
            }

    fake_queue = _run_stream(
        monkeypatch,
        session,
        "stream_delivered_answer_failed",
        DeliveredThenFailedAgent,
        workspace=str(tmp_path),
    )
    saved = Session.load("delivered_answer_failed")
    assert saved is not None

    events = _events(fake_queue)
    apperrors = [data for event, data in events if event == "apperror"]
    assert apperrors, "expected apperror for a hard failed result"
    assert apperrors[-1]["type"] == "no_response"
    assert saved.messages[-1]["_error"] is True
