"""The silent-failure retry must not replay work that already happened.

Drives the real `_run_agent_streaming` settlement path through the
`worker_scene` fixture (its external Agent is synthetic — no provider call, no
tool execution). The four shapes below are the ones the review pinned on the
first revision of this fix: a turn that ran a tool, a turn whose tool rows were
already persisted, a turn that exited on the tool-iteration limit, and a turn
that truly produced nothing.
"""

import threading

import pytest

from api import streaming
from tests.test_steer_worker_boundaries import worker_scene as worker_scene


@pytest.fixture(autouse=True)
def _enable_silent_retry(monkeypatch):
    """These worker cases exercise the retry, so opt in for the module.

    The retry is off by default (see ``_silent_retry_enabled``): the Agent has
    its own budgeted empty-response ladder and a WebUI whole-turn replay resets
    that budget, so an operator must enable it. Every test here is about the
    retry's behaviour once enabled; the default-off shape is pinned separately
    at the end of this module.
    """
    monkeypatch.setenv("HERMES_WEBUI_SILENT_FAILURE_RETRY", "1")


def _user_row():
    return {"role": "user", "content": "Do the task."}


def _materialized_user_row():
    """A user row as the worker persists it (turn bookkeeping attached)."""
    row = _user_row()
    row.update({"timestamp": 1.0, "_source": "webui", "attachments": []})
    return row


def _tool_rows():
    """A completed tool round: the assistant call plus its result row."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]


def _silent_result():
    """A turn that produced nothing: no reply, no error, no text."""
    return {"messages": [_user_row()]}


def _runs(scene):
    return scene.calls.count("run")


def _session_text(scene):
    return "\n".join(str(row.get("content") or "") for row in scene.session.messages)


def _error_rows(scene):
    return [row for row in scene.session.messages if row.get("_error")]


# ── Work already done is never replayed ──────────────────────────────────────


def test_tool_activity_blocks_the_silent_retry(worker_scene):
    """A turn that ran a tool already produced side effects."""
    scene = worker_scene
    scene.result = {"messages": [_user_row()] + _tool_rows()}

    scene.run()

    assert _runs(scene) == 1, "a tool-running turn must not be replayed"
    assert "No response from provider" in _session_text(scene)


class _LiveToolProgressAgent:
    """Synthetic agent that reports one streamed tool round, then stays silent.

    Mirrors production: a turn that ran a tool always emits tool progress, and
    the worker's per-turn guard (`_live_tool_calls`) reads that signal — not the
    transcript — because the two attempts' rows are identical once the first
    attempt's tool rows have been persisted.
    """

    instances = []
    result_messages = []
    emit_tool_progress = True

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.context_compressor = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_estimated_cost_usd = None
        self.reasoning_config = None
        self.ephemeral_system_prompt = None
        self._last_error = None
        self.pending = []
        self.pending_lock = threading.Lock()
        self.runs = 0
        _LiveToolProgressAgent.instances.append(self)

    def run_conversation(self, **_kwargs):
        self.runs += 1
        if self.emit_tool_progress and self.tool_progress_callback is not None:
            self.tool_progress_callback("tool.started", "shell", "", "{}")
        return {"messages": [dict(row) for row in self.result_messages]}

    def interrupt(self, _message):
        return None

    def steer(self, _text):
        return True

    def _drain_pending_steer(self):
        return None


def test_persisted_tool_rows_do_not_replay_the_turn(worker_scene, monkeypatch):
    """Tool work already in the transcript must not be executed a second time."""
    scene = worker_scene
    persisted = [_materialized_user_row()] + _tool_rows()
    scene.session.messages = [dict(row) for row in persisted]
    scene.session.context_messages = [dict(row) for row in persisted]
    _LiveToolProgressAgent.instances = []
    _LiveToolProgressAgent.result_messages = persisted
    monkeypatch.setattr(streaming, "_get_ai_agent", lambda: _LiveToolProgressAgent)

    scene.run()

    assert [agent.runs for agent in _LiveToolProgressAgent.instances] == [1], (
        "a turn that streamed tool progress must not be replayed"
    )
    tool_rows = [row for row in scene.session.messages if row.get("role") == "tool"]
    assert len(tool_rows) == 1, "the tool round must not be executed a second time"


def test_an_echoed_transcript_is_not_retried(worker_scene, monkeypatch):
    """A result identical to the pre-turn context is not a retryable cut."""
    scene = worker_scene
    echoed = [_user_row(), {"role": "assistant", "content": "Earlier answer"}]
    scene.session.messages = [dict(row) for row in echoed]
    scene.session.context_messages = [dict(row) for row in echoed]
    _LiveToolProgressAgent.instances = []
    _LiveToolProgressAgent.result_messages = echoed
    _LiveToolProgressAgent.emit_tool_progress = False
    monkeypatch.setattr(streaming, "_get_ai_agent", lambda: _LiveToolProgressAgent)
    try:
        scene.run()
    finally:
        _LiveToolProgressAgent.emit_tool_progress = True

    assert [agent.runs for agent in _LiveToolProgressAgent.instances] == [1], (
        "re-sending an identical context cannot change the result"
    )


def test_tool_limit_exit_keeps_its_own_card(worker_scene):
    """The tool-iteration limit owns the card; the silent branch must not shadow it."""
    scene = worker_scene
    scene.result = {
        "turn_exit_reason": "max_iterations_reached(30/30)",
        "messages": [_user_row()] + _tool_rows(),
    }

    scene.run()

    assert _runs(scene) == 1, "a tool-limit exit is not a silent failure"
    assert "Tool iteration limit reached" in _session_text(scene)


# ── The genuinely silent turn is retried, once ───────────────────────────────


def _silent_then(scene, second_result):
    """Return an on_run hook: silent first attempt, `second_result` afterwards."""

    def on_run():
        scene.result = _silent_result() if _runs(scene) == 1 else second_result

    return on_run


def test_truly_silent_turn_retries_once_and_recovers(worker_scene):
    scene = worker_scene
    scene.on_run = _silent_then(scene, None)  # second attempt: fixture success result

    scene.run()

    assert _runs(scene) == 2, "a silent turn earns exactly one retry"
    assert _error_rows(scene) == [], "the recovered turn must not carry an error card"
    assert "Finished." in _session_text(scene)


def test_silent_retry_does_not_route_through_the_credential_self_heal(
    worker_scene, monkeypatch,
):
    """A silent cut is not a credential problem: never re-read auth.json for it."""
    scene = worker_scene
    credential_calls = []

    def spy(*args, **kwargs):
        credential_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(streaming, "_attempt_credential_self_heal", spy)
    monkeypatch.setattr(streaming, "_attempt_model_alias_credential_self_heal", spy)
    scene.on_run = _silent_then(scene, None)

    scene.run()

    assert _runs(scene) == 2, (
        "the retry must proceed on the runtime already resolved for the turn, "
        "not wait on a credential refresh it does not need"
    )
    assert credential_calls == [], "the credential self-heal was consulted"


def test_a_failed_retry_keeps_the_no_response_card(worker_scene):
    """Both attempts silent: the card stays the pre-retry no-response wording."""
    scene = worker_scene
    scene.result = _silent_result()

    scene.run()

    assert _runs(scene) == 2
    errors = _error_rows(scene)
    assert len(errors) == 1
    assert "No response from provider" in errors[0]["content"]
    assert "Authentication failed" not in errors[0]["content"], (
        "a non-auth failure must not be labelled as an auth failure"
    )


def test_a_failed_retry_reports_the_retrys_own_error(worker_scene):
    """The second attempt's real cause owns the card, not the silent first one."""
    scene = worker_scene
    scene.on_run = _silent_then(scene, {
        "error": "429 rate limit exceeded",
        "messages": [_user_row()],
    })

    scene.run()

    assert _runs(scene) == 2
    errors = _error_rows(scene)
    assert len(errors) == 1
    assert "No response from provider" not in errors[0]["content"], (
        "the retry's own failure must replace the silent-first-attempt wording"
    )


def test_gate_off_means_a_silent_turn_is_not_retried(worker_scene, monkeypatch):
    """Default posture: no opt-in, no whole-turn replay.

    The module-wide fixture enables the retry; this case removes the only place
    it can be enabled, so the worker must settle the silent turn exactly once
    and let the Agent's own empty-response ladder own recovery. Without this
    case a regression that hard-codes the retry on would pass every other test
    in this file.
    """
    monkeypatch.delenv("HERMES_WEBUI_SILENT_FAILURE_RETRY", raising=False)
    monkeypatch.setattr(
        streaming, "_silent_retry_enabled", lambda *a, **k: False,
    )

    scene = worker_scene
    scene.on_run = _silent_then(scene, None)

    scene.run()

    assert _runs(scene) == 1, "the gate is off: no second provider call"


# ── Blocker: an answer is owned by the turn, not by its text ─────────────────


def _history_with_an_earlier_yes():
    """History already holds an assistant ``Yes.`` from an earlier turn."""
    return [
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "Yes."},
        _materialized_user_row(),
    ]


def test_retry_accepts_an_equal_text_answer_owned_by_the_current_turn(
    worker_scene, monkeypatch,
):
    """A new answer may repeat an older one verbatim; it is still the answer.

    The reviewer's case: history holds ``Yes.``, the first attempt is silent,
    and the retry answers the current turn with the same text the earlier
    assistant used. Text novelty rejected it, settled ``no_response`` and threw
    the answer away. Ownership comes from the current turn's checkpoint, not
    from the text.
    """
    scene = worker_scene
    history = _history_with_an_earlier_yes()
    scene.session.messages = [dict(row) for row in history]
    scene.session.context_messages = [dict(row) for row in history]

    def answer_now():
        if _runs(scene) == 1:
            scene.result = {"messages": [dict(row) for row in history]}
        else:
            scene.result = {"messages": [dict(row) for row in history] + [
                {"role": "assistant", "content": "Yes."},
            ]}

    scene.on_run = answer_now

    scene.run()

    assert _runs(scene) == 2, "the silent turn still earns exactly one retry"
    assert _error_rows(scene) == [], (
        "an equal-text answer owned by the current turn must settle, not error"
    )
    assert "Yes." in _session_text(scene), (
        "the retry's answer must reach the session"
    )


def test_retry_rejects_a_bare_replay_of_historical_rows(worker_scene):
    """The negative control: replaying only history is not a recovery."""
    scene = worker_scene
    history = _history_with_an_earlier_yes()
    scene.session.messages = [dict(row) for row in history]
    scene.session.context_messages = [dict(row) for row in history]
    scene.result = {"messages": [dict(row) for row in history]}

    scene.run()

    errors = _error_rows(scene)
    assert len(errors) == 1, "a replayed historical transcript must not settle"
    assert "No response from provider" in errors[0]["content"]


# ── Blocker: the retry's own failure owns the card ───────────────────────────


class _RaisingRetryAgent:
    """Synthetic agent whose call raises, as a provider failure does.

    The retry builds a *fresh* agent, so a per-instance counter cannot tell the
    first attempt from the retry. The counter is class-level: the first
    ``run_conversation`` returns the silent result, every one after raises.
    """

    instances = []
    result_messages = []
    raise_on_run = None
    calls = 0

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.context_compressor = None
        self.session_prompt_tokens = self.session_completion_tokens = 0
        self.session_cache_read_tokens = self.session_cache_write_tokens = 0
        self.session_estimated_cost_usd = None
        self.reasoning_config = self.ephemeral_system_prompt = self._last_error = None
        self.pending = []
        self.pending_lock = threading.Lock()
        self.runs = 0
        _RaisingRetryAgent.instances.append(self)

    def run_conversation(self, **_kwargs):
        self.runs += 1
        _RaisingRetryAgent.calls += 1
        if _RaisingRetryAgent.calls >= 2 and _RaisingRetryAgent.raise_on_run is not None:
            raise _RaisingRetryAgent.raise_on_run
        return {"messages": [dict(row) for row in self.result_messages]}

    def interrupt(self, _message):
        return None

    def steer(self, _text):
        return True

    def _drain_pending_steer(self):
        return None


def _run_with_raising_retry(scene, monkeypatch, exc, silent_rows):
    _RaisingRetryAgent.instances = []
    _RaisingRetryAgent.calls = 0
    _RaisingRetryAgent.result_messages = silent_rows
    _RaisingRetryAgent.raise_on_run = exc
    monkeypatch.setattr(streaming, "_get_ai_agent", lambda: _RaisingRetryAgent)
    try:
        scene.run()
    finally:
        _RaisingRetryAgent.raise_on_run = None
    return _RaisingRetryAgent.instances


def test_raised_rate_limit_on_the_retry_is_classified_as_rate_limit(
    worker_scene, monkeypatch,
):
    """A retry that *raises* 429 must not be reported as a silent no-response."""
    scene = worker_scene
    agents = _run_with_raising_retry(
        scene, monkeypatch,
        RuntimeError("429 rate limit exceeded, please retry later"),
        [_user_row()],
    )

    assert len(agents) == 2, "the retry builds a fresh agent: two instances"
    assert _RaisingRetryAgent.calls == 2, "the retry must have run"
    errors = _error_rows(scene)
    assert len(errors) == 1
    body = errors[0]["content"]
    assert "No response from provider" not in body, (
        "a raised retry error is not a silent turn"
    )
    assert "rate limit" in body.lower(), (
        f"the retry's own cause must reach the card, got: {body[:200]}"
    )


def test_raised_quota_error_on_the_retry_is_classified_as_quota(
    worker_scene, monkeypatch,
):
    """A retry that raises ``402 insufficient_quota`` classifies as quota."""
    scene = worker_scene
    _run_with_raising_retry(
        scene, monkeypatch,
        RuntimeError("402 insufficient_quota: your balance is exhausted"),
        [_user_row()],
    )

    errors = _error_rows(scene)
    assert len(errors) == 1
    body = errors[0]["content"]
    assert "No response from provider" not in body
    assert "quota" in body.lower() or "balance" in body.lower() or "credit" in body.lower(), (
        f"quota wording expected, got: {body[:200]}"
    )


# ── Blocker: every form of current-turn activity counts ──────────────────────


class _ActivityAgent:
    """Synthetic agent whose first attempt shows activity but no final answer."""

    instances = []
    calls = 0
    result_messages = []
    reasoning_content = None
    fire_interim = False

    def __init__(self, session_id=None, tool_progress_callback=None,
                 interim_assistant_callback=None, **_kwargs):
        self.session_id = session_id
        self.tool_progress_callback = tool_progress_callback
        self.context_compressor = None
        self.session_prompt_tokens = self.session_completion_tokens = 0
        self.session_cache_read_tokens = self.session_cache_write_tokens = 0
        self.session_estimated_cost_usd = None
        self.reasoning_config = self.ephemeral_system_prompt = self._last_error = None
        self.pending = []
        self.pending_lock = threading.Lock()
        # Named explicitly so the worker's signature probe wires the callback,
        # exactly as it does for a real AIAgent.
        self.interim_assistant_callback = interim_assistant_callback
        _ActivityAgent.instances.append(self)

    def run_conversation(self, **_kwargs):
        _ActivityAgent.calls += 1
        first = _ActivityAgent.calls == 1
        if first and _ActivityAgent.fire_interim and self.interim_assistant_callback:
            self.interim_assistant_callback("Let me look into that.")
        result = {"messages": [dict(row) for row in _ActivityAgent.result_messages]}
        if first and _ActivityAgent.reasoning_content:
            result["reasoning_content"] = _ActivityAgent.reasoning_content
        return result

    def interrupt(self, _message):
        return None

    def steer(self, _text):
        return True

    def _drain_pending_steer(self):
        return None


def _run_activity_scene(scene, monkeypatch, *, reasoning=None, interim=False):
    _ActivityAgent.instances = []
    _ActivityAgent.calls = 0
    _ActivityAgent.result_messages = [_user_row()]
    _ActivityAgent.reasoning_content = reasoning
    _ActivityAgent.fire_interim = interim
    monkeypatch.setattr(streaming, "_get_ai_agent", lambda: _ActivityAgent)
    try:
        scene.run()
    finally:
        _ActivityAgent.reasoning_content = None
        _ActivityAgent.fire_interim = False
    return _ActivityAgent.calls


def test_result_only_reasoning_blocks_the_silent_retry(worker_scene, monkeypatch):
    """Reasoning returned on the result — not streamed — is still activity."""
    scene = worker_scene
    calls = _run_activity_scene(
        scene, monkeypatch,
        reasoning="I considered the request and decided to answer shortly.",
    )

    assert calls == 1, (
        "a turn that produced reasoning did work and must not be replayed"
    )


def test_an_interim_assistant_message_blocks_the_silent_retry(
    worker_scene, monkeypatch,
):
    """Visible interim output is work already shown to the user."""
    scene = worker_scene
    calls = _run_activity_scene(scene, monkeypatch, interim=True)

    assert calls == 1, (
        "a turn that emitted an interim assistant message must not be replayed"
    )


# ── Follow-up: a genuinely empty fresh turn is retryable ─────────────────────


def test_a_fresh_silent_turn_with_no_rows_is_retried(worker_scene):
    """An empty result in a fresh session is a cut, not an echoed context."""
    scene = worker_scene
    scene.session.messages = []
    scene.session.context_messages = []
    scene.result = {"messages": []}

    scene.run()

    assert _runs(scene) == 2, (
        "an empty fresh turn produced nothing to echo: it earns its retry"
    )


def test_a_fresh_silent_turn_with_no_messages_key_is_retried(worker_scene):
    """A result carrying neither messages nor an answer is still retryable."""
    scene = worker_scene
    scene.session.messages = []
    scene.session.context_messages = []
    scene.result = {}

    scene.run()

    assert _runs(scene) == 2, (
        "an empty result dict must not be read as an echoed transcript"
    )
