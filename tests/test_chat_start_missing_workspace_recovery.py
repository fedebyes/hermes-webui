"""`/api/chat/start` must survive a session whose workspace directory is gone.

The browser posts ``workspace: S.session.workspace`` on every send, so an
"explicit" workspace that equals the session's own stored binding is the
sidecar echoed back, not a fresh choice. When that directory had been deleted
(a pruned worktree) the route answered 400 forever and the conversation could
never be resumed. The fix recovers through the same last-workspace/default path
the implicit branch uses and reports the recovery in the start response.

A workspace the user actually picked for this turn (one that differs from the
stored binding) stays strict: picking a non-existent directory is a real error.
"""

import json
import shutil
from types import SimpleNamespace

import pytest

from api import models, routes, workspace


def _session_dir(monkeypatch, tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "_write_session_index", lambda **_kwargs: None)
    return session_dir


def _write_sidecar(session_dir, session, workspace_value=None):
    sidecar = session_dir / f"{session.session_id}.json"
    sidecar.write_text(
        json.dumps(
            {
                "session_id": session.session_id,
                "workspace": (
                    session.workspace if workspace_value is None else workspace_value
                ),
                "messages": [{"role": "user", "content": "preserve me"}],
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _trust_local(monkeypatch, tmp_path, fallback):
    monkeypatch.setattr(workspace, "_home_path", lambda: tmp_path)
    monkeypatch.setattr(workspace, "load_workspaces", lambda: [])
    monkeypatch.setattr(routes, "get_last_workspace", lambda *a, **k: str(fallback))


def test_echoed_missing_workspace_is_recovered_and_reported(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    stale = tmp_path / "deleted-worktree"
    session = SimpleNamespace(
        session_id="sess-echo", workspace=str(stale), profile="default"
    )
    sidecar = _write_sidecar(_session_dir(monkeypatch, tmp_path), session)
    _trust_local(monkeypatch, tmp_path, fallback)

    resolved, recovered = routes._resolve_chat_workspace_with_fallback(
        session, str(stale)
    )

    assert resolved == str(fallback.resolve())
    assert recovered == str(stale)
    assert session.workspace == str(fallback.resolve())
    assert str(fallback.resolve()) in sidecar.read_text(encoding="utf-8")


def test_echo_is_matched_across_path_spellings(monkeypatch, tmp_path):
    """The browser round-trips whatever the sidecar held (e.g. a trailing slash)."""
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    stale = tmp_path / "deleted-worktree"
    session = SimpleNamespace(
        session_id="sess-echo-spelling", workspace=str(stale), profile="default"
    )
    _write_sidecar(_session_dir(monkeypatch, tmp_path), session)
    _trust_local(monkeypatch, tmp_path, fallback)

    resolved, recovered = routes._resolve_chat_workspace_with_fallback(
        session, str(stale) + "/"
    )

    assert resolved == str(fallback.resolve())
    assert recovered is not None


def test_nested_worktree_chain_deletion_is_recovered(monkeypatch, tmp_path):
    """Card shape: a 5-level nested worktree chain that was deleted wholesale."""
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    chain = tmp_path
    for level in ("a", "b", "c", "d", "e"):
        chain = chain / level
    worktree = chain / "worktree"
    chain.mkdir(parents=True)
    (chain / "worktree").mkdir()
    assert worktree.is_dir()
    # `git worktree remove` of an outer nest takes the whole chain with it.
    shutil.rmtree(tmp_path / "a")
    assert not worktree.exists()

    session = SimpleNamespace(
        session_id="sess-96347531b162", workspace=str(worktree), profile="default"
    )
    sidecar = _write_sidecar(_session_dir(monkeypatch, tmp_path), session)
    _trust_local(monkeypatch, tmp_path, fallback)

    resolved, recovered = routes._resolve_chat_workspace_with_fallback(
        session, str(worktree)
    )

    assert resolved == str(fallback.resolve())
    assert recovered == str(worktree)
    persisted = json.loads(sidecar.read_text(encoding="utf-8"))
    assert persisted["workspace"] == str(fallback.resolve())
    assert persisted["messages"] == [{"role": "user", "content": "preserve me"}]


def test_fresh_explicit_pick_of_missing_path_stays_strict(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    stale = tmp_path / "deleted-worktree"
    session = SimpleNamespace(
        session_id="sess-pick", workspace=str(existing), profile="default"
    )
    sidecar = _write_sidecar(_session_dir(monkeypatch, tmp_path), session)
    before = sidecar.read_bytes()
    _trust_local(monkeypatch, tmp_path, fallback)

    with pytest.raises(ValueError, match="Path does not exist"):
        routes._resolve_chat_workspace_with_fallback(session, str(stale))

    assert session.workspace == str(existing)
    assert sidecar.read_bytes() == before


def test_existing_echoed_workspace_is_untouched(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    session = SimpleNamespace(
        session_id="sess-ok", workspace=str(existing), profile="default"
    )
    sidecar = _write_sidecar(_session_dir(monkeypatch, tmp_path), session)
    before = sidecar.read_bytes()

    def _no_fallback(*_a, **_k):
        raise AssertionError("fallback must not be consulted")

    _trust_local(monkeypatch, tmp_path, fallback)
    monkeypatch.setattr(routes, "get_last_workspace", _no_fallback)

    resolved, recovered = routes._resolve_chat_workspace_with_fallback(
        session, str(existing)
    )

    assert resolved == str(existing)
    assert recovered is None
    assert session.workspace == str(existing)
    assert sidecar.read_bytes() == before


def test_resolver_carries_recovery_marker_only_when_it_recovers(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    stale = tmp_path / "deleted-worktree"
    _session_dir(monkeypatch, tmp_path)
    _trust_local(monkeypatch, tmp_path, fallback)

    recovered = SimpleNamespace(
        session_id="sess-carrier-1", workspace=str(stale), profile="default"
    )
    _write_sidecar(models.SESSION_DIR, recovered)
    resolved = routes._resolve_chat_workspace_with_recovery(recovered, str(stale))
    assert isinstance(resolved, str)
    assert str(resolved) == str(fallback.resolve())
    assert resolved.recovered_from == str(stale)

    healthy = SimpleNamespace(
        session_id="sess-carrier-2", workspace=str(existing), profile="default"
    )
    _write_sidecar(models.SESSION_DIR, healthy)
    resolved = routes._resolve_chat_workspace_with_recovery(healthy, str(existing))
    assert str(resolved) == str(existing)
    assert getattr(resolved, "recovered_from", None) is None


def _post_chat_start(monkeypatch, session, body):
    """Drive the real ``_handle_chat_start`` with only the periphery stubbed."""
    import api.compression_continuation as compression_continuation

    captured = {}

    monkeypatch.setattr(routes, "require", lambda _body, _key: None)
    monkeypatch.setattr(routes, "run_admission_closed_response", lambda: None)
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **_k: None)
    monkeypatch.setattr(routes, "_get_or_materialize_session", lambda *_a, **_k: session)
    monkeypatch.setattr(routes, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_a: True)
    monkeypatch.setattr(
        compression_continuation, "durable_compression_continuation", lambda _s: (False, None)
    )
    monkeypatch.setattr(routes, "compression_recovery_payload_for_session", lambda _s: None)
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda *_a: (None, None, {}))
    monkeypatch.setattr(routes, "get_config_snapshot", lambda: {})
    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _c: False)
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda model, *_a, **_k: (model, "deepseek", model),
    )
    monkeypatch.setattr(routes, "_external_model_alias_lane_response", lambda *_a, **_k: None)
    monkeypatch.setattr(
        routes,
        "_repair_foreign_session_model_provider",
        lambda _s, **kwargs: kwargs.get("resolved_provider"),
    )
    monkeypatch.setattr(
        routes,
        "_start_run",
        lambda _s, **_kwargs: {"stream_id": "stream-1", "session_id": session.session_id},
    )
    monkeypatch.setattr(
        routes,
        "j",
        lambda _h, payload, status=200, **_k: captured.setdefault("response", (status, payload)),
    )
    monkeypatch.setattr(
        routes,
        "bad",
        lambda _h, message, status=400, **_k: captured.setdefault("error", (status, message)),
    )

    result = routes._handle_chat_start(SimpleNamespace(), body)
    return captured, result


def test_chat_start_returns_200_and_names_the_fallback(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    stale = tmp_path / "deleted-worktree"
    session = SimpleNamespace(
        session_id="sess-chat-echo",
        workspace=str(stale),
        profile="default",
        model="deepseek-v3",
        model_provider="deepseek",
        messages=[{"role": "user", "content": "hello"}],
    )
    _write_sidecar(_session_dir(monkeypatch, tmp_path), session)
    _trust_local(monkeypatch, tmp_path, fallback)

    captured, _result = _post_chat_start(
        monkeypatch,
        session,
        {
            "session_id": session.session_id,
            "message": "carry on where we stopped",
            "workspace": str(stale),
            "profile": "default",
        },
    )

    assert "error" not in captured
    status, payload = captured["response"]
    assert status == 200
    assert payload["workspace"] == str(fallback.resolve())
    assert payload["workspace_recovered"] is True
    assert payload["workspace_recovered_from"] == str(stale)
    assert session.workspace == str(fallback.resolve())


def test_chat_start_with_healthy_workspace_stays_byte_identical(monkeypatch, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    session = SimpleNamespace(
        session_id="sess-chat-ok",
        workspace=str(existing),
        profile="default",
        model="deepseek-v3",
        model_provider="deepseek",
        messages=[{"role": "user", "content": "hello"}],
    )
    _write_sidecar(_session_dir(monkeypatch, tmp_path), session)

    captured, _result = _post_chat_start(
        monkeypatch,
        session,
        {
            "session_id": session.session_id,
            "message": "another turn",
            "workspace": str(existing),
            "profile": "default",
        },
    )

    assert "error" not in captured
    status, payload = captured["response"]
    assert status == 200
    assert "workspace_recovered" not in payload
    assert "workspace" not in payload
    assert payload == {"stream_id": "stream-1", "session_id": session.session_id}


def test_chat_start_still_400s_a_user_picked_missing_workspace(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    stale = tmp_path / "deleted-worktree"
    session = SimpleNamespace(
        session_id="sess-chat-pick",
        workspace=str(existing),
        profile="default",
        model="deepseek-v3",
        model_provider="deepseek",
        messages=[{"role": "user", "content": "hello"}],
    )
    _write_sidecar(_session_dir(monkeypatch, tmp_path), session)
    _trust_local(monkeypatch, tmp_path, fallback)

    captured, _result = _post_chat_start(
        monkeypatch,
        session,
        {
            "session_id": session.session_id,
            "message": "run this in a deleted folder",
            "workspace": str(stale),
            "profile": "default",
        },
    )

    assert captured["error"][0] == 400
    assert "does not exist" in captured["error"][1].lower()
    assert session.workspace == str(existing)
