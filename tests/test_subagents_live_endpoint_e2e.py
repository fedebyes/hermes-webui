"""End-to-end: GET /api/subagents/live over a real HTTP server + real state.db.

The structural tests in `test_session_live_subagent_badge.py` assert the shape;
this file proves the route is actually dispatched and that the payload matches
what the delegate runner leaves in state.db. It spins up the shared test server
(conftest fixture), writes subagent rows directly into a throwaway state.db, and
reads the endpoint over HTTP — no mocks on the path that matters.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request

import pytest

from tests.conftest import TEST_BASE

pytestmark = pytest.mark.requires_agent


def _get(path):
    req = urllib.request.Request(TEST_BASE + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - surfaced as a test failure
        return exc.code, {"error": exc.read().decode("utf-8", "replace")}


def test_live_endpoint_is_dispatched_and_shaped(test_server):
    status, body = _get("/api/subagents/live")
    assert status == 200, body
    assert isinstance(body.get("counts"), dict), body
    assert isinstance(body.get("total"), int), body
    assert body.get("read_only") is True, body
    assert body["total"] == sum(body["counts"].values()), body


def test_live_endpoint_counts_an_open_child_row(test_server, tmp_path, monkeypatch):
    """A child row with `ended_at IS NULL` for a parent must appear as live=1.

    This exercises the real query against a real sqlite file, so a change to the
    predicate (or to the state.db path resolution) fails here instead of silently
    showing zero in the picker forever.
    """
    import api.routes as routes

    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT,
            content TEXT,
            timestamp REAL NOT NULL
        );
        """
    )
    now = time.time()
    con.execute(
        "INSERT INTO sessions (id, source, parent_session_id, started_at, ended_at) VALUES (?,?,?,?,?)",
        ("child-live", "subagent", "parent-x", now - 30, None),
    )
    con.execute(
        "INSERT INTO sessions (id, source, parent_session_id, started_at, ended_at, end_reason) VALUES (?,?,?,?,?,?)",
        ("child-done", "subagent", "parent-x", now - 90, now - 60, "agent_close"),
    )
    con.commit()
    con.close()

    monkeypatch.setattr(routes, "_active_state_db_path", lambda: db)
    routes._live_subagents_cache.clear()

    assert routes._live_subagent_counts(now, db) == {"parent-x": 1}

    # A finished child must not count, even though it belongs to the same parent.
    con = sqlite3.connect(db)
    con.execute("UPDATE sessions SET ended_at=? WHERE id='child-live'", (now,))
    con.commit()
    con.close()
    assert routes._live_subagent_counts(now, db) == {}


def test_live_endpoint_ignores_a_stale_crashed_row(test_server, tmp_path, monkeypatch):
    """A row left open by a crash must not inflate the badge forever."""
    import api.routes as routes

    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT,
            content TEXT,
            timestamp REAL NOT NULL
        );
        """
    )
    now = time.time()
    stale = now - (routes._LIVE_SUBAGENTS_GRACE_SECONDS + 3600)
    con.execute(
        "INSERT INTO sessions (id, source, parent_session_id, started_at, ended_at) VALUES (?,?,?,?,?)",
        ("child-crashed", "subagent", "parent-y", stale, None),
    )
    con.commit()
    con.close()

    monkeypatch.setattr(routes, "_active_state_db_path", lambda: db)
    assert routes._live_subagent_counts(now, db) == {}


def test_live_endpoint_retires_a_wedged_row_before_two_hours(test_server, tmp_path, monkeypatch):
    """A child the runner never closed must stop counting within the window.

    Measured regression: a subagent row with `ended_at IS NULL` whose last message
    was 145 minutes old was reported as "1 live" while nothing was running, because
    the grace window was 2 hours. The window must sit above the real long-run tail
    (~15 min, 44 of ~790 runs) and below a genuinely dead row.
    """
    import api.routes as routes

    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT,
            content TEXT,
            timestamp REAL NOT NULL
        );
        """
    )
    now = time.time()
    started = now - 145 * 60
    con.execute(
        "INSERT INTO sessions (id, source, parent_session_id, started_at, ended_at) VALUES (?,?,?,?,?)",
        ("child-wedged", "subagent", "parent-z", started, None),
    )
    con.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        ("child-wedged", "assistant", "done", started + 21),
    )
    con.commit()
    con.close()

    monkeypatch.setattr(routes, "_active_state_db_path", lambda: db)
    assert routes._live_subagent_counts(now, db) == {}

    # ...while a child still inside the window keeps counting.
    con = sqlite3.connect(db)
    con.execute("UPDATE messages SET timestamp=? WHERE session_id='child-wedged'", (now - 60,))
    con.commit()
    con.close()
    assert routes._live_subagent_counts(now, db) == {"parent-z": 1}


def test_grace_window_covers_the_real_long_run_tail():
    """The window must stay above the measured long-run tail (~15 min)."""
    import api.routes as routes

    assert routes._LIVE_SUBAGENTS_GRACE_SECONDS >= 20 * 60, (
        "a window below the long-run tail hides a badge for a live child"
    )
