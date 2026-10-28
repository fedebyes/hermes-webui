"""``state.db`` stores input tokens NET of cache, whichever surface writes them.

``api/state_sync.sync_session_usage`` mirrors the WebUI's own counters into
``state.db``. Its callers hand it ``session.input_tokens``, which the streaming
path fills from ``agent.session_prompt_tokens`` — the GROSS prompt total. The
runtime's per-call writer stores ``CanonicalUsage.input_tokens``, which
``agent/usage_pricing.normalize_usage`` has already stripped of the cache
buckets. Consumers sum ``input + cache_read + output``
(``usage_tracker.deepseek_tokens``, the ``peak-cost-notice`` family query,
``api/session_cost.family_tokens``), so a gross input value counts the cache a
second time and a warm session reads about twice its real size.

These tests pin the write, and only the write: the WebUI's own counters stay
gross on purpose, because the cache-hit percentage needs
``cache_read / (input + cache_read + cache_write)``.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def profile_home(tmp_path, monkeypatch):
    """A minimal profile home whose ``state.db`` has the real schema."""
    pytest.importorskip("hermes_state")
    from hermes_state import SessionDB

    home = tmp_path / "profile"
    home.mkdir(parents=True)
    # Let SessionDB create the file and run its own schema/migrations: a
    # hand-rolled ``touch`` leaves a 0-byte db it then quarantines as damaged.
    SessionDB(home / "state.db").close()

    import api.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "get_active_hermes_home", lambda: home)
    return home


def _row(db_path: Path, session_id: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT input_tokens, output_tokens, cache_read_tokens,"
            " cache_write_tokens FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()


def test_cache_is_subtracted_out_of_the_written_input(profile_home):
    from api.state_sync import sync_session_usage

    # A warm turn: 40k uncached, 5M served from cache, 20k out. The gross prompt
    # total is what ``session_prompt_tokens`` holds (and what the WebUI passes).
    gross_prompt = 40_000 + 5_000_000
    sync_session_usage(
        session_id="sess-net",
        input_tokens=gross_prompt,
        output_tokens=20_000,
        cache_read_tokens=5_000_000,
        cache_write_tokens=0,
        model="deepseek-v4",
    )

    row = _row(profile_home / "state.db", "sess-net")
    assert row is not None
    assert row["input_tokens"] == 40_000
    assert row["cache_read_tokens"] == 5_000_000
    # The quantity every consumer sums (input + cache_read + output) is the real
    # prompt size plus the completion, not ~2x it.
    assert row["input_tokens"] + row["cache_read_tokens"] + row["output_tokens"] == (
        gross_prompt + 20_000
    ) == 5_060_000


def test_cache_writes_are_not_counted_either(profile_home):
    """A third bucket: ``cache_write`` is excluded from the net input too.

    No consumer sums ``cache_write_tokens`` (DeepSeek reports no separate write
    bucket, and ``usage_tracker.deepseek_tokens`` adds read + input + output), so
    subtracting it keeps the written input equal to the uncached prompt. The
    runtime's ``normalize_usage`` subtracts both buckets for the same reason.
    """
    from api.state_sync import sync_session_usage

    sync_session_usage(
        session_id="sess-write",
        input_tokens=40_000 + 5_000_000 + 1_000_000,  # gross, write included
        output_tokens=20_000,
        cache_read_tokens=5_000_000,
        cache_write_tokens=1_000_000,
    )
    row = _row(profile_home / "state.db", "sess-write")
    assert row["input_tokens"] == 40_000
    assert row["cache_write_tokens"] == 1_000_000
    # exact uncached prompt (40k) + cache read (5M) + completion (20k)
    assert row["input_tokens"] + row["cache_read_tokens"] + row["output_tokens"] == 5_060_000


def test_rerun_with_no_growth_is_idempotent(profile_home):
    """``absolute=True``: a title-only re-sync must not subtract twice."""
    from api.state_sync import sync_session_usage

    def sync(title: str) -> None:
        sync_session_usage(
            session_id="sess-idem",
            input_tokens=120_000,
            output_tokens=1_000,
            cache_read_tokens=100_000,
            cache_write_tokens=0,
            title=title,
        )

    sync("first")
    first = _row(profile_home / "state.db", "sess-idem")["input_tokens"]
    sync("renamed")
    second = _row(profile_home / "state.db", "sess-idem")["input_tokens"]
    assert first == second == 20_000


def test_a_cold_first_turn_is_unchanged(profile_home):
    """No cache read yet: gross == net, so the first turn cannot go negative."""
    from api.state_sync import sync_session_usage

    sync_session_usage(
        session_id="sess-cold",
        input_tokens=9_500,
        output_tokens=500,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    row = _row(profile_home / "state.db", "sess-cold")
    assert row["input_tokens"] == 9_500
    assert row["output_tokens"] == 500


def test_broken_counters_never_write_a_negative_input(profile_home):
    """Cache counters larger than the prompt (a stale sidecar) clamp at 0."""
    from api.state_sync import sync_session_usage

    sync_session_usage(
        session_id="sess-clamp",
        input_tokens=1_000,
        output_tokens=10,
        cache_read_tokens=900_000,
        cache_write_tokens=0,
    )
    assert _row(profile_home / "state.db", "sess-clamp")["input_tokens"] == 0
