"""The WebUI meter and the ``[session-cost]`` notice must quote one number.

The two used to disagree by orders of magnitude on the same session because they
were different quantities: the WebUI quoted the runtime's per-process USD
accumulator, the plugin quoted durable tokens from ``state.db`` at the measured
CNY rate. These tests pin the aligned arithmetic (``api/session_cost.py``) and
the refusals that keep it from inventing a figure:

  * the money is ``tokens / 1e6 * price.effective_cny_per_mtok * eur_per_cny``,
    i.e. exactly the plugin's formula, with the price read from the feed instead
    of a second constant;
  * ``tokens`` is the recursive family sum (the plugin's own query), so a
    delegated subagent's spend counts in the session's quote;
  * feed missing / stale / non-finite / non-positive, and a non-DeepSeek model,
    all yield no money rather than a confident ``€0.00``;
  * ``state.db`` missing or schema-poor never raises.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from api import session_cost


NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _feed(path: Path, *, cny_per_mtok=0.0544, eur_per_cny=0.1335,
          generated_at=None, **overrides) -> Path:
    payload = {
        "version": 1,
        "generated_at": (generated_at or NOW - timedelta(minutes=5)).isoformat(),
        "price": {"effective_cny_per_mtok": cny_per_mtok, "window_days": 7},
        "eur_per_cny": eur_per_cny,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _db(path: Path, rows) -> Path:
    """A ``sessions`` table with the columns the family query reads."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT,"
        " input_tokens INTEGER, cache_read_tokens INTEGER, output_tokens INTEGER)"
    )
    conn.executemany(
        "INSERT INTO sessions (id, parent_session_id, input_tokens, cache_read_tokens,"
        " output_tokens) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return path


def test_price_is_read_from_the_feed_and_scaled_to_eur(tmp_path):
    feed = _feed(tmp_path / "data.json")
    price = session_cost.read_price(feed, now=NOW)
    assert price is not None
    assert price["cny_per_mtok"] == pytest.approx(0.0544)
    # One multiplication for callers: EUR per million tokens, not CNY.
    assert price["eur_per_mtok"] == pytest.approx(0.0544 * 0.1335)


def test_money_matches_the_notice_arithmetic(tmp_path):
    feed = _feed(tmp_path / "data.json")
    db = _db(tmp_path / "state.db", [("s1", None, 1000, 5_000_000, 200_000)])
    quote = session_cost.session_cost_eur(
        "s1", db_path=db, model="deepseek-flash", feed_path=feed, now=NOW)
    assert quote["available"] is True
    assert quote["tokens"] == 5_201_000
    assert quote["source"] == "state.db"
    assert quote["eur"] == pytest.approx(5_201_000 / 1e6 * 0.0544 * 0.1335)


def test_family_tokens_include_delegated_children(tmp_path):
    db = _db(tmp_path / "state.db", [
        ("parent", None, 0, 1_000_000, 0),
        ("child", "parent", 0, 500_000, 0),
        ("grandchild", "child", 0, 250_000, 0),
        ("unrelated", None, 0, 9_000_000, 0),
    ])
    assert session_cost.family_tokens(db, "parent") == 1_750_000


@pytest.mark.parametrize("age_minutes,expected", [(5, True), (119, True), (121, False)])
def test_stale_feed_is_refused(tmp_path, age_minutes, expected):
    feed = _feed(tmp_path / "data.json", generated_at=NOW - timedelta(minutes=age_minutes))
    price = session_cost.read_price(feed, now=NOW)
    assert (price is not None) is expected


@pytest.mark.parametrize("generated_at", [None, "", "not-a-date"])
def test_feed_without_a_usable_timestamp_is_refused(tmp_path, generated_at):
    feed = _feed(tmp_path / "data.json")
    data = json.loads(feed.read_text())
    if generated_at is None:
        data.pop("generated_at")
    else:
        data["generated_at"] = generated_at
    feed.write_text(json.dumps(data), encoding="utf-8")
    assert session_cost.read_price(feed, now=NOW) is None


@pytest.mark.parametrize("raw", [0, -1, "0.05", True, None, float("nan"), float("inf"), 10 ** 400])
def test_unusable_price_is_refused(tmp_path, raw):
    """A price that is not a positive finite number is no price at all."""
    feed = _feed(tmp_path / "data.json")
    data = json.loads(feed.read_text())
    data["price"]["effective_cny_per_mtok"] = raw
    feed.write_text(json.dumps(data), encoding="utf-8")
    assert session_cost.read_price(feed, now=NOW) is None


def test_missing_feed_is_refused_not_guessed(tmp_path):
    assert session_cost.read_price(tmp_path / "nope.json", now=NOW) is None
    quote = session_cost.session_cost_eur(
        "s1", db_path=tmp_path / "state.db", model="deepseek-flash",
        feed_path=tmp_path / "nope.json", now=NOW)
    assert quote["available"] is False
    assert quote["eur"] is None
    assert quote["reason"] == "no-feed"


def test_non_deepseek_model_is_not_priced_with_a_deepseek_rate(tmp_path):
    feed = _feed(tmp_path / "data.json")
    db = _db(tmp_path / "state.db", [("s1", None, 0, 1_000_000, 0)])
    quote = session_cost.session_cost_eur(
        "s1", db_path=db, model="claude-opus-5", feed_path=feed, now=NOW)
    assert quote["eur"] is None
    assert quote["reason"] == "unpriced-model"


def test_live_tokens_are_used_only_without_a_durable_row(tmp_path):
    feed = _feed(tmp_path / "data.json")
    db = _db(tmp_path / "state.db", [])
    quote = session_cost.session_cost_eur(
        "brand-new", db_path=db, model="deepseek-flash",
        fallback_tokens=42_000, feed_path=feed, now=NOW)
    assert quote["source"] == "live"
    assert quote["tokens"] == 42_000
    assert quote["eur"] == pytest.approx(42_000 / 1e6 * 0.0544 * 0.1335)


def test_no_tokens_means_no_money(tmp_path):
    feed = _feed(tmp_path / "data.json")
    db = _db(tmp_path / "state.db", [("s1", None, 0, 0, 0)])
    quote = session_cost.session_cost_eur(
        "s1", db_path=db, model="deepseek-flash", feed_path=feed, now=NOW)
    assert quote["available"] is False
    assert quote["reason"] == "no-tokens"


def test_broken_database_never_raises(tmp_path):
    broken = tmp_path / "state.db"
    broken.write_bytes(b"not a database")
    assert session_cost.family_tokens(broken, "s1") is None
    assert session_cost.family_tokens(tmp_path / "absent.db", "s1") is None


def test_usage_fields_never_raise_and_report_none(tmp_path):
    fields = session_cost.usage_fields(
        "s1", db_path=tmp_path / "absent.db", feed_path=tmp_path / "absent.json")
    assert fields == {"estimated_cost_eur": None, "cost_tokens": 0}
