"""Coverage for the metadata-head read used by ``WebUIJsonSessionDB.list_sessions()``.

``list_sessions()`` only needs the scalar metadata of each sidecar, but it used to
``json.loads`` the whole file -- 13.8-17.8 s per refresh over the live store (637
files / 1.18 GB), every byte of the transcripts included. The sidecar layout puts
all metadata before ``messages``, so the row source now parses the head only.

These tests pin the three properties that make that safe:
  * the metadata prefix decoded from the head is identical to the one a full
    parse yields (so the sidebar cannot drift from the full reader),
  * a sidecar without a usable ``message_count`` still reports a truthful count
    (falls back to a full parse instead of showing 0 messages),
  * ``messages`` is never part of a head-read result, and a prefix that overflows
    the window is served by the shared budgeted reader rather than silently
    dropping fields.

Timing is deliberately not asserted: it depends on the host's page cache.
"""
import json

import pytest

import api.models as models
import api.webui_session_db as session_db

DB = session_db.WebUIJsonSessionDB


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(models, "SESSION_DIR", sdir)
    return sdir


def _big_messages(count=400, text_size=2000):
    return [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"message {i} " + "X" * text_size,
         "timestamp": 1000.0 + i}
        for i in range(count)
    ]


def _save_modern_session(session_store, sid="s1", messages=None, title="Metadata Head"):
    """Write a sidecar through the real writer, so the key order is the contract."""
    session = models.Session(
        session_id=sid,
        title=title,
        workspace=str(session_store.parent),
        model="test-model",
        messages=messages if messages is not None else _big_messages(),
    )
    session.save()
    return session_store / f"{sid}.json"


def _full_row(path):
    data = DB._read_path(path)
    assert isinstance(data, dict)
    return DB._metadata_row(str(data.get("session_id") or path.stem), data)


def test_head_metadata_matches_full_parse_on_large_sidecar(session_store):
    path = _save_modern_session(session_store)
    assert path.stat().st_size > DB._METADATA_HEAD_BYTES * 4

    head = DB._read_metadata_head(path)
    assert isinstance(head, dict)
    assert "messages" not in head

    source = DB._read_metadata_row_source(path)
    assert isinstance(source, dict)
    assert "messages" not in source

    full = DB._read_path(path)
    assert isinstance(full, dict)
    sid = str(full["session_id"])
    assert DB._metadata_row(sid, source) == DB._metadata_row(sid, full)

    row = DB(session_store).list_sessions()[0]
    assert row == DB._metadata_row(sid, full)
    assert row["message_count"] == 400
    assert row["title"] == "Metadata Head"


def test_head_metadata_covers_every_adapter_field(session_store):
    path = _save_modern_session(session_store, sid="s2")
    source = DB._read_metadata_row_source(path)
    full = DB._read_path(path)
    assert isinstance(source, dict) and isinstance(full, dict)

    expected_fields = (session_db._METADATA_FIELDS | {"session_id", "message_count"}) & set(full)
    for field in expected_fields:
        assert source.get(field) == full.get(field), field


def test_missing_message_count_falls_back_to_full_parse(session_store):
    """A sidecar without ``message_count`` must not render as 0 messages."""
    payload = {
        "session_id": "legacy",
        "title": "Legacy",
        "created_at": 100.0,
        "updated_at": 200.0,
        "messages": [{"role": "user", "content": "hello " + "X" * 500}, {"role": "user", "content": "again"}],
    }
    path = session_store / "legacy.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # The head read alone cannot answer: the count is derived from `messages`.
    head = DB._read_metadata_head(path)
    assert isinstance(head, dict)
    assert "message_count" not in head

    source = DB._read_metadata_row_source(path)
    assert isinstance(source, dict)
    row = DB._metadata_row("legacy", source)
    assert row["message_count"] == 2
    assert DB(session_store).list_sessions()[0]["message_count"] == 2


def test_oversized_metadata_prefix_uses_budgeted_reader(session_store, monkeypatch):
    """Metadata past the head window is still read without a full parse."""
    payload = {
        "session_id": "fat",
        "title": "Fat Prefix",
        "created_at": 100.0,
        "updated_at": 200.0,
        "compression_anchor_summary": "S" * (DB._METADATA_HEAD_BYTES + 16 * 1024),
        "message_count": 3,
        "messages": [{"role": "user", "content": "hi"}] * 3,
    }
    path = session_store / "fat.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    assert DB._read_metadata_head(path) is None

    def _no_full_parse(_path):
        raise AssertionError("oversized prefix should be served by the budgeted reader")

    monkeypatch.setattr(DB, "_read_path", staticmethod(_no_full_parse))
    source = DB._read_metadata_row_source(path)
    assert isinstance(source, dict)
    assert "messages" not in source
    assert source["session_id"] == "fat"
    assert source["message_count"] == 3


def test_compact_sidecar_is_read_from_the_head(session_store):
    payload = {
        "session_id": "compact",
        "title": "Compact",
        "created_at": 100.0,
        "message_count": 2,
        "messages": [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b"}],
    }
    path = session_store / "compact.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    source = DB._read_metadata_row_source(path)
    assert isinstance(source, dict)
    assert "messages" not in source
    assert source["title"] == "Compact"
    assert DB._metadata_row("compact", source)["message_count"] == 2


def test_malformed_sidecar_is_skipped(session_store):
    (session_store / "broken.json").write_text("{not json", encoding="utf-8")
    _save_modern_session(session_store, sid="healthy", messages=[{"role": "user", "content": "hi"}])

    assert DB._read_metadata_row_source(session_store / "broken.json") is None
    rows = DB(session_store).list_sessions()
    assert [row["session_id"] for row in rows] == ["healthy"]
