"""Session cost on ONE basis: durable tokens x the calibrated price.

Why this module exists
----------------------
The WebUI used to quote ``agent.session_estimated_cost_usd`` — the runtime's own
accumulator, priced per call against the vendor card hardcoded in the agent
package (``agent/usage_pricing.py``). Two things make that number disagree with
the ``[session-cost]`` notice the ``peak-cost-notice`` plugin injects for the
same conversation:

  * **It is per-process, not per-session.** The accumulator lives on the agent
    object and starts at 0 on every resume. A long session that survived three
    restarts reports only what the *current* process spent (measured here: 450 of
    615 sessions in ``state.db`` disagree with the WebUI index, and the index was
    the smaller one in every heavy case — 460.3 M durable tokens shown as 13.5 M
    for one Docker-monitoring session, i.e. a 34x under-report).
  * **It is priced on the card, not on the ledger.** The card is DeepSeek's
    published off-peak rate; what actually leaves the account also carries the
    peak multiplier and the real cache mix. ``usage_tracker.effective_price``
    measures that from the CNY ledger, and the daily spend feed publishes the
    result — that measured price is the project's single implementation of the
    arithmetic and nothing here re-derives it.

So this module quotes, exactly like the plugin does:

    EUR = tokens / 1e6 * price.effective_cny_per_mtok * eur_per_cny

where ``tokens`` is the session family's durable total from ``state.db`` and the
price pair comes from the peak-spend feed (written every 15 min by the
``webui-spend-feed`` cron, itself quoting ``effective_price``).

Token definition — deliberate, and the reason the two numbers now agree
-----------------------------------------------------------------------
``tokens`` is ``input_tokens + cache_read_tokens + output_tokens``, summed over
the session and everything it delegated to (the plugin's ``_family_tokens``
query, verbatim). That is the *same sum* ``usage_tracker.deepseek_tokens`` puts
in the denominator of ``effective_price``: price and token count are defined
over the same quantity, so the product reproduces the ledger drain. Swapping in
the "net of cache" input count here would halve the figure while the price kept
its gross denominator — the exact silent drift this module exists to remove.

(The WebUI used to write ``input_tokens`` to ``state.db`` cache-INCLUSIVE, from
``session_prompt_tokens`` — the gross prompt total — while the runtime's own
writer stores it net of cache, so for WebUI-written rows the cache was counted
twice in that sum. ``api/state_sync.sync_session_usage`` now subtracts both
cache buckets before the write, so a new row means the same thing whichever
surface wrote it. Rows written before that fix stay gross and are NOT rewritten:
history is history, the calibration absorbs the residual, and the product above
is only ever as good as the token definition it inherits.)

Refusals
--------
Every degenerate input yields ``available: False`` and no money, never a
fabricated one: feed missing / unreadable / not finite / older than
``STALE_AFTER_MINUTES`` / non-positive price, database missing or schema-poor,
no durable row and no live fallback. A ``€0.00`` printed as if measured is the
confident lie this module must never tell; so is a session on somebody else's
pricing (the measured price is DeepSeek-only, so a non-deepseek model is
refused rather than priced with a rate that never applied to it).
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: The peak-spend feed the 15-min cron writes. Its ``price`` block is
#: ``usage_tracker.effective_price`` verbatim, and ``eur_per_cny`` the FX the
#: panel already uses — reading both from here is what keeps the WebUI, the
#: plugin notice and the extension panel from quoting different money.
SPEND_FEED = Path.home() / ".hermes" / "webui" / "extensions" / "peak-spend" / "data.json"

#: The feed is written every 15 minutes; two hours is eight missed ticks — a
#: broken cron, not a hiccup — and a stale price presented as current is a lie.
STALE_AFTER_MINUTES = 120

#: The calibrated price is measured on DeepSeek traffic only. Anything else gets
#: no figure rather than a rate that never applied to it.
PRICED_MODEL_PREFIX = "deepseek"

_FAMILY_TOKENS_SQL = (
    "WITH RECURSIVE fam(id) AS ("
    "  SELECT id FROM sessions WHERE id = ?"
    "  UNION"
    "  SELECT s.id FROM sessions s JOIN fam f ON s.parent_session_id = f.id"
    ") SELECT COALESCE(SUM(input_tokens + cache_read_tokens + output_tokens), 0),"
    "         COUNT(*)"
    "  FROM sessions WHERE id IN (SELECT id FROM fam)"
)


def _finite_positive(value: Any) -> Optional[float]:
    """``value`` as a finite float > 0, else None.

    ``json`` happily parses ``NaN``/``Infinity`` literals and a huge integer
    raises ``OverflowError`` in ``float()``; both must read as "no price"
    rather than propagate into a money figure.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _as_utc(value: Any) -> Optional[datetime]:
    """Parse a feed timestamp (ISO-8601, ``Z`` or offset, or epoch seconds)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def read_price(
    feed_path: Optional[Path | str] = None,
    *,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """``{'eur_per_mtok', 'cny_per_mtok', 'eur_per_cny', 'age_minutes'}``, or None.

    None for every reason listed in the module docstring. The EUR-per-million
    figure is computed once here so callers multiply and nothing else has to
    remember the CNY->EUR step.
    """
    path = Path(feed_path) if feed_path is not None else SPEND_FEED
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    price = data.get("price")
    if not isinstance(price, dict):
        return None
    cny_per_mtok = _finite_positive(price.get("effective_cny_per_mtok"))
    if cny_per_mtok is None:
        return None

    eur_per_cny = _finite_positive(data.get("eur_per_cny"))
    if eur_per_cny is None:
        return None

    generated = _as_utc(data.get("generated_at"))
    if generated is None:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age_minutes = (current - generated).total_seconds() / 60.0
    # A clock that runs backwards is not a fresh feed. Negative age beyond a
    # small skew means the file was written by a machine whose clock disagrees.
    if age_minutes > STALE_AFTER_MINUTES or age_minutes < -STALE_AFTER_MINUTES:
        return None

    return {
        "eur_per_mtok": cny_per_mtok * eur_per_cny,
        "cny_per_mtok": cny_per_mtok,
        "eur_per_cny": eur_per_cny,
        "age_minutes": age_minutes,
    }


def family_tokens(db_path: Path | str, session_id: str) -> Optional[int]:
    """Durable tokens for ``session_id`` and everything it delegated to.

    A recursive walk (not a flat ``parent_session_id`` join): a depth-2
    orchestrator's workers hang off the subagent, so a flat join would miss
    them. Same query as the plugin's ``_family_tokens`` on purpose — the two
    quotes must be the same quantity or they drift again.

    Returns None when the database or the table is unavailable, 0 when the
    session has no row yet (a brand-new conversation).
    """
    if not session_id:
        return None
    try:
        connection = sqlite3.connect(
            f"file:{Path(db_path).expanduser()}?mode=ro", uri=True, timeout=2.0
        )
    except sqlite3.Error:
        return None
    try:
        row = connection.execute(_FAMILY_TOKENS_SQL, (session_id,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
    if not row:
        return None
    try:
        return max(0, int(row[0] or 0))
    except (TypeError, ValueError):
        return None


def session_cost_eur(
    session_id: str,
    *,
    db_path: Optional[Path | str] = None,
    model: Optional[str] = None,
    fallback_tokens: int = 0,
    feed_path: Optional[Path | str] = None,
    now: Optional[datetime] = None,
) -> dict:
    """The aligned quote for one session.

    Returns ``{'available', 'eur', 'tokens', 'price_eur_per_mtok', 'source',
    'reason'}``. ``eur`` is None whenever the figure cannot be defended;
    ``reason`` then says why in one word (``no-feed``, ``stale-feed``,
    ``unpriced-model``, ``no-tokens``, ``no-db``).

    ``fallback_tokens`` is used only when ``state.db`` has no row for the
    session — a conversation that has not been flushed once yet. On such a
    session the first call carries no cache reads, so the net/gross difference
    cannot bite there.
    """
    result: dict = {
        "available": False,
        "eur": None,
        "tokens": 0,
        "price_eur_per_mtok": None,
        "source": None,
        "reason": None,
    }

    if str(model or "").strip().lower() and not str(model).strip().lower().startswith(
        PRICED_MODEL_PREFIX
    ):
        result["reason"] = "unpriced-model"
        return result

    price = read_price(feed_path, now=now)
    if price is None:
        result["reason"] = "no-feed"
        return result

    tokens = None
    source = "state.db"
    if db_path is not None:
        tokens = family_tokens(db_path, session_id)
    if not tokens:
        try:
            live = int(fallback_tokens or 0)
        except (TypeError, ValueError):
            live = 0
        if live > 0:
            tokens, source = live, "live"
        elif db_path is None:
            result["reason"] = "no-db"
            return result
        else:
            tokens = 0

    tokens = int(tokens or 0)
    result["tokens"] = tokens
    result["price_eur_per_mtok"] = price["eur_per_mtok"]
    if tokens <= 0:
        result["reason"] = "no-tokens"
        return result

    result.update(
        available=True,
        eur=tokens / 1e6 * price["eur_per_mtok"],
        source=source,
    )
    return result


def _default_db_path() -> Optional[Path]:
    """The active profile's ``state.db``, or None when the app can't name it.

    Imported lazily: ``api.models`` is a huge module and this helper is called
    from the streaming worker, which must not pay that import on a hot path it
    already paid for.
    """
    try:
        from api.models import _active_state_db_path

        path = _active_state_db_path()
        return Path(path) if path else None
    except Exception:
        return None


def usage_fields(
    session_id: str,
    *,
    model: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    fallback_tokens: int = 0,
    feed_path: Optional[Path | str] = None,
    now: Optional[datetime] = None,
) -> dict:
    """The extra keys to merge into a usage payload. Never raises, never invents.

    ``estimated_cost_eur`` is None (not 0) when the figure cannot be defended,
    so a frontend can fall back to the legacy ``estimated_cost`` instead of
    printing a confident zero. ``cost_tokens`` is the durable quantity the money
    was computed from, so a caller can show the basis alongside it.
    """
    try:
        quote = session_cost_eur(
            session_id,
            db_path=db_path if db_path is not None else _default_db_path(),
            model=model,
            fallback_tokens=fallback_tokens,
            feed_path=feed_path,
            now=now,
        )
    except Exception:
        return {"estimated_cost_eur": None, "cost_tokens": 0}
    return {"estimated_cost_eur": quote["eur"], "cost_tokens": quote["tokens"]}
