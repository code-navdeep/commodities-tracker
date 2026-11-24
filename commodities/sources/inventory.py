"""Inventory source — weekly EIA stocks (and the crude production flow).

For crude these are the numbers behind the Wednesday Weekly Petroleum Status
Report; for gas, the Thursday Weekly Natural Gas Storage Report. Both are
published a few days after the Friday week-ending date they describe, which
is why the analytics join uses INVENTORY_STALE_DAYS rather than COT's 14.

Each configured series is fetched whole and windowed client-side — EIA's
`start` parameter is ignored for weekly series (see sources/eia.py). Series
are small (≤ ~2,300 rows), so this costs one request each and the store
upserts only what is new.
"""

import logging
import math
from datetime import datetime, timedelta

from ..config import ASSETS, INVENTORY_HISTORY_YEARS, INVENTORY_OVERLAP_WEEKS
from ..domain import InventoryPoint
from .eia import MAX_ROWS, eia_rows, to_series, units_of

logger = logging.getLogger(__name__)


def _backfill_cutoff() -> str:
    """Start of the window on a fresh store: enough history for a 5-year
    seasonal band at every point shown."""
    return (datetime.now() - timedelta(days=365 * INVENTORY_HISTORY_YEARS)
            ).strftime('%Y-%m-%d')


def request_plan(latest: str | None, today: datetime = None) -> tuple[int, str]:
    """(rows_to_request, keep_from) for one series.

    Nothing stored → one full backfill. Otherwise ask only for the weeks
    we're missing plus INVENTORY_OVERLAP_WEEKS, because EIA revises recent
    figures and the upsert should see those revisions. `length` is the lever
    that bounds the request; `start` is ignored by this route (see eia.py).
    """
    backfill_from = _backfill_cutoff()
    if not latest:
        return MAX_ROWS, backfill_from

    today = today or datetime.now()
    gap_days = max((today - datetime.strptime(latest, '%Y-%m-%d')).days, 0)
    weeks_missing = math.ceil(gap_days / 7)
    rows = min(weeks_missing + INVENTORY_OVERLAP_WEEKS, MAX_ROWS)

    keep_from = (datetime.strptime(latest, '%Y-%m-%d')
                 - timedelta(weeks=INVENTORY_OVERLAP_WEEKS)).strftime('%Y-%m-%d')
    # ISO dates compare lexicographically, so max() is chronological
    return rows, max(keep_from, backfill_from)


def fetch_inventory(asset: str, since=None) -> list[InventoryPoint]:
    """Every configured inventory series for one asset. Assets without an
    `inventory` registry entry (the metals) return nothing.

    `since(series_name) -> iso_date | None` reports what the store already
    holds, so each series asks EIA only for what is missing. Without it the
    call falls back to a full backfill.
    """
    series_map = ASSETS.get(asset, {}).get('inventory') or {}
    if not series_map:
        return []

    points = []
    for name, series_id in series_map.items():
        latest = since(name) if since else None
        rows_wanted, keep_from = request_plan(latest)
        try:
            rows = eia_rows(series_id, length=rows_wanted)
        except Exception as e:
            logger.error(f"{asset}: inventory series '{name}' ({series_id}) failed: {e}")
            continue
        if not rows:
            continue
        units = units_of(rows)
        for period, value in to_series(rows, cutoff=keep_from).items():
            points.append(InventoryPoint(asset=asset, series=name, period=period,
                                         value=value, units=units))
    return points
