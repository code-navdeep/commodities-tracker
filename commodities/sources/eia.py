"""EIA API access — shared by the spot-price and inventory sources.

Two hard-won conventions live here so neither caller has to remember them:

1. **Full legacy series IDs only.** The `/v2/seriesid/` route wants
   `<dataset>.<series>.<frequency>` (`PET.RWTC.D`, not `RWTC`). A bare name
   returns `404 "Series ID is not valid"` — and because EIA validates the API
   key *first*, that mistake is invisible without a working key: every
   request fails at the 403 auth check instead, which looks like a key
   problem rather than a series problem.

2. **The `start` parameter is not reliable.** For the weekly series it is
   silently ignored (asking for `start=2019-01-01` returns rows back to
   1982). Callers must window the returned rows themselves; `since` here is
   passed through as a hint but never trusted.

3. **`length` is the lever that actually bounds a request.** The route
   returns rows **newest-first**, so `length=N` yields the N most recent
   observations — that, not `start`, is how an incremental fetch asks only
   for what it is missing. Measured on `PET.WCESTUS1.W`: `length=5000`
   is 676 KB / 2,289 rows (back to 1982), `length=6` is 2.3 KB.

Requests need `EIA_API_KEY` (free registration). Without it this returns an
empty list and logs why — the caller reports the stream as skipped rather
than failing the whole refresh.
"""

import logging

from ..config import EIA_API_KEY
from .http import get_json

logger = logging.getLogger(__name__)

EIA_URL = 'https://api.eia.gov/v2/seriesid/{series}'
MAX_ROWS = 5000            # EIA's own per-request cap


def eia_rows(series_id: str, since: str = None, length: int = None) -> list[dict]:
    """Raw rows for one series, newest-first as EIA returns them. Each row
    carries at least `period`, `value` and `units`. Empty list when no key.

    `length` caps the response to that many most-recent observations — the
    only reliable way to keep a request small (see convention 3 above)."""
    if not EIA_API_KEY:
        logger.error(
            f"EIA_API_KEY is not set — cannot fetch {series_id}. Register a free key "
            "at https://www.eia.gov/opendata/register.php and put EIA_API_KEY=... in .env")
        return []

    params = {'api_key': EIA_API_KEY,
              'length': str(min(int(length), MAX_ROWS)) if length else str(MAX_ROWS)}
    if since:
        params['start'] = since   # honoured for daily series, ignored for weekly
    payload = get_json(EIA_URL.format(series=series_id), params=params)
    return payload.get('response', {}).get('data', []) or []


def to_series(rows: list[dict], cutoff: str = None) -> dict:
    """{period: float} from raw rows, dropping nulls and anything before
    `cutoff` — the client-side window that convention 2 above requires."""
    values = {}
    for row in rows:
        period, value = row.get('period'), row.get('value')
        if not period or value is None:
            continue
        if cutoff and period < cutoff:
            continue
        try:
            values[period] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def units_of(rows: list[dict], default: str = '') -> str:
    """The units EIA reports for this series ('MBBL', 'BCF', 'MBBL/D')."""
    for row in rows:
        if row.get('units'):
            return row['units']
    return default
