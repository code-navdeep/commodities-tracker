"""COT source — CFTC Public Reporting API.

Dataset kh3c-gbw2: Disaggregated report, Futures + Options Combined — chosen
because it captures total commercial/speculator positioning including options
hedging.

**All four reportable categories are stored as published.** Nothing is summed
here. The legacy groupings —

    commercial     = Producer/Merchant + Swap Dealers
    non-commercial = Managed Money + Other Reportables

— are derived at read time in analytics/features.py. Summing them at write
time (which this module used to do) both persisted a computed value and threw
away the split that carries the signal: a producer short is a miner selling
future output, while a swap-dealer short is usually a bank hedging investor
length, and the two mean opposite things for price.

Records are validated here, before they leave the source: a week whose total
long or short positions exceed twice the open interest is not a state real
COT data can be in — it means a field-mapping error (e.g. CFTC renamed an API
field), and it is logged and dropped rather than stored. Deliberately loose:
it exists to catch upstream breakage, not to second-guess CFTC.
"""

import logging
from datetime import datetime, timedelta

from ..config import ASSETS, HISTORY_DAYS
from ..domain import CotWeek
from .http import get_json

logger = logging.getLogger(__name__)

API_URL = 'https://publicreporting.cftc.gov/resource/kh3c-gbw2.json'


def _field(record: dict, *candidates: str) -> int:
    """Read a numeric field trying several name variants — the dataset is
    inconsistent about '_all' suffixes and even contains double underscores
    (e.g. 'swap__positions_short_all' next to 'prod_merc_positions_long')."""
    for name in candidates:
        if name in record and record[name] is not None:
            return int(float(record[name]))
    raise KeyError(f"None of {candidates} present in CFTC record {record.get('id')}")


def map_record(api_record: dict, asset: str) -> CotWeek:
    prod_long   = _field(api_record, 'prod_merc_positions_long', 'prod_merc_positions_long_all')
    prod_short  = _field(api_record, 'prod_merc_positions_short', 'prod_merc_positions_short_all')
    swap_long   = _field(api_record, 'swap_positions_long_all', 'swap_positions_long')
    swap_short  = _field(api_record, 'swap__positions_short_all', 'swap_positions_short_all', 'swap_positions_short')
    swap_spread = _field(api_record, 'swap__positions_spread_all', 'swap_positions_spread_all', 'swap_positions_spread')
    mm_long     = _field(api_record, 'm_money_positions_long_all', 'm_money_positions_long')
    mm_short    = _field(api_record, 'm_money_positions_short_all', 'm_money_positions_short')
    mm_spread   = _field(api_record, 'm_money_positions_spread', 'm_money_positions_spread_all')
    oth_long    = _field(api_record, 'other_rept_positions_long', 'other_rept_positions_long_all')
    oth_short   = _field(api_record, 'other_rept_positions_short', 'other_rept_positions_short_all')
    oth_spread  = _field(api_record, 'other_rept_positions_spread', 'other_rept_positions_spread_all')
    nr_long     = _field(api_record, 'nonrept_positions_long_all', 'nonrept_positions_long')
    nr_short    = _field(api_record, 'nonrept_positions_short_all', 'nonrept_positions_short')
    oi          = _field(api_record, 'open_interest_all', 'open_interest')

    return CotWeek(
        asset=asset,
        report_date=api_record['report_date_as_yyyy_mm_dd'][:10],
        prod_long=prod_long, prod_short=prod_short,
        swap_long=swap_long, swap_short=swap_short, swap_spreading=swap_spread,
        mm_long=mm_long, mm_short=mm_short, mm_spreading=mm_spread,
        other_long=oth_long, other_short=oth_short, other_spreading=oth_spread,
        nonreportable_long=nr_long,
        nonreportable_short=nr_short,
        open_interest=oi,
    )


def validate(week: CotWeek) -> str | None:
    """Returns a rejection reason, or None when the week is sane."""
    values = [week.prod_long, week.prod_short,
              week.swap_long, week.swap_short, week.swap_spreading,
              week.mm_long, week.mm_short, week.mm_spreading,
              week.other_long, week.other_short, week.other_spreading,
              week.nonreportable_long, week.nonreportable_short, week.open_interest]
    if any(v < 0 for v in values):
        return "negative position value"
    total_long = (week.prod_long + week.swap_long + week.mm_long
                  + week.other_long + week.nonreportable_long)
    total_short = (week.prod_short + week.swap_short + week.mm_short
                   + week.other_short + week.nonreportable_short)
    if total_long > week.open_interest * 2 or total_short > week.open_interest * 2:
        return (f"total positions ({total_long}/{total_short}) inconsistent "
                f"with OI ({week.open_interest})")
    return None


def fetch_cot(asset: str, since: str | None) -> list[CotWeek]:
    """All valid weeks for one asset from `since` (exclusive of nothing —
    re-fetching the boundary week is a harmless upsert). since=None →
    HISTORY_DAYS backfill."""
    spec = ASSETS[asset]
    if since is None:
        since = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')

    api_records = get_json(API_URL, params={
        '$where': (f"cftc_contract_market_code='{spec['cftc_code']}' "
                   f"AND report_date_as_yyyy_mm_dd>='{since}T00:00:00.000'"),
        '$order': 'report_date_as_yyyy_mm_dd ASC',
        '$limit': '5000',
    })

    weeks = []
    for api_record in api_records:
        try:
            week = map_record(api_record, asset)
        except Exception as e:
            logger.error(f"{asset}: could not map CFTC record: {e}")
            continue
        reason = validate(week)
        if reason:
            logger.error(f"{asset}: invalid COT record ({week.report_date}): {reason}")
            continue
        weeks.append(week)
    return weeks
