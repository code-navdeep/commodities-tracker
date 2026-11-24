"""Refresh orchestration — the one place fetching meets the store.

refresh(store, streams) runs the selected streams for every registry asset,
incrementally: each source is pointed at the store's high-water mark so only
missing data is fetched. Failures never propagate — every error lands in the
RefreshReport, per stream, and the rest keeps going.
"""

import logging

from .config import ASSETS, STREAMS
from .domain import RefreshReport, StreamResult
from .sources.cot import fetch_cot
from .sources.delivery import fetch_delivery
from .sources.inventory import fetch_inventory
from .sources.options import fetch_options
from .sources.prices import fetch_futures, fetch_macro, fetch_prices
from .store import Store

logger = logging.getLogger(__name__)


def _refresh_delivery(store: Store) -> StreamResult:
    result = StreamResult()
    try:
        notices = fetch_delivery()
    except Exception as e:
        logger.error(f"delivery: fetch failed: {e}")
        return StreamResult(ok=False, errors=[str(e)])
    written = store.put_delivery(notices)
    for notice in notices:
        result.counts[notice.asset] = result.counts.get(notice.asset, 0) + 1
    result.counts['stored'] = written
    return result


def _refresh_prices(store: Store) -> StreamResult:
    result = StreamResult()
    macro = fetch_macro()
    for asset in ASSETS:
        try:
            # Spot + macro travel as columns; the futures bar travels whole.
            written = store.put_prices(fetch_prices(asset, macro))
            written += store.put_futures(fetch_futures(asset))
            result.counts[asset] = written
        except Exception as e:
            logger.error(f"prices: {asset} failed: {e}")
            result.ok = False
            result.errors.append(f"{asset}: {e}")
    return result


def _refresh_cot(store: Store) -> StreamResult:
    result = StreamResult()
    for asset in ASSETS:
        try:
            weeks = fetch_cot(asset, since=store.latest_cot_date(asset))
            result.counts[asset] = store.put_cot(weeks)
        except Exception as e:
            logger.error(f"cot: {asset} failed: {e}")
            result.ok = False
            result.errors.append(f"{asset}: {e}")
    return result


def _refresh_inventory(store: Store) -> StreamResult:
    result = StreamResult()
    for asset in ASSETS:
        if not ASSETS[asset].get('inventory'):
            continue          # metals have no inventory series configured
        try:
            # Ask the store what it already has, so each series requests only
            # the weeks it is missing (plus a revision overlap).
            points = fetch_inventory(
                asset, since=lambda series, a=asset: store.latest_inventory_period(a, series))
            result.counts[asset] = store.put_inventory(points)
        except Exception as e:
            logger.error(f"inventory: {asset} failed: {e}")
            result.ok = False
            result.errors.append(f"{asset}: {e}")
    return result


def _refresh_options(store: Store) -> StreamResult:
    result = StreamResult()
    for asset in ASSETS:
        try:
            chains = fetch_options(
                asset,
                skip=lambda expiry, date, a=asset: store.has_chain(a, expiry, date))
            result.counts[asset] = sum(store.put_chain(chain) for chain in chains)
        except Exception as e:
            logger.error(f"options: {asset} failed: {e}")
            result.ok = False
            result.errors.append(f"{asset}: {e}")
    return result


_RUNNERS = {
    'delivery': _refresh_delivery,
    'prices': _refresh_prices,
    'cot': _refresh_cot,
    'inventory': _refresh_inventory,
    'options': _refresh_options,
}


def refresh(store: Store, streams: list[str]) -> RefreshReport:
    invalid = [s for s in streams if s not in STREAMS]
    if invalid:
        raise ValueError(f"Unknown streams: {invalid} (valid: {STREAMS})")

    report = RefreshReport()
    for stream in STREAMS:                # canonical run order
        if stream not in streams:
            continue
        logger.info(f"refresh: {stream} starting")
        try:
            report.streams[stream] = _RUNNERS[stream](store)
        except Exception as e:            # belt and braces — runners catch their own
            logger.error(f"refresh: {stream} crashed: {e}", exc_info=True)
            report.streams[stream] = StreamResult(ok=False, errors=[str(e)])
        logger.info(f"refresh: {stream} done — {report.streams[stream].counts}")
    return report
