"""Price sources — spot, the front-month futures bar, and macro indicators.

All free, no scraping:
  futures — CME settlements for the nearest listed contract month: the whole
      published row (month, OHLC, last, settle, volume, open interest), for
      each of the ~5 recent trade dates CME exposes. Older trade dates return
      nothing, so this history accrues one refresh at a time.
  spot — per the registry: LBMA official JSON (gold PM / silver, USD) or
      EIA's API (WTI Cushing / Henry Hub daily spot, published next business
      day; needs the free EIA_API_KEY — without it those assets are skipped
      with a clear log line). Copper has no clean free source.
  usd_index / vix — Yahoo's chart API, called directly through the shared
      client (no yfinance dependency).

Everything is windowed to HISTORY_DAYS. Functions return values; they never
touch the database.
"""

import csv
import io
import logging
import re
from datetime import datetime, timedelta

from ..config import ASSETS, HISTORY_DAYS
from ..domain import FuturesBar, PriceSeries
from .eia import eia_rows, to_series
from .http import get_json, get_text

logger = logging.getLogger(__name__)

CME_TRADE_DATES_URL = 'https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/TradeDate/{fut_id}'
CME_SETTLEMENTS_URL = 'https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/Settlements/{fut_id}/FUT'
LBMA_URL = 'https://prices.lbma.org.uk/json/{series}.json'
FRED_CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
YAHOO_CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'

LBMA_SERIES = {'lbma_gold': 'gold_pm', 'lbma_silver': 'silver'}
MACRO_SYMBOLS = {'usd_index': 'DX-Y.NYB', 'vix': '^VIX'}

# CME contract-month codes. Lives here because both this module and
# sources/options.py parse them, and options.py already imports from here.
MONTH_ABBREV = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}


def parse_number(value):
    """CME numbers arrive as strings, possibly with thousands separators or
    bid/ask marker suffixes ('4346.4A'); '-' means no value."""
    if value is None:
        return None
    text = re.sub(r'[A-Za-z]+$', '', str(value).replace(',', '').strip())
    if not text or text in ('-', '.'):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _cutoff() -> str:
    return (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')


def _contract_month(raw) -> str | None:
    """CME's 'SEP 26' -> '2026-09'."""
    match = re.match(r'([A-Z]{3})\s+(\d{2})$', str(raw or '').strip().upper())
    if not match:
        return None
    month = MONTH_ABBREV.get(match.group(1))
    return f"{2000 + int(match.group(2))}-{month:02d}" if month else None


def fetch_futures(asset: str) -> list[FuturesBar]:
    """Front-month settlement bars for every trade date CME currently exposes.

    CME serves a rolling ~5-business-day window and nothing older (an older
    tradeDate is accepted and echoed back but returns zero rows), so history
    accrues one refresh at a time and a gap beyond that window is permanent.

    Every offered date is re-fetched and re-upserted, deliberately: that is
    what lets a missed day inside the window heal itself on the next refresh.
    Do NOT add a store high-water 'since' skip here — the inventory stream has
    one for an unrelated reason (bandwidth over a 2,300-row series), and
    copying it to this function would silently remove that recovery.
    """
    fut_id = ASSETS[asset]['fut_id']
    bars = []
    for entry in get_json(CME_TRADE_DATES_URL.format(fut_id=fut_id)):
        date_us = entry[0] if isinstance(entry, (list, tuple)) else entry
        try:
            iso_date = datetime.strptime(date_us, '%m/%d/%Y').strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        try:
            data = get_json(CME_SETTLEMENTS_URL.format(fut_id=fut_id), params={
                'strategy': 'DEFAULT', 'tradeDate': date_us, 'pageSize': '500'})
        except Exception as e:
            logger.warning(f"CME settlements failed (product {fut_id}, {date_us}): {e}")
            continue
        # Months come nearest-first; front month = first with a real settle.
        for row in data.get('settlements', []):
            settle = parse_number(row.get('settle'))
            if not settle:
                continue
            bars.append(FuturesBar(
                asset=asset,
                date=iso_date,
                contract_month=_contract_month(row.get('month')),
                open=parse_number(row.get('open')),
                high=parse_number(row.get('high')),
                low=parse_number(row.get('low')),
                last=parse_number(row.get('last')),
                settle=settle,
                volume=int(parse_number(row.get('volume')) or 0),
                open_interest=int(parse_number(row.get('openInterest')) or 0),
            ))
            break
    return bars


def fetch_lbma(series: str) -> dict:
    """{iso_date: usd} — LBMA entry 'v' is [USD, GBP, EUR]."""
    cutoff = _cutoff()
    prices = {}
    for entry in get_json(LBMA_URL.format(series=series)):
        date, values = entry.get('d'), entry.get('v') or []
        if date and date >= cutoff and values and values[0]:
            prices[date] = float(values[0])
    return prices


def fetch_eia(series: str) -> dict:
    """{iso_date: value} for a daily EIA spot series, windowed to
    HISTORY_DAYS. Empty (with a loud log) when EIA_API_KEY is unset — see
    sources/eia.py for the series-ID and windowing conventions."""
    cutoff = _cutoff()
    return to_series(eia_rows(series, since=cutoff), cutoff=cutoff)


def fetch_fred(series: str) -> dict:
    """{iso_date: value} from FRED's keyless CSV ('.' = missing). Used for
    the DTB3 risk-free rate (options source)."""
    cutoff = _cutoff()
    reader = csv.reader(io.StringIO(get_text(FRED_CSV_URL.format(series=series))))
    next(reader, None)
    values = {}
    for row in reader:
        if len(row) >= 2 and row[0] >= cutoff and row[1] not in ('.', ''):
            try:
                values[row[0]] = float(row[1])
            except ValueError:
                continue
    return values


def fetch_yahoo_daily(symbol: str) -> dict:
    """{iso_date: close} from Yahoo's chart API (2y daily)."""
    payload = get_json(YAHOO_CHART_URL.format(symbol=symbol),
                       params={'range': '2y', 'interval': '1d'})
    result = payload.get('chart', {}).get('result') or [{}]
    timestamps = result[0].get('timestamp') or []
    closes = (result[0].get('indicators', {}).get('quote') or [{}])[0].get('close') or []
    values = {}
    for ts, close in zip(timestamps, closes):
        if close is not None:
            values[datetime.fromtimestamp(ts).strftime('%Y-%m-%d')] = float(close)
    return values


def fetch_spot(asset: str) -> dict:
    source = ASSETS[asset]['spot']
    if source is None:
        return {}
    if source in LBMA_SERIES:
        return fetch_lbma(LBMA_SERIES[source])
    if source.startswith('eia:'):
        return fetch_eia(source.split(':', 1)[1])
    raise ValueError(f"Unknown spot source '{source}'")


def fetch_prices(asset: str, macro: dict | None = None) -> list[PriceSeries]:
    """All price series for one asset. `macro` ({column: series}) is fetched
    once per refresh run and passed in, since it's asset-independent."""
    series = [PriceSeries(asset, 'spot', fetch_spot(asset))]
    for column, values in (macro or {}).items():
        series.append(PriceSeries(asset, column, values))
    return series


def fetch_macro() -> dict:
    """{column: {iso_date: value}} — DXY and VIX, shared across assets."""
    macro = {}
    for column, symbol in MACRO_SYMBOLS.items():
        try:
            macro[column] = fetch_yahoo_daily(symbol)
        except Exception as e:
            logger.warning(f"Macro fetch failed for {symbol}: {e}")
    return macro
