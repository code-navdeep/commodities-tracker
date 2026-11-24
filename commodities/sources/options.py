"""Options source — CME settlement chains.

Per asset: the standard monthly option product's expirations (weeklies,
dailies, and exotics are separate CME products and never fetched), capped at
OPTIONS_MAX_MONTHS_AHEAD calendar months; each contract's last-trade date
from the product calendar; and, for each (expiration, trade date) the caller
hasn't stored yet, the full settlement chain — strike, side, settle, OHLC,
volume, open interest. Rows with neither open interest nor volume are
dropped, as is CME's 'Total' row.

Underlying = the first futures month >= the option's contract month from the
same trade date's futures settlements (serial options exercise into the next
listed future). Rate = latest 3-month T-bill (FRED DTB3) at/before the trade
date, decimal. IV/Greeks are NOT computed here — they are read-time
analytics over these stored facts.

The chain endpoint's parameter format (monthYear='U26',
optionExpiration='OGU26') was reverse-engineered from CME's own JS bundle
and verified live.
"""

import logging
import re
from datetime import datetime

from ..config import ASSETS, OPTIONS_MAX_MONTHS_AHEAD
from ..domain import OptionChain, OptionQuote
from .http import get_json
from .prices import MONTH_ABBREV, fetch_fred, parse_number

logger = logging.getLogger(__name__)

EXPIRATIONS_URL = 'https://www.cmegroup.com/CmeWS/mvc/Settlements/Options/TradeDateAndExpirations/{opt_id}'
CALENDAR_URL = 'https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/Options/{opt_id}'
CHAIN_URL = 'https://www.cmegroup.com/CmeWS/mvc/Settlements/Options/Settlements/{opt_id}/OOF'
FUT_SETTLEMENTS_URL = 'https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/Settlements/{fut_id}/FUT'


def _expiration_cap() -> tuple:
    """(year, month) of the furthest expiration to fetch."""
    today = datetime.now()
    total = today.year * 12 + (today.month - 1) + OPTIONS_MAX_MONTHS_AHEAD
    return total // 12, total % 12 + 1


def _fetch_expirations(opt_id: int) -> list:
    """[(year, month, contract_id, ticker, [trade_datetimes])] — standard
    monthly product only, within the cap."""
    cap = _expiration_cap()
    expirations = []
    for group in get_json(EXPIRATIONS_URL.format(opt_id=opt_id)):
        for exp in group.get('expirations', []):
            if exp.get('productId') != opt_id:
                continue
            code = exp.get('expiration', {})
            year, month = code.get('year'), code.get('month')
            contract_id = exp.get('contractId')
            ticker = code.get('twoDigitsCode') or code.get('tickerCode')
            if not (year and month and contract_id and ticker):
                continue
            if (year, month) > cap:
                continue
            trade_dates = []
            for td in exp.get('tradeDates', []):
                try:
                    trade_dates.append(datetime.strptime(td.get('formatedDate'), '%m/%d/%Y'))
                except (ValueError, TypeError):
                    continue
            expirations.append((year, month, contract_id, ticker, trade_dates))
    return expirations


def _fetch_expiry_dates(opt_id: int) -> dict:
    """{contract_id: last_trade_datetime} from the options product calendar."""
    expiry_dates = {}
    try:
        groups = get_json(CALENDAR_URL.format(opt_id=opt_id))
    except Exception as e:
        logger.warning(f"Options calendar failed (product {opt_id}): {e}")
        return expiry_dates
    for group in groups:
        for entry in group.get('calendarEntries', []):
            code, last_trade = entry.get('productCode'), entry.get('lastTrade')
            if code and last_trade:
                try:
                    expiry_dates[code] = datetime.strptime(last_trade, '%d %b %Y')
                except ValueError:
                    continue
    return expiry_dates


def _fetch_futures_months(fut_id: int, trade_date_us: str) -> list:
    """[((year, month), settle)] nearest-first for one trade date."""
    data = get_json(FUT_SETTLEMENTS_URL.format(fut_id=fut_id), params={
        'strategy': 'DEFAULT', 'tradeDate': trade_date_us, 'pageSize': '500'})
    months = []
    for row in data.get('settlements', []):
        match = re.match(r'([A-Z]{3})\s+(\d{2})$', str(row.get('month', '')).strip().upper())
        if not match:
            continue
        month_num = MONTH_ABBREV.get(match.group(1))
        settle = parse_number(row.get('settle'))
        if month_num and settle:
            months.append(((2000 + int(match.group(2)), month_num), settle))
    return months


def _underlying_for(year: int, month: int, futures_months: list):
    for (fy, fm), settle in futures_months:
        if (fy, fm) >= (year, month):
            return settle
    return futures_months[-1][1] if futures_months else None


def _rate_for(date_iso: str, rates: dict) -> float:
    candidates = [d for d in rates if d <= date_iso]
    return (rates[max(candidates)] / 100.0) if candidates else 0.0


def fetch_options(asset: str, skip=None) -> list[OptionChain]:
    """New chains for one asset. `skip(expiry_month, data_date) -> bool` lets
    the caller exclude already-stored snapshots (incremental fetching)."""
    spec = ASSETS[asset]
    opt_id, fut_id = spec['opt_id'], spec['fut_id']
    skip = skip or (lambda expiry_month, data_date: False)

    expirations = _fetch_expirations(opt_id)
    if not expirations:
        return []
    expiry_dates = _fetch_expiry_dates(opt_id)
    rates = fetch_fred('DTB3')
    futures_cache = {}
    chains = []

    for year, month, contract_id, ticker, trade_dates in expirations:
        expiry_month = f"{year}-{month:02d}"
        expiry_dt = expiry_dates.get(contract_id)

        for trade_dt in trade_dates:
            data_date = trade_dt.strftime('%Y-%m-%d')
            if skip(expiry_month, data_date):
                continue
            trade_date_us = trade_dt.strftime('%m/%d/%Y')

            try:
                payload = get_json(CHAIN_URL.format(opt_id=opt_id), params={
                    'strategy': 'DEFAULT', 'optionProductId': str(opt_id),
                    'monthYear': ticker, 'optionExpiration': contract_id,
                    'tradeDate': trade_date_us, 'pageSize': '500'})
            except Exception as e:
                logger.warning(f"{asset}: chain failed ({contract_id} @ {data_date}): {e}")
                continue

            quotes = []
            for raw in payload.get('settlements', []):
                if str(raw.get('strike', '')).strip().lower() == 'total':
                    continue
                strike = parse_number(raw.get('strike'))
                side = raw.get('type')
                volume = int(parse_number(raw.get('volume')) or 0)
                open_interest = int(parse_number(raw.get('openInterest')) or 0)
                if strike is None or side not in ('Call', 'Put'):
                    continue
                if volume == 0 and open_interest == 0:
                    continue
                quotes.append(OptionQuote(
                    strike=strike, side=side,
                    settle=parse_number(raw.get('settle')),
                    open=parse_number(raw.get('open')),
                    high=parse_number(raw.get('high')),
                    low=parse_number(raw.get('low')),
                    volume=volume, open_interest=open_interest))
            if not quotes:
                continue

            if trade_date_us not in futures_cache:
                try:
                    futures_cache[trade_date_us] = _fetch_futures_months(fut_id, trade_date_us)
                except Exception as e:
                    logger.warning(f"{asset}: futures settlements failed ({data_date}): {e}")
                    futures_cache[trade_date_us] = []

            chains.append(OptionChain(
                asset=asset, expiry_month=expiry_month,
                expiry_date=expiry_dt.strftime('%Y-%m-%d') if expiry_dt else None,
                data_date=data_date,
                underlying=_underlying_for(year, month, futures_cache[trade_date_us]),
                rate=_rate_for(data_date, rates),
                quotes=tuple(quotes)))
    return chains
