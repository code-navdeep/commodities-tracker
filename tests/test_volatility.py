"""Parkinson realized volatility over the stored futures bar.

The scale is load-bearing: realized vol is subtracted from implied vol to make
VRP, so both sides must be annualised decimals. A missing sqrt(252), or log10
instead of ln, would still produce plausible-looking numbers.
"""

import math

import numpy as np
import pandas as pd
import pytest

from commodities.analytics.features import features
from commodities.config import REALIZED_VOL_WINDOWS
from commodities.domain import FuturesBar, PriceSeries
from commodities.store import Store

WINDOW = REALIZED_VOL_WINDOWS[0]          # 20


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'vol.db')
    s.init()
    return s


def _dates(n: int) -> list:
    """n consecutive weekdays as ISO strings."""
    out, day = [], pd.Timestamp('2026-01-05')      # a Monday
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime('%Y-%m-%d'))
        day += pd.Timedelta(days=1)
    return out


def _load(store, dates, high, low, settle=100.0):
    store.put_futures([FuturesBar('GC', d, '2026-03', open=settle, high=high,
                                  low=low, last=settle, settle=settle,
                                  volume=1, open_interest=1) for d in dates])
    store.put_prices([PriceSeries('GC', 'spot', {d: settle for d in dates})])


def _parkinson_reference(high: float, low: float) -> float:
    """Written out longhand, deliberately not the vectorised implementation."""
    daily_variance = math.log(high / low) ** 2 / (4 * math.log(2))
    return math.sqrt(daily_variance * 252)


def test_matches_reference_value(store):
    dates = _dates(WINDOW + 5)
    _load(store, dates, high=110.0, low=100.0)
    df = features(store, 'GC')
    expected = _parkinson_reference(110.0, 100.0)
    assert df[f'realized_vol_{WINDOW}'].dropna().iloc[-1] == pytest.approx(expected)
    # sanity on the magnitude: a 10% daily range is a very volatile market
    assert 0.85 < expected < 0.95


def test_nan_until_the_window_fills(store):
    dates = _dates(WINDOW + 3)
    _load(store, dates, high=101.0, low=100.0)
    series = features(store, 'GC').set_index('date')[f'realized_vol_{WINDOW}']
    assert series.iloc[:WINDOW - 1].isna().all()     # not enough bars yet
    assert series.iloc[WINDOW - 1:].notna().all()


def test_wider_range_means_higher_volatility(store, tmp_path):
    dates = _dates(WINDOW + 2)
    _load(store, dates, high=101.0, low=100.0)
    calm = features(store, 'GC')[f'realized_vol_{WINDOW}'].dropna().iloc[-1]

    other = Store(tmp_path / 'vol2.db')
    other.init()
    _load(other, dates, high=120.0, low=100.0)
    wild = features(other, 'GC')[f'realized_vol_{WINDOW}'].dropna().iloc[-1]

    assert wild > calm * 5


def test_absent_ohlc_yields_no_column_rather_than_zeros(store):
    """Rows predating the futures-bar change have no high/low. They must stay
    NaN — never fall back to close-to-close, which the front-month roll makes
    unusable."""
    dates = _dates(WINDOW + 2)
    store.put_prices([PriceSeries('GC', 'spot', {d: 100.0 for d in dates})])
    df = features(store, 'GC')
    assert f'realized_vol_{WINDOW}' not in df.columns


def test_survives_bars_with_no_high_low(store):
    """CME publishes a settle with '-' for high/low when the front month did
    not trade, so some real bars have OHLC missing. Those days must drop out
    of the window rather than poison it."""
    dates = _dates(WINDOW + 5)
    bars = []
    for i, d in enumerate(dates):
        blank = (i == 3)
        bars.append(FuturesBar('GC', d, '2026-03', open=100.0,
                               high=None if blank else 110.0,
                               low=None if blank else 100.0,
                               last=100.0, settle=100.0,
                               volume=0 if blank else 10, open_interest=1))
    store.put_futures(bars)
    store.put_prices([PriceSeries('GC', 'spot', {d: 100.0 for d in dates})])
    series = features(store, 'GC')[f'realized_vol_{WINDOW}']
    assert series.notna().any()
    assert np.isfinite(series.dropna()).all()
    assert (series.dropna() > 0).all()


def test_roll_does_not_contaminate(store):
    """The contract month changing mid-series is a change of instrument, not a
    price move. Parkinson reads one bar at a time, so it must be unaffected."""
    dates = _dates(WINDOW + 2)
    bars = []
    for i, d in enumerate(dates):
        # settle jumps 60 -> 62 at the roll; high/low ranges stay identical
        month, base = ('2026-03', 60.0) if i < WINDOW // 2 else ('2026-04', 62.0)
        bars.append(FuturesBar('GC', d, month, open=base, high=base + 1.0,
                               low=base - 1.0, last=base, settle=base,
                               volume=1, open_interest=1))
    store.put_futures(bars)
    store.put_prices([PriceSeries('GC', 'spot', {d: 60.0 for d in dates})])
    series = features(store, 'GC')[f'realized_vol_{WINDOW}'].dropna()
    # ranges differ only through the base price, so vol stays in a tight band
    assert series.max() / series.min() < 1.1
    assert np.isfinite(series).all()
