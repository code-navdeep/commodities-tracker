"""View smoke tests — the modules import, and the new figures actually build.

Nothing else in the suite imports views/, so without this a NameError or a
stale column reference in a chart only shows up when the dashboard is opened.
"""

import numpy as np
import pandas as pd
import pytest

from commodities.analytics.features import features
from commodities.domain import (CotWeek, FuturesBar, OptionChain, OptionQuote,
                                PriceSeries)
from commodities.sources.greeks import black76_price
from commodities.store import Store
from commodities.views import delivery_cot, options

UNDERLYING = 100.0
STRIKES = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]


def _weekdays(n, start='2026-01-05'):
    out, day = [], pd.Timestamp(start)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime('%Y-%m-%d'))
        day += pd.Timedelta(days=1)
    return out


def _chain(data_date, expiry_month, expiry_date, sigma):
    T = (pd.Timestamp(expiry_date) - pd.Timestamp(data_date)).days / 365.0
    quotes = [
        OptionQuote(strike, side,
                    round(black76_price(side, UNDERLYING, strike, T, 0.04, sigma), 4),
                    None, None, None, 10, 100)
        for strike in STRIKES for side in ('Call', 'Put')
    ]
    return OptionChain(asset='GC', expiry_month=expiry_month, expiry_date=expiry_date,
                       data_date=data_date, underlying=UNDERLYING, rate=0.04,
                       quotes=tuple(quotes))


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'views.db')
    s.init()
    dates = _weekdays(40)

    s.put_prices([PriceSeries('GC', 'spot', {d: UNDERLYING for d in dates}),
                  PriceSeries('GC', 'vix', {d: 18.0 for d in dates})])
    s.put_futures([FuturesBar('GC', d, '2026-03', open=100.0, high=103.0, low=98.0,
                              last=100.0, settle=100.0, volume=10, open_interest=100)
                   for d in dates])
    s.put_cot([CotWeek('GC', d,
                       prod_long=100 + i, prod_short=200 - i,
                       swap_long=300, swap_short=50 + i, swap_spreading=10,
                       mm_long=40, mm_short=30, mm_spreading=5,
                       other_long=20, other_short=15, other_spreading=5,
                       nonreportable_long=12, nonreportable_short=8,
                       open_interest=1000)
               for i, d in enumerate(dates[::5])])
    for d in dates[-6:]:
        s.put_chain(_chain(d, '2026-03', '2026-03-20', sigma=0.30))
        s.put_chain(_chain(d, '2026-04', '2026-04-20', sigma=0.25))
    return s


def test_new_cot_and_vol_figures_build(store):
    df = features(store, 'GC')
    assert delivery_cot.fig_producer_vs_swap(df) is not None
    assert delivery_cot.fig_implied_vs_realized(df) is not None


def test_figures_return_none_rather_than_raising_on_missing_columns():
    """Metals with no options, or a fresh DB with no futures bars, must render
    the rest of the page rather than blowing up."""
    empty = pd.DataFrame({'date': pd.to_datetime(['2026-01-05'])})
    assert delivery_cot.fig_producer_vs_swap(empty) is None
    assert delivery_cot.fig_implied_vs_realized(empty) is None
    assert options.fig_options_history(pd.DataFrame()) is None


def test_cot_only_path_includes_the_new_charts(store):
    figs = delivery_cot.cot_only_figures(features(store, 'GC'))
    assert 'producer_vs_swap' in figs
    assert 'implied_vs_realized' in figs


def test_options_view_builds_with_history_outside_the_body(store):
    view = options.build(store, 'GC', '2026-03', store.chain_dates('GC', '2026-03')[0])
    ids = _collect_ids(view)
    assert 'options-body' in ids
    assert 'options-history' in ids


def _collect_ids(component, found=None):
    found = found if found is not None else set()
    component_id = getattr(component, 'id', None)
    if isinstance(component_id, str):
        found.add(component_id)
    children = getattr(component, 'children', None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _collect_ids(child, found)
    elif children is not None:
        _collect_ids(children, found)
    return found
