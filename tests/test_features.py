"""Feature engineering semantics over a constructed store."""

import numpy as np
import pandas as pd
import pytest

from commodities.analytics.features import features, firm_hhi
from commodities.domain import (CotWeek, DeliveryNotice, FirmLine, FuturesBar,
                                PriceSeries)
from commodities.store import Store


def _notice(intent: str, issued: int, stopped: int, mtd: int = 1000):
    return DeliveryNotice(
        asset='GC', contract_month=intent[:7], intent_date=intent,
        delivery_date=intent, total_issued=issued, total_stopped=stopped,
        month_to_date=mtd, settlement=None,
        firms=(FirmLine('661', 'C', 'JPM', issued, stopped),))


def _cot(date: str, comm_long=100, comm_short=40):
    """Totals are what the assertions read, but they are split unevenly across
    producer and swap dealer so features.py's aggregation is really exercised —
    a wrong pairing would show up rather than cancel out."""
    prod_long, swap_long = comm_long - comm_long // 3, comm_long // 3
    prod_short, swap_short = comm_short - comm_short // 4, comm_short // 4
    return CotWeek('GC', date,
                   prod_long=prod_long, prod_short=prod_short,
                   swap_long=swap_long, swap_short=swap_short, swap_spreading=5,
                   mm_long=80, mm_short=60, mm_spreading=10,
                   other_long=0, other_short=0, other_spreading=0,
                   nonreportable_long=20, nonreportable_short=15,
                   open_interest=400)


def _bar(date: str, settle: float, high: float = None, low: float = None):
    return FuturesBar('GC', date, '2026-12', open=settle, high=high or settle,
                      low=low or settle, last=settle, settle=settle,
                      volume=100, open_interest=1000)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'features.db')
    s.init()
    return s


def test_net_flow_and_targets(store):
    store.put_delivery([_notice('2026-08-04', 100, 150), _notice('2026-08-05', 80, 30)])
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-04': 100.0, '2026-08-05': 110.0})])
    store.put_futures([_bar('2026-08-04', 102.0)])
    df = features(store, 'GC')
    assert list(df['net_flow']) == [50, -50]
    assert df.iloc[0]['future_return_1d'] == pytest.approx(0.10)
    assert df.iloc[0]['future_direction_1d'] == 1


def test_basis_nan_when_either_leg_missing(store):
    store.put_delivery([_notice('2026-08-04', 1, 1)])
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-04': 100.0})])   # no future_close
    df = features(store, 'GC')
    assert np.isnan(df.iloc[0]['basis'])


def test_basis_when_both_legs_present(store):
    store.put_delivery([_notice('2026-08-04', 1, 1)])
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-04': 100.0})])
    store.put_futures([_bar('2026-08-04', 103.0)])
    df = features(store, 'GC')
    assert df.iloc[0]['basis'] == pytest.approx(3.0)
    assert df.iloc[0]['basis_pct'] == pytest.approx(3.0)


def test_cot_joined_backward_and_staleness_nulled(store):
    store.put_cot([_cot('2026-08-04')])
    store.put_delivery([
        _notice('2026-08-05', 1, 1),    # 1 day after report → joined
        _notice('2026-08-25', 1, 1),    # 21 days after → stale, nulled
    ])
    df = features(store, 'GC').set_index('date')
    assert df.loc[pd.Timestamp('2026-08-05'), 'comm_net_long'] == 60
    assert np.isnan(df.loc[pd.Timestamp('2026-08-25'), 'comm_long'])
    assert df.loc[pd.Timestamp('2026-08-25'), 'has_cot'] == 0


def test_legacy_aggregates_are_derived_from_categories(store):
    """comm_* no longer exists in the store — features.py rebuilds it, and it
    must equal producer + swap exactly."""
    store.put_cot([_cot('2026-08-04', comm_long=100, comm_short=40)])
    store.put_delivery([_notice('2026-08-05', 1, 1)])
    row = features(store, 'GC').set_index('date').loc[pd.Timestamp('2026-08-05')]
    assert row['comm_long'] == row['prod_long'] + row['swap_long'] == 100
    assert row['comm_short'] == row['prod_short'] + row['swap_short'] == 40
    assert row['noncomm_long'] == row['mm_long'] + row['other_long'] == 80
    assert row['comm_spreading'] == row['swap_spreading'] == 5
    assert row['comm_net_long'] == 60


def test_category_nets_present(store):
    store.put_cot([_cot('2026-08-04', comm_long=100, comm_short=40)])
    store.put_delivery([_notice('2026-08-05', 1, 1)])
    row = features(store, 'GC').set_index('date').loc[pd.Timestamp('2026-08-05')]
    # 100 -> prod 67 / swap 33 ; 40 -> prod 30 / swap 10
    assert row['producer_net'] == 67 - 30
    assert row['swap_net'] == 33 - 10
    assert row['mm_net'] == 80 - 60
    assert row['producer_hedging_ratio'] == pytest.approx(30 / 400 * 100)


def test_staleness_nulls_derived_categories_too(store):
    """A stale report must take its derived columns with it — otherwise a
    three-month-old producer_net renders as current."""
    store.put_cot([_cot('2026-08-04')])
    store.put_delivery([_notice('2026-08-25', 1, 1)])       # 21 days later
    row = features(store, 'GC').set_index('date').loc[pd.Timestamp('2026-08-25')]
    for column in ('prod_long', 'swap_long', 'producer_net', 'swap_net',
                   'mm_net', 'producer_hedging_ratio'):
        assert pd.isna(row[column]), f'{column} survived staleness nulling'


def test_comm_net_ratio_nan_on_zero_denominator(store):
    store.put_cot([_cot('2026-08-04', comm_long=0, comm_short=0)])
    store.put_delivery([_notice('2026-08-05', 1, 1)])
    df = features(store, 'GC').set_index('date')
    assert np.isnan(df.loc[pd.Timestamp('2026-08-05'), 'comm_net_ratio'])


def test_union_axis_keeps_dense_history_with_sparse_delivery(store):
    """The axis is price ∪ delivery ∪ COT dates: one delivery day must not
    collapse two weeks of price history (the predecessor app's defect)."""
    price_days = {f'2026-08-{day:02d}': 100.0 + day for day in range(3, 15)}
    store.put_prices([PriceSeries('GC', 'spot', price_days)])
    store.put_delivery([_notice('2026-08-10', 5, 25)])
    df = features(store, 'GC')
    assert len(df) == len(price_days)                      # full daily axis
    by_date = df.set_index('date')
    assert by_date.loc[pd.Timestamp('2026-08-10'), 'net_flow'] == 20
    assert np.isnan(by_date.loc[pd.Timestamp('2026-08-11'), 'net_flow'])
    # price features exist across the whole axis, not just delivery days
    assert by_date['price_return_1d'].notna().sum() >= len(price_days) - 1


def test_cot_only_mode_uses_report_dates(store):
    store.put_cot([_cot('2026-07-28'), _cot('2026-08-04')])
    df = features(store, 'GC')
    assert len(df) == 2
    assert 'net_flow' not in df.columns
    assert df.iloc[-1]['comm_net_long'] == 60


def test_empty_store_yields_empty_frame(store):
    assert features(store, 'GC').empty


def test_cache_hits_until_revision_changes(store):
    store.put_cot([_cot('2026-08-04')])
    first = features(store, 'GC')
    assert features(store, 'GC') is first            # same revision → same object
    store.put_cot([_cot('2026-08-11')])
    assert features(store, 'GC') is not first        # revision bumped → recomputed


def test_firm_hhi_single_firm_is_one(store):
    store.put_delivery([_notice('2026-08-04', 10, 10)])
    hhi = firm_hhi(store, 'GC')
    assert hhi.iloc[0] == pytest.approx(1.0)
