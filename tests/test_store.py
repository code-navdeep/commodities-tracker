"""Store contract: schema, revision bumps, upsert semantics, reads."""

import pandas as pd
import pytest

from commodities.domain import (CotWeek, DeliveryNotice, FirmLine, FuturesBar,
                                OptionChain, OptionQuote, PriceSeries)
from commodities.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'test.db')
    s.init()
    return s


def _cot_week(date='2026-08-04', oi=1000):
    """prod 100/200, swap 300/50 (spread 10), mm 40/30 (spread 5),
    other 20/15 (spread 5), non-reportable 12/8."""
    return CotWeek('GC', date,
                   prod_long=100, prod_short=200,
                   swap_long=300, swap_short=50, swap_spreading=10,
                   mm_long=40, mm_short=30, mm_spreading=5,
                   other_long=20, other_short=15, other_spreading=5,
                   nonreportable_long=12, nonreportable_short=8,
                   open_interest=oi)


def _bar(date='2026-08-10', settle=4361.8):
    return FuturesBar('GC', date, '2026-12', open=4350.0, high=4380.0,
                      low=4340.0, last=4360.5, settle=settle,
                      volume=1758, open_interest=2811)


def _notice(intent='2026-08-10'):
    return DeliveryNotice(
        asset='GC', contract_month='2026-08', intent_date=intent,
        delivery_date='2026-08-12', total_issued=100, total_stopped=100,
        month_to_date=5000, settlement=4340.7,
        firms=(FirmLine('661', 'C', 'JP MORGAN SECURITIES', 0, 60),
               FirmLine('661', 'H', 'JP MORGAN SECURITIES', 40, 0),
               FirmLine('905', 'C', 'ADM', 60, 40)))


def _chain(data_date='2026-08-10'):
    return OptionChain(
        asset='GC', expiry_month='2026-09', expiry_date='2026-08-26',
        data_date=data_date, underlying=4371.4, rate=0.037,
        quotes=(OptionQuote(4300.0, 'Call', 134.0, None, None, None, 36, 3023),
                OptionQuote(4300.0, 'Put', 47.1, None, None, None, 278, 1400)))


def test_init_is_idempotent_and_revision_starts_at_zero(store):
    store.init()
    assert store.revision() == 0


def test_each_write_batch_bumps_revision_once(store):
    store.put_cot([_cot_week(), _cot_week('2026-08-11')])
    assert store.revision() == 1
    store.put_delivery([_notice()])
    assert store.revision() == 2
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-10': 4324.45})])
    assert store.revision() == 3
    store.put_chain(_chain())
    assert store.revision() == 4


def test_empty_writes_do_not_bump_revision(store):
    store.put_cot([])
    store.put_delivery([])
    store.put_prices([])
    assert store.revision() == 0


def test_cot_upsert_overwrites_same_week(store):
    store.put_cot([_cot_week(oi=1000)])
    store.put_cot([_cot_week(oi=2000)])
    df = store.cot('GC')
    assert len(df) == 1
    assert df.iloc[0]['open_interest'] == 2000


def test_delivery_skips_existing_reports_entirely(store):
    assert store.put_delivery([_notice()]) == 1
    revision = store.revision()
    assert store.put_delivery([_notice()]) == 0        # same natural key → skipped
    assert store.revision() == revision                # nothing written → no bump
    reports, firms = store.delivery('GC')
    assert len(reports) == 1
    assert len(firms) == 3                             # incl. both 661 H and C rows


def test_house_and_customer_rows_coexist(store):
    store.put_delivery([_notice()])
    _, firms = store.delivery('GC')
    org_codes = set(firms[firms['firm_code'] == '661']['org_code'])
    assert org_codes == {'C', 'H'}


def test_price_columns_create_rows_but_macro_only_fills_existing(store):
    written = store.put_prices([
        PriceSeries('GC', 'spot', {'2026-08-10': 4324.45}),
        PriceSeries('GC', 'vix', {'2026-08-10': 15.46, '2026-08-11': 15.5}),
    ])
    # spot creates the 08-10 row; vix fills it; vix on 08-11 has no row → dropped
    assert written == 2
    df = store.prices('GC')
    assert len(df) == 1
    assert df.iloc[0]['vix'] == 15.46


def test_price_overwrite_touches_only_its_own_column(store):
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-10': 4324.45})])
    store.put_futures([_bar()])
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-10': 4330.00})])
    row = store.prices('GC').iloc[0]
    assert row['spot'] == 4330.00
    assert row['future_close'] == 4361.8        # untouched by the spot rewrite


def test_futures_are_not_a_price_column(store):
    """Futures travel as a whole bar; put_prices must refuse them, so no
    column ever has two write paths."""
    with pytest.raises(ValueError):
        store.put_prices([PriceSeries('GC', 'future_close', {'2026-08-10': 1.0})])


def test_put_futures_round_trip_and_creates_rows(store):
    assert store.put_futures([_bar()]) == 1
    row = store.prices('GC').iloc[0]
    assert row['future_close'] == 4361.8
    assert row['future_high'] == 4380.0
    assert row['future_low'] == 4340.0
    assert row['future_open'] == 4350.0
    assert row['future_last'] == 4360.5
    assert row['future_month'] == '2026-12'
    assert row['future_volume'] == 1758
    assert row['future_oi'] == 2811
    assert row['spot'] is None or pd.isna(row['spot'])


def test_put_futures_upserts_and_preserves_spot(store):
    store.put_futures([_bar()])
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-10': 4324.45})])
    store.put_futures([_bar(settle=4400.0)])
    df = store.prices('GC')
    assert len(df) == 1                          # upsert, not a second row
    assert df.iloc[0]['future_close'] == 4400.0
    assert df.iloc[0]['spot'] == 4324.45         # futures write never clobbers spot


def test_option_dates_and_chains_on(store):
    store.put_chain(_chain('2026-08-10'))
    store.put_chain(_chain('2026-08-11'))
    assert store.option_dates('GC') == ['2026-08-10', '2026-08-11']
    day = store.chains_on('GC', '2026-08-11')
    assert len(day) == 2
    assert set(day['expiry_month']) == {'2026-09'}


def test_chain_replace_and_reads(store):
    store.put_chain(_chain())
    store.put_chain(_chain())                          # replace, not duplicate
    df = store.chain('GC', '2026-09', '2026-08-10')
    assert len(df) == 2
    assert store.expiries('GC') == ['2026-09']
    assert store.chain_dates('GC', '2026-09') == ['2026-08-10']
    assert store.has_chain('GC', '2026-09', '2026-08-10')
    assert not store.has_chain('GC', '2026-09', '2026-08-11')


def test_high_water_reads(store):
    assert store.latest_cot_date('GC') is None
    store.put_cot([_cot_week('2026-07-28'), _cot_week('2026-08-04')])
    assert store.latest_cot_date('GC') == '2026-08-04'
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-07': 1.0, '2026-08-10': 2.0})])
    assert store.latest_price_date('GC', 'spot') == '2026-08-10'
    assert store.latest_price_date('GC', 'vix') is None       # column with no rows yet


def test_stream_presence(store):
    store.put_cot([_cot_week()])
    presence = store.stream_presence()
    assert presence['cot'] == {'GC'}
    assert presence['options'] == set()


def test_unknown_price_column_rejected(store):
    with pytest.raises(ValueError):
        store.put_prices([PriceSeries('GC', 'bogus', {'2026-08-10': 1.0})])
