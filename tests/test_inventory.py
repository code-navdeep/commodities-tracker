"""Inventory: store contract, EIA row mapping, and the seasonal band math."""

import numpy as np
import pandas as pd
import pytest

from commodities.analytics.features import features
from commodities.analytics.inventory import (headline_series, inventory_series,
                                              inventory_summary, series_names,
                                              stored_series)
from commodities.config import INVENTORY_OVERLAP_WEEKS
from commodities.domain import InventoryPoint, PriceSeries
from commodities.sources.eia import MAX_ROWS, to_series, units_of
from commodities.sources.inventory import fetch_inventory, request_plan
from commodities.store import Store

# Shape of a real /v2/seriesid/ response row (captured from EIA).
EIA_ROWS = [
    {'period': '2026-08-07', 'value': '22566', 'units': 'MBBL',
     'series': 'W_EPC0_SAX_YCUOK_MBBL', 'area-name': 'CUSHING, OK'},
    {'period': '2026-07-31', 'value': '20955', 'units': 'MBBL'},
    {'period': '2019-01-04', 'value': '11111', 'units': 'MBBL'},   # outside a recent window
    {'period': '2026-07-24', 'value': None, 'units': 'MBBL'},      # EIA nulls happen
]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'inventory.db')
    s.init()
    return s


def _points(asset, series, values, units='MBBL'):
    return [InventoryPoint(asset, series, period, value, units)
            for period, value in values.items()]


# --- source mapping ---------------------------------------------------------

def test_rows_map_to_values_dropping_nulls():
    values = to_series(EIA_ROWS)
    assert values == {'2026-08-07': 22566.0, '2026-07-31': 20955.0, '2019-01-04': 11111.0}
    assert '2026-07-24' not in values          # null value dropped


def test_client_side_cutoff_windows_the_series():
    # EIA ignores `start` for weekly series, so the cutoff must be applied here
    values = to_series(EIA_ROWS, cutoff='2026-01-01')
    assert set(values) == {'2026-08-07', '2026-07-31'}


def test_units_read_from_the_response():
    assert units_of(EIA_ROWS) == 'MBBL'
    assert units_of([{'period': 'x', 'value': 1}], default='BCF') == 'BCF'


# --- incremental request planning -------------------------------------------

def test_empty_store_asks_for_a_full_backfill():
    rows, keep_from = request_plan(None)
    assert rows == MAX_ROWS
    # window is the multi-year backfill, not a recent slice
    assert keep_from < '2021-01-01'


def test_current_store_asks_for_only_a_handful_of_rows():
    today = pd.Timestamp('2026-08-17').to_pydatetime()
    rows, keep_from = request_plan('2026-08-07', today=today)
    # 10 days missing -> 2 weeks, plus the revision overlap
    assert rows == 2 + INVENTORY_OVERLAP_WEEKS
    assert rows < 10                       # nothing like the 5000-row pull
    # keeps the overlap window so revisions are re-read
    assert keep_from == '2026-07-17'


def test_gap_drives_the_row_count():
    today = pd.Timestamp('2026-08-17').to_pydatetime()
    near = request_plan('2026-08-14', today=today)[0]
    far = request_plan('2026-02-01', today=today)[0]
    assert near < far                       # bigger gap -> bigger request
    assert far < MAX_ROWS                   # but still bounded


def test_row_count_never_exceeds_the_api_cap():
    today = pd.Timestamp('2026-08-17').to_pydatetime()
    # a decades-long gap is still only ~1,900 weeks — under the cap
    assert request_plan('1990-01-01', today=today)[0] < MAX_ROWS
    # only an absurd gap (>96 years of weeks) reaches it, and it clamps
    assert request_plan('1900-01-01', today=today)[0] == MAX_ROWS


def test_fetch_consults_the_high_water_mark(monkeypatch):
    """The bug this fix addresses: `since` existed but was never called."""
    asked = {}

    def fake_eia_rows(series_id, since=None, length=None):
        asked[series_id] = length
        return [{'period': '2026-08-07', 'value': '1', 'units': 'MBBL'}]

    monkeypatch.setattr('commodities.sources.inventory.eia_rows', fake_eia_rows)
    seen = []

    def since(series):
        seen.append(series)
        return '2026-08-07'

    points = fetch_inventory('CL', since=since)
    assert seen, "high-water lookup was never consulted"
    assert all(length < 50 for length in asked.values()), asked
    assert all(p.asset == 'CL' for p in points)


# --- store ------------------------------------------------------------------

def test_put_and_read_inventory(store):
    written = store.put_inventory(_points('CL', 'cushing',
                                          {'2026-08-07': 22566.0, '2026-07-31': 20955.0}))
    assert written == 2
    assert store.revision() == 1
    frame = store.inventory('CL')
    assert len(frame) == 2
    assert set(frame['series']) == {'cushing'}
    assert store.latest_inventory_period('CL', 'cushing') == '2026-08-07'


def test_inventory_upserts_and_filters_by_series(store):
    store.put_inventory(_points('CL', 'cushing', {'2026-08-07': 22566.0}))
    store.put_inventory(_points('CL', 'cushing', {'2026-08-07': 23000.0}))   # revision
    store.put_inventory(_points('CL', 'spr', {'2026-08-07': 298694.0}))
    assert len(store.inventory('CL')) == 2                     # not 3 — upsert, not insert
    assert store.inventory('CL', 'cushing')['value'].iloc[0] == 23000.0
    assert len(store.inventory('CL', 'spr')) == 1


def test_empty_write_does_not_bump_revision(store):
    store.put_inventory([])
    assert store.revision() == 0


def test_stream_presence_includes_inventory(store):
    store.put_inventory(_points('CL', 'cushing', {'2026-08-07': 1.0}))
    assert store.stream_presence()['inventory'] == {'CL'}


# --- registry ---------------------------------------------------------------

def test_registry_series_order_and_metals_have_none():
    assert headline_series('CL') == 'us_crude'      # first entry is the headline
    assert headline_series('NG') == 'working_gas'
    assert 'cushing' in series_names('CL')
    assert series_names('GC') == []
    assert headline_series('GC') is None


# --- seasonal band ----------------------------------------------------------

def _weekly(start: str, weeks: int, value_fn):
    """A weekly series starting at `start`, value_fn(index) -> level."""
    dates = pd.date_range(start, periods=weeks, freq='7D')
    return {d.strftime('%Y-%m-%d'): float(value_fn(i)) for i, d in enumerate(dates)}


def test_band_uses_prior_years_at_the_same_week(store):
    # Six years of a flat-per-year series: each calendar year sits at a known
    # level, so the band over the 5 prior years is exactly predictable.
    values = {}
    for offset, year in enumerate(range(2020, 2027)):
        values.update(_weekly(f'{year}-01-06', 52, lambda i, o=offset: 100 + o * 10))
    store.put_inventory(_points('CL', 'cushing', values))

    frame = inventory_series(store, 'CL', 'cushing')
    banded = frame.dropna(subset=['band_avg'])
    assert not banded.empty

    row = banded[banded['period'].dt.year == 2026].iloc[0]
    # prior five years are 2021..2025 -> levels 110,120,130,140,150
    assert row['band_min'] == pytest.approx(110)
    assert row['band_max'] == pytest.approx(150)
    assert row['band_avg'] == pytest.approx(130)
    assert row['value'] == pytest.approx(160)
    assert row['dev_5y'] == pytest.approx(30)
    assert row['dev_5y_pct'] == pytest.approx(30 / 130 * 100)


def test_earliest_years_have_no_band(store):
    values = {}
    for offset, year in enumerate(range(2020, 2027)):
        values.update(_weekly(f'{year}-01-06', 52, lambda i, o=offset: 100 + o * 10))
    store.put_inventory(_points('CL', 'cushing', values))
    frame = inventory_series(store, 'CL', 'cushing')
    # 2020 and 2021 lack the minimum prior years — NaN, by design
    assert frame[frame['period'].dt.year == 2020]['band_avg'].isna().all()
    assert frame[frame['period'].dt.year == 2026]['band_avg'].notna().any()


def test_weekly_change_is_the_build_or_draw(store):
    store.put_inventory(_points('NG', 'working_gas',
                                {'2026-07-24': 3000.0, '2026-07-31': 3117.0,
                                 '2026-08-07': 3153.0}, units='BCF'))
    frame = inventory_series(store, 'NG', 'working_gas')
    assert np.isnan(frame['weekly_change'].iloc[0])          # nothing before the first
    assert frame['weekly_change'].iloc[1] == pytest.approx(117)
    assert frame['weekly_change'].iloc[2] == pytest.approx(36)


def test_summary_reports_latest_reading(store):
    store.put_inventory(_points('NG', 'working_gas',
                                {'2026-07-31': 3117.0, '2026-08-07': 3153.0}, units='BCF'))
    summary = inventory_summary(store, 'NG', 'working_gas')
    assert summary['period'] == '2026-08-07'
    assert summary['value'] == 3153.0
    assert summary['units'] == 'BCF'
    assert summary['weekly_change'] == pytest.approx(36)


def test_stored_series_follows_registry_order(store):
    store.put_inventory(_points('CL', 'spr', {'2026-08-07': 1.0}))
    store.put_inventory(_points('CL', 'us_crude', {'2026-08-07': 2.0}))
    # registry order is us_crude, cushing, spr... — not insertion order
    assert stored_series(store, 'CL') == ['us_crude', 'spr']
    assert stored_series(store, 'GC') == []


def test_missing_inventory_yields_empty_frame(store):
    assert inventory_series(store, 'GC').empty
    assert inventory_summary(store, 'GC') == {}


# --- feature join -----------------------------------------------------------

def test_inventory_joins_features_backward_and_goes_stale(store):
    store.put_inventory(_points('CL', 'us_crude', {'2026-07-06': 400.0}))
    store.put_prices([PriceSeries('CL', 'spot', {
        '2026-07-03': 70.0,    # before the reading — must stay empty (no lookahead)
        '2026-07-08': 71.0,    # 2 days after — carries the reading
        '2026-08-05': 72.0,    # 30 days after — past INVENTORY_STALE_DAYS
    })])
    frame = features(store, 'CL').set_index('date')
    assert np.isnan(frame.loc[pd.Timestamp('2026-07-03'), 'inventory_level'])
    assert frame.loc[pd.Timestamp('2026-07-08'), 'inventory_level'] == 400.0
    assert np.isnan(frame.loc[pd.Timestamp('2026-08-05'), 'inventory_level'])


def test_features_without_inventory_have_no_inventory_columns(store):
    store.put_prices([PriceSeries('GC', 'spot', {'2026-08-07': 4300.0})])
    frame = features(store, 'GC')
    assert 'inventory_level' not in frame.columns
