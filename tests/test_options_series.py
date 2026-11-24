"""The daily options summary — interpolation helpers and an end-to-end pass.

Chains here are priced with Black-76 at a KNOWN sigma, so the implied-vol
inversion inside enriched_chain must hand that same sigma back. That makes
these tests check the real path rather than restate the implementation.
"""

import numpy as np
import pandas as pd
import pytest

from commodities.analytics.options_series import (_atm_iv, _constant_maturity_iv,
                                                  _interpolate, _skew_25_delta,
                                                  options_timeseries)
from commodities.config import IV_TARGET_DAYS
from commodities.domain import OptionChain, OptionQuote
from commodities.sources.greeks import black76_price
from commodities.store import Store

DATA_DATE = '2026-01-05'
UNDERLYING = 100.0
RATE = 0.04
STRIKES = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'opt.db')
    s.init()
    return s


def _chain(expiry_month, expiry_date, sigma):
    """A chain priced at a flat `sigma`, so IV inversion recovers it exactly."""
    days = (pd.Timestamp(expiry_date) - pd.Timestamp(DATA_DATE)).days
    T = days / 365.0
    quotes = []
    for strike in STRIKES:
        for side in ('Call', 'Put'):
            price = black76_price(side, UNDERLYING, strike, T, RATE, sigma)
            quotes.append(OptionQuote(strike, side, round(price, 4),
                                      None, None, None, 10, 100))
    return OptionChain(asset='GC', expiry_month=expiry_month,
                       expiry_date=expiry_date, data_date=DATA_DATE,
                       underlying=UNDERLYING, rate=RATE, quotes=tuple(quotes))


# --- helpers -----------------------------------------------------------------

def test_interpolate_between_and_clamps():
    xs, ys = [10.0, 20.0], [1.0, 2.0]
    assert _interpolate(xs, ys, 15.0) == pytest.approx(1.5)
    assert _interpolate(xs, ys, 5.0) == pytest.approx(1.0)      # clamp low
    assert _interpolate(xs, ys, 99.0) == pytest.approx(2.0)     # clamp high
    assert _interpolate([], [], 1.0) is None


def test_atm_iv_interpolates_between_bracketing_strikes():
    """Underlying sits between two listed strikes; the ATM reading must be the
    interpolated value, not whichever strike happens to be nearest."""
    chain = pd.DataFrame({
        'side': ['Call', 'Call', 'Put', 'Put'],
        'strike': [100.0, 110.0, 100.0, 110.0],
        'iv': [0.20, 0.30, 0.20, 0.30],
        'underlying': [102.5] * 4,
    })
    assert _atm_iv(chain) == pytest.approx(0.225)


def test_skew_is_put_minus_call_at_equal_delta():
    chain = pd.DataFrame({
        'side': ['Call', 'Call', 'Put', 'Put'],
        'strike': [105.0, 110.0, 95.0, 90.0],
        'delta': [0.35, 0.15, -0.35, -0.15],
        'iv': [0.20, 0.24, 0.30, 0.34],
    })
    # at |delta| 0.25: call 0.22, put 0.32 -> skew +0.10
    assert _skew_25_delta(chain) == pytest.approx(0.10)


def test_constant_maturity_interpolates_in_variance_time():
    """sigma^2 * T is what is additive across expiries, not sigma."""
    near, far = (15, 0.30), (45, 0.20)
    total = 0.30 ** 2 * 15 + ((0.20 ** 2 * 45) - (0.30 ** 2 * 15)) * 0.5
    expected = np.sqrt(total / IV_TARGET_DAYS)
    assert _constant_maturity_iv([near, far]) == pytest.approx(expected)
    # interpolating in vol instead would give 0.25 — materially different
    assert abs(_constant_maturity_iv([near, far]) - 0.25) > 0.015


def test_constant_maturity_clamps_and_never_extrapolates():
    assert _constant_maturity_iv([(90, 0.22)]) == pytest.approx(0.22)
    assert _constant_maturity_iv([(60, 0.22), (90, 0.25)]) == pytest.approx(0.22)
    assert _constant_maturity_iv([(5, 0.40), (10, 0.35)]) == pytest.approx(0.35)
    assert _constant_maturity_iv([]) is None


# --- end to end ---------------------------------------------------------------

def test_timeseries_recovers_known_vols(store):
    store.put_chain(_chain('2026-01', '2026-01-20', sigma=0.30))   # 15 days
    store.put_chain(_chain('2026-02', '2026-02-19', sigma=0.20))   # 45 days

    series = options_timeseries(store, 'GC')
    assert len(series) == 1
    row = series.iloc[0]

    total = 0.30 ** 2 * 15 + ((0.20 ** 2 * 45) - (0.30 ** 2 * 15)) * 0.5
    assert row['atm_iv'] == pytest.approx(np.sqrt(total / IV_TARGET_DAYS), rel=1e-3)
    assert row['atm_iv_front'] == pytest.approx(0.30, rel=1e-3)
    assert row['iv_term_slope'] == pytest.approx(-0.10, rel=1e-2)


def test_timeseries_reports_chain_wide_readings(store):
    store.put_chain(_chain('2026-01', '2026-01-20', sigma=0.30))
    row = options_timeseries(store, 'GC').iloc[0]
    assert row['pcr_oi'] == pytest.approx(1.0)          # symmetric synthetic chain
    assert row['total_oi'] == 100 * len(STRIKES) * 2
    assert np.isfinite(row['net_gex'])


def test_flat_smile_has_no_skew(store):
    store.put_chain(_chain('2026-01', '2026-01-20', sigma=0.30))
    row = options_timeseries(store, 'GC').iloc[0]
    assert row['iv_skew_25d'] == pytest.approx(0.0, abs=5e-3)


def test_empty_store_yields_empty_frame(store):
    assert options_timeseries(store, 'GC').empty
