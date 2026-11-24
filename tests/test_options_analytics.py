"""Options analytics over a stored two-date chain."""

import numpy as np
import pytest

from commodities.analytics.options import (chain_summary, enriched_chain,
                                            extreme_oi_changes, gex, pcr, top_strikes)
from commodities.domain import OptionChain, OptionQuote
from commodities.store import Store

F = 100.0


def _chain(data_date: str, oi_bump: int = 0):
    quotes = []
    for strike in (90.0, 100.0, 110.0):
        # plausible settles around F=100, ~30 days out, ~30% vol
        call_settle = max(F - strike, 0) + 3.5 * np.exp(-((strike - F) / 10) ** 2)
        put_settle = max(strike - F, 0) + 3.5 * np.exp(-((strike - F) / 10) ** 2)
        quotes.append(OptionQuote(strike, 'Call', round(call_settle, 2),
                                  None, None, None, 10, 100 + oi_bump))
        quotes.append(OptionQuote(strike, 'Put', round(put_settle, 2),
                                  None, None, None, 20, 200 + oi_bump * 2))
    return OptionChain(asset='GC', expiry_month='2026-09', expiry_date='2026-09-25',
                       data_date=data_date, underlying=F, rate=0.04, quotes=tuple(quotes))


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'options.db')
    s.init()
    s.put_chain(_chain('2026-08-13'))
    s.put_chain(_chain('2026-08-14', oi_bump=50))
    return s


def test_enrichment_produces_iv_and_greeks(store):
    chain = enriched_chain(store, 'GC', '2026-09', '2026-08-14')
    atm_call = chain[(chain['strike'] == 100.0) & (chain['side'] == 'Call')].iloc[0]
    assert 0.05 < atm_call['iv'] < 1.5
    assert 0.3 < atm_call['delta'] < 0.7
    assert atm_call['gamma'] > 0
    assert atm_call['theta'] < 0
    assert atm_call['vega'] > 0
    atm_put = chain[(chain['strike'] == 100.0) & (chain['side'] == 'Put')].iloc[0]
    assert atm_put['delta'] < 0
    # ATM skew is zero by construction
    assert atm_call['iv_skew'] == pytest.approx(0.0)


def test_oi_change_vs_previous_date(store):
    chain = enriched_chain(store, 'GC', '2026-09', '2026-08-14')
    call = chain[(chain['strike'] == 100.0) & (chain['side'] == 'Call')].iloc[0]
    put = chain[(chain['strike'] == 100.0) & (chain['side'] == 'Put')].iloc[0]
    assert call['oi_change'] == 50
    assert put['oi_change'] == 100
    earliest = enriched_chain(store, 'GC', '2026-09', '2026-08-13')
    assert earliest['oi_change'].isna().all()      # no earlier date to diff against


def test_pcr_per_strike_and_summary(store):
    chain = enriched_chain(store, 'GC', '2026-09', '2026-08-14')
    ratios = pcr(chain)
    row = ratios[ratios['strike'] == 100.0].iloc[0]
    assert row['pcr_oi'] == pytest.approx(300 / 150)
    assert row['pcr_volume'] == pytest.approx(2.0)
    summary = chain_summary(chain)
    assert summary['total_call_oi'] == 450
    assert summary['total_put_oi'] == 900
    assert summary['pcr_oi'] == pytest.approx(2.0)
    assert summary['oi_diff'] == 450


def test_gex_structure(store):
    chain = enriched_chain(store, 'GC', '2026-09', '2026-08-14')
    exposure = gex(chain)
    assert set(exposure['strike']) == {90.0, 100.0, 110.0}
    # Puts carry 2x the OI here, so signed GEX is negative where gamma exists
    atm = exposure[exposure['strike'] == 100.0].iloc[0]
    assert atm['gex'] < 0


def test_filter_strikes_counts_from_underlying_price(store):
    """The window counts listed strikes from the underlying price itself:
    n at/below it, n strictly above it — symmetric even when the price sits
    between strikes."""
    from commodities.views.options import filter_strikes
    chain = enriched_chain(store, 'GC', '2026-09', '2026-08-14')  # F = 100.0
    # strikes present: 90, 100, 110; underlying 100.0
    one = filter_strikes(chain, 1)
    assert sorted(set(one['strike'])) == [100.0, 110.0]   # 1 at/below + 1 above
    two = filter_strikes(chain, 2)
    assert sorted(set(two['strike'])) == [90.0, 100.0, 110.0]
    assert len(filter_strikes(chain, None)) == len(chain)  # All


def test_top_strikes_and_extremes(store):
    chain = enriched_chain(store, 'GC', '2026-09', '2026-08-14')
    tops = top_strikes(chain)
    assert len(tops['call_oi']) == 2
    extremes = extreme_oi_changes(chain)
    assert 'max_increase' in extremes.values()     # uniform +50 → one max pick per side
