"""Options as a time series — the surface's daily summary, not one chain.

analytics/options.py answers "what does this chain look like today". This
module answers "how has the options market moved", which nothing in the app
could do: every stored chain is keyed to (asset, expiry_month, data_date) and
the only cross-date read is a one-day OI difference.

One row per stored trade date, with the four numbers that carry information a
positioning/price/inventory feature set does not already hold:

- **atm_iv** — at-the-money implied volatility at a CONSTANT 30-day horizon.
  The constant horizon is the whole difficulty. A given expiry's remaining
  life shrinks from ~90 days to 1, and implied vol is term-dependent, so an
  "IV of the front expiry" series sawtooths every time the front rolls — by
  about the size of the signal being measured. Interpolating to a fixed
  horizon removes that artefact.
- **iv_term_slope** — second expiry's ATM IV minus the nearest's. Inversion
  (front above back) is an acute-stress signal with no analogue elsewhere in
  the feature set.
- **iv_skew_25d** — the 25-delta risk reversal: put IV minus call IV at equal
  and opposite delta. Delta, not strike distance, because CME's strike
  spacing is uneven (5-wide near the money, 25-wide in the wings), so any
  strike-based measure silently changes meaning across the surface.
- **pcr_oi / pcr_volume / net_gex / total_oi** — chain-wide sentiment and
  dealer-gamma readings.

Everything except the term slope is measured on the NEAREST expiry, so every
series sits on one consistent basis.

IV, delta and gamma are not recomputed here — enriched_chain() already
inverts Black-76 over the stored facts, and this module consumes it.
"""

import numpy as np
import pandas as pd

from ..config import IV_TARGET_DAYS, SKEW_DELTA
from .cache import cached
from .options import enriched_chain, gex


def _interpolate(x_values, y_values, x):
    """Linear interpolation of y at x over points sorted by x. None when fewer
    than two usable points exist; clamps rather than extrapolating."""
    pairs = sorted((float(a), float(b)) for a, b in zip(x_values, y_values)
                   if a is not None and b is not None
                   and np.isfinite(a) and np.isfinite(b))
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0][1]
    if x <= pairs[0][0]:
        return pairs[0][1]
    if x >= pairs[-1][0]:
        return pairs[-1][1]
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pairs[-1][1]


def _atm_iv(chain: pd.DataFrame):
    """ATM implied vol: interpolated between the two strikes bracketing the
    underlying, per side, then averaged. Not the nearest strike — that jumps
    discontinuously as the underlying drifts across the midpoint between two
    listed strikes, which reads as a volatility move that never happened."""
    if chain.empty or 'underlying' not in chain.columns:
        return None
    underlying = chain['underlying'].iloc[0]
    if underlying is None or not np.isfinite(underlying):
        return None

    per_side = []
    for side in ('Call', 'Put'):
        rows = chain[(chain['side'] == side) & chain['iv'].notna()]
        if rows.empty:
            continue
        value = _interpolate(rows['strike'], rows['iv'], underlying)
        if value is not None:
            per_side.append(value)
    return float(np.mean(per_side)) if per_side else None


def _skew_25_delta(chain: pd.DataFrame):
    """Put IV at delta -0.25 minus call IV at delta +0.25. Positive means the
    market is paying more for downside protection than for upside."""
    calls = chain[(chain['side'] == 'Call') & chain['iv'].notna() & chain['delta'].notna()]
    puts = chain[(chain['side'] == 'Put') & chain['iv'].notna() & chain['delta'].notna()]
    if calls.empty or puts.empty:
        return None
    call_iv = _interpolate(calls['delta'], calls['iv'], SKEW_DELTA)
    put_iv = _interpolate(puts['delta'], puts['iv'], -SKEW_DELTA)
    if call_iv is None or put_iv is None:
        return None
    return float(put_iv - call_iv)


def _constant_maturity_iv(points: list):
    """Interpolate ATM IV to IV_TARGET_DAYS across expiries, in total variance
    (sigma^2 * T) rather than in vol — variance is what is additive in time.
    `points` is [(days_to_expiry, atm_iv)]. Clamps to the nearest expiry when
    the target sits outside the available range; never extrapolates."""
    usable = [(d, v) for d, v in points if d and d > 0 and v and np.isfinite(v)]
    if not usable:
        return None
    usable.sort()
    if len(usable) == 1:
        return usable[0][1]

    target = IV_TARGET_DAYS
    if target <= usable[0][0]:
        return usable[0][1]
    if target >= usable[-1][0]:
        return usable[-1][1]

    variances = [(d, v * v * d) for d, v in usable]
    total = _interpolate([d for d, _ in variances], [w for _, w in variances], target)
    if total is None or total <= 0:
        return None
    return float(np.sqrt(total / target))


@cached
def options_timeseries(store, asset: str, start=None, end=None) -> pd.DataFrame:
    """One row per stored trade date. Empty frame when nothing is stored."""
    dates = store.option_dates(asset)
    if start:
        dates = [d for d in dates if d >= str(start)[:10]]
    if end:
        dates = [d for d in dates if d <= str(end)[:10]]
    if not dates:
        return pd.DataFrame()

    rows = []
    for data_date in dates:
        day = store.chains_on(asset, data_date)
        if day.empty:
            continue

        per_expiry = []
        for expiry_month in sorted(set(day['expiry_month'])):
            chain = enriched_chain(store, asset, expiry_month, data_date)
            if chain.empty:
                continue
            expiry_date = chain['expiry_date'].iloc[0]
            days = ((pd.Timestamp(expiry_date) - pd.Timestamp(data_date)).days
                    if expiry_date else None)
            per_expiry.append((days, expiry_month, chain, _atm_iv(chain)))

        if not per_expiry:
            continue
        per_expiry.sort(key=lambda item: (item[0] is None, item[0]))
        nearest = per_expiry[0]
        _, _, near_chain, near_iv = nearest

        second_iv = per_expiry[1][3] if len(per_expiry) > 1 else None

        # gex() sums to 0.0 when no strike has a computable gamma — on a chain
        # whose IVs are all undefined that is a fabricated observation, not a
        # measurement, so report it as missing.
        has_gamma = 'gamma' in near_chain.columns and near_chain['gamma'].notna().any()
        gamma_exposure = gex(near_chain) if has_gamma else pd.DataFrame()

        calls = near_chain[near_chain['side'] == 'Call']
        puts = near_chain[near_chain['side'] == 'Put']
        call_oi = float(calls['open_interest'].sum())
        put_oi = float(puts['open_interest'].sum())
        call_volume = float(calls['volume'].sum())
        put_volume = float(puts['volume'].sum())

        rows.append({
            'date': pd.Timestamp(data_date),
            'atm_iv': _constant_maturity_iv([(d, iv) for d, _, _, iv in per_expiry]),
            'atm_iv_front': near_iv,
            'iv_term_slope': (second_iv - near_iv
                              if second_iv is not None and near_iv is not None else np.nan),
            'iv_skew_25d': _skew_25_delta(near_chain),
            'pcr_oi': put_oi / call_oi if call_oi else np.nan,
            'pcr_volume': put_volume / call_volume if call_volume else np.nan,
            'net_gex': float(gamma_exposure['gex'].sum()) if not gamma_exposure.empty else np.nan,
            'total_oi': call_oi + put_oi,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
