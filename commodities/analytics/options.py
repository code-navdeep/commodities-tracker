"""Options analytics — everything derived from a stored chain, at read time.

The options table stores market facts only (settle, OHLC, volume, OI,
underlying, rate, expiry date). This module computes the rest on demand,
memoized by store revision: IV + Greeks (Black-76 over the stored inputs),
skew vs ATM, put/call ratios, day-over-day OI changes, top strikes, GEX,
and the aggregate row the chain table shows at the underlying price.
"""

import numpy as np
import pandas as pd

from ..sources.greeks import greeks, implied_vol
from .cache import cached


# scratch=True: one entry per (asset, expiry, trade date) — hundreds of them,
# and they must never evict the per-asset results the UI waits on. See cache.py.
@cached(scratch=True)
def enriched_chain(store, asset: str, expiry_month: str, data_date: str) -> pd.DataFrame:
    """The stored chain plus iv/delta/gamma/theta/vega/iv_skew columns, and
    oi_change vs the previous stored date (NaN when none exists)."""
    df = store.chain(asset, expiry_month, data_date)
    if df.empty:
        return df

    expiry_date = df['expiry_date'].iloc[0]
    T = None
    if expiry_date:
        T = max((pd.Timestamp(expiry_date) - pd.Timestamp(data_date)).days, 0) / 365.0

    ivs, deltas, gammas, thetas, vegas = [], [], [], [], []
    for row in df.itertuples():
        iv = implied_vol(row.side, row.settle, row.underlying, row.strike, T, row.rate) \
            if T else None
        if iv:
            g = greeks(row.side, row.underlying, row.strike, T, row.rate, iv)
            deltas.append(g['delta']); gammas.append(g['gamma'])
            thetas.append(g['theta']); vegas.append(g['vega'])
        else:
            deltas.append(np.nan); gammas.append(np.nan)
            thetas.append(np.nan); vegas.append(np.nan)
        ivs.append(iv if iv else np.nan)
    df['iv'], df['delta'], df['gamma'], df['theta'], df['vega'] = ivs, deltas, gammas, thetas, vegas

    # Skew vs the ATM strike of the same side
    underlying = df['underlying'].iloc[0]
    df['iv_skew'] = np.nan
    if underlying and not np.isnan(underlying):
        for side in ('Call', 'Put'):
            side_rows = df[(df['side'] == side) & df['iv'].notna()]
            if side_rows.empty:
                continue
            atm_iv = side_rows.loc[(side_rows['strike'] - underlying).abs().idxmin(), 'iv']
            df.loc[df['side'] == side, 'iv_skew'] = df.loc[df['side'] == side, 'iv'] - atm_iv

    # Day-over-day OI change vs the previous stored date for this expiry
    dates = store.chain_dates(asset, expiry_month)
    previous = next((d for d in dates if d < data_date), None)
    if previous:
        prev = store.chain(asset, expiry_month, previous)[['strike', 'side', 'open_interest']]
        prev = prev.rename(columns={'open_interest': 'prev_oi'})
        df = df.merge(prev, on=['strike', 'side'], how='left')
        df['oi_change'] = df['open_interest'] - df['prev_oi'].fillna(0)
        df = df.drop(columns=['prev_oi'])
    else:
        df['oi_change'] = np.nan
    return df


def pcr(chain: pd.DataFrame) -> pd.DataFrame:
    """Per strike: put/call ratio by OI and by volume."""
    calls = chain[chain['side'] == 'Call'].set_index('strike')
    puts = chain[chain['side'] == 'Put'].set_index('strike')
    strikes = sorted(set(chain['strike']))
    rows = []
    for strike in strikes:
        call_oi = calls['open_interest'].get(strike, 0)
        put_oi = puts['open_interest'].get(strike, 0)
        call_vol = calls['volume'].get(strike, 0)
        put_vol = puts['volume'].get(strike, 0)
        rows.append({'strike': strike,
                     'pcr_oi': put_oi / call_oi if call_oi else 0.0,
                     'pcr_volume': put_vol / call_vol if call_vol else 0.0})
    return pd.DataFrame(rows)


def top_strikes(chain: pd.DataFrame, n: int = 2) -> dict:
    """{'call_oi': [...], 'put_oi': [...], 'call_volume': [...], 'put_volume': [...]}"""
    out = {}
    for side in ('Call', 'Put'):
        side_df = chain[chain['side'] == side]
        out[f'{side.lower()}_oi'] = list(side_df.nlargest(n, 'open_interest')['strike'])
        out[f'{side.lower()}_volume'] = list(side_df.nlargest(n, 'volume')['strike'])
    return out


def extreme_oi_changes(chain: pd.DataFrame) -> dict:
    """{(strike, side): 'max_increase'|'max_decrease'} — the single largest
    OI build and unwind per side."""
    extremes = {}
    if 'oi_change' not in chain.columns or chain['oi_change'].isna().all():
        return extremes
    for side in ('Call', 'Put'):
        side_df = chain[(chain['side'] == side) & chain['oi_change'].notna()]
        gains = side_df[side_df['oi_change'] > 0]
        losses = side_df[side_df['oi_change'] < 0]
        if not gains.empty:
            extremes[(gains.loc[gains['oi_change'].idxmax(), 'strike'], side)] = 'max_increase'
        if not losses.empty:
            extremes[(losses.loc[losses['oi_change'].idxmin(), 'strike'], side)] = 'max_decrease'
    return extremes


def gex(chain: pd.DataFrame) -> pd.DataFrame:
    """Gamma exposure per strike: call gamma·OI·100 − put gamma·OI·100.
    Positive GEX dampens moves (dealers fade), negative amplifies."""
    rows = []
    for strike in sorted(set(chain['strike'])):
        total = 0.0
        for side, sign in (('Call', 1), ('Put', -1)):
            row = chain[(chain['strike'] == strike) & (chain['side'] == side)]
            if not row.empty and pd.notna(row['gamma'].iloc[0]):
                total += sign * row['gamma'].iloc[0] * row['open_interest'].iloc[0] * 100
        rows.append({'strike': strike, 'gex': total})
    return pd.DataFrame(rows)


def chain_summary(chain: pd.DataFrame) -> dict:
    """Aggregates for the chain table's underlying row."""
    calls = chain[chain['side'] == 'Call']
    puts = chain[chain['side'] == 'Put']
    call_iv = calls[calls['iv'].notna()]['iv']
    put_iv = puts[puts['iv'].notna()]['iv']
    total_call_oi = int(calls['open_interest'].sum())
    total_put_oi = int(puts['open_interest'].sum())
    total_call_volume = int(calls['volume'].sum())
    total_put_volume = int(puts['volume'].sum())
    has_oi_change = 'oi_change' in chain.columns and chain['oi_change'].notna().any()
    return {
        'total_call_oi': total_call_oi, 'total_put_oi': total_put_oi,
        'total_call_volume': total_call_volume, 'total_put_volume': total_put_volume,
        'avg_call_iv': float(call_iv.mean()) if len(call_iv) else 0.0,
        'avg_put_iv': float(put_iv.mean()) if len(put_iv) else 0.0,
        'net_call_oi_change': float(calls['oi_change'].sum()) if has_oi_change else 0.0,
        'net_put_oi_change': float(puts['oi_change'].sum()) if has_oi_change else 0.0,
        'pcr_oi': total_put_oi / total_call_oi if total_call_oi else 0.0,
        'pcr_volume': total_put_volume / total_call_volume if total_call_volume else 0.0,
        'oi_diff': total_put_oi - total_call_oi,
        'iv_skew': (float(put_iv.mean() - call_iv.mean()) * 100
                    if len(call_iv) and len(put_iv) else 0.0),
    }
