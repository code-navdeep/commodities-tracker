"""Feature engineering — the analytical core, as pure functions.

features() produces one row per date with the full engineered feature set.
The date axis is the UNION of every date the store knows anything about —
daily price dates, delivery intent dates, and COT report dates. Delivery is
sparse (it accrues one day per refresh) and simply carries NaN on days
without a notice; the dense price/COT/volatility families get their full
history regardless of how many delivery days exist. (The predecessor app
keyed the whole frame to delivery dates, which starved two years of price
and COT data down to however many delivery days existed.)

Weekly COT is joined backward onto the axis (only reports published BEFORE
each date — no lookahead) and nulled once older than COT_STALE_DAYS.

Every derived value in the app is computed here (or in the sibling modules),
never in SQL: comm_net_long, comm_net_ratio (NaN on zero denominator), and
friends exist in exactly one place.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

from ..config import (COT_INDEX_MIN_WEEKS, COT_INDEX_WINDOW_WEEKS, COT_STALE_DAYS,
                      INVENTORY_STALE_DAYS, REALIZED_VOL_WINDOWS, ZSCORE_WINDOW)
from .cache import cached


def _window(df: pd.DataFrame, column: str, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out[column] = pd.to_datetime(out[column])
    if start:
        out = out[out[column] >= pd.to_datetime(start)]
    if end:
        out = out[out[column] <= pd.to_datetime(end)]
    return out


def _add_delivery_features(df: pd.DataFrame) -> None:
    """Delivery columns are sparse on the union axis — NaN on days without a
    notice. Every computation here tolerates that: NaN rows stay NaN, and
    z-scores/rolling means are taken over the delivery days that exist."""
    df['net_flow'] = df['total_stopped'] - df['total_issued']
    month_key = df['date'].dt.to_period('M')
    df['cumulative_net_flow'] = df.groupby(month_key)['net_flow'].cumsum()

    for window in (7, 14, 30):
        for col in ('total_issued', 'total_stopped', 'net_flow'):
            name = col.replace('total_', '')
            df[f'rolling_{name}_{window}d'] = df[col].rolling(window, min_periods=1).mean()

    if df['total_issued'].notna().sum() > 1:
        df['issued_zscore'] = zscore(df['total_issued'], nan_policy='omit')
        df['stopped_zscore'] = zscore(df['total_stopped'], nan_policy='omit')
        df['mtd_zscore'] = zscore(df['month_to_date'], nan_policy='omit')
    else:
        df['issued_zscore'] = df['stopped_zscore'] = df['mtd_zscore'] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        df['issued_pct_mtd'] = df['total_issued'] / df['month_to_date'] * 100
        df['stopped_pct_mtd'] = df['total_stopped'] / df['month_to_date'] * 100

    roll_mean = df['net_flow'].rolling(ZSCORE_WINDOW).mean()
    roll_std = df['net_flow'].rolling(ZSCORE_WINDOW).std()
    df['net_flow_zscore'] = (df['net_flow'] - roll_mean) / roll_std
    df['delivery_alert'] = ((df['net_flow_zscore'].abs() > 2) |
                            (df['stopped_zscore'].abs() > 2))


def _add_price_features(df: pd.DataFrame) -> None:
    if 'spot' not in df.columns:
        return
    df['price_return_1d'] = df['spot'].pct_change()
    df['price_return_5d'] = df['spot'].pct_change(periods=5)
    df['future_return_1d'] = df['spot'].pct_change().shift(-1)
    df['future_return_5d'] = df['spot'].pct_change(periods=5).shift(-5)
    df['future_direction_1d'] = (df['future_return_1d'] > 0).astype(int)
    df['future_direction_5d'] = (df['future_return_5d'] > 0).astype(int)


def _add_basis_features(df: pd.DataFrame, rates: pd.DataFrame) -> None:
    if 'spot' not in df.columns or 'future_close' not in df.columns:
        return
    valid = (df['spot'].notna() & (df['spot'] > 0) &
             df['future_close'].notna() & (df['future_close'] > 0))
    df['basis'] = np.where(valid, df['future_close'] - df['spot'], np.nan)
    df['basis_pct'] = np.where(valid, df['basis'] / df['spot'] * 100, np.nan)

    if 'vix' in df.columns:
        df['vol_adjusted_basis'] = df['basis_pct'] / df['vix'].replace(0, np.nan)

    if not rates.empty:
        rate_series = rates.assign(date=pd.to_datetime(rates['date'])).set_index('date')['rate']
        df['risk_free_rate'] = df['date'].map(rate_series)
        carry_valid = valid & df['risk_free_rate'].notna()
        days_to_expiry = 30   # front-month approximation
        df['carry_adjusted_basis'] = np.where(
            carry_valid,
            df['basis'] - (df['spot'] * df['risk_free_rate'] * days_to_expiry / 365),
            np.nan)


def _add_realized_vol_features(df: pd.DataFrame) -> None:
    """Parkinson realized volatility of the asset's own front-month future.

        sigma^2_day = ln(H/L)^2 / (4 ln 2)

    Range-based rather than close-to-close for two reasons: it is ~5x more
    statistically efficient, and both inputs come from the same day's bar, so
    the monthly front-month roll — which makes a close-to-close return on
    future_close a change of instrument rather than a price move — cannot
    contaminate it. Annualised by sqrt(252), decimal scale to match the `iv`
    coming out of sources/greeks.py. NaN until a window fills; no fallback.
    """
    if 'future_high' not in df.columns or 'future_low' not in df.columns:
        return
    high, low = df['future_high'], df['future_low']
    valid = high.notna() & low.notna() & (high > 0) & (low > 0) & (high >= low)
    if not valid.any():
        return
    log_range = np.log(high.where(valid) / low.where(valid))
    daily_variance = log_range ** 2 / (4 * np.log(2))
    for window in REALIZED_VOL_WINDOWS:
        df[f'realized_vol_{window}'] = np.sqrt(
            daily_variance.rolling(window, min_periods=window).mean() * 252)


def _add_volatility_features(df: pd.DataFrame) -> None:
    if 'vix' not in df.columns:
        return
    df['vol_zscore'] = zscore(df['vix'], nan_policy='omit') if len(df) > 1 else np.nan
    df['vol_ma_20'] = df['vix'].rolling(20, min_periods=1).mean()
    df['vol_ma_60'] = df['vix'].rolling(60, min_periods=1).mean()
    df['vol_change_20d'] = df['vix'].pct_change(periods=20)
    df['vol_regime'] = pd.cut(df['vol_zscore'], bins=[-np.inf, -1.0, 1.0, np.inf],
                              labels=['low', 'normal', 'high'])
    df['vol_regime_encoded'] = df['vol_regime'].map(
        {'low': 0, 'normal': 1, 'high': 2}).astype('Int64')


def _cot_index(net: pd.Series) -> pd.Series:
    """The 0-100 percentile desks actually quote: where this week's net sits
    inside its own range over COT_INDEX_WINDOW_WEEKS. More robust than a
    z-score on positioning data, which is skewed and fat-tailed.

    min_periods matters here and is not a detail: the conventional window is
    156 weeks but HISTORY_DAYS backfills only ~104, so a strict window would
    make this column permanently NaN on every database. It therefore computes
    over whatever history exists past COT_INDEX_MIN_WEEKS and tightens toward
    the full three years as the store accrues. NaN below that floor, and
    wherever the range is flat."""
    low = net.rolling(COT_INDEX_WINDOW_WEEKS, min_periods=COT_INDEX_MIN_WEEKS).min()
    high = net.rolling(COT_INDEX_WINDOW_WEEKS, min_periods=COT_INDEX_MIN_WEEKS).max()
    span = high - low
    return pd.Series(np.where(span > 0, (net - low) / span * 100, np.nan),
                     index=net.index)


def _weekly_cot_derivations(cot: pd.DataFrame) -> pd.DataFrame:
    """Category analytics, computed on the WEEKLY frame before it is joined to
    the daily axis — a 156-row window there would mean 156 days, not weeks.
    Must run on full history, not a windowed slice, or the lookbacks starve."""
    frame = cot.sort_values('report_date').copy()

    frame['producer_net'] = frame['prod_long'] - frame['prod_short']
    frame['swap_net'] = frame['swap_long'] - frame['swap_short']
    frame['mm_net'] = frame['mm_long'] - frame['mm_short']
    frame['other_net'] = frame['other_long'] - frame['other_short']

    # What share of the market is genuine forward selling by physical players.
    open_interest = frame['open_interest'].replace(0, np.nan)
    frame['producer_hedging_ratio'] = frame['prod_short'] / open_interest * 100

    frame['cot_index_producer'] = _cot_index(frame['producer_net'])
    frame['cot_index_swap'] = _cot_index(frame['swap_net'])
    frame['cot_index_mm'] = _cot_index(frame['mm_net'])

    # Weeks where the two halves of "commercial" moved opposite ways and
    # largely cancelled: the aggregate reads as "nothing happened" while
    # producers were selling into swap dealers buying (or the reverse).
    delta_prod = frame['producer_net'].diff()
    delta_swap = frame['swap_net'].diff()
    combined = (delta_prod + delta_swap).abs()
    separate = delta_prod.abs() + delta_swap.abs()
    frame['cot_divergence'] = (
        (np.sign(delta_prod) != np.sign(delta_swap))
        & (delta_prod != 0) & (delta_swap != 0)
        & (combined < 0.5 * separate)
    ).astype('Int64')
    return frame


def _add_cot_features(df: pd.DataFrame) -> None:
    if 'prod_long' not in df.columns or not df['prod_long'].notna().any():
        return

    # The legacy groupings, derived here rather than persisted by the source.
    df['comm_long'] = df['prod_long'] + df['swap_long']
    df['comm_short'] = df['prod_short'] + df['swap_short']
    df['comm_spreading'] = df['swap_spreading']
    df['noncomm_long'] = df['mm_long'] + df['other_long']
    df['noncomm_short'] = df['mm_short'] + df['other_short']
    df['noncomm_spreading'] = df['mm_spreading'] + df['other_spreading']

    df['comm_net_long'] = df['comm_long'] - df['comm_short']
    df['noncomm_net_long'] = df['noncomm_long'] - df['noncomm_short']
    denominator = df['comm_long'] + df['comm_short']
    df['comm_net_ratio'] = np.where(denominator != 0,
                                    df['comm_net_long'] / denominator, np.nan)
    df['comm_net_change'] = df['comm_net_long'].diff(periods=7)
    df['spec_net_change'] = df['noncomm_net_long'].diff(periods=7)
    df['oi_change'] = df['open_interest'].diff(periods=7)
    df['spec_positioning_ratio'] = (
        df['noncomm_net_long'] / df['open_interest'] * 100
    ).replace([np.inf, -np.inf], np.nan)
    df['comm_spec_spread'] = df['comm_net_long'] - df['noncomm_net_long']
    df['comm_momentum_4w'] = df['comm_net_change'].rolling(28, min_periods=1).mean()
    mask = df['comm_net_long'].notna()
    if mask.sum() > 1:
        df.loc[mask, 'comm_extreme_z'] = zscore(df.loc[mask, 'comm_net_long'],
                                                nan_policy='omit')


# Everything carried across from the weekly COT frame — nulled together once
# the underlying report is older than COT_STALE_DAYS.
COT_RAW_COLUMNS = ['prod_long', 'prod_short',
                   'swap_long', 'swap_short', 'swap_spreading',
                   'mm_long', 'mm_short', 'mm_spreading',
                   'other_long', 'other_short', 'other_spreading',
                   'nonreportable_long', 'nonreportable_short', 'open_interest',
                   'producer_net', 'swap_net', 'mm_net', 'other_net',
                   'producer_hedging_ratio', 'cot_index_producer',
                   'cot_index_swap', 'cot_index_mm', 'cot_divergence']

INVENTORY_COLUMNS = ['inventory_level', 'inventory_weekly_change', 'inventory_dev_5y']


def _add_inventory_features(df: pd.DataFrame, store, asset: str) -> None:
    """Join the headline weekly inventory series onto the daily axis — same
    backward-only pattern as COT (a reading is carried forward until the next
    one, never backwards into days before it was published), nulled once
    older than INVENTORY_STALE_DAYS. Assets with no inventory get nothing."""
    from .inventory import inventory_series          # local: avoids a cycle

    weekly = inventory_series(store, asset)
    if weekly.empty:
        return

    readings = (weekly[['period', 'value', 'weekly_change', 'dev_5y']]
                .rename(columns={'period': 'inventory_period',
                                 'value': 'inventory_level',
                                 'weekly_change': 'inventory_weekly_change',
                                 'dev_5y': 'inventory_dev_5y'})
                .sort_values('inventory_period'))

    merged = pd.merge_asof(df.sort_values('date'), readings,
                           left_on='date', right_on='inventory_period',
                           direction='backward')
    stale = (merged['date'] - merged['inventory_period']).dt.days.fillna(999) > INVENTORY_STALE_DAYS
    merged.loc[stale, INVENTORY_COLUMNS] = np.nan
    for column in INVENTORY_COLUMNS + ['inventory_period']:
        df[column] = merged[column].values


OPTIONS_COLUMNS = ['atm_iv', 'iv_term_slope', 'iv_skew_25d',
                   'pcr_oi', 'pcr_volume', 'net_gex']


def _add_options_features(df: pd.DataFrame, store, asset: str) -> None:
    """Join the daily options summary onto the axis, then derive the
    volatility risk premium.

    Exact date join, deliberately — unlike COT and inventory, which publish
    weekly and are carried forward until superseded, options reprice every
    session. A day with no stored chain is missing, not stale, and
    forward-filling last week's implied vol would invent a reading.
    """
    from .options_series import options_timeseries      # local: avoids a cycle

    series = options_timeseries(store, asset)
    if series.empty:
        return

    merged = df.merge(series[['date'] + OPTIONS_COLUMNS], on='date', how='left')
    for column in OPTIONS_COLUMNS:
        df[column] = merged[column].values

    # Rich or cheap: what the market charges for future movement against what
    # the underlying has actually delivered. Both sides are decimal vol.
    if 'realized_vol_20' in df.columns:
        df['vrp'] = df['atm_iv'] - df['realized_vol_20']


@cached
def features(store, asset: str, start=None, end=None) -> pd.DataFrame:
    """One row per date with the full engineered feature set. Empty DataFrame
    when the asset has no data at all in the window."""
    reports, _ = store.delivery(asset)
    prices = store.prices(asset)
    cot = store.cot(asset)

    # Category analytics first, on FULL weekly history: the COT index looks
    # back COT_INDEX_WINDOW_WEEKS, which a windowed slice cannot supply.
    if not cot.empty:
        cot = _weekly_cot_derivations(cot)

    reports = _window(reports, 'intent_date', start, end)
    prices = _window(prices, 'date', start, end)
    cot = _window(cot, 'report_date', start, end)

    # The axis: every date the store knows anything about.
    all_dates = set()
    if not prices.empty:
        all_dates |= set(prices['date'])
    if not reports.empty:
        all_dates |= set(reports['intent_date'])
    if not cot.empty:
        all_dates |= set(cot['report_date'])
    if not all_dates:
        return pd.DataFrame()
    df = pd.DataFrame({'date': sorted(all_dates)})

    if not reports.empty:
        df = pd.merge(df, reports.drop(columns=['asset', 'id'], errors='ignore')
                      .rename(columns={'intent_date': 'date'}),
                      on='date', how='left')
    if not prices.empty:
        df = pd.merge(df, prices.drop(columns=['asset']), on='date', how='left')
    # SQLite returns all-NULL columns as object dtype (None values), which
    # crashes scipy/numpy — force numeric.
    for column in ('spot', 'future_close', 'future_open', 'future_high',
                   'future_low', 'future_last', 'future_volume', 'future_oi',
                   'usd_index', 'vix',
                   'total_issued', 'total_stopped', 'month_to_date'):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    if not cot.empty:
        cot_sorted = cot.drop(columns=['asset']).sort_values('report_date')
        df = pd.merge_asof(df.sort_values('date'), cot_sorted,
                           left_on='date', right_on='report_date',
                           direction='backward')
        stale = (df['date'] - df['report_date']).dt.days.fillna(999) > COT_STALE_DAYS
        df.loc[stale, COT_RAW_COLUMNS] = np.nan
        df = df.rename(columns={'report_date': 'cot_report_date'})

    if 'total_issued' in df.columns and df['total_issued'].notna().any():
        _add_delivery_features(df)
    _add_price_features(df)
    _add_basis_features(df, store.option_rates(asset))
    _add_volatility_features(df)
    _add_realized_vol_features(df)
    _add_cot_features(df)

    _add_inventory_features(df, store, asset)
    _add_options_features(df, store, asset)

    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['has_cot'] = (df['comm_net_long'].notna().astype(int)
                     if 'comm_net_long' in df.columns else 0)
    df['has_price'] = df['spot'].notna().astype(int) if 'spot' in df.columns else 0
    df['has_basis'] = (df['basis'].notna().astype(int)
                       if 'basis' in df.columns else 0)
    return df


@cached
def seasonality(store, asset: str) -> dict:
    """{metric: DataFrame(month → mean/std)} for the seasonal chart."""
    df = features(store, asset)
    if df.empty:
        return {}
    metrics = [c for c in ('net_flow', 'price_return_1d', 'comm_net_long')
               if c in df.columns]
    if not metrics:
        return {}
    monthly = df.groupby('month')[metrics].agg(['mean', 'std'])
    return {metric: monthly[metric] for metric in metrics}


@cached
def firm_hhi(store, asset: str) -> pd.Series:
    """Daily Herfindahl-Hirschman index of stopped-contract concentration."""
    reports, firms = store.delivery(asset)
    if reports.empty or firms.empty:
        return pd.Series(dtype=float)
    merged = firms.merge(reports[['id', 'intent_date']].rename(columns={'id': 'report_id'}),
                         on='report_id')

    def hhi(group):
        total = group['stopped'].sum()
        if total == 0:
            return np.nan
        return ((group['stopped'] / total) ** 2).sum()

    return merged.groupby('intent_date').apply(hhi, include_groups=False).rename('hhi')
