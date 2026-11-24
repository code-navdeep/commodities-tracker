"""Inventory analytics — what the stored weekly levels actually mean.

The store holds only what EIA published (level, units, week-ending date).
Everything a trader reads off those levels is computed here, on demand:

- **weekly change** — the build (positive) or draw (negative) headline.
- **5-year seasonal band** — for each week-of-year, the min/max/average of
  that same week across the five preceding calendar years. Inventories are
  strongly seasonal (gas injects all summer and withdraws all winter), so a
  raw level says almost nothing on its own; the level *relative to its own
  seasonal norm* is the signal.
- **deviation from the 5-year average** — absolute, percent, and z-scored:
  the "is this market tight or loose right now" read.

A band needs prior years to exist, so the earliest years of stored history
carry NaN there (MIN_BAND_YEARS below). That is expected, not missing data.
"""

import numpy as np
import pandas as pd

from ..config import ASSETS, SEASONAL_BAND_YEARS
from .cache import cached

# Fewer than this many prior years and a "5-year band" would be noise.
MIN_BAND_YEARS = 3


def series_names(asset: str) -> list:
    """Configured series for an asset, headline first (registry order)."""
    return list(ASSETS.get(asset, {}).get('inventory') or {})


def headline_series(asset: str):
    """The series used for stat cards and ML features — first in the registry."""
    names = series_names(asset)
    return names[0] if names else None


def _seasonal_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach band_min / band_max / band_avg from the preceding
    SEASONAL_BAND_YEARS calendar years at the same week-of-year."""
    iso = frame['period'].dt.isocalendar()
    frame['year'] = iso.year.astype(int)
    frame['week'] = iso.week.astype(int)

    # week × year matrix of levels; a (year, week) may legitimately be absent
    # (holiday weeks, week 53), which just narrows that week's band.
    grid = frame.pivot_table(index='week', columns='year', values='value', aggfunc='mean')

    mins, maxs, avgs = [], [], []
    for year, week in zip(frame['year'], frame['week']):
        prior = [y for y in grid.columns if year - SEASONAL_BAND_YEARS <= y <= year - 1]
        values = grid.loc[week, prior].dropna() if (week in grid.index and prior) else pd.Series(dtype=float)
        if len(values) >= MIN_BAND_YEARS:
            mins.append(float(values.min()))
            maxs.append(float(values.max()))
            avgs.append(float(values.mean()))
        else:
            mins.append(np.nan)
            maxs.append(np.nan)
            avgs.append(np.nan)

    frame['band_min'], frame['band_max'], frame['band_avg'] = mins, maxs, avgs
    return frame


@cached
def inventory_series(store, asset: str, series: str = None) -> pd.DataFrame:
    """One series with every derived column. Empty frame when nothing stored."""
    series = series or headline_series(asset)
    if not series:
        return pd.DataFrame()

    frame = store.inventory(asset, series)
    if frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    frame['period'] = pd.to_datetime(frame['period'])
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    frame = frame.dropna(subset=['value']).sort_values('period').reset_index(drop=True)

    frame['weekly_change'] = frame['value'].diff()
    frame = _seasonal_band(frame)

    frame['dev_5y'] = frame['value'] - frame['band_avg']
    with np.errstate(divide='ignore', invalid='ignore'):
        frame['dev_5y_pct'] = np.where(frame['band_avg'].notna() & (frame['band_avg'] != 0),
                                       frame['dev_5y'] / frame['band_avg'] * 100, np.nan)
    deviations = frame['dev_5y'].dropna()
    if len(deviations) > 1 and deviations.std():
        frame['dev_5y_z'] = (frame['dev_5y'] - deviations.mean()) / deviations.std()
    else:
        frame['dev_5y_z'] = np.nan
    return frame


@cached
def inventory_summary(store, asset: str, series: str = None) -> dict:
    """Latest reading plus its context — what the stat cards show."""
    frame = inventory_series(store, asset, series)
    if frame.empty:
        return {}
    latest = frame.iloc[-1]
    return {
        'series': series or headline_series(asset),
        'period': latest['period'].strftime('%Y-%m-%d'),
        'value': float(latest['value']),
        'units': latest['units'],
        'weekly_change': (float(latest['weekly_change'])
                          if pd.notna(latest['weekly_change']) else None),
        'band_avg': float(latest['band_avg']) if pd.notna(latest['band_avg']) else None,
        'dev_5y': float(latest['dev_5y']) if pd.notna(latest['dev_5y']) else None,
        'dev_5y_pct': float(latest['dev_5y_pct']) if pd.notna(latest['dev_5y_pct']) else None,
    }


@cached
def stored_series(store, asset: str) -> list:
    """Configured series that actually have rows, headline first — drives the
    view's series chip strip."""
    present = set(store.inventory(asset)['series']) if not store.inventory(asset).empty else set()
    return [name for name in series_names(asset) if name in present]
