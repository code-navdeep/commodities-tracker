"""The coverage-aware feature gate.

Training rows are the intersection of every admitted column, so admitting a
column purely because it clears ML_MIN_ROWS lets one thin family dictate the
training size for all of them. These tests pin the property that prevents it.
"""

import numpy as np
import pandas as pd
import pytest

from commodities.analytics.ml import _prepare, screen_regression
from commodities.config import ML_COVERAGE_FLOOR, ML_MIN_ROWS

TARGET = 'future_return_5d'
DENSE = ['vix', 'usd_index', 'price_return_1d']
SPARSE = 'atm_iv'


def _frame(rows=400, sparse_rows=70):
    """Dense columns over `rows` days, plus one column present only on the
    most recent `sparse_rows` — the shape the options family really has."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=rows, freq='B'),
        'vix': rng.normal(18, 2, rows),
        'usd_index': rng.normal(100, 1, rows),
        'price_return_1d': rng.normal(0, 0.01, rows),
        TARGET: rng.normal(0, 0.02, rows),
    })
    column = np.full(rows, np.nan)
    column[-sparse_rows:] = rng.normal(0.25, 0.02, sparse_rows)
    frame[SPARSE] = column
    return frame


def test_sparse_column_is_rejected_rather_than_shrinking_the_training_set():
    frame = _frame(rows=400, sparse_rows=70)
    X, y, kept, rejected = _prepare(frame, TARGET)

    assert SPARSE in rejected
    assert SPARSE not in kept
    assert set(DENSE) <= set(kept)
    # the whole point: rows survive
    assert len(X) == 400
    assert len(X) > ML_MIN_ROWS * 5


def test_same_column_is_admitted_once_coverage_is_high():
    frame = _frame(rows=400, sparse_rows=380)      # 95% > ML_COVERAGE_FLOOR
    X, y, kept, rejected = _prepare(frame, TARGET)

    assert SPARSE in kept
    assert SPARSE not in rejected
    assert len(X) == 380


def test_floor_is_relative_not_absolute():
    """A column clearing ML_MIN_ROWS is still refused when it would cut the
    intersection below the floor — the old absolute test admitted it."""
    below = int(400 * ML_COVERAGE_FLOOR) - 20
    assert below > ML_MIN_ROWS                      # clears the old gate
    _, _, kept, rejected = _prepare(_frame(rows=400, sparse_rows=below), TARGET)
    assert SPARSE in rejected

    above = int(400 * ML_COVERAGE_FLOOR) + 20
    _, _, kept, rejected = _prepare(_frame(rows=400, sparse_rows=above), TARGET)
    assert SPARSE in kept


def test_columns_below_min_rows_never_appear():
    frame = _frame(rows=400, sparse_rows=ML_MIN_ROWS - 1)
    _, _, kept, rejected = _prepare(frame, TARGET)
    # too thin to even be a candidate, so it is not a coverage rejection
    assert SPARSE not in kept
    assert SPARSE not in rejected


def test_metadata_reports_both_lists():
    results = screen_regression(_frame(rows=400, sparse_rows=70))
    metadata = results['metadata']
    assert SPARSE in metadata['coverage_rejected']
    assert SPARSE not in metadata['features']
    assert metadata['train_size'] + metadata['test_size'] == 400


def test_missing_target_returns_none():
    assert _prepare(_frame().drop(columns=[TARGET]), TARGET) is None
