"""ML screening — does the delivery/COT/price/vol feature set carry any
predictive signal for 5-day-forward returns? Output is a screening aid, not
a trading signal; the feature engineering is the product, the models grade it.

Design (unchanged from the validated original): chronological 80/20 split for
the predicted-vs-actual chart; 5-fold TimeSeriesSplit for honest CV metrics;
3-fold TimeSeriesSplit GridSearchCV for Ridge/Lasso alphas; StandardScaler on
linear models only; features with pairwise |r| > 0.95 pruned first.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import ML_COVERAGE_FLOOR, ML_MIN_ROWS
from .cache import cached

warnings.filterwarnings('ignore')

FEATURE_COLUMNS = [
    'total_issued', 'total_stopped', 'net_flow', 'month_to_date',
    'issued_zscore', 'stopped_zscore', 'mtd_zscore',
    'rolling_net_flow_7d', 'rolling_net_flow_14d',
    'price_return_1d', 'price_return_5d', 'usd_index', 'vix',
    # Trader categories rather than the legacy aggregates: comm_net_long is
    # exactly producer_net + swap_net, so carrying both guarantees
    # collinearity. The aggregates remain in features() for the views.
    'producer_net', 'swap_net', 'mm_net', 'producer_hedging_ratio', 'cot_index_mm',
    'comm_net_change', 'spec_net_change',
    'comm_net_ratio', 'spec_positioning_ratio', 'comm_spec_spread', 'comm_extreme_z',
    'inventory_level', 'inventory_weekly_change', 'inventory_dev_5y',
    'month_sin', 'month_cos',
    'vol_zscore', 'vol_ma_20', 'vol_ma_60',
    'vol_change_20d', 'vol_regime_encoded', 'vol_adjusted_basis',
    # The asset's own volatility, and the forward-looking options family —
    # the only daily, market-priced expectations in the set.
    'realized_vol_20',
    'atm_iv', 'vrp', 'iv_skew_25d', 'iv_term_slope', 'pcr_oi', 'net_gex',
]

REGRESSION_TARGET = 'future_return_5d'
CLASSIFICATION_TARGET = 'future_direction_5d'


def _select_features(df: pd.DataFrame, columns: list, threshold: float = 0.95) -> list:
    if len(columns) < 2:
        return columns
    corr = df[columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in columns if c not in dropped]


def _prepare(features_df: pd.DataFrame, target: str):
    """Select feature columns, then the rows on which all of them exist.

    Coverage-aware, and it has to be. Training rows are the INTERSECTION of
    every admitted column (`dropna()` below), so admitting a column purely
    because it clears ML_MIN_ROWS lets one thin family dictate the training
    size for all of them: the day the options family reaches 60 values, a
    ~500-row training set would collapse to 60 while the feature count rose
    past 40 — strictly worse than not having the family at all.

    So columns are offered densest-first and each is kept only while the
    surviving rows stay at ML_COVERAGE_FLOOR of what the densest column alone
    supports. A sparse family is not rejected forever, it simply waits until
    it covers enough of the window — and narrowing the dashboard's date range
    to that family's era admits it sooner, since features() is window-scoped.

    Returns (X, y, kept, rejected) or None.
    """
    if target not in features_df.columns:
        return None
    base = features_df[features_df[target].notna()]
    candidates = [c for c in FEATURE_COLUMNS
                  if c in base.columns and base[c].notna().sum() >= ML_MIN_ROWS]
    if not candidates:
        return None
    candidates.sort(key=lambda c: base[c].notna().sum(), reverse=True)

    kept, rejected, baseline = [], [], None
    for column in candidates:
        rows = len(base[kept + [column, target]].dropna())
        if baseline is None:                      # densest column sets the bar
            baseline, kept = rows, [column]
            continue
        if rows >= max(ML_MIN_ROWS, ML_COVERAGE_FLOOR * baseline):
            kept.append(column)
        else:
            rejected.append(column)

    model_df = base[kept + [target]].dropna()
    if len(model_df) < ML_MIN_ROWS:
        return None
    kept = _select_features(model_df, kept)
    return model_df[kept], model_df[target], kept, rejected


def screen_regression(features_df: pd.DataFrame) -> dict:
    """Per model: rmse, r2, cv_mean/std_r2, predictions; plus metadata.
    Empty dict below ML_MIN_ROWS."""
    prepared = _prepare(features_df, REGRESSION_TARGET)
    if prepared is None:
        return {}
    X, y, columns, rejected = prepared

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    outer = TimeSeriesSplit(n_splits=5)
    inner = TimeSeriesSplit(n_splits=3)
    X_scaled_full = StandardScaler().fit_transform(X)

    results = {}

    def record(name, model, predictions, scaled: bool, importances=None):
        cv = cross_val_score(model, X_scaled_full if scaled else X, y, cv=outer, scoring='r2')
        results[name] = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
            'r2': float(r2_score(y_test, predictions)),
            'cv_mean_r2': float(cv.mean()), 'cv_std_r2': float(cv.std()),
            'predictions': predictions,
        }
        if importances is not None:
            results[name]['feature_importance'] = dict(zip(columns, importances))

    linear = LinearRegression().fit(X_train_scaled, y_train)
    record('linear', linear, linear.predict(X_test_scaled), scaled=True)

    ridge = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
                         cv=inner, scoring='r2').fit(X_train_scaled, y_train)
    record('ridge', ridge.best_estimator_, ridge.predict(X_test_scaled), scaled=True)

    lasso = GridSearchCV(Lasso(max_iter=10000),
                         {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]},
                         cv=inner, scoring='r2').fit(X_train_scaled, y_train)
    record('lasso', lasso.best_estimator_, lasso.predict(X_test_scaled), scaled=True)

    forest = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    record('random_forest', forest, forest.predict(X_test), scaled=False,
           importances=forest.feature_importances_)

    boosting = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)
    record('gradient_boosting', boosting, boosting.predict(X_test), scaled=False,
           importances=boosting.feature_importances_)

    results['metadata'] = {'features': columns, 'coverage_rejected': rejected,
                           'y_test': y_test,
                           'train_size': len(X_train), 'test_size': len(X_test)}
    return results


def screen_classification(features_df: pd.DataFrame) -> dict:
    """Per model: accuracy, cv_mean/std_acc; plus metadata."""
    prepared = _prepare(features_df, CLASSIFICATION_TARGET)
    if prepared is None:
        return {}
    X, y, columns, rejected = prepared

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    splitter = TimeSeriesSplit(n_splits=5)
    X_scaled_full = StandardScaler().fit_transform(X)

    results = {}

    def record(name, model, predictions, scaled: bool, importances=None):
        cv = cross_val_score(model, X_scaled_full if scaled else X, y,
                             cv=splitter, scoring='accuracy')
        results[name] = {
            'accuracy': float(accuracy_score(y_test, predictions)),
            'cv_mean_acc': float(cv.mean()), 'cv_std_acc': float(cv.std()),
        }
        if importances is not None:
            results[name]['feature_importance'] = dict(zip(columns, importances))

    logistic = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
    record('logistic', logistic, logistic.predict(X_test_scaled), scaled=True)

    forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    record('random_forest', forest, forest.predict(X_test), scaled=False,
           importances=forest.feature_importances_)

    boosting = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)
    record('gradient_boosting', boosting, boosting.predict(X_test), scaled=False,
           importances=boosting.feature_importances_)

    results['metadata'] = {'features': columns, 'coverage_rejected': rejected,
                           'train_size': len(X_train), 'test_size': len(X_test)}
    return results


@cached
def screen(store, asset: str, start=None, end=None) -> dict:
    """{'regression': ..., 'classification': ...} over the asset's features.
    Cached by store revision — the heavy call views share."""
    from .features import features
    df = features(store, asset, start, end)
    if df.empty:
        return {'regression': {}, 'classification': {}}
    return {'regression': screen_regression(df),
            'classification': screen_classification(df)}
