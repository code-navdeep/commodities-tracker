"""Convenience-yield model — does physical delivery pressure show up in the
futures basis? Correlations of delivery/COT activity against basis, plus a
small linear model of basis_pct."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MIN_ROWS = 20


def convenience_yield(features_df) -> dict:
    """{'correlations': {...}, 'model': {r2, rmse, features}} or {} when the
    asset lacks basis data or history."""
    required = ['basis', 'basis_pct', 'total_stopped', 'net_flow']
    if features_df.empty or any(c not in features_df.columns for c in required):
        return {}

    activity = [c for c in ('total_stopped', 'total_issued', 'net_flow', 'month_to_date')
                if c in features_df.columns]
    cot = [c for c in ('comm_net_long', 'noncomm_net_long', 'comm_extreme_z')
           if c in features_df.columns]
    frame = features_df[['basis', 'basis_pct'] + activity + cot].dropna()
    if len(frame) < MIN_ROWS:
        return {}

    correlations = {f'{column}_vs_basis': float(frame[column].corr(frame['basis']))
                    for column in activity + cot}

    X = frame[activity + cot].values
    y = frame['basis_pct'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        'correlations': correlations,
        'model': {
            'r2': float(r2_score(y_test, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
            'features': activity + cot,
        },
    }
