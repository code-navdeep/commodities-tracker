"""Delivery/COT view — the main analytical screen for one asset.

Everything renders from analytics over the store: features() drives the
charts, screen() the ML panels, convenience_yield() the basis model. Assets
without delivery data (COT-only) get the COT chart subset.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from ..analytics.basis import convenience_yield
from ..analytics.features import features, firm_hhi, seasonality
from ..analytics.inventory import (headline_series, inventory_series,
                                    inventory_summary, stored_series)
from ..analytics.ml import screen
from ..config import ML_MIN_ROWS
from .components import chip_strip, date_axes, stat_card, ticker_rule
from .theme import PALETTE

CORRELATION_COLUMNS = [
    'total_issued', 'total_stopped', 'net_flow', 'month_to_date',
    'issued_zscore', 'stopped_zscore', 'spot', 'future_close',
    'usd_index', 'vix', 'comm_net_long', 'noncomm_net_long', 'open_interest',
    'comm_net_change', 'spec_net_change', 'comm_net_ratio', 'comm_extreme_z',
    'vol_ma_20', 'vol_ma_60', 'vol_change_20d', 'vol_regime_encoded',
    'vol_adjusted_basis',
]

MODEL_LABELS = {'linear': 'Linear', 'ridge': 'Ridge', 'lasso': 'Lasso',
                'logistic': 'Logistic', 'random_forest': 'Random Forest',
                'gradient_boosting': 'Gradient Boosting'}

# The app's only continuous colorbar (MTD volume on the delivery-vs-price
# scatter). Plotly's built-in scales are all off-palette, so this ramps from
# near-black to amber.
MTD_COLORSCALE = [[0.0, '#26221a'], [1.0, PALETTE['amber']]]

# Display names for the registry's inventory series keys.
INVENTORY_LABELS = {
    'us_crude': 'US Crude (ex-SPR)', 'cushing': 'Cushing', 'spr': 'SPR',
    'gasoline': 'Gasoline', 'distillate': 'Distillate', 'production': 'Production',
    'working_gas': 'Lower 48', 'east': 'East', 'midwest': 'Midwest',
    'mountain': 'Mountain', 'pacific': 'Pacific', 'south_central': 'South Central',
}


def inventory_label(series: str) -> str:
    return INVENTORY_LABELS.get(series, series.replace('_', ' ').title())


# --- figures -----------------------------------------------------------------

def fig_delivery_timeseries(df):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
                        subplot_titles=('Daily Issued Contracts', 'Daily Stopped Contracts'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['total_issued'], mode='lines+markers',
                             name='Issued', line=dict(color=PALETTE['down'], width=2),
                             marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['total_stopped'], mode='lines+markers',
                             name='Stopped', line=dict(color=PALETTE['up'], width=2),
                             marker=dict(size=4)), row=2, col=1)
    fig.update_yaxes(title_text='Contracts')
    fig.update_layout(height=700, title_text='Daily Delivery Volumes', hovermode='x unified')
    return date_axes(fig)


def fig_net_flow(df):
    fig = go.Figure(go.Scatter(x=df['date'], y=df['net_flow'], mode='lines+markers',
                               name='Net Flow', line=dict(color=PALETTE['text'], width=2),
                               marker=dict(size=4), fill='tozeroy',
                               fillcolor='rgba(232, 234, 237, 0.08)'))
    fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'])
    fig.update_layout(title='Net Delivery Flow (Stopped − Issued)', xaxis_title='Date',
                      yaxis_title='Net Flow (Contracts)', height=500, hovermode='x unified')
    return date_axes(fig)


def fig_mtd(df):
    fig = go.Figure(go.Scatter(x=df['date'], y=df['month_to_date'], mode='lines+markers',
                               name='MTD', line=dict(color=PALETTE['dim'], width=2),
                               marker=dict(size=4)))
    fig.update_layout(title='Month-to-Date Delivery Volumes', xaxis_title='Date',
                      yaxis_title='MTD Contracts', height=500, hovermode='x unified')
    return date_axes(fig)


def fig_top_firms(firms):
    totals = firms.groupby(['firm_code', 'org_code', 'firm_name'], as_index=False).agg(
        issued=('issued', 'sum'), stopped=('stopped', 'sum'))
    totals['activity'] = totals['issued'] + totals['stopped']
    totals['net'] = totals['stopped'] - totals['issued']
    totals['account'] = totals['org_code'].map({'H': 'House', 'C': 'Customer'}).fillna(totals['org_code'])
    totals['label'] = totals['firm_code'] + '-' + totals['org_code']
    top = totals.nlargest(10, 'activity').copy()
    # Group a firm's House/Customer bars together, firm-groups by combined activity
    group_activity = top.groupby('firm_code')['activity'].transform('sum')
    top = top.assign(_g=group_activity).sort_values(['_g', 'activity'], ascending=False)

    hover = np.stack([top['firm_name'], top['account']], axis=-1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 10 Most Active Firms', 'Net Positions'))
    fig.add_trace(go.Bar(y=top['label'], x=top['activity'], orientation='h',
                         marker_color=PALETTE['gray'], customdata=hover,
                         hovertemplate='<b>%{customdata[0]}</b> — %{customdata[1]} (%{y})'
                                       '<br>Total activity: %{x}<extra></extra>'),
                  row=1, col=1)
    colors = [PALETTE['up'] if v > 0 else PALETTE['down'] for v in top['net']]
    fig.add_trace(go.Bar(y=top['label'], x=top['net'], orientation='h', marker_color=colors,
                         customdata=hover,
                         hovertemplate='<b>%{customdata[0]}</b> — %{customdata[1]} (%{y})'
                                       '<br>Net position: %{x}<extra></extra>'),
                  row=1, col=2)
    # category axes: numeric-looking firm codes otherwise become continuous axes
    fig.update_yaxes(type='category', automargin=True)
    fig.update_xaxes(title_text='Contracts', row=1, col=1)
    fig.update_xaxes(title_text='Net Contracts', row=1, col=2)
    fig.update_layout(height=600, showlegend=False,
                      title_text='Firm Activity (hover for firm name and account type)')
    return fig


def fig_firm_behavior(reports, firms, firm_code):
    detail = firms[firms['firm_code'] == firm_code].merge(
        reports[['id', 'intent_date']].rename(columns={'id': 'report_id'}), on='report_id')
    if detail.empty:
        return None
    detail = detail.sort_values('intent_date')
    labels = {'H': 'House', 'C': 'Customer'}
    dashes = {'H': 'solid', 'C': 'dash'}
    fig = go.Figure()
    for org in sorted(detail['org_code'].unique()):
        rows = detail[detail['org_code'] == org]
        fig.add_trace(go.Scatter(x=rows['intent_date'], y=rows['issued'], mode='lines+markers',
                                 name=f"Issued ({labels.get(org, org)})",
                                 line=dict(color=PALETTE['down'], width=2, dash=dashes.get(org, 'dot'))))
        fig.add_trace(go.Scatter(x=rows['intent_date'], y=rows['stopped'], mode='lines+markers',
                                 name=f"Stopped ({labels.get(org, org)})",
                                 line=dict(color=PALETTE['up'], width=2, dash=dashes.get(org, 'dot'))))
    name = detail['firm_name'].iloc[0]
    accounts = ' & '.join(labels.get(o, o) for o in sorted(detail['org_code'].unique()))
    fig.update_layout(title=f"Firm {firm_code} ({name}, {accounts}) — Delivery Activity",
                      xaxis_title='Date', yaxis_title='Contracts', height=500,
                      hovermode='x unified')
    return date_axes(fig)


def fig_correlation_heatmap(df):
    columns = [c for c in CORRELATION_COLUMNS if c in df.columns]
    if len(columns) < 2:
        return None
    corr = df[columns].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                               colorscale='RdBu', zmid=0, text=np.round(corr.values, 2),
                               texttemplate='%{text}', textfont={'size': 8},
                               colorbar=dict(title='Correlation')))
    fig.update_layout(title='Correlation Matrix of Key Variables', height=784, width=896,
                      xaxis=dict(automargin=True), yaxis=dict(automargin=True))
    return fig


def fig_delivery_vs_price(df):
    frame = df.dropna(subset=['net_flow', 'future_return_5d']) \
        if 'future_return_5d' in df.columns else pd.DataFrame()
    if len(frame) < 10:
        return None
    fig = go.Figure(go.Scatter(
        x=frame['net_flow'], y=frame['future_return_5d'] * 100, mode='markers',
        showlegend=False,
        marker=dict(size=8, color=frame['month_to_date'], colorscale=MTD_COLORSCALE,
                    showscale=True, colorbar=dict(title='MTD Volume')),
        text=frame['date'].dt.strftime('%Y-%m-%d'),
        hovertemplate='<b>%{text}</b><br>Net Flow: %{x}<br>5d Return: %{y:.2f}%<extra></extra>'))
    if frame['net_flow'].nunique() > 1:
        slope, intercept = np.polyfit(frame['net_flow'], frame['future_return_5d'] * 100, 1)
        xs = np.linspace(frame['net_flow'].min(), frame['net_flow'].max(), 100)
        # The fitted trend is what this chart exists to show — it takes amber.
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode='lines', name='Trend',
                                 line=dict(color=PALETTE['amber'], dash='dash')))
    fig.update_layout(title='Net Delivery Flow vs Future 5-Day Returns',
                      xaxis_title='Net Flow (Stopped − Issued)',
                      yaxis_title='5-Day Forward Return (%)', height=600,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='right', x=1))
    return fig


def fig_regression_performance(regression):
    models = [k for k in regression if k != 'metadata']
    if not models:
        return None
    labels = [MODEL_LABELS.get(m, m) for m in models]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('RMSE (lower is better)', 'R² Score (higher is better)'))
    fig.add_trace(go.Bar(x=labels, y=[regression[m]['rmse'] for m in models],
                         marker_color=PALETTE['gray']), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[regression[m]['r2'] for m in models],
                         marker_color=PALETTE['gray']), row=1, col=2)
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_layout(height=500, title_text='Regression Model Performance', showlegend=False)
    return fig


def fig_classification_performance(classification):
    models = [k for k in classification if k != 'metadata']
    if not models:
        return None
    labels = [MODEL_LABELS.get(m, m) for m in models]
    accuracy = [classification[m]['accuracy'] * 100 for m in models]
    fig = go.Figure(go.Bar(x=labels, y=accuracy, marker_color=PALETTE['gray'],
                           text=[f"{v:.1f}%" for v in accuracy], textposition='auto'))
    fig.add_hline(y=50, line_dash='dash', line_color=PALETTE['dim'],
                  annotation_text='Random Guess (50%)')
    fig.update_layout(title='Classification Accuracy (Direction Prediction)',
                      xaxis_title='Model', yaxis_title='Accuracy (%)', height=500)
    fig.update_xaxes(tickangle=-45, automargin=True)
    return fig


def fig_feature_importance(regression):
    importance = regression.get('random_forest', {}).get('feature_importance')
    if not importance:
        return None
    ordered = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    fig = go.Figure(go.Bar(y=[k for k, _ in ordered], x=[v for _, v in ordered],
                           orientation='h', marker_color=PALETTE['gray']))
    fig.update_layout(title='Feature Importance (Random Forest Regression)',
                      xaxis_title='Importance', yaxis_title='Feature', height=600,
                      yaxis=dict(automargin=True))
    return fig


def fig_predicted_vs_actual(regression):
    metadata = regression.get('metadata')
    forest = regression.get('random_forest')
    if not metadata or not forest:
        return None
    y_test = metadata['y_test']
    predictions = forest['predictions']
    fig = go.Figure(go.Scatter(x=np.array(y_test) * 100, y=np.array(predictions) * 100,
                               mode='markers', name='Predictions',
                               marker=dict(size=8, color=PALETTE['text'], opacity=0.6)))
    lo = min(min(y_test), min(predictions)) * 100
    hi = max(max(y_test), max(predictions)) * 100
    # A guide line, not an alert — dim, like every other reference rule.
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect Prediction',
                             line=dict(color=PALETTE['dim'], dash='dash')))
    fig.update_layout(title='Predicted vs Actual 5-Day Returns (Random Forest)',
                      xaxis_title='Actual (%)', yaxis_title='Predicted (%)', height=600)
    return fig


def fig_basis(df, cy: dict | None = None):
    """Panel 1 needs only basis rows; panel 2 needs basis ∩ delivery days —
    on the union axis those coverages differ (basis accrues with futures
    settles, delivery with notices)."""
    basis_rows = df.dropna(subset=['basis']) if 'basis' in df.columns else pd.DataFrame()
    if len(basis_rows) < 2:
        return None
    both = basis_rows.dropna(subset=['total_stopped']) \
        if 'total_stopped' in basis_rows.columns else pd.DataFrame()
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
                        subplot_titles=('Basis Over Time', 'Basis vs Delivery Activity'))
    fig.add_trace(go.Scatter(x=basis_rows['date'], y=basis_rows['basis'],
                             mode='lines+markers', name='Basis',
                             line=dict(color=PALETTE['text'], width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=both['total_stopped'], y=both['basis'], mode='markers',
                             name='Stopped vs Basis',
                             marker=dict(size=8, color=PALETTE['amber'])), row=2, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Total Stopped Contracts', row=2, col=1)
    fig.update_yaxes(title_text='Basis ($)')
    fig.update_xaxes(tickformat='%Y-%m-%d', hoverformat='%Y-%m-%d', row=1, col=1)
    title = 'Futures Basis Analysis'
    if cy and cy.get('model'):
        # Convenience-yield model: how much of basis_pct the delivery/COT
        # activity explains (analytics/basis.py)
        title += (f" — convenience-yield model R² {cy['model']['r2']:.3f}, "
                  f"stopped↔basis r {cy['correlations'].get('total_stopped_vs_basis', float('nan')):.2f}")
    fig.update_layout(height=800, title_text=title,
                      legend=dict(yanchor='middle', y=0.5, xanchor='left', x=1.02))
    return fig


def fig_cot_vs_delivery(df):
    if 'comm_net_long' not in df.columns or df['comm_net_long'].isna().all():
        return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Commercial Net Position (COT)',
                                        'Daily Net Delivery Flow', 'Spot Price'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['comm_net_long'], mode='lines+markers',
                             name='Commercial Net Long',
                             line=dict(color=PALETTE['amber'], width=2), fill='tozeroy',
                             fillcolor='rgba(232, 163, 61, 0.10)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['net_flow'], mode='lines+markers',
                             name='Net Delivery Flow',
                             line=dict(color=PALETTE['text'], width=2), fill='tozeroy',
                             fillcolor='rgba(232, 234, 237, 0.08)'), row=2, col=1)
    if 'spot' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['spot'], mode='lines+markers',
                                 name='Spot Price',
                                 line=dict(color=PALETTE['amber'], width=2)), row=3, col=1)
    for row in (1, 2):
        fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'], row=row, col=1)
    fig.update_layout(title='COT Positioning vs Delivery Activity', height=900,
                      hovermode='x unified')
    return date_axes(fig)


def fig_cot_extremes(df):
    if 'comm_extreme_z' not in df.columns or df['comm_extreme_z'].isna().all():
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=('Commercial Net Position',
                                        'Positioning Z-Score (Extreme Indicator)'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['comm_net_long'], mode='lines+markers',
                             name='Commercial Net Long',
                             line=dict(color=PALETTE['amber'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['comm_extreme_z'], mode='lines+markers',
                             name='Z-Score', line=dict(color=PALETTE['dim'], width=2),
                             fill='tozeroy', fillcolor='rgba(232, 234, 237, 0.08)'),
                  row=2, col=1)
    fig.add_hline(y=2, line_dash='dash', line_color=PALETTE['down'], row=2, col=1,
                  annotation_text='Extreme High')
    fig.add_hline(y=-2, line_dash='dash', line_color=PALETTE['down'], row=2, col=1,
                  annotation_text='Extreme Low')
    fig.add_hline(y=0, line_color=PALETTE['dim'], row=2, col=1)
    highs = df[df['comm_extreme_z'] > 2]
    lows = df[df['comm_extreme_z'] < -2]
    if len(highs):
        fig.add_trace(go.Scatter(x=highs['date'], y=highs['comm_extreme_z'], mode='markers',
                                 name='Extreme High',
                                 marker=dict(color=PALETTE['down'], size=8, symbol='triangle-up')),
                      row=2, col=1)
    if len(lows):
        fig.add_trace(go.Scatter(x=lows['date'], y=lows['comm_extreme_z'], mode='markers',
                                 name='Extreme Low',
                                 marker=dict(color=PALETTE['down'], size=8,
                                             symbol='triangle-down')), row=2, col=1)
    fig.update_layout(title='COT Extreme Positioning', height=800, hovermode='x unified')
    return date_axes(fig)


def fig_comm_vs_spec(df):
    if 'comm_net_long' not in df.columns or df['comm_net_long'].isna().all():
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=('Net Positions: Commercial vs Speculator',
                                        'Position Spread (Commercial − Speculator)'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['comm_net_long'], mode='lines+markers',
                             name='Commercial Net Long',
                             line=dict(color=PALETTE['amber'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['noncomm_net_long'], mode='lines+markers',
                             name='Speculator Net Long',
                             line=dict(color=PALETTE['text'], width=2)), row=1, col=1)
    if 'comm_spec_spread' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['comm_spec_spread'], mode='lines+markers',
                                 name='Position Spread',
                                 line=dict(color=PALETTE['dim'], width=2), fill='tozeroy',
                                 fillcolor='rgba(232, 234, 237, 0.08)'), row=2, col=1)
    for row in (1, 2):
        fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'], row=row, col=1)
    fig.update_layout(title='Commercial vs Speculator Positioning', height=800,
                      hovermode='x unified')
    return date_axes(fig)


def fig_producer_vs_swap(df):
    """The split the legacy commercial number hides. Producers selling forward
    is real supply arriving at a price; swap dealers are usually the mirror of
    investor demand. Added together they can cancel — the marked weeks are
    exactly those, where the aggregate says 'nothing happened'."""
    if 'producer_net' not in df.columns or df['producer_net'].isna().all():
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['producer_net'], mode='lines+markers',
                             name='Producer / Merchant Net',
                             line=dict(color=PALETTE['amber'], width=2)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['swap_net'], mode='lines+markers',
                             name='Swap Dealer Net',
                             line=dict(color=PALETTE['steel'], width=2)))
    if 'mm_net' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['mm_net'], mode='lines',
                                 name='Managed Money Net',
                                 line=dict(color=PALETTE['dim'], width=1.5, dash='dot')))
    if 'cot_divergence' in df.columns:
        marks = df[df['cot_divergence'] == 1]
        if not marks.empty:
            fig.add_trace(go.Scatter(
                x=marks['date'], y=marks['producer_net'], mode='markers',
                name='Divergence (components cancelled)',
                marker=dict(color=PALETTE['down'], size=9, symbol='x'),
                hovertemplate='%{x|%Y-%m-%d}<br>producer and swap moved '
                              'opposite ways<extra></extra>'))
    fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'])
    fig.update_layout(title='Producer vs Swap Dealer — the two halves of "commercial"',
                      xaxis_title='Date', yaxis_title='Net Contracts',
                      height=500, hovermode='x unified')
    return date_axes(fig)


def fig_implied_vs_realized(df):
    """What options charge for future movement against what the underlying has
    actually delivered. The gap is the volatility risk premium: positive means
    options are rich, negative means the market is under-pricing movement that
    is already happening."""
    if 'atm_iv' not in df.columns or df['atm_iv'].isna().all():
        return None
    if 'realized_vol_20' not in df.columns or df['realized_vol_20'].isna().all():
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=('Implied (30d constant) vs Realized (20d Parkinson)',
                                        'Volatility Risk Premium (Implied − Realized)'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['atm_iv'] * 100, mode='lines',
                             name='Implied Vol',
                             line=dict(color=PALETTE['amber'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['realized_vol_20'] * 100, mode='lines',
                             name='Realized Vol',
                             line=dict(color=PALETTE['text'], width=2)), row=1, col=1)
    if 'vrp' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['vrp'] * 100, mode='lines',
                                 name='VRP', line=dict(color=PALETTE['dim'], width=2),
                                 fill='tozeroy',
                                 fillcolor='rgba(232, 163, 61, 0.16)'), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'], row=2, col=1)
    fig.update_yaxes(title_text='Vol (%)', row=1, col=1)
    fig.update_yaxes(title_text='Vol points', row=2, col=1)
    fig.update_layout(title='Volatility Risk Premium', height=640, hovermode='x unified')
    return date_axes(fig)


def fig_volatility_regime(df):
    if 'vix' not in df.columns or df['vix'].isna().all():
        return None
    vix = df['vix'].dropna()
    current = float(vix.iloc[-1]) if len(vix) else None
    regime = '—'
    if 'vol_regime' in df.columns:
        valid = df['vol_regime'].dropna()
        if len(valid):
            regime = str(valid.iloc[-1])
    top = max(float(df['vix'].max()) * 1.1, 35)
    fig = go.Figure(go.Scatter(x=df['date'], y=df['vix'], mode='lines', name='VIX',
                               line=dict(color=PALETTE['dim'], width=2)))
    fig.add_hline(y=20, line_dash='dash', line_color=PALETTE['dim'],
                  annotation_text='Fear Threshold (20)', annotation_position='bottom right')
    fig.add_hline(y=30, line_dash='dash', line_color=PALETTE['down'],
                  annotation_text='Elevated Fear (30)', annotation_position='bottom right')
    fig.add_hrect(y0=0, y1=20, fillcolor=PALETTE['up'], opacity=0.04, line_width=0)
    fig.add_hrect(y0=30, y1=top, fillcolor=PALETTE['down'], opacity=0.04, line_width=0)
    shown = f"{current:.1f}" if current is not None else '—'
    fig.update_layout(title=f"VIX | Current: {shown} ({regime})", xaxis_title='Date',
                      yaxis_title='VIX', height=400, hovermode='x unified')
    return date_axes(fig)


def fig_inventory_band(frame, series: str):
    """Level against its own 5-year seasonal envelope — the shape energy desks
    read first. The band is min–max of the same week across the prior five
    years; the dashed line is that window's average."""
    if frame.empty:
        return None
    units = frame['units'].iloc[-1]
    label = inventory_label(series)
    banded = frame.dropna(subset=['band_min', 'band_max'])

    fig = go.Figure()
    if not banded.empty:
        fig.add_trace(go.Scatter(x=banded['period'], y=banded['band_max'], mode='lines',
                                 line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=banded['period'], y=banded['band_min'], mode='lines',
                                 line=dict(width=0), fill='tonexty',
                                 fillcolor='rgba(232, 234, 237, 0.08)',
                                 name='5-year range', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=banded['period'], y=banded['band_avg'], mode='lines',
                                 name='5-year average',
                                 line=dict(color=PALETTE['dim'], width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=frame['period'], y=frame['value'], mode='lines',
                             name=label, line=dict(color=PALETTE['amber'], width=2)))
    fig.update_layout(title=f'{label} vs 5-Year Seasonal Range',
                      xaxis_title='Week ending', yaxis_title=units,
                      height=460, hovermode='x unified')
    return date_axes(fig)


def fig_inventory_change(frame, series: str):
    """Weekly build/draw. Bars follow the sign of the change; the title says
    outright which direction is bearish so the colors aren't read as a call."""
    if frame.empty or frame['weekly_change'].isna().all():
        return None
    units = frame['units'].iloc[-1]
    changes = frame.dropna(subset=['weekly_change'])
    colors = [PALETTE['up'] if v >= 0 else PALETTE['down'] for v in changes['weekly_change']]
    fig = go.Figure(go.Bar(x=changes['period'], y=changes['weekly_change'],
                           marker_color=colors, opacity=0.85, name='Weekly change'))
    fig.add_hline(y=0, line_color=PALETTE['dim'], line_width=1)
    fig.update_layout(
        title=f'{inventory_label(series)} — Weekly Build / Draw '
              '(builds add supply: bearish for price; draws tighten: bullish)',
        xaxis_title='Week ending', yaxis_title=f'Change ({units})',
        height=400, hovermode='x unified')
    return date_axes(fig)


def fig_inventory_deviation(frame, series: str):
    """Surplus/deficit against the 5-year average — the tight-or-loose read."""
    if frame.empty or frame['dev_5y'].isna().all():
        return None
    units = frame['units'].iloc[-1]
    deviations = frame.dropna(subset=['dev_5y'])
    fig = go.Figure(go.Scatter(x=deviations['period'], y=deviations['dev_5y'],
                               mode='lines', name='vs 5-year average',
                               line=dict(color=PALETTE['amber'], width=2),
                               fill='tozeroy', fillcolor='rgba(232, 163, 61, 0.10)'))
    fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'])
    fig.update_layout(
        title=f'{inventory_label(series)} — Surplus / Deficit vs 5-Year Average',
        xaxis_title='Week ending', yaxis_title=f'Deviation ({units})',
        height=400, hovermode='x unified')
    return date_axes(fig)


def fig_seasonality(monthly: dict):
    if not monthly:
        return None
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    metrics = list(monthly)
    fig = make_subplots(rows=len(metrics), cols=1, vertical_spacing=0.12,
                        subplot_titles=[f"Avg {m} by Month" for m in metrics])
    for i, metric in enumerate(metrics, start=1):
        frame = monthly[metric].reindex(range(1, 13))
        fig.add_trace(go.Bar(x=labels, y=frame['mean'].values,
                             error_y=dict(type='data', array=frame['std'].fillna(0).values),
                             showlegend=False), row=i, col=1)
    fig.update_layout(title='Seasonal Patterns by Calendar Month', height=400 * len(metrics))
    return fig


def fig_hhi(hhi: pd.Series):
    if hhi is None or hhi.empty:
        return None
    fig = go.Figure(go.Scatter(x=hhi.index, y=hhi.values, mode='lines', name='HHI',
                               line=dict(color=PALETTE['text'], width=2)))
    fig.add_hline(y=0.25, line_dash='dash', line_color=PALETTE['amber'],
                  annotation_text='Moderate (0.25)')
    fig.add_hline(y=0.50, line_dash='dash', line_color=PALETTE['down'],
                  annotation_text='High (0.50)')
    fig.update_layout(title='Delivery Firm Concentration (HHI)', xaxis_title='Date',
                      yaxis_title='HHI', height=450, hovermode='x unified')
    return date_axes(fig)


# --- COT-only charts ---------------------------------------------------------

def cot_only_figures(df):
    figs = {}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Net Positions', 'Open Interest'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['comm_net_long'], name='Commercial Net',
                             line=dict(color=PALETTE['amber'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['noncomm_net_long'], name='Speculator Net',
                             line=dict(color=PALETTE['text'])), row=1, col=1)
    fig.add_trace(go.Bar(x=df['date'], y=df['open_interest'], name='Open Interest',
                         marker_color=PALETTE['dim']), row=2, col=1)
    fig.update_layout(title='COT Net Positions', height=700, hovermode='x unified')
    figs['net_positions'] = date_axes(fig)

    fig = go.Figure(go.Scatter(x=df['date'], y=df['comm_spec_spread'], mode='lines',
                               name='Comm − Spec Spread'))
    fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'])
    fig.update_layout(title='Commercial vs Speculator Spread', xaxis_title='Date',
                      yaxis_title='Net Contracts', height=450)
    figs['spread'] = date_axes(fig)

    roll_mean = df['comm_net_long'].rolling(52).mean()
    roll_std = df['comm_net_long'].rolling(52).std().replace(0, np.nan)
    z = (df['comm_net_long'] - roll_mean) / roll_std
    fig = go.Figure(go.Scatter(x=df['date'], y=z, mode='lines', name='Z-Score'))
    for band in (2, -2):
        fig.add_hline(y=band, line_dash='dash', line_color=PALETTE['down'])
    fig.update_layout(title='Commercial Net Z-Score (52-week)', xaxis_title='Date',
                      yaxis_title='Z-Score', height=450)
    figs['comm_zscore'] = date_axes(fig)

    fig = go.Figure()
    for year, group in df.groupby(df['date'].dt.year):
        fig.add_trace(go.Scatter(x=group['date'].dt.dayofyear, y=group['open_interest'],
                                 mode='lines', name=str(year)))
    fig.update_layout(title='Open Interest YoY Comparison', xaxis_title='Day of Year',
                      yaxis_title='Open Interest', height=500)
    figs['oi_yoy'] = fig

    fig = go.Figure(go.Scatter(x=df['date'], y=df['comm_net_ratio'], mode='lines',
                               name='Comm Net Ratio'))
    fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'])
    fig.update_layout(title='Commercial Net Ratio', xaxis_title='Date',
                      yaxis_title='Ratio', height=450)
    figs['comm_net_ratio'] = date_axes(fig)

    # CL/NG spend most of the year COT-only, which is precisely when the
    # category split and the vol premium are the only fundamentals on screen.
    for key, figure in (('producer_vs_swap', fig_producer_vs_swap(df)),
                        ('implied_vs_realized', fig_implied_vs_realized(df))):
        if figure is not None:
            figs[key] = figure
    return figs


# --- assembly ----------------------------------------------------------------

def firm_codes_by_activity(firms: pd.DataFrame) -> list:
    if firms.empty:
        return []
    totals = firms.groupby('firm_code').agg(issued=('issued', 'sum'),
                                            stopped=('stopped', 'sum'))
    totals['activity'] = totals['issued'] + totals['stopped']
    return totals.sort_values('activity', ascending=False).index.tolist()


def default_firm(firms: pd.DataFrame):
    if firms.empty:
        return None
    return str(firms.groupby('firm_code')['stopped'].sum().idxmax())


def _inventory_cards(store, asset: str, series: str) -> list:
    """Latest level, the week's build/draw, and the gap to the 5-year average."""
    summary = inventory_summary(store, asset, series)
    if not summary:
        return []
    label = inventory_label(summary['series'])
    units = summary['units']
    cards = [stat_card(f'{label} ({units})', f"{summary['value']:,.0f}")]
    if summary['weekly_change'] is not None:
        cards.append(stat_card('Weekly Change', f"{summary['weekly_change']:+,.0f}"))
    if summary['dev_5y'] is not None:
        cards.append(stat_card('vs 5-Year Average', f"{summary['dev_5y']:+,.0f}",
                               summary.get('dev_5y_pct')))
    cards.append(stat_card('Week Ending', summary['period']))
    return cards


def inventory_children(store, asset: str, selected_series: str = None) -> list:
    """Chip strip + the three inventory charts. Empty list for assets with no
    inventory configured (the metals), so nothing renders for them. Public
    because switching series replaces just this region — see
    layout.update_inventory_series."""
    available = stored_series(store, asset)
    if not available:
        return []
    series = selected_series if selected_series in available else available[0]
    frame = inventory_series(store, asset, series)
    if frame.empty:
        return []

    blocks = [ticker_rule(f'{asset} · INVENTORY')]
    if len(available) > 1:
        blocks.append(chip_strip([inventory_label(s) for s in available], 'inv-chip',
                                 inventory_label(series)))
    for figure in (fig_inventory_band(frame, series),
                   fig_inventory_change(frame, series),
                   fig_inventory_deviation(frame, series)):
        if figure is not None:
            blocks.append(html.Div([dcc.Graph(figure=figure)], className='graph-container'))
    return blocks


def stat_cards(store, asset: str, start, end, selected_inventory: str = None) -> list:
    """The summary strip: COT positioning, the latest delivery day, and the
    selected inventory series. Rebuilt whenever the inventory selection
    changes, so it never shows one series' numbers beside another's charts."""
    df = features(store, asset, start, end)
    if df.empty:
        return []
    delivery_rows = (df[df['total_issued'].notna()]
                     if 'net_flow' in df.columns else df.iloc[0:0])

    cards = []
    if 'comm_net_long' in df.columns and df['comm_net_long'].notna().any():
        cot_rows = df[df['comm_net_long'].notna()]
        latest = cot_rows.iloc[-1]
        comm_change = spec_change = None
        distinct = cot_rows.drop_duplicates(subset='cot_report_date') \
            if 'cot_report_date' in cot_rows.columns else cot_rows
        if len(distinct) > 1:
            previous = distinct.iloc[-2]
            if previous['comm_net_long']:
                comm_change = ((latest['comm_net_long'] - previous['comm_net_long'])
                               / abs(previous['comm_net_long'])) * 100
            if previous['noncomm_net_long']:
                spec_change = ((latest['noncomm_net_long'] - previous['noncomm_net_long'])
                               / abs(previous['noncomm_net_long'])) * 100
        cards += [
            stat_card('Commercial Net Position', f"{latest['comm_net_long']:,.0f}", comm_change),
            stat_card('Speculator Net Position', f"{latest['noncomm_net_long']:,.0f}", spec_change),
            stat_card('Open Interest', f"{latest['open_interest']:,.0f}"),
        ]
        if 'cot_report_date' in df.columns and pd.notna(latest.get('cot_report_date')):
            cards.append(stat_card('Latest COT Date',
                                   pd.Timestamp(latest['cot_report_date']).strftime('%Y-%m-%d')))
    if len(delivery_rows):
        last = delivery_rows.iloc[-1]
        cards += [
            stat_card('Latest Issued', f"{last['total_issued']:,.0f}"),
            stat_card('Latest Stopped', f"{last['total_stopped']:,.0f}"),
            stat_card('Month-to-Date', f"{last['month_to_date']:,.0f}"),
            stat_card('Net Flow', f"{last['total_stopped'] - last['total_issued']:+,.0f}"),
        ]
    available = stored_series(store, asset)
    if available:
        chosen = selected_inventory if selected_inventory in available else available[0]
        cards += _inventory_cards(store, asset, chosen)
    return cards


def _graph(fig, name, ml_rows=None, graph_id=None):
    if fig is None:
        if name in ('regression_performance', 'classification_performance',
                    'feature_importance', 'predicted_vs_actual') and ml_rows is not None \
                and ml_rows < ML_MIN_ROWS:
            reason = (f"Needs at least {ML_MIN_ROWS} rows of aligned data to train the ML "
                      f"models — the selected range has {ml_rows}. Widen the date range "
                      f"(roughly {ML_MIN_ROWS} trading days, about 3 months).")
        else:
            reason = "Insufficient data for this visualization in the selected date range."
        return html.Div([html.H4(f"Could not generate: {name.replace('_', ' ').title()}"),
                         html.P(reason)], className='graph-container')
    graph = dcc.Graph(figure=fig, id=graph_id) if graph_id else dcc.Graph(figure=fig)
    style = {'display': 'flex', 'justifyContent': 'center'} if name == 'correlation_heatmap' else None
    return html.Div([graph], className='graph-container', style=style)


def build(store, asset: str, start, end, selected_firm=None,
          selected_inventory=None) -> html.Div:
    df = features(store, asset, start, end)
    if df.empty:
        return html.Div("No delivery or COT data for this asset in the selected range.",
                        className='asset-not-selected')

    reports, firms = store.delivery(asset)
    if not reports.empty:
        reports['intent_date'] = pd.to_datetime(reports['intent_date'], errors='coerce')
    delivery_mode = 'net_flow' in df.columns
    # Rows that carry an actual delivery notice — the union axis is mostly
    # price days, so banner/cards/delivery charts read from these.
    delivery_rows = df[df['total_issued'].notna()] if delivery_mode else df.iloc[0:0]

    content = [ticker_rule(f"{asset} · {start} to {end}")]

    # Alert banner — judged on the most recent delivery day
    if len(delivery_rows) and bool(delivery_rows.iloc[-1].get('delivery_alert', False)):
        last = delivery_rows.iloc[-1]
        content.append(html.Div(
            f"DELIVERY ALERT — Unusual activity detected. "
            f"Net Flow Z-Score: {last.get('net_flow_zscore', float('nan')):.2f}  |  "
            f"Stopped Z-Score: {last.get('stopped_zscore', float('nan')):.2f}",
            style={'backgroundColor': PALETTE['down'], 'color': PALETTE['text'],
                   'padding': '12px 20px', 'borderRadius': '6px',
                   'fontWeight': 'bold', 'marginBottom': '16px'}))

    # Stat cards. Always emitted (class-less and empty when there is nothing
    # to show, so invisible) — switching the inventory series rewrites this
    # container in place, and a callback Output must exist in the layout.
    cards = stat_cards(store, asset, start, end, selected_inventory)
    content.append(html.Div(cards, id='stats-container',
                            className='stats-container' if cards else ''))

    # Inventory sits directly under the stat cards, above the positioning
    # charts it helps explain — and in both paths, since CL/NG are COT-only
    # for most of the year. Same always-emitted reasoning as above.
    content.append(html.Div(inventory_children(store, asset, selected_inventory),
                            id='inventory-section'))

    if not delivery_mode:
        for fig in cot_only_figures(df).values():
            content.append(html.Div([dcc.Graph(figure=fig)], className='graph-container'))
        return html.Div(content, className='sub-tab-content')

    ml = screen(store, asset, start, end)
    regression = ml['regression']
    classification = ml['classification']
    cy = convenience_yield(df)
    ml_rows = int(len(df.dropna(subset=[c for c in ('future_return_5d',) if c in df.columns])))

    active_firm = selected_firm or default_firm(firms)
    window_reports = reports[(reports['intent_date'] >= pd.to_datetime(start)) &
                             (reports['intent_date'] <= pd.to_datetime(end))] \
        if not reports.empty else reports

    ordered = [
        ('delivery_timeseries', fig_delivery_timeseries(delivery_rows)),
        ('net_flow', fig_net_flow(delivery_rows)),
        ('mtd', fig_mtd(delivery_rows)),
        ('top_firms', fig_top_firms(firms) if not firms.empty else None),
        ('firm_behavior', fig_firm_behavior(window_reports, firms, active_firm)
            if active_firm else None),
        ('correlation_heatmap', fig_correlation_heatmap(df)),
        ('delivery_vs_price', fig_delivery_vs_price(df)),
        ('regression_performance', fig_regression_performance(regression)),
        ('classification_performance', fig_classification_performance(classification)),
        ('feature_importance', fig_feature_importance(regression)),
        ('predicted_vs_actual', fig_predicted_vs_actual(regression)),
        ('basis_analysis', fig_basis(df, cy)),
        ('cot_vs_delivery', fig_cot_vs_delivery(df)),
        ('cot_extremes', fig_cot_extremes(df)),
        ('comm_vs_spec', fig_comm_vs_spec(df)),
        ('producer_vs_swap', fig_producer_vs_swap(df)),
        ('volatility_regime', fig_volatility_regime(df)),
        ('implied_vs_realized', fig_implied_vs_realized(df)),
        ('seasonality', fig_seasonality(seasonality(store, asset))),
        ('firm_concentration', fig_hhi(firm_hhi(store, asset))),
    ]
    for name, fig in ordered:
        if name == 'firm_behavior':
            content.append(chip_strip(firm_codes_by_activity(firms), 'firm-chip',
                                      active_firm, strip_id='firm-chip-strip'))
            content.append(_graph(fig, name, ml_rows, graph_id='firm-behavior-graph'))
        else:
            content.append(_graph(fig, name, ml_rows))
    return html.Div(content, className='sub-tab-content')
