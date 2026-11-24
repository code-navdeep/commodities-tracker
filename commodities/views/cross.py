"""Cross-asset view — comparable overlays across 2+ assets on a common
date axis, built from each asset's features() frame."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from ..analytics.features import features
from .components import date_axes
from .theme import PALETTE

CORRELATION_WINDOW = 60


def _aligned(store, assets, start, end) -> dict:
    """{asset: features frame indexed by date} — only assets with data."""
    frames = {}
    for asset in assets:
        df = features(store, asset, start, end)
        if not df.empty:
            frames[asset] = df.set_index('date')
    return frames


def fig_net_flow_overlay(frames):
    fig = go.Figure()
    for asset, df in frames.items():
        if 'net_flow' not in df.columns:
            continue
        series = df['net_flow'].dropna()
        if series.empty:
            continue
        std = series.std()
        normalized = (series - series.mean()) / std if std else series - series.mean()
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized.values, mode='lines',
                                 name=asset,
                                 hovertemplate=f'<b>{asset}</b><br>%{{x|%Y-%m-%d}}'
                                               '<br>Z-Score: %{y:.2f}<extra></extra>'))
    fig.add_hline(y=0, line_dash='dot', line_color=PALETTE['border'])
    fig.update_layout(title='Net Flow — Z-Score Normalised', xaxis_title='Date',
                      yaxis_title='Z-Score', hovermode='x unified', height=380,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='right', x=1))
    return date_axes(fig)


def fig_comm_ratio_overlay(frames):
    fig = go.Figure()
    skipped = []
    for asset, df in frames.items():
        if 'comm_net_ratio' not in df.columns:
            skipped.append(asset)
            continue
        series = df['comm_net_ratio']
        if 'has_cot' in df.columns:
            series = series.where(df['has_cot'] == 1)
        series = series.dropna()
        if series.empty:
            skipped.append(asset)
            continue
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=asset,
                                 hovertemplate=f'<b>{asset}</b><br>%{{x|%Y-%m-%d}}'
                                               '<br>Comm Net Ratio: %{y:.3f}<extra></extra>'))
    note = f"  (no COT: {', '.join(skipped)})" if skipped else ''
    fig.update_layout(title=f'Commercial Net Ratio (COT){note}', xaxis_title='Date',
                      yaxis_title='Commercial Net Ratio', hovermode='x unified', height=380,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='right', x=1))
    return date_axes(fig)


def fig_basis_overlay(frames):
    fig = go.Figure()
    skipped = []
    for asset, df in frames.items():
        if 'basis_pct' not in df.columns:
            skipped.append(asset)
            continue
        series = df['basis_pct']
        if 'has_basis' in df.columns:
            series = series.where(df['has_basis'] == 1)
        series = series.dropna()
        if series.empty:
            skipped.append(asset)
            continue
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=asset,
                                 hovertemplate=f'<b>{asset}</b><br>%{{x|%Y-%m-%d}}'
                                               '<br>Basis %: %{y:.3f}%<extra></extra>'))
    fig.add_hline(y=0, line_dash='dot', line_color=PALETTE['border'])
    note = f"  (no basis: {', '.join(skipped)})" if skipped else ''
    fig.update_layout(title=f'Basis % of Spot Price{note}', xaxis_title='Date',
                      yaxis_title='Basis (%)', hovermode='x unified', height=380,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='right', x=1))
    return date_axes(fig)


def fig_net_flow_correlation(frames):
    net_flows = pd.DataFrame({asset: df['net_flow'] for asset, df in frames.items()
                              if 'net_flow' in df.columns})
    if net_flows.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text='Insufficient net-flow data across selected assets.',
                           xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=380)
        return fig
    window = net_flows.tail(CORRELATION_WINDOW)
    corr = window.corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                               colorscale='RdBu', zmin=-1, zmax=1,
                               text=np.round(corr.values, 2), texttemplate='%{text}',
                               hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>',
                               colorbar=dict(title='r')))
    fig.update_layout(title=f'Net Flow Pairwise Correlation — Last {len(window)}-Row Window',
                      yaxis=dict(autorange='reversed'), height=380)
    return fig


def build(store, assets: list, start, end) -> html.Div:
    if not assets or len(assets) < 2:
        return html.Div('Select at least 2 assets to view the cross-asset comparison.',
                        className='asset-not-selected')
    frames = _aligned(store, assets, start, end)
    if not frames:
        return html.Div('No data could be loaded for the selected assets and range.',
                        className='asset-not-selected')
    total_dates = len(set().union(*[set(df.index) for df in frames.values()]))
    return html.Div([
        html.Div([
            html.H3('Cross-Asset Comparison', style={'marginTop': '0', 'marginBottom': '4px'}),
            html.P(f"Assets: {', '.join(frames)}  |  Period: {start} to {end}  |  "
                   f"{total_dates} trading dates",
                   style={'color': '#888', 'fontSize': '0.85em', 'marginTop': '0'}),
        ], style={'marginBottom': '12px'}),
        dcc.Graph(figure=fig_net_flow_overlay(frames), config={'scrollZoom': True}),
        dcc.Graph(figure=fig_comm_ratio_overlay(frames), config={'scrollZoom': True}),
        dcc.Graph(figure=fig_basis_overlay(frames), config={'scrollZoom': True}),
        dcc.Graph(figure=fig_net_flow_correlation(frames), config={'scrollZoom': True}),
    ])
