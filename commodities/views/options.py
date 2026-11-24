"""Options view — chain table + distribution/IV/GEX/Greeks charts for one
(asset, expiry, data date). A pure read over analytics.options; nothing here
touches feature engineering or ML."""

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from ..analytics.options import (chain_summary, enriched_chain, extreme_oi_changes,
                                  gex, pcr, top_strikes)
from ..analytics.options_series import options_timeseries
from .components import chip_strip, date_axes, ticker_rule
from .theme import PALETTE

CALL_COLOR = PALETTE['down']
PUT_COLOR = PALETTE['up']


# Strike-window selector: N listed strikes on each side of the underlying
# price (the brass ATM row itself — not any strike). Counting from the price
# keeps both sides symmetric; anchoring on a "nearest strike" would leave one
# side a strike short, since that strike lies on one side of the price.
STRIKE_RANGES = {'ATM ±10': 10, 'ATM ±20': 20, 'ATM ±50': 50, 'All': None}
DEFAULT_STRIKE_RANGE = 10


def filter_strikes(chain: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """The n listed strikes at/below the underlying plus the n strictly above
    it. n=None → everything; no underlying stored → everything."""
    if n is None or chain.empty:
        return chain
    underlying = chain['underlying'].iloc[0]
    if pd.isna(underlying):
        return chain
    strikes = sorted(set(chain['strike']))
    below = [s for s in strikes if s <= underlying][-n:]
    above = [s for s in strikes if s > underlying][:n]
    return chain[chain['strike'].isin(set(below) | set(above))]


def range_label(n: int | None) -> str:
    for label, value in STRIKE_RANGES.items():
        if value == n:
            return label
    return 'All'


def format_expiry(expiry_month: str) -> str:
    try:
        return pd.Timestamp(expiry_month + '-01').strftime('%b %Y')
    except Exception:
        return str(expiry_month)


def format_chip_date(iso_date: str) -> str:
    return pd.Timestamp(iso_date).strftime('%d/%m/%y')


def _fmt_count(value) -> str:
    if pd.isna(value) or value == 0:
        return '0'
    return f"{value / 1000:.1f}K" if value >= 1000 else str(int(value))


# --- chain table -------------------------------------------------------------

_TABLE_CSS = """
<style>
/* This table renders inside an iframe (srcDoc), which inherits nothing from
   the app's stylesheet — so it carries its own copy of the fonts and tokens.
   The font URLs are same-origin (Dash serves /assets/), i.e. still the local
   bundle: nothing is fetched from a CDN. Only the latin subsets are declared;
   the table's content is digits and ASCII. */
@font-face {
    font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 400;
    font-display: swap;
    src: url('/assets/fonts/plex-mono-400-latin.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 600;
    font-display: swap;
    src: url('/assets/fonts/plex-mono-600-latin.woff2') format('woff2');
}
body { margin: 0; background: #0a0b0d; }
.options-table { width: 100%; border-collapse: collapse; font-size: 0.82em;
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums; color: #e8eaed; }
.options-table th { background: #111316; color: #7d838c; padding: 7px 6px;
    text-align: center; font-weight: 600; letter-spacing: 0.6px;
    border: 1px solid #1e2126; }
.options-table td { padding: 5px 6px; text-align: center; border: 1px solid #1e2126; }
.call-column, .put-column { background-color: #111316; }
.strike-column { background-color: #0a0b0d !important; font-weight: 600 !important;
    color: #e8a33d !important; }
.otm { background-color: #0a0b0d !important; }
.pcr-column { background-color: #0a0b0d; width: 60px; color: #7d838c; }
/* The one solid amber surface in the app: you are here, at the underlying. */
.underlying-row td { background-color: #e8a33d !important; color: #0a0b0d !important;
    font-weight: 600; border-top: 2px solid #e8a33d; border-bottom: 2px solid #e8a33d; }
.underlying-row td.strike-column { background-color: #e8a33d !important;
    color: #0a0b0d !important; font-size: 15px; }
.rank-1 { background-color: rgba(76, 195, 138, 0.16) !important; color: #4cc38a !important;
    font-weight: 600; }
.rank-2 { background-color: rgba(232, 163, 61, 0.16) !important; color: #e8a33d !important;
    font-weight: 600; }
.oi-up { color: #4cc38a !important; font-weight: 600 !important; }
.oi-down { color: #e5484d !important; font-weight: 600 !important; }
.oi-max-up { border: 2px solid #4cc38a !important; }
.oi-max-down { border: 2px solid #e5484d !important; }
</style>
"""


def _side_cells(row, max_oi, max_volume, tops, extremes, side):
    """(iv, oi_chg, oi, volume, ltp) display cells + css classes for one side."""
    if row is None:
        return ['-'] * 5, [''] * 5
    iv = f"{row['iv'] * 100:.1f}%" if pd.notna(row['iv']) and row['iv'] > 0 else '-'
    has_change = pd.notna(row.get('oi_change'))
    oi_change = f"{int(row['oi_change']):+d}" if has_change else '-'
    oi_pct = (row['open_interest'] / max_oi * 100) if max_oi and row['open_interest'] else 0
    oi = f"{_fmt_count(row['open_interest'])}<br>{oi_pct:.1f}%"
    vol_pct = (row['volume'] / max_volume * 100) if max_volume and row['volume'] else 0
    volume = f"{_fmt_count(row['volume'])}<br>{vol_pct:.1f}%"
    ltp = f"{row['settle']:.2f}" if pd.notna(row['settle']) else '-'

    side_key = side.lower()
    oi_class = ('rank-1' if row['strike'] in tops[f'{side_key}_oi'][:1]
                else 'rank-2' if row['strike'] in tops[f'{side_key}_oi'][1:2] else '')
    vol_class = ('rank-1' if row['strike'] in tops[f'{side_key}_volume'][:1]
                 else 'rank-2' if row['strike'] in tops[f'{side_key}_volume'][1:2] else '')
    change_class = ''
    if has_change:
        change_class = 'oi-up' if row['oi_change'] > 0 else ('oi-down' if row['oi_change'] < 0 else '')
        extreme = extremes.get((row['strike'], side))
        if extreme == 'max_increase':
            change_class += ' oi-max-up'
        elif extreme == 'max_decrease':
            change_class += ' oi-max-down'
    return [iv, oi_change, oi, volume, ltp], ['', change_class, oi_class, vol_class, '']


def chain_table(chain: pd.DataFrame) -> html.Iframe:
    ratios = pcr(chain).set_index('strike')
    tops = top_strikes(chain)
    extremes = extreme_oi_changes(chain)
    summary = chain_summary(chain)
    underlying = chain['underlying'].iloc[0]

    calls = chain[chain['side'] == 'Call'].set_index('strike')
    puts = chain[chain['side'] == 'Put'].set_index('strike')
    max_call_oi = calls['open_interest'].max() or 1
    max_put_oi = puts['open_interest'].max() or 1
    max_call_vol = calls['volume'].max() or 1
    max_put_vol = puts['volume'].max() or 1

    strikes = sorted(set(chain['strike']))

    # The underlying row is a marker BETWEEN two rungs of the strike ladder,
    # never a rung itself — so it is placed by position and must not be
    # selected by matching a level against the price. Value-matching broke
    # whenever the settle landed exactly on a listed strike: the price was
    # concatenated onto the strike list that already held it, so the marker
    # printed twice and the at-the-money strike's own row was skipped.
    underlying_row = ''
    if pd.notna(underlying):
        call_oi_pct = summary['total_call_oi'] / max_call_oi * 100 if summary['total_call_oi'] else 0
        put_oi_pct = summary['total_put_oi'] / max_put_oi * 100 if summary['total_put_oi'] else 0
        call_vol_pct = summary['total_call_volume'] / max_call_vol * 100 if summary['total_call_volume'] else 0
        put_vol_pct = summary['total_put_volume'] / max_put_vol * 100 if summary['total_put_volume'] else 0
        underlying_row = f"""
            <tr class="underlying-row">
              <td class="pcr-column">{summary['pcr_oi']:.2f}<br>{summary['pcr_volume']:.2f}</td>
              <td class="call-column">{summary['avg_call_iv'] * 100:.1f}%</td>
              <td class="call-column">{_fmt_count(summary['net_call_oi_change'])}</td>
              <td class="call-column">{_fmt_count(summary['total_call_oi'])}<br>{call_oi_pct:.1f}%</td>
              <td class="call-column">{_fmt_count(summary['total_call_volume'])}<br>{call_vol_pct:.1f}%</td>
              <td class="call-column"></td>
              <td class="strike-column">{underlying:.2f}<br>({summary['oi_diff']:+,.0f} | {summary['iv_skew']:+.1f}%)</td>
              <td class="put-column"></td>
              <td class="put-column">{_fmt_count(summary['total_put_volume'])}<br>{put_vol_pct:.1f}%</td>
              <td class="put-column">{_fmt_count(summary['total_put_oi'])}<br>{put_oi_pct:.1f}%</td>
              <td class="put-column">{_fmt_count(summary['net_put_oi_change'])}</td>
              <td class="put-column">{summary['avg_put_iv'] * 100:.1f}%</td>
            </tr>"""

    rows_html = []
    underlying_placed = False
    for strike in strikes:
        # Strictly greater: a strike sitting exactly ON the price keeps its own
        # row and the marker follows it, matching filter_strikes' <= / > split.
        if pd.notna(underlying) and not underlying_placed and strike > underlying:
            rows_html.append(underlying_row)
            underlying_placed = True

        call = calls.reset_index()[calls.reset_index()['strike'] == strike]
        put = puts.reset_index()[puts.reset_index()['strike'] == strike]
        call_row = call.iloc[0] if len(call) else None
        put_row = put.iloc[0] if len(put) else None
        (c_iv, c_chg, c_oi, c_vol, c_ltp), (c_iv_cls, c_chg_cls, c_oi_cls, c_vol_cls, _) = \
            _side_cells(call_row, max_call_oi, max_call_vol, tops, extremes, 'Call')
        (p_iv, p_chg, p_oi, p_vol, p_ltp), (p_iv_cls, p_chg_cls, p_oi_cls, p_vol_cls, _) = \
            _side_cells(put_row, max_put_oi, max_put_vol, tops, extremes, 'Put')
        call_moneyness = '' if (pd.notna(underlying) and strike < underlying) else 'otm'
        put_moneyness = '' if (pd.notna(underlying) and strike > underlying) else 'otm'
        ratio = ratios.loc[strike] if strike in ratios.index else None
        pcr_cell = f"{ratio['pcr_oi']:.2f}<br>{ratio['pcr_volume']:.2f}" if ratio is not None else '-'
        rows_html.append(f"""
        <tr>
          <td class="pcr-column">{pcr_cell}</td>
          <td class="call-column {call_moneyness}">{c_iv}</td>
          <td class="call-column {call_moneyness} {c_chg_cls}">{c_chg}</td>
          <td class="call-column {call_moneyness} {c_oi_cls}">{c_oi}</td>
          <td class="call-column {call_moneyness} {c_vol_cls}">{c_vol}</td>
          <td class="call-column {call_moneyness}">{c_ltp}</td>
          <td class="strike-column">{strike:.2f}</td>
          <td class="put-column {put_moneyness}">{p_ltp}</td>
          <td class="put-column {put_moneyness} {p_vol_cls}">{p_vol}</td>
          <td class="put-column {put_moneyness} {p_oi_cls}">{p_oi}</td>
          <td class="put-column {put_moneyness} {p_chg_cls}">{p_chg}</td>
          <td class="put-column {put_moneyness}">{p_iv}</td>
        </tr>""")

    # Price above every listed strike (common on a narrow window after a run):
    # the loop's insertion never fires, so the marker goes at the bottom.
    if pd.notna(underlying) and not underlying_placed:
        rows_html.append(underlying_row)

    table = (_TABLE_CSS + '<table class="options-table"><thead><tr>'
             '<th class="pcr-column">PCR OI/Vol</th>'
             '<th class="call-column">IV</th><th class="call-column">OI Chg</th>'
             '<th class="call-column">OI</th><th class="call-column">Vol</th>'
             '<th class="call-column">LTP</th><th class="strike-column">Strike</th>'
             '<th class="put-column">LTP</th><th class="put-column">Vol</th>'
             '<th class="put-column">OI</th><th class="put-column">OI Chg</th>'
             '<th class="put-column">IV</th></tr></thead><tbody>'
             + ''.join(rows_html) + '</tbody></table>')
    height = 40 + len(rows_html) * 45 + 5
    return html.Iframe(srcDoc=table,
                       style={'width': '100%', 'height': f'{height}px', 'border': 'none'})


# --- charts ------------------------------------------------------------------

def _underlying_line(fig, underlying):
    if pd.notna(underlying):
        fig.add_vline(x=underlying, line_dash='dot', line_color=PALETTE['amber'],
                      line_width=2, annotation_text=f"Price: {underlying:.2f}",
                      annotation_position='top')


def fig_oi_distribution(chain):
    calls = chain[chain['side'] == 'Call']
    puts = chain[chain['side'] == 'Put']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=calls['strike'], y=calls['open_interest'], name='Call OI',
                         marker_color=CALL_COLOR, opacity=0.7))
    fig.add_trace(go.Bar(x=puts['strike'], y=puts['open_interest'], name='Put OI',
                         marker_color=PUT_COLOR, opacity=0.7))
    fig.update_layout(title='Open Interest Distribution — Calls vs Puts',
                      xaxis_title='Strike', yaxis_title='Open Interest',
                      barmode='group', hovermode='x unified')
    _underlying_line(fig, chain['underlying'].iloc[0])
    return fig


def fig_volume_distribution(chain):
    calls = chain[chain['side'] == 'Call']
    puts = chain[chain['side'] == 'Put']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=calls['strike'], y=calls['volume'], name='Call Volume',
                         marker_color=CALL_COLOR, opacity=0.7))
    fig.add_trace(go.Bar(x=puts['strike'], y=puts['volume'], name='Put Volume',
                         marker_color=PUT_COLOR, opacity=0.7))
    fig.update_layout(title='Volume Distribution — Calls vs Puts', xaxis_title='Strike',
                      yaxis_title='Volume', barmode='group', hovermode='x unified')
    _underlying_line(fig, chain['underlying'].iloc[0])
    return fig


def fig_oi_change(chain):
    if 'oi_change' not in chain.columns or chain['oi_change'].isna().all():
        fig = go.Figure()
        fig.add_annotation(text='No previous data date stored for OI change calculation',
                           xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title='OI Change Distribution',
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    fig = go.Figure()
    for side, group_offset in (('Call', 1), ('Put', 2)):
        rows = chain[chain['side'] == side]
        colors = [PUT_COLOR if v >= 0 else CALL_COLOR for v in rows['oi_change'].fillna(0)]
        fig.add_trace(go.Bar(x=rows['strike'], y=rows['oi_change'], name=f'{side} OI Δ',
                             marker_color=colors, opacity=0.7, offsetgroup=group_offset))
    fig.add_hline(y=0, line_color=PALETTE['dim'], line_width=1)
    fig.update_layout(title='OI Change Distribution', xaxis_title='Strike',
                      yaxis_title='OI Change', barmode='group', hovermode='x unified')
    _underlying_line(fig, chain['underlying'].iloc[0])
    return fig


def fig_iv_smile(chain):
    calls = chain[chain['side'] == 'Call']
    puts = chain[chain['side'] == 'Put']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=calls['strike'], y=calls['iv'], mode='lines+markers',
                             name='Call IV', line=dict(color=CALL_COLOR, width=2),
                             marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=puts['strike'], y=puts['iv'], mode='lines+markers',
                             name='Put IV', line=dict(color=PUT_COLOR, width=2),
                             marker=dict(size=8)))
    fig.update_layout(title='Implied Volatility Smile', xaxis_title='Strike',
                      yaxis_title='Implied Volatility', hovermode='x unified')
    _underlying_line(fig, chain['underlying'].iloc[0])
    return fig


def fig_gex(chain):
    exposure = gex(chain)
    colors = [PUT_COLOR if v >= 0 else CALL_COLOR for v in exposure['gex']]
    fig = go.Figure(go.Bar(x=exposure['strike'], y=exposure['gex'], marker_color=colors,
                           name='GEX', opacity=0.8))
    fig.add_hline(y=0, line_color=PALETTE['dim'], line_width=1)
    fig.update_layout(title='Gamma Exposure (GEX) by Strike', xaxis_title='Strike',
                      yaxis_title='Gamma Exposure', hovermode='x unified')
    _underlying_line(fig, chain['underlying'].iloc[0])
    return fig


def fig_greeks_panel(chain):
    calls = chain[chain['side'] == 'Call']
    puts = chain[chain['side'] == 'Put']
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega'))
    positions = {'delta': (1, 1), 'gamma': (1, 2), 'theta': (2, 1), 'vega': (2, 2)}
    for greek, (row, col) in positions.items():
        show = greek == 'delta'
        fig.add_trace(go.Scatter(x=calls['strike'], y=calls[greek], name=f'Call {greek}',
                                 mode='lines+markers', line=dict(color=CALL_COLOR, width=2),
                                 marker=dict(size=6), showlegend=show), row=row, col=col)
        fig.add_trace(go.Scatter(x=puts['strike'], y=puts[greek], name=f'Put {greek}',
                                 mode='lines+markers', line=dict(color=PUT_COLOR, width=2),
                                 marker=dict(size=6), showlegend=show), row=row, col=col)
    underlying = chain['underlying'].iloc[0]
    if pd.notna(underlying):
        for row in (1, 2):
            for col in (1, 2):
                fig.add_vline(x=underlying, line_dash='dot', line_color=PALETTE['amber'],
                              line_width=2, row=row, col=col)
    fig.update_layout(title_text='Greeks Distribution', height=800)
    return fig


def fig_options_history(series):
    """How the surface has moved, rather than what it looks like today.

    Implied vol is shown at a constant 30-day horizon; without that the series
    sawtooths every time the front expiry rolls. Skew is the 25-delta risk
    reversal — positive means downside protection costs more than upside.
    """
    if series.empty or series['atm_iv'].isna().all():
        return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=('ATM Implied Volatility (30d constant maturity)',
                                        'Put/Call Ratio (open interest)',
                                        '25-Delta Skew (put IV − call IV)'))
    fig.add_trace(go.Scatter(x=series['date'], y=series['atm_iv'] * 100, mode='lines',
                             name='ATM IV', line=dict(color=PALETTE['amber'], width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=series['date'], y=series['pcr_oi'], mode='lines',
                             name='PCR (OI)', line=dict(color=PALETTE['text'], width=2)),
                  row=2, col=1)
    fig.add_hline(y=1.0, line_dash='dash', line_color=PALETTE['dim'], row=2, col=1)
    fig.add_trace(go.Scatter(x=series['date'], y=series['iv_skew_25d'] * 100, mode='lines',
                             name='25Δ Skew', line=dict(color=PALETTE['steel'], width=2)),
                  row=3, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color=PALETTE['dim'], row=3, col=1)
    fig.update_yaxes(title_text='Vol (%)', row=1, col=1)
    fig.update_yaxes(title_text='Ratio', row=2, col=1)
    fig.update_yaxes(title_text='Vol points', row=3, col=1)
    fig.update_layout(title='Options History', height=760, hovermode='x unified',
                      showlegend=False)
    return date_axes(fig)


# --- assembly ----------------------------------------------------------------

def options_body_children(store, asset: str, expiry_month: str, data_date: str,
                          strike_range: int | None = DEFAULT_STRIKE_RANGE) -> list:
    """Everything that depends on the strike window: the chain table and all
    six charts. Split out of build() so switching the range can replace just
    this region instead of the whole page (see layout.update_strike_window)."""
    chain = enriched_chain(store, asset, expiry_month, data_date)
    if chain.empty:
        return [html.Div('No options data for this selection.',
                         className='asset-not-selected')]

    # Table highlights, percentages, aggregates and charts are all computed
    # over the selected window, so the display stays internally consistent.
    chain = filter_strikes(chain, strike_range)

    return [
        html.Div([chain_table(chain)], className='graph-container',
                 style={'marginBottom': '30px'}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_oi_distribution(chain))],
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_volume_distribution(chain))],
                     style={'width': '50%', 'display': 'inline-block'}),
        ], className='graph-container'),
        html.Div([dcc.Graph(figure=fig_oi_change(chain))], className='graph-container'),
        html.Div([
            html.Div([dcc.Graph(figure=fig_iv_smile(chain))],
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_gex(chain))],
                     style={'width': '50%', 'display': 'inline-block'}),
        ], className='graph-container'),
        html.Div([dcc.Graph(figure=fig_greeks_panel(chain))], className='graph-container'),
    ]


def build(store, asset: str, expiry_month: str, data_date: str,
          strike_range: int | None = DEFAULT_STRIKE_RANGE) -> html.Div:
    if not (asset and expiry_month and data_date):
        return html.Div('No options data selected.', className='asset-not-selected')
    if enriched_chain(store, asset, expiry_month, data_date).empty:
        return html.Div('No options data for this selection.', className='asset-not-selected')

    dates = [format_chip_date(d) for d in store.chain_dates(asset, expiry_month)]

    # Chain-wide and across dates, so it is independent of the strike window —
    # it lives OUTSIDE options-body, which the range chips replace wholesale.
    history = fig_options_history(options_timeseries(store, asset))

    return html.Div([
        ticker_rule(f"{asset} · {format_expiry(expiry_month)}"),
        chip_strip(dates, 'date-chip', format_chip_date(data_date)),
        chip_strip(list(STRIKE_RANGES), 'range-chip', range_label(strike_range),
                   strip_id='range-chip-strip'),
        html.Div(options_body_children(store, asset, expiry_month, data_date, strike_range),
                 id='options-body'),
        html.Div([dcc.Graph(figure=history)] if history else [],
                 id='options-history',
                 className='graph-container' if history else ''),
    ], className='sub-tab-content')
