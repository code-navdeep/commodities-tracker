"""Application shell and callbacks.

Layout: header → collapsible builder (row 1: mode/asset/dates/view/expiry;
row 2: the four stream checkboxes + Refresh button + status line) → main
content. Mutually-exclusive controls share grid cells and differ only by
visibility, so nothing reflows when the mode changes.
"""

import pandas as pd
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from datetime import date, timedelta

from ..config import ASSETS, STREAMS
from ..refresh import refresh
from . import cross, delivery_cot, options
from .options import DEFAULT_STRIKE_RANGE, STRIKE_RANGES, format_expiry

SINGLE_RANGE_DAYS = 30
CROSS_RANGE_YEARS = 2


def build_layout(store) -> html.Div:
    asset_options = [{'label': a, 'value': a} for a in sorted(ASSETS)]
    today = date.today()
    return html.Div(id='main-container', children=[
        html.Div(className='header-bar', children=[
            html.H1('Commodities Tracker'),
            html.A('ChorBazaar', href='https://chorbazaar.co.in', target='_blank',
                   className='header-link'),
        ]),

        html.Div(className='builder-section', children=[
            html.Button('Select  ▾', id='builder-toggle-btn', n_clicks=0,
                        className='builder-toggle-btn'),
            html.Div(id='builder-body', className='builder-body', children=[
                html.Div(className='control-row', children=[
                    html.Div([dcc.Dropdown(
                        id='mode-selector',
                        options=[{'label': 'Single Asset', 'value': 'single'},
                                 {'label': 'Cross-Asset', 'value': 'cross'}],
                        value='single', clearable=False, searchable=False,
                    )], id='mode-selector-wrapper', className='control-element'),

                    html.Div([dcc.Dropdown(
                        id='asset-dropdown', options=asset_options,
                        value=sorted(ASSETS)[0], searchable=False,
                    )], id='asset-dropdown-wrapper', className='control-element'),

                    html.Div([dcc.Dropdown(
                        id='cross-asset-selector', options=asset_options, value=[],
                        multi=True, placeholder='Select 2+ assets', searchable=False,
                    )], id='cross-asset-selector-wrapper', className='control-element',
                       style={'visibility': 'hidden'}),

                    html.Div([dcc.DatePickerRange(
                        id='date-picker',
                        min_date_allowed=date(2010, 1, 1), max_date_allowed=today,
                        start_date=today - timedelta(days=SINGLE_RANGE_DAYS),
                        end_date=today, display_format='YYYY-MM-DD',
                        className='date-picker-inline',
                    )], id='single-date-wrapper', className='control-element',
                       style={'display': 'flex', 'align-items': 'center'}),

                    html.Div([dcc.DatePickerRange(
                        id='cross-date-picker',
                        min_date_allowed=date(2010, 1, 1), max_date_allowed=today,
                        start_date=date(today.year - CROSS_RANGE_YEARS, today.month, 1),
                        end_date=today, display_format='YYYY-MM-DD',
                        className='date-picker-inline',
                    )], id='cross-date-wrapper', className='control-element',
                       style={'display': 'flex', 'align-items': 'center',
                              'visibility': 'hidden'}),

                    html.Div([dcc.Dropdown(
                        id='view-toggle',
                        options=[{'label': 'Delivery/COT', 'value': 'delivery_cot'},
                                 {'label': 'Options', 'value': 'options'}],
                        value='delivery_cot', clearable=False, searchable=False,
                    )], id='view-toggle-wrapper', className='control-element'),

                    html.Div([dcc.Dropdown(
                        id='expiry-dropdown', options=[], value=None,
                        placeholder='Options Expiry', disabled=True, searchable=False,
                    )], id='expiry-dropdown-wrapper', className='control-element',
                       style={'visibility': 'hidden'}),
                ]),

                # Refresh row — the four stream checkboxes + button + status,
                # on their own line so nothing stacks or overlaps.
                html.Div(className='fetch-row', children=[
                    dcc.Checklist(
                        id='refresh-streams',
                        options=[{'label': s.capitalize(), 'value': s} for s in STREAMS],
                        value=list(STREAMS), inline=True,
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'marginRight': '14px'},
                    ),
                    html.Button('Refresh', id='refresh-btn', n_clicks=0,
                                className='date-chip'),
                    html.Span(id='refresh-status'),
                ]),
                dcc.Interval(id='refresh-poll', interval=1500, disabled=True),
            ]),
        ]),

        dcc.Store(id='selected-firm', data=None),
        dcc.Store(id='selected-data-date', data=None),
        # Options strike window (count of strikes each side of the underlying
        # row) — a display preference, kept across asset/expiry/date changes.
        dcc.Store(id='strike-range', data=DEFAULT_STRIKE_RANGE),
        # Which inventory series the CL/NG view is showing (registry key).
        dcc.Store(id='inventory-series', data=None),
        dcc.Loading(id='loading-main', type='default',
                    target_components={'main-content': 'children'},
                    children=html.Div(id='main-content', className='tab-content')),
    ])


def _chip_date_to_iso(chip_label: str) -> str:
    """'14/08/26' (chip display format) -> '2026-08-14'."""
    day, month, year = chip_label.split('/')
    return f"20{year}-{month}-{day}"


def _render_report(report) -> html.Span:
    if isinstance(report, Exception) or report is None:
        return html.Span('✗ Refresh failed — check the server log',
                         className='stat-change-negative')
    parts = []
    for stream, result in report.streams.items():
        mark = '✓' if result.ok else '✗'
        counts = ' '.join(f"{k}:{v}" for k, v in result.counts.items() if v)
        parts.append(f"{stream} {mark}" + (f" ({counts})" if counts else ''))
    text = ' · '.join(parts) if parts else 'nothing selected'
    css = 'stat-change-positive' if report.all_ok else 'stat-change-negative'
    prefix = '✓ Done — ' if report.all_ok else '⚠ Finished with failures — '
    return html.Span(prefix + text, className=css)


def register_callbacks(app, store, runner):

    @app.callback(
        [Output('builder-body', 'style'), Output('builder-toggle-btn', 'children')],
        Input('builder-toggle-btn', 'n_clicks'))
    def toggle_builder(n_clicks):
        if n_clicks and n_clicks % 2 == 1:
            return {'display': 'none'}, 'Select  ▸'
        return {'display': 'block'}, 'Select  ▾'

    @app.callback(
        [Output('asset-dropdown-wrapper', 'style'),
         Output('cross-asset-selector-wrapper', 'style'),
         Output('single-date-wrapper', 'style'),
         Output('cross-date-wrapper', 'style'),
         Output('view-toggle-wrapper', 'style'),
         Output('expiry-dropdown-wrapper', 'style')],
        [Input('mode-selector', 'value'), Input('view-toggle', 'value')])
    def toggle_mode_fields(mode, view):
        visible, hidden = {'visibility': 'visible'}, {'visibility': 'hidden'}
        date_style = {'display': 'flex', 'align-items': 'center'}
        if mode == 'cross':
            return (hidden, visible, {**date_style, **hidden},
                    {**date_style, **visible}, hidden, hidden)
        expiry = visible if view == 'options' else hidden
        return (visible, hidden, {**date_style, **visible},
                {**date_style, **hidden}, visible, expiry)

    @app.callback(
        [Output('view-toggle', 'options'), Output('view-toggle', 'value')],
        Input('asset-dropdown', 'value'), State('view-toggle', 'value'))
    def gate_options_view(asset, current_view):
        has_options = bool(asset and store.expiries(asset))
        view_options = [
            {'label': 'Delivery/COT', 'value': 'delivery_cot'},
            {'label': 'Options', 'value': 'options', 'disabled': not has_options},
        ]
        value = 'delivery_cot' if (not has_options and current_view == 'options') else current_view
        return view_options, value

    @app.callback(
        [Output('expiry-dropdown', 'options'), Output('expiry-dropdown', 'disabled'),
         Output('expiry-dropdown', 'value')],
        Input('asset-dropdown', 'value'))
    def update_expiries(asset):
        expiries = store.expiries(asset) if asset else []
        dropdown = [{'label': format_expiry(e), 'value': e} for e in expiries]
        return dropdown, not dropdown, (expiries[0] if expiries else None)

    @app.callback(
        Output('selected-data-date', 'data'),
        [Input('asset-dropdown', 'value'), Input('expiry-dropdown', 'value')])
    def default_data_date(asset, expiry):
        dates = store.chain_dates(asset, expiry) if (asset and expiry) else []
        return dates[0] if dates else None

    @app.callback(
        Output('selected-data-date', 'data', allow_duplicate=True),
        Input({'type': 'date-chip', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True)
    def pick_data_date(clicks):
        if not any(clicks):
            raise PreventUpdate
        triggered = callback_context.triggered_id
        if not triggered:
            raise PreventUpdate
        return _chip_date_to_iso(triggered['index'])

    @app.callback(
        Output('strike-range', 'data'),
        Input({'type': 'range-chip', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True)
    def pick_strike_range(clicks):
        if not any(clicks):
            raise PreventUpdate
        triggered = callback_context.triggered_id
        if not triggered:
            raise PreventUpdate
        return STRIKE_RANGES.get(triggered['index'], DEFAULT_STRIKE_RANGE)

    @app.callback(Output('selected-firm', 'data'), Input('asset-dropdown', 'value'))
    def reset_firm(asset):
        return None

    # No reset-on-asset-change here, deliberately: `delivery_cot.build` already
    # falls back to the asset's first series when the stored key doesn't apply,
    # and a reset would fire the targeted callback below against a page that
    # hasn't been rebuilt for the new asset yet.
    @app.callback(
        Output('inventory-series', 'data', allow_duplicate=True),
        Input({'type': 'inv-chip', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True)
    def pick_inventory_series(clicks):
        if not any(clicks):
            raise PreventUpdate
        triggered = callback_context.triggered_id
        if not triggered:
            raise PreventUpdate
        # Chips carry display labels; map back to the registry key.
        label = triggered['index']
        for key, text in delivery_cot.INVENTORY_LABELS.items():
            if text == label:
                return key
        return label

    @app.callback(
        Output('selected-firm', 'data', allow_duplicate=True),
        Input({'type': 'firm-chip', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True)
    def pick_firm(clicks):
        if not any(clicks):
            raise PreventUpdate
        triggered = callback_context.triggered_id
        if not triggered:
            raise PreventUpdate
        return triggered['index']

    # Firm switch: recompute ONLY the firm chart + chip strip, not the page.
    @app.callback(
        [Output('firm-behavior-graph', 'figure'), Output('firm-chip-strip', 'children')],
        Input('selected-firm', 'data'),
        [State('asset-dropdown', 'value'), State('date-picker', 'start_date'),
         State('date-picker', 'end_date')],
        prevent_initial_call=True)
    def update_firm_chart(selected_firm, asset, start, end):
        if not asset:
            raise PreventUpdate
        reports, firms = store.delivery(asset)
        if reports.empty:
            raise PreventUpdate
        reports['intent_date'] = pd.to_datetime(reports['intent_date'])
        window = reports[(reports['intent_date'] >= pd.to_datetime(start)) &
                         (reports['intent_date'] <= pd.to_datetime(end))]
        firm = selected_firm or delivery_cot.default_firm(firms)
        figure = delivery_cot.fig_firm_behavior(window, firms, firm)
        if figure is None:
            raise PreventUpdate
        from .components import chip
        codes = delivery_cot.firm_codes_by_activity(firms)
        chips = [chip(code, 'firm-chip', code == firm) for code in codes]
        return figure, chips

    # strike-range and inventory-series are State, not Input: changing either
    # swaps only its own region via the targeted callbacks below. As Inputs
    # they rebuilt the entire page — spinner, scroll reset and all — for what
    # is a two-panel change.
    @app.callback(
        Output('main-content', 'children'),
        [Input('mode-selector', 'value'), Input('asset-dropdown', 'value'),
         Input('date-picker', 'start_date'), Input('date-picker', 'end_date'),
         Input('view-toggle', 'value'), Input('expiry-dropdown', 'value'),
         Input('selected-data-date', 'data'), Input('cross-asset-selector', 'value'),
         Input('cross-date-picker', 'start_date'), Input('cross-date-picker', 'end_date')],
        [State('strike-range', 'data'), State('inventory-series', 'data'),
         State('selected-firm', 'data')])
    def update_main(mode, asset, start, end, view, expiry, data_date,
                    cross_assets, cross_start, cross_end,
                    strike_range, inventory_series, selected_firm):
        if mode == 'cross':
            return cross.build(store, cross_assets, cross_start, cross_end)
        if not asset:
            return html.Div('Please select an asset.', className='asset-not-selected')
        if view == 'options':
            return options.build(store, asset, expiry, data_date, strike_range)
        return delivery_cot.build(store, asset, start, end, selected_firm, inventory_series)

    # --- targeted region swaps (no full-page rebuild) ---

    @app.callback(
        [Output('inventory-section', 'children'),
         Output('stats-container', 'children')],
        Input('inventory-series', 'data'),
        [State('mode-selector', 'value'), State('view-toggle', 'value'),
         State('asset-dropdown', 'value'), State('date-picker', 'start_date'),
         State('date-picker', 'end_date')],
        prevent_initial_call=True)
    def update_inventory_series(series, mode, view, asset, start, end):
        # Both outputs only exist on the single-asset Delivery/COT page; if the
        # user has navigated away since clicking, write nothing.
        if mode == 'cross' or view == 'options' or not asset:
            raise PreventUpdate
        if not delivery_cot.stored_series(store, asset):
            raise PreventUpdate
        return (delivery_cot.inventory_children(store, asset, series),
                delivery_cot.stat_cards(store, asset, start, end, series))

    @app.callback(
        [Output('options-body', 'children'),
         Output('range-chip-strip', 'children')],
        Input('strike-range', 'data'),
        [State('mode-selector', 'value'), State('view-toggle', 'value'),
         State('asset-dropdown', 'value'), State('expiry-dropdown', 'value'),
         State('selected-data-date', 'data')],
        prevent_initial_call=True)
    def update_strike_window(strike_range, mode, view, asset, expiry, data_date):
        if mode == 'cross' or view != 'options' or not (asset and expiry and data_date):
            raise PreventUpdate
        from .components import chip
        chips = [chip(label, 'range-chip', label == options.range_label(strike_range))
                 for label in options.STRIKE_RANGES]
        return (options.options_body_children(store, asset, expiry, data_date, strike_range),
                chips)

    @app.callback(
        [Output('refresh-status', 'children'),
         Output('refresh-poll', 'disabled')],
        Input('refresh-btn', 'n_clicks'), State('refresh-streams', 'value'),
        prevent_initial_call=True)
    def start_refresh(n_clicks, streams):
        streams = streams or []
        if not streams:
            return html.Span('nothing selected'), True
        started = runner.start(lambda: refresh(store, streams))
        if not started:
            raise PreventUpdate
        return html.Span(f"⏳ Refreshing {', '.join(streams)}..."), False

    @app.callback(
        [Output('refresh-status', 'children', allow_duplicate=True),
         Output('refresh-poll', 'disabled', allow_duplicate=True)],
        Input('refresh-poll', 'n_intervals'),
        prevent_initial_call=True)
    def poll_refresh(n_intervals):
        running, result = runner.status()
        if running or result is None:
            raise PreventUpdate
        return _render_report(result), True
