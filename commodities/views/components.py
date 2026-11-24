"""Small shared UI builders used by every view."""

from dash import html


def stat_card(title: str, value: str, change: float | None = None) -> html.Div:
    children = [html.Div([
        html.Div(title, className='stat-title'),
        html.Div(value, className='stat-value'),
    ], className='stat-text')]
    if change is not None:
        css = 'stat-change-positive' if change >= 0 else 'stat-change-negative'
        arrow = '▲' if change >= 0 else '▼'
        children.append(html.Div(f"{arrow} {abs(change):.1f}%", className=f'stat-change {css}'))
    return html.Div(children, className='stat-card')


def ticker_rule(label: str) -> html.Div:
    return html.Div(html.Span(label, className='ticker-rule-label'), className='ticker-rule')


def chip(label: str, chip_type: str, active: bool) -> object:
    from dash import html as h
    return h.Button(label, id={'type': chip_type, 'index': label}, n_clicks=0,
                    className='date-chip' + (' date-chip-active' if active else ''))


def chip_strip(labels: list, chip_type: str, active: str, strip_id: str = None) -> html.Div:
    if not labels:
        return html.Div("No data available.", className='date-chip-strip-empty',
                        **({'id': strip_id} if strip_id else {}))
    chips = [chip(label, chip_type, label == active) for label in labels]
    return html.Div(chips, className='date-chip-strip',
                    **({'id': strip_id} if strip_id else {}))


def date_axes(fig):
    """Force date-only tick/hover format — Plotly falls to millisecond
    precision on narrow ranges otherwise."""
    fig.update_xaxes(tickformat='%Y-%m-%d', hoverformat='%Y-%m-%d')
    return fig
