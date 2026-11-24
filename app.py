"""Commodities Tracker — application entry point (the only executable).

Boot: open the store (creating the schema on first run), build the Dash app,
serve. All data enters through the dashboard's Refresh button; all analytics
compute on demand, cached by store revision.
"""

import logging

import dash

from commodities.config import DB_PATH
from commodities.jobs import JobRunner
from commodities.store import Store
from commodities.views import theme  # noqa: F401  (registers the Plotly template)
from commodities.views.layout import build_layout, register_callbacks

# The dashboard is long-running and interactive — INFO-level logging from
# every callback would flood the terminal. Refresh outcomes surface in the UI.
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

store = Store(DB_PATH)
store.init()
runner = JobRunner()

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Commodities Tracker'
app.layout = build_layout(store)
register_callbacks(app, store, runner)

# Security posture: the actual access control is the loopback-only publish in
# docker-compose.yml; these headers are defense-in-depth for anything already
# on the machine probing localhost. script-src needs unsafe-inline/eval —
# Dash injects inline scripts and Plotly relies on eval-like patterns.
_CSP = ("default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'")


@app.server.after_request
def security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:8057'
    response.headers['Content-Security-Policy'] = _CSP
    response.headers['Server'] = 'commodities'
    return response


if __name__ == '__main__':
    # host 0.0.0.0 is required inside Docker's bridge network; the host-side
    # publish in docker-compose.yml is loopback-only.
    app.run(debug=False, host='0.0.0.0', port=8057,
            dev_tools_silence_routes_logging=True)
