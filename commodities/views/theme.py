"""Visual theme: the Graphite Terminal palette + Plotly template.

Importing this module registers 'commodities_dark' and sets it as the
default, so every figure built anywhere in the views picks it up.

The palette is deliberately narrow: grayscale carries structure, amber marks
the one thing you are looking at, and red/green carry market semantics only
(issued/stopped, calls/puts, positive/negative). `steel` exists solely so a
fourth series in a multi-asset overlay stays distinguishable — it never
appears in UI chrome.
"""

import plotly.graph_objects as go
import plotly.io as pio

PALETTE = {
    'bg':     '#0a0b0d',   # page ground
    'panel':  '#111316',   # raised surfaces, chart paper
    'border': '#1e2126',   # hairlines, gridlines, zerolines
    'text':   '#e8eaed',   # primary text and primary chart series
    'dim':    '#7d838c',   # labels, secondary/analytical lines, reference rules
    'gray':   '#565c66',   # neutral bars and fills (metrics carry no direction)
    'amber':  '#e8a33d',   # THE accent: selection, emphasis, "you are here"
    'up':     '#4cc38a',   # stopped / calls / positive
    'down':   '#e5484d',   # issued / puts / negative / alerts
    'steel':  '#7fa5c4',   # chart-aux only: extra series in overlays
}

FONT_BODY = "'IBM Plex Sans', -apple-system, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, monospace"

_template = go.layout.Template()
_template.layout = go.Layout(
    dragmode='pan',
    paper_bgcolor=PALETTE['panel'],
    plot_bgcolor=PALETTE['panel'],
    font=dict(family=FONT_BODY, color=PALETTE['text'], size=12),
    title=dict(font=dict(family=FONT_BODY, color=PALETTE['text'], size=14)),
    # Order matters only for unspecified traces (multi-asset overlays);
    # semantic series always set their own color explicitly.
    colorway=[
        PALETTE['text'], PALETTE['amber'], PALETTE['steel'],
        PALETTE['up'], PALETTE['down'], PALETTE['dim'],
    ],
    xaxis=dict(gridcolor=PALETTE['border'], zerolinecolor=PALETTE['border'],
               linecolor=PALETTE['border'],
               tickfont=dict(family=FONT_MONO, color=PALETTE['dim'], size=11)),
    yaxis=dict(gridcolor=PALETTE['border'], zerolinecolor=PALETTE['border'],
               linecolor=PALETTE['border'],
               tickfont=dict(family=FONT_MONO, color=PALETTE['dim'], size=11)),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=PALETTE['text'])),
    # Mono in hover labels so figures line up column-wise; namelength=-1
    # because several trace names exceed Plotly's 15-character truncation
    # ("Commercial Net Long", ...).
    hoverlabel=dict(bgcolor=PALETTE['panel'], bordercolor=PALETTE['border'],
                    font=dict(family=FONT_MONO, color=PALETTE['text']),
                    namelength=-1),
)

pio.templates['commodities_dark'] = _template
pio.templates.default = 'commodities_dark'
