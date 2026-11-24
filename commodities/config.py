"""Asset registry and application constants — the single source of truth.

Adding an asset to the app means adding one entry to ASSETS. Every fetcher,
view, and analytic derives what it does from this registry; nothing anywhere
else hardcodes an asset.

Identifiers verified live against each source (2026-08):
  cftc_code  — cftc_contract_market_code in CFTC dataset kh3c-gbw2
               (Disaggregated, Futures + Options Combined)
  fut_id     — CME product id for the futures settlements endpoints
  opt_id     — CME product id for the standard monthly options product
               (weeklies/dailies/exotics are separate products and never fetched)
  opt_style  — 'AME' or 'EUR'; Black-76 is exact for EUR and a close
               approximation for AME futures options
  spot       — daily USD spot source: 'lbma_gold' | 'lbma_silver' |
               'eia:<SERIES>' | None (no clean free source — copper)
  delivery_keywords — substrings identifying this asset's CONTRACT lines in
               CME's Issues & Stops delivery PDFs
"""

import os
from pathlib import Path

REBUILD_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REBUILD_ROOT / 'data' / 'fundamentals.db'

# Fetch windows and limits
HISTORY_DAYS = 730                # backfill horizon for prices/COT on a fresh DB
OPTIONS_MAX_MONTHS_AHEAD = 3      # options: only expirations this many months out
COT_STALE_DAYS = 14               # null COT joined to delivery when older than this
INVENTORY_HISTORY_YEARS = 8       # 5-year band needs 5 *prior* years at every
                                  # point shown, so fetch well past the window
INVENTORY_STALE_DAYS = 21         # weekly cadence + EIA's 5-12 day publication lag,
                                  # so COT's 14 would null perfectly current data
INVENTORY_OVERLAP_WEEKS = 3       # incremental refresh re-pulls this many recent
                                  # weeks: EIA revises them, and the upsert then
                                  # overwrites our stored copy

# Analytics
ML_MIN_ROWS = 60                  # minimum aligned rows before ML runs (5-fold TSSplit)
ML_COVERAGE_FLOOR = 0.7           # a feature joins the model only while the rows
                                  # surviving dropna() stay at this fraction of the
                                  # densest column's count — without it one sparse
                                  # family collapses the whole training set
ZSCORE_WINDOW = 30                # rolling window for the delivery alert z-score
SEASONAL_BAND_YEARS = 5           # prior years forming the min/max/avg band
COT_INDEX_WINDOW_WEEKS = 156      # 3y lookback for the 0-100 COT percentile —
                                  # the convention desks quote
COT_INDEX_MIN_WEEKS = 52          # ...but HISTORY_DAYS only backfills ~104 weeks,
                                  # so the full window can never fill on a fresh
                                  # DB. Compute over whatever exists past a year
                                  # and let it converge as history accrues;
                                  # without this the index is permanently NaN
REALIZED_VOL_WINDOWS = (20, 60)   # trading days for Parkinson realized volatility
IV_TARGET_DAYS = 30               # constant-maturity horizon for the ATM IV series
SKEW_DELTA = 0.25                 # |delta| of the risk-reversal skew wings

# Refreshable data streams, in run order
STREAMS = ['delivery', 'prices', 'cot', 'inventory', 'options']

# Free key from https://www.eia.gov/opendata/register.php — CL/NG daily spot.
# Read from the environment (docker-compose forwards it from .env).
EIA_API_KEY = os.environ.get('EIA_API_KEY', '')

ASSETS = {
    'GC': {
        'name': 'Gold',
        'cftc_code': '088691',
        'fut_id': 437,
        'opt_id': 192,            # OG
        'opt_style': 'AME',
        'spot': 'lbma_gold',
        'delivery_keywords': ['COMEX 100 GOLD'],
    },
    'SI': {
        'name': 'Silver',
        'cftc_code': '084691',
        'fut_id': 458,
        'opt_id': 193,            # SO
        'opt_style': 'AME',
        'spot': 'lbma_silver',
        'delivery_keywords': ['COMEX 5000 SILVER'],
    },
    'HG': {
        'name': 'Copper',
        'cftc_code': '085692',
        'fut_id': 438,
        'opt_id': 797,            # HX
        'opt_style': 'AME',
        'spot': None,             # no clean free daily copper spot (LME is licensed)
        'delivery_keywords': ['COMEX COPPER'],
    },
    'CL': {
        'name': 'WTI Crude Oil',
        'cftc_code': '067651',
        'fut_id': 425,
        'opt_id': 190,            # LO
        'opt_style': 'AME',
        # EIA's /v2/seriesid/ route wants the full legacy series ID
        # (<dataset>.<series>.<frequency>) — the bare name 404s.
        'spot': 'eia:PET.RWTC.D',   # WTI Cushing daily spot
        'delivery_keywords': ['CRUDE OIL'],
        # Weekly EIA stocks/flows. First entry is the headline series (stat
        # cards, ML features); the rest are selectable in the view. Cushing
        # is the contract's physical delivery point, so it moves the front
        # month hardest. `production` is a flow (MBBL/D), not a stock — kept
        # here because it is the supply side of the same weekly report.
        'inventory': {
            'us_crude':   'PET.WCESTUS1.W',               # US stocks excl. SPR
            'cushing':    'PET.W_EPC0_SAX_YCUOK_MBBL.W',  # NYMEX delivery point
            'spr':        'PET.WCSSTUS1.W',               # Strategic Petroleum Reserve
            'gasoline':   'PET.WGTSTUS1.W',
            'distillate': 'PET.WDISTUS1.W',
            'production': 'PET.WCRFPUS2.W',               # MBBL/D
        },
    },
    'NG': {
        'name': 'Henry Hub Natural Gas',
        'cftc_code': '023651',
        'fut_id': 444,
        'opt_id': 1352,           # LN (European)
        'opt_style': 'EUR',
        'spot': 'eia:NG.RNGWHHD.D',  # Henry Hub daily spot (see CL note above)
        'delivery_keywords': ['NATURAL GAS'],
        # Working gas in underground storage (BCF). Lower 48 is the headline
        # number the market trades; the five regions decompose it and should
        # sum back to it.
        'inventory': {
            'working_gas':   'NG.NW2_EPG0_SWO_R48_BCF.W',  # Lower 48 total
            'east':          'NG.NW2_EPG0_SWO_R31_BCF.W',
            'midwest':       'NG.NW2_EPG0_SWO_R32_BCF.W',
            'mountain':      'NG.NW2_EPG0_SWO_R33_BCF.W',
            'pacific':       'NG.NW2_EPG0_SWO_R34_BCF.W',
            'south_central': 'NG.NW2_EPG0_SWO_R35_BCF.W',
        },
    },
}

# CFTC codes for assets not yet tracked end-to-end — data for future entries.
COT_ONLY_CODES = {
    'PL': '076651',   # Platinum
    'PA': '075651',   # Palladium
    'HO': '022651',   # NY Harbor ULSD
    'RB': '111659',   # RBOB Gasoline
}
