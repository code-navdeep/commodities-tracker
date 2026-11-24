"""Black-76 option pricing, implied volatility, and Greeks.

CME settlement chains carry prices/volume/OI but no IV or Greeks — both are
computed locally from stored facts. Black-76 prices European options on
futures:

    d1 = (ln(F/K) + 0.5 sigma^2 T) / (sigma sqrt(T));   d2 = d1 - sigma sqrt(T)
    call = e^(-rT) [F N(d1) - K N(d2)]
    put  = e^(-rT) [K N(-d2) - F N(-d1)]

Exact for the European NG product (LN); a close approximation for the
American metals/crude products (OG/SO/HX/LO) — the early-exercise premium on
futures options is small, and delta/gamma were verified to 3 decimals against
a professional reference chain.

Output scales:
    iv    — decimal (0.3611 = 36.11%)
    delta — decimal, puts negative
    gamma — per $1 move of the underlying
    theta — per calendar day (negative)
    vega  — per 1 percentage point of vol
"""

import math

from scipy.optimize import brentq

MIN_VOL = 0.0001
MAX_VOL = 5.0
MIN_T = 1.0 / 365.0 / 24.0   # one hour — below this IV inversion is meaningless

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF of a single scalar.

    Mathematically identical to scipy.stats.norm.cdf — both reduce to erfc,
    and the two agree to 1.1e-16 across [-6, 6] — but ~290x faster (measured
    15.11us vs 0.05us per call). scipy's distribution object validates and
    broadcasts every call through NumPy, which is the right design for whole
    arrays and pure overhead for one number.

    This is the hottest line in the app: brentq evaluates black76_price ~15
    times per option while inverting for IV, so one 856-strike chain reaches
    here ~29,000 times. Profiled at 89% of total runtime before the switch.
    Do not "tidy" this back to scipy.stats.
    """
    return 0.5 * math.erfc(-x / _SQRT2)


def _norm_pdf(x: float) -> float:
    """Standard normal PDF of a single scalar — same reasoning as _norm_cdf."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def black76_price(side: str, F: float, K: float, T: float, r: float, sigma: float) -> float:
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = math.exp(-r * T)
    if side == 'Call':
        return discount * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return discount * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def implied_vol(side: str, price: float, F: float, K: float, T: float, r: float):
    """Invert Black-76 for sigma. None when no vol reproduces the price —
    e.g. a deep-ITM settle at discounted intrinsic, where IV is undefined."""
    if price is None or price <= 0 or F is None or F <= 0 or K <= 0 or T is None or T < MIN_T:
        return None

    discount = math.exp(-r * T)
    intrinsic = discount * max((F - K) if side == 'Call' else (K - F), 0.0)
    if price <= intrinsic:
        return None

    def objective(sigma):
        return black76_price(side, F, K, T, r, sigma) - price

    try:
        if objective(MIN_VOL) > 0 or objective(MAX_VOL) < 0:
            return None
        return brentq(objective, MIN_VOL, MAX_VOL, xtol=1e-8, maxiter=200)
    except Exception:
        return None


def greeks(side: str, F: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Closed-form Black-76 Greeks in the app's scales (see module docstring)."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = math.exp(-r * T)
    pdf_d1 = _norm_pdf(d1)

    if side == 'Call':
        delta = discount * _norm_cdf(d1)
        theta_per_year = (
            -F * discount * pdf_d1 * sigma / (2 * sqrt_T)
            + r * F * discount * _norm_cdf(d1)
            - r * K * discount * _norm_cdf(d2)
        )
    else:
        delta = -discount * _norm_cdf(-d1)
        theta_per_year = (
            -F * discount * pdf_d1 * sigma / (2 * sqrt_T)
            - r * F * discount * _norm_cdf(-d1)
            + r * K * discount * _norm_cdf(-d2)
        )

    return {
        'delta': delta,
        'gamma': discount * pdf_d1 / (F * sigma * sqrt_T),
        'theta': theta_per_year / 365.0,
        'vega': discount * F * pdf_d1 * sqrt_T / 100.0,
    }
