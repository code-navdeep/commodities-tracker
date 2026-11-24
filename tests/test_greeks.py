"""Black-76 — round-trip precision, a textbook value, and tolerance against a
professional reference chain (Barchart CL greeks, 2025-10-31, F=60.98,
expiry 2025-11-17, r≈4%)."""

import math

from scipy.stats import norm

from commodities.sources.greeks import black76_price, greeks, implied_vol

F, T, R = 60.98, 17 / 365, 0.04

# (strike, side, barchart_iv, barchart_delta, barchart_gamma)
BARCHART_ROWS = [
    (58.0, 'Put', 0.3350, -0.220, 0.071),
    (60.0, 'Call', 0.3262, 0.609, 0.095),
    (60.0, 'Put', 0.3259, -0.390, 0.095),
    (62.0, 'Call', 0.3343, 0.416, 0.094),
]


def test_round_trip_price_iv_price():
    worst = 0.0
    for underlying in (60.0, 2650.0, 4340.0, 3.5):
        for moneyness in (0.9, 1.0, 1.1):
            for t in (30 / 365, 90 / 365):
                for sigma in (0.15, 0.35, 0.7):
                    for side in ('Call', 'Put'):
                        strike = underlying * moneyness
                        price = black76_price(side, underlying, strike, t, 0.042, sigma)
                        if price < 1e-6:
                            continue
                        recovered = implied_vol(side, price, underlying, strike, t, 0.042)
                        assert recovered is not None
                        worst = max(worst, abs(recovered - sigma))
    assert worst < 1e-6


def test_textbook_atm_value():
    # F=K=100, T=1, r=5%, sigma=20%: call = e^-0.05 * 100 * (N(0.1) - N(-0.1))
    expected = math.exp(-0.05) * 100 * (norm.cdf(0.1) - norm.cdf(-0.1))
    assert abs(black76_price('Call', 100, 100, 1.0, 0.05, 0.2) - expected) < 1e-9


def test_iv_undefined_at_discounted_intrinsic():
    intrinsic = math.exp(-R * T) * (60.98 - 51.0)
    assert implied_vol('Call', intrinsic, F, 51.0, T, R) is None
    assert implied_vol('Call', intrinsic * 0.99, F, 51.0, T, R) is None


def test_greeks_match_barchart_reference():
    # Tolerance note: Barchart's deltas pair with THEIR day-count convention;
    # combining their IV with our T shifts wings by ~0.01 (live verification
    # with self-consistent IV+delta matched to 3 decimals).
    for strike, side, iv, ref_delta, ref_gamma in BARCHART_ROWS:
        g = greeks(side, F, strike, T, R, iv)
        assert abs(g['delta'] - ref_delta) < 0.015, (strike, side)
        assert abs(g['gamma'] - ref_gamma) < 0.008, (strike, side)
        assert g['theta'] < 0
        assert g['vega'] > 0


def test_put_call_delta_relationship():
    # Black-76: delta_call - delta_put = e^(-rT)
    call = greeks('Call', F, 60.0, T, R, 0.33)['delta']
    put = greeks('Put', F, 60.0, T, R, 0.33)['delta']
    assert abs((call - put) - math.exp(-R * T)) < 1e-9
