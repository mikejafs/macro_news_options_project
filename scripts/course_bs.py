"""Black-Scholes pricing and delta, copied unchanged from the Erdos course
notebooks (Lecture 6 "Complete BS Greeks", Lecture 7 "Completed_Delta_Hedging",
Lecture 8 "Completed BS_Assumptions_Delta_Backtesting").

S0 spot, K strike, sigma annualized volatility, t time to expiry in years,
r continuously compounded risk-free rate. All functions accept numpy arrays.
"""
import numpy as np
from scipy.stats import norm


def bs_call(S0, K, sigma, t, r):
    d1 = (np.log(S0 / K) + (r + (0.5) * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S0 * norm.cdf(d1) - np.exp(-r * t) * K * norm.cdf(d2)


def bs_put(S0, K, sigma, t, r):
    d1 = (np.log(S0 / K) + (r + (0.5) * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return -S0 * norm.cdf(-d1) + np.exp(-r * t) * K * norm.cdf(-d2)


def bs_call_delta(S0, K, sigma, t, r):
    d1 = (np.log(S0 / K) + (r + (0.5) * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return norm.cdf(d1)


def bs_put_delta(S0, K, sigma, t, r):
    d1 = (np.log(S0 / K) + (r + (0.5) * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return -norm.cdf(-d1)


def bs_gamma(S0, K, sigma, t, r):
    d1 = (np.log(S0 / K) + (r + (0.5) * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(t))


# Straddle = call + put at the same strike (project addition, built from the above)
def straddle(S0, K, sigma, t, r):
    return bs_call(S0, K, sigma, t, r) + bs_put(S0, K, sigma, t, r)


def straddle_delta(S0, K, sigma, t, r):
    return bs_call_delta(S0, K, sigma, t, r) + bs_put_delta(S0, K, sigma, t, r)
