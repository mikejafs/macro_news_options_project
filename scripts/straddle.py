#!/usr/bin/env python
"""Delta-hedged short one-day ATM straddle on SPY, CPI eves vs non-event eves.

Trade (pre-specified, brief step 6):
    At the eve 16:00 close, sell one ATM straddle (K = S_eve) priced by
    Black-Scholes with sigma = VIX1D_eve / 100, t = 1/252, r = 0
    (sigma^2 t = V_imp). Hold the Black-Scholes straddle delta in SPY.
    Overnight: no rebalancing. At the 09:30 open, mark and rebalance; then
    rebalance every --every minutes; the straddle expires at the 16:00 close
    with payoff |S_close - K|.

Open mark: business time puts no decay on the overnight gap, so the implied
budget is split. Remaining implied variance at the open is (1 - phi) V_bg;
intraday sigma_d solves sigma_d^2 / 252 = (1 - phi) V_bg, with remaining time
tau_i = (1 - f_i) / 252 at session fraction f_i. V_bg = v_bg_med on CPI eves
and the eve's own V_imp on non-event eves. Total P&L does not depend on the
mark; only the overnight / intraday split does.

P&L of the short straddle (r = 0, so no financing or discounting terms):
    overnight = premium - mark_open + Delta_eve (S_open - S_eve)
    intraday  = mark_open - |S_close - K| + sum_i Delta_i (S_{i+1} - S_i)
reported in bp of S_eve (Bakshi-Kapadia normalisation).

Positive = the option seller made money = implied variance exceeded realized.

Samples: CPI eves (core panel, by period) and non-event eves (no CPI, FOMC or
approximate NFP the next day) as the placebo / benchmark.

Trading rule (if data/ss_panel_<sig>.csv exists): short the straddle when
excess_t = EV_t - fair_t > 0, long otherwise. fair_t uses beta_{t|t-1} only;
the filter's two hyperparameters are fit on the full sample (stated caveat).

Run:   python straddle.py --validate          # GBM checks of the engine
       python straddle.py                     # open + close rebalancing (primary)
       python straddle.py --every 30          # robustness: every 30 minutes
       python straddle.py --every 5
Reads data/cache_alpaca, data/cache_cboe/VIX1D.csv, data/decomposition_panel.csv.
Writes data/straddle_<every>.csv and figures/straddle_<every>.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette import RED, NAVY, GREY, INK, MUTED, PERIOD  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_bs import straddle, straddle_delta  # noqa: E402
from narrow_window import load_bars, nfp_dates  # noqa: E402
import ss_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
YEAR = 252.0
R = 0.0
REGIME_SPLIT = pd.Timestamp("2024-01-01")
FOMC = pd.to_datetime("""2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21
2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20
2023-11-01 2023-12-13 2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18
2024-11-07 2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17
2025-10-29 2025-12-10 2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16
2026-10-28 2026-12-09""".split())


# ----------------------------------------------------------------- engine

def hedge_pnl(S_eve, S_grid, sigma_eve, w_open):
    """Short-straddle P&L for arrays of trades.

    S_eve     (n,)      eve close = strike
    S_grid    (n, m+1)  open, m-1 intraday rebalance prices, close
    sigma_eve (n,)      annualized implied vol at the eve close
    w_open    (n,)      remaining implied variance at the open (decimal)
    Returns premium, overnight, intraday (all in price units, arrays (n,)).
    """
    K = S_eve
    t0 = 1.0 / YEAR
    premium = straddle(S_eve, K, sigma_eve, t0, R)
    d_eve = straddle_delta(S_eve, K, sigma_eve, t0, R)

    m = S_grid.shape[1] - 1
    sig_d = np.sqrt(w_open * YEAR)[:, None]
    tau = ((1.0 - np.arange(m) / m) / YEAR)[None, :]            # (1, m)
    S_open = S_grid[:, 0]
    mark_open = straddle(S_open, K, sig_d[:, 0], t0, R)
    deltas = straddle_delta(S_grid[:, :m], K[:, None], sig_d, tau, R)
    hedge_intra = np.sum(deltas * np.diff(S_grid, axis=1), axis=1)
    payoff = np.abs(S_grid[:, -1] - K)

    overnight = premium - mark_open + d_eve * (S_open - S_eve)
    intraday = mark_open - payoff + hedge_intra
    return premium, overnight, intraday


# -------------------------------------------------------------- validation

def validate(n=40_000, seed=0):
    """GBM checks: E[P&L] = straddle(V_imp) - straddle(V_true) exactly (mu = r = 0,
    hedge gains are martingale increments); P&L sd falls as rebalancing rises."""
    rng = np.random.default_rng(seed)
    S0, vix, phi = 100.0, 15.0, 0.44
    w_imp = (vix / 100) ** 2 / YEAR
    print("GBM validation (S0 = 100, VIX1D = 15, phi = 0.44; P&L in bp of S0)")
    print(f"{'w_true/w_imp':>13s}{'rebal/day':>10s}{'mean':>9s}{'theory':>9s}{'mc se':>8s}{'sd':>9s}")
    for ratio in (1.0, 1.5, 0.7):
        w_true = ratio * w_imp
        w_on, w_in = phi * w_true, (1 - phi) * w_true
        for m in (1, 13, 78, 390):
            S_open = S0 * np.exp(-0.5 * w_on + np.sqrt(w_on) * rng.standard_normal(n))
            z = rng.standard_normal((n, m))
            steps = -0.5 * w_in / m + np.sqrt(w_in / m) * z
            S_grid = np.column_stack([S_open, S_open[:, None] * np.exp(np.cumsum(steps, 1))])
            prem, on, intra = hedge_pnl(np.full(n, S0), S_grid, np.full(n, vix / 100),
                                        np.full(n, (1 - phi) * w_imp))
            pnl = 1e4 * (on + intra) / S0
            theory = 1e4 * (straddle(S0, S0, vix / 100, 1 / YEAR, R)
                            - straddle(S0, S0, np.sqrt(w_true * YEAR), 1 / YEAR, R)) / S0
            print(f"{ratio:13.2f}{m:10d}{pnl.mean():9.3f}{theory:9.3f}"
                  f"{pnl.std() / np.sqrt(n):8.3f}{pnl.std():9.2f}")
    print("Expect: mean = theory within ~2 mc se; sd falls roughly as m^(-1/2) intraday "
          "(the overnight gap is unhedged, so sd levels off).")


# -------------------------------------------------------------- real data

def grid_times(every):
    t = pd.date_range("2000-01-01 09:30", "2000-01-01 16:00", freq=f"{every}min")
    return [x.strftime("%H:%M") for x in t[1:-1]] if every < 390 else []


def session_table(bars, every):
    """Per trading day: open, intraday rebalance prices, close (raw SPY)."""
    times = grid_times(every)
    rows = {}
    for day, g in bars.groupby("day", sort=True):
        g = g[(g.hm >= "09:30") & (g.hm <= "15:59")]
        if len(g) < 300 or g.hm.iloc[0] != "09:30":
            continue
        hm, c = g.hm.values, g.close.values
        prices = [g.open.iloc[0]]
        for T in times:            # price at T = close of last bar starting before T
            j = np.searchsorted(hm, T) - 1
            prices.append(c[j])
        prices.append(c[-1])
        rows[day] = prices
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=390, help="rebalance interval, minutes")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--signal", default="win45", help="ss_panel tag for the trading rule")
    a = ap.parse_args()
    if a.validate:
        validate()
        return

    panel = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date", "eve"])
    phi = panel["phi"].iloc[0]
    vix = pd.read_csv(DATA / "cache_cboe" / "VIX1D.csv")
    vix.columns = [c.strip().lower() for c in vix.columns]
    vix = pd.Series(vix["close"].values, index=pd.to_datetime(vix["date"]).dt.normalize())

    sess = session_table(load_bars(), a.every)
    days = sorted(sess)
    cpi = set(panel["date"])
    skip = cpi | set(FOMC) | nfp_dates(pd.Series(days))
    bg = panel.set_index("date")["v_bg_med"]

    rows = []
    for eve, day in zip(days[:-1], days[1:]):
        if eve not in vix.index:
            continue
        kind = "CPI" if day in cpi else ("other_event" if day in skip else "non_event")
        if kind == "other_event":
            continue
        v_imp = vix[eve] ** 2 / YEAR                       # %^2
        v_bg = bg.get(day, np.nan) if kind == "CPI" else v_imp
        if np.isnan(v_bg):
            continue
        S_eve = sess[eve][-1]
        rows.append(dict(date=day, eve=eve, kind=kind, S_eve=S_eve, vix1d=vix[eve],
                         sigma=vix[eve] / 100, w_open=max((1 - phi) * v_bg, 1e-4) / 1e4,
                         grid=sess[day]))
    df = pd.DataFrame(rows)
    S_grid = np.array(df.pop("grid").tolist())
    prem, on, intra = hedge_pnl(df.S_eve.values, S_grid, df.sigma.values, df.w_open.values)
    df["premium_bp"] = 1e4 * prem / df.S_eve
    df["pnl_on_bp"] = 1e4 * on / df.S_eve
    df["pnl_intra_bp"] = 1e4 * intra / df.S_eve
    df["pnl_bp"] = df.pnl_on_bp + df.pnl_intra_bp
    df["period"] = np.where(df.date < REGIME_SPLIT, "2021-23", "2024-26")

    print(f"Short ATM straddle, delta-hedged, rebalancing: "
          f"{'open + close' if a.every >= 390 else f'every {a.every} min'}; bp of S_eve")
    print(f"{'sample':22s}{'n':>5s}{'premium':>9s}{'total':>8s}{'NW se':>7s}{'t':>6s}"
          f"{'overnight':>10s}{'intraday':>9s}{'P(>0)':>7s}")
    for (k, p), g in df.groupby(["kind", "period"]):
        mu, se = sm.newey_west_mean(g.pnl_bp.values, lags=0 if k == "CPI" else 5)
        print(f"{k + ' ' + p:22s}{len(g):5d}{g.premium_bp.mean():9.1f}{mu:8.1f}{se:7.1f}"
              f"{mu / se:6.2f}{g.pnl_on_bp.mean():10.1f}{g.pnl_intra_bp.mean():9.1f}"
              f"{(g.pnl_bp > 0).mean():7.2f}")
    print("\nCPI minus non-event (same period), Welch t:")
    for p in ["2021-23", "2024-26"]:
        x = df[(df.kind == "CPI") & (df.period == p)].pnl_bp
        y = df[(df.kind == "non_event") & (df.period == p)].pnl_bp
        d = x.mean() - y.mean()
        se = np.sqrt(x.var() / len(x) + y.var() / len(y))
        print(f"  {p}: {d:6.1f} bp (se {se:.1f}, t {d / se:.2f})")

    sig_file = DATA / f"ss_panel_{a.signal}.csv"
    if sig_file.exists():
        s = pd.read_csv(sig_file, parse_dates=["date"])[["date", "excess"]]
        c = df[df.kind == "CPI"].merge(s, on="date", how="inner").dropna(subset=["excess"])
        c["pos"] = np.where(c.excess > 0, 1.0, -1.0)       # +1 = short straddle
        c["rule_bp"] = c.pos * c.pnl_bp
        print(f"\nTrading rule on CPI eves (signal {a.signal}; +1 = short):")
        for lab, x in [("always short", c.pnl_bp), ("rule", c.rule_bp)]:
            mu, se = sm.newey_west_mean(x.values, lags=0)
            print(f"  {lab:13s} n {len(x):3d}  mean {mu:7.1f} bp  se {se:5.1f}  t {mu / se:5.2f}"
                  f"  hit {(x > 0).mean():.2f}")
        print(f"  share of releases flagged short: {(c.pos > 0).mean():.2f}")

    tag = "oc" if a.every >= 390 else str(a.every)
    df.to_csv(DATA / f"straddle_{tag}.csv", index=False)
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ne = df[df.kind == "non_event"]
    ax.plot(ne.date, ne.pnl_bp.rolling(60, min_periods=20).mean(), c=GREY,
            label="non-event eves, 60-day rolling mean")
    cp = df[df.kind == "CPI"]
    ax.bar(cp.date, cp.pnl_on_bp, width=12, color=NAVY, label="CPI: overnight")
    ax.bar(cp.date, cp.pnl_intra_bp, width=12, bottom=cp.pnl_on_bp, color=RED, label="CPI: intraday")
    ax.axhline(0, c="k", lw=0.5)
    ax.axvline(REGIME_SPLIT, ls=":", c="k")
    ax.set_ylabel("short straddle P&L (bp of S)")
    ax.legend(fontsize=8)
    ax.set_title("Delta-hedged short 1-day ATM straddle at VIX1D")
    fig.tight_layout()
    fig.savefig(FIGS / f"straddle_{tag}.png", dpi=150)
    print(f"\nSaved data/straddle_{tag}.csv and figures/straddle_{tag}.png")


if __name__ == "__main__":
    main()
