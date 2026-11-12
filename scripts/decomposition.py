#!/usr/bin/env python
"""Step 3: static decomposition of CPI event variance, by period.

Model. On a release day the overnight return is r = beta*s + e, with s the
CPI surprise against the Kalshi eve-close mean. If Kalshi's eve distribution
is calibrated, E[s^2] = sigma_K^2. The CPI part of the move then has expected
variance beta^2 * sigma_K^2.

Quantities per period (means over releases, %^2 unless stated)
  EV            options-implied event variance (VIX1D eve minus background)
  beta_hat      realized sensitivity: OLS slope of r_on on s (%/pp), HC1 SE
  beta_impl     options-implied sensitivity: sqrt(mean EV / mean sigma_K^2),
                the |beta| that reconciles the options price with Kalshi's
                uncertainty
  fund          beta_hat^2 * mean sigma_K^2: fair CPI event variance given
                Kalshi uncertainty and realized sensitivity
  cpi_real      beta_hat^2 * mean s^2: CPI-explained realized event variance
  ev_real       mean(r_on^2 - phi*V_bg): realized event variance, all sources

Decomposition of options' pricing of the CPI move
  EV / cpi_real = (beta_impl / beta_hat)^2  x  (mean sigma_K^2 / mean s^2)
                   sensitivity factor           uncertainty factor
A sensitivity factor > 1 means options price stocks as more CPI-sensitive
than they turned out to be. An uncertainty factor > 1 means Kalshi's ex-ante
spread exceeded realized surprises (if options used the same uncertainty,
they over-priced it too). 95% CIs from a release-level bootstrap.

Run:   python decomposition.py                (core, all releases)
       python decomposition.py --measure headline
Reads  data/event_variance_panel.csv, data/go_no_go_panel_eve.csv
Writes data/decomposition_panel.csv, figures/decomposition_beta.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

# Releases contaminated by non-CPI news, fixed by external events, not by
# residual size. Results are reported with and without them.
CONTAMINATED = {
    "2024-06-12": "FOMC decision same day",
    "2025-04-10": "day after the 2025-04-09 tariff-pause rally",
}
N_BOOT = 2000
ROLL = 8   # releases per rolling beta window in the figure


def ols_hc1(x, y):
    """Slope and HC1 standard error of y = a + b x."""
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    V = n / (n - k) * XtX_inv @ (X.T * e ** 2) @ X @ XtX_inv
    return b[1], np.sqrt(V[1, 1])


def stats_for(g):
    """Decomposition quantities for one set of releases."""
    s, sk2, r, ev = g["s"].to_numpy(), g["sk2"].to_numpy(), g["r_on"].to_numpy(), g["ev_med"].to_numpy()
    b, se = ols_hc1(s, r)
    m_sk2, m_s2, m_ev = sk2.mean(), (s ** 2).mean(), ev.mean()
    b_impl = np.sqrt(max(m_ev, 0) / m_sk2)
    return {
        "n": len(g), "beta_hat": b, "beta_se": se, "beta_impl": b_impl,
        "sigma_K2": m_sk2, "s2": m_s2, "EV": m_ev,
        "fund": b ** 2 * m_sk2, "cpi_real": b ** 2 * m_s2,
        "ev_real": (g["r_on"] ** 2 - g["phi"] * g["v_bg_med"]).mean(),
        "sens": (b_impl / b) ** 2 if b != 0 else np.nan,
        "unc": m_sk2 / m_s2,
        "total": m_ev / (b ** 2 * m_s2) if b != 0 else np.nan,
    }


def bootstrap(g, keys, rng):
    draws = []
    idx = np.arange(len(g))
    for _ in range(N_BOOT):
        gb = g.iloc[rng.choice(idx, len(idx), replace=True)]
        if gb["s"].std() == 0:
            continue
        st = stats_for(gb)
        draws.append([st[k] for k in keys])
    d = np.array(draws, float)
    return {k: np.nanpercentile(d[:, i], [2.5, 97.5]) for i, k in enumerate(keys)}


def report(df, label, rng):
    print(f"\n=== {label} ===")
    keys = ["beta_hat", "beta_impl", "sens", "unc", "total"]
    rows = []
    for per, g in [("all", df), *df.groupby("period")]:
        st = stats_for(g)
        ci = bootstrap(g, keys, rng)
        st.update({f"{k}_ci": ci[k] for k in keys})
        st["period"] = per
        rows.append(st)
    print(f"  {'period':8s} {'n':>3s} {'beta_hat':>15s} {'|beta_impl|':>18s} "
          f"{'sigma_K2':>8s} {'s2':>7s} {'EV':>6s} {'fund':>6s} {'cpi_real':>8s} {'ev_real':>7s}")
    for st in rows:
        print(f"  {st['period']:8s} {st['n']:3d} {st['beta_hat']:7.2f} ({st['beta_se']:4.2f}) "
              f"{st['beta_impl']:6.2f} [{st['beta_impl_ci'][0]:4.1f},{st['beta_impl_ci'][1]:4.1f}] "
              f"{st['sigma_K2']:8.4f} {st['s2']:7.4f} {st['EV']:6.2f} {st['fund']:6.2f} "
              f"{st['cpi_real']:8.2f} {st['ev_real']:7.2f}")
    print("  EV / cpi_real = sensitivity factor x uncertainty factor   [95% bootstrap CI]")
    for st in rows:
        f = lambda k: f"{st[k]:6.2f} [{st[k + '_ci'][0]:5.2f},{st[k + '_ci'][1]:6.2f}]"
        print(f"  {st['period']:8s} total {f('total')}  = sens {f('sens')}  x  unc {f('unc')}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", choices=["core", "headline"], default="core")
    args = ap.parse_args()
    m = args.measure

    ev = pd.read_csv(DATA / "event_variance_panel.csv", parse_dates=["date"])
    eve = pd.read_csv(DATA / "go_no_go_panel_eve.csv", parse_dates=["date"])
    eve = eve[["date", f"k_sd_{m}", f"surprise_{m}", f"k_mean_{m}"]].rename(
        columns={f"k_sd_{m}": "k_sd", f"surprise_{m}": "s", f"k_mean_{m}": "k_mean"})
    df = ev.merge(eve, on="date", how="inner")
    df["sk2"] = df["k_sd"] ** 2
    df = df.dropna(subset=["s", "sk2", "r_on", "ev_med", "phi", "v_bg_med"]).sort_values("date")
    df["contaminated"] = df["date"].dt.strftime("%Y-%m-%d").map(CONTAMINATED).fillna("")
    print(f"Measure: {m} CPI, Kalshi eve-close snapshot. Releases: {len(df)} "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}); "
          f"phi = {df['phi'].iloc[0]:.2f}")

    rng = np.random.default_rng(0)
    report(df, "All releases", rng)
    clean = df[df["contaminated"] == ""]
    report(clean, f"Excluding contaminated: {', '.join(CONTAMINATED)}", rng)

    # Per-release implied beta and rolling realized beta
    df["beta_impl_t"] = np.sqrt(df["ev_med"].clip(lower=0)) / df["k_sd"]
    roll = []
    for i in range(len(df)):
        w = df.iloc[max(0, i - ROLL + 1): i + 1]
        roll.append(ols_hc1(w["s"].to_numpy(), w["r_on"].to_numpy())[0] if len(w) == ROLL else np.nan)
    df["beta_roll"] = roll
    df.to_csv(DATA / "decomposition_panel.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(df["date"], df["beta_impl_t"], s=18, color="tab:purple",
               label="|β| implied by options: √EV / σ_K (per release)")
    ax.plot(df["date"], df["beta_roll"].abs(), color="tab:orange", lw=2,
            label=f"|β| realized: rolling OLS, last {ROLL} releases")
    ax.axvline(pd.Timestamp("2024-01-01"), color="grey", lw=0.8, ls="--")
    ax.set(ylabel="CPI sensitivity |β| (% SPY per pp surprise)",
           title=f"Options-implied vs realized SPY sensitivity to {m} CPI surprises")
    ax.legend(loc="upper right")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "decomposition_beta.png", dpi=150)
    print(f"\nSaved {DATA / 'decomposition_panel.csv'} and {FIGS / 'decomposition_beta.png'}")


if __name__ == "__main__":
    main()