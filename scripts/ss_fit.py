#!/usr/bin/env python
"""Fit the beta_t / pi_t state-space model to the real core-CPI panel.

Return observation (pre-specified primary: win45 once minute bars exist):
    --ret on      overnight r_on,  noise v_t = phi * V_bg,t
    --ret win45   08:30 -> 08:45 window return, noise v_t = v_win45,t
                  (also win35, win60 as robustness)

Outputs
    filtered and smoothed beta_t with 95% bands, against the options-implied
    |beta_impl,t| = sqrt(EV_t / sK2_t) from decomposition.py (sign set to -)
    excess_t = EV_t - (m_{t|t-1}^2 + P_{t|t-1}) sK2_t   (no look-ahead)
    by period: mean excess with Newey-West SE, and the slope of excess on
    sK2 (flat premium -> 0; mispriced sensitivity -> beta_opt^2 - beta^2 > 0)
    joint EKF parameters, reported for comparison only

Run:   python ss_fit.py --ret on
       python ss_fit.py --ret win45 [--ex-contaminated]
Reads data/decomposition_panel.csv (+ data/narrow_window_panel.csv);
writes data/ss_panel_<ret>.csv and figures/ss_beta_<ret>.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ss_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"


def load(ret, ex_contaminated):
    d = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date"])
    if ret == "on":
        d["r_obs"], d["v_obs"] = d["r_on"], d["phi"] * d["v_bg_med"]
    else:
        w = pd.read_csv(DATA / "narrow_window_panel.csv", parse_dates=["date"])
        k = ret.replace("win", "")
        d = d.merge(w[["date", f"r_win{k}", f"v_win{k}"]], on="date", how="left")
        d["r_obs"], d["v_obs"] = d[f"r_win{k}"], d[f"v_win{k}"]
    if ex_contaminated:
        d.loc[d["contaminated"].notna(), ["r_obs", "ev_med"]] = np.nan
    return d.sort_values("date").reset_index(drop=True)


def slope_se(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (X.T * e ** 2) @ X @ XtX * len(x) / (len(x) - 2)   # HC1
    return b[1], np.sqrt(V[1, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret", default="on")
    ap.add_argument("--ex-contaminated", action="store_true")
    a = ap.parse_args()
    tag = a.ret + ("_exc" if a.ex_contaminated else "")
    d = load(a.ret, a.ex_contaminated)
    data = dict(s=d["s"].values, r=d["r_obs"].values, v=d["v_obs"].values,
                ev=d["ev_med"].values, sk2=d["sk2"].values)
    data["v"] = np.where(np.isnan(data["v"]), np.nanmedian(data["v"]), data["v"])

    fr = sm.fit("returns", data)
    fj = sm.fit("joint", data)
    for f in (fr, fj):
        print(f"{f.model:8s} loglik {f.loglik:8.2f}  " + "  ".join(
            f"{k} {v:.4g} (se_log {f.se_log[k]:.2f})" for k, v in f.params.items()))

    d["beta_f"] = fr.filt.m_filt[:, 0]
    d["beta_f_sd"] = np.sqrt(fr.filt.P_filt[:, 0, 0])
    d["beta_s"] = fr.m_smooth[:, 0]
    d["beta_s_sd"] = np.sqrt(fr.P_smooth[:, 0, 0])
    d["fair_ev"] = sm.fair_event_variance(fr, data["sk2"])
    d["excess"] = d["ev_med"] - d["fair_ev"]
    d["beta_impl_signed"] = -np.sqrt(np.clip(d["ev_med"], 0, None) / d["sk2"])
    d["pi_joint"] = fj.m_smooth[:, 1]

    print(f"\n{'period':10s}{'n':>4s}{'excess':>9s}{'NW se':>8s}{'t':>7s}"
          f"{'slope sK2':>11s}{'se':>8s}{'beta_s mean':>13s}{'|b_impl| med':>14s}")
    for p, g in [("all", d)] + list(d.groupby("period")):
        mu, se = sm.newey_west_mean(g["excess"].values)
        b, bse = slope_se(g["sk2"].values, g["excess"].values)
        print(f"{p:10s}{g['excess'].notna().sum():4d}{mu:9.3f}{se:8.3f}{mu / se:7.2f}"
              f"{b:11.1f}{bse:8.1f}{g['beta_s'].mean():13.2f}"
              f"{-g['beta_impl_signed'].median():14.2f}")

    d.to_csv(DATA / f"ss_panel_{tag}.csv", index=False)
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = d["date"]
    ax[0].fill_between(x, d.beta_s - 1.96 * d.beta_s_sd, d.beta_s + 1.96 * d.beta_s_sd,
                       alpha=0.25, label="smoothed beta_t, 95%")
    ax[0].plot(x, d.beta_s, lw=2)
    ax[0].plot(x, d.beta_f, ls="--", label="filtered beta_t|t")
    ax[0].scatter(x, d.beta_impl_signed, s=14, c="k", label="-|beta_impl| (options/Kalshi)")
    ax[0].axhline(0, c="grey", lw=0.5)
    ax[0].set_ylabel("beta  (%/pp)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"CPI sensitivity, return window = {a.ret}")
    ax[1].bar(x, d.excess, width=20)
    ax[1].axhline(0, c="grey", lw=0.5)
    ax[1].set_ylabel("EV - fair EV  (%^2)")
    fig.tight_layout()
    fig.savefig(FIGS / f"ss_beta_{tag}.png", dpi=150)
    print(f"\nSaved data/ss_panel_{tag}.csv and figures/ss_beta_{tag}.png")


if __name__ == "__main__":
    main()
