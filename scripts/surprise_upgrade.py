#!/usr/bin/env python
"""Does a better surprise variable explain more of the CPI-morning move?

Compares, by period and for the 08:35 and 08:45 SPY windows (eve snapshot):
    A  core, rounded        s = actual_core - Kalshi mean          (baseline)
    B  core, unrounded      s_u = unrounded core MoM - Kalshi mean (needs fred_unrounded.py)
    C  core + headline, rounded
    D  core + headline, unrounded
Reports each coefficient with its HC1 t, R^2, adjusted R^2 and residual sd.
The comparison to read is adjusted R^2 and residual sd: a lower residual sd
means beta_t is measured with less noise per release.

Run:   python surprise_upgrade.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    p = pd.read_csv(DATA / "go_no_go_panel_eve.csv", parse_dates=["date"])
    w = pd.read_csv(DATA / "narrow_window_panel.csv", parse_dates=["date"])
    dec = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date"])[["date", "contaminated"]]
    d = dec.merge(p, on="date").merge(w[["date", "r_win35", "r_win45"]], on="date")
    d["core"], d["head"] = d["surprise_core"], d["surprise_headline"]
    specs = {"A core": ["core"], "C core+head": ["core", "head"]}
    f = DATA / "cpi_unrounded.csv"
    if f.exists():
        u = pd.read_csv(f, parse_dates=["date"])
        d = d.merge(u[["date", "core_u", "headline_u"]], on="date", how="left")
        d["core_un"] = d["core_u"] - d["k_mean_core"]
        d["head_un"] = d["headline_u"] - d["k_mean_headline"]
        specs = {"A core": ["core"], "B core unrounded": ["core_un"],
                 "C core+head": ["core", "head"], "D core+head unrounded": ["core_un", "head_un"]}
        allc = ["core", "head", "core_un", "head_un"]
    else:
        print("data/cpi_unrounded.csv not found: run fred_unrounded.py for specs B and D.\n")
        allc = ["core", "head"]
    for per, g in d.groupby("period"):
        for col in ["r_win35", "r_win45"]:
            x = g.dropna(subset=[col] + allc)      # same releases for every spec
            print(f"{per}  {col}  n = {len(x)}")
            for lab, X in specs.items():
                m = sm.OLS(x[col], sm.add_constant(x[X])).fit(cov_type="HC1")
                co = "  ".join(f"{k} {m.params[k]:6.2f} (t {m.tvalues[k]:5.2f})" for k in X)
                print(f"   {lab:24s} R2 {m.rsquared:.2f}  adjR2 {m.rsquared_adj:.2f}  "
                      f"resid sd {np.sqrt(m.scale):.3f}   {co}")
        print(f"   corr(core, head) = {g[['core', 'head']].corr().iloc[0, 1]:.2f}\n")


if __name__ == "__main__":
    main()
