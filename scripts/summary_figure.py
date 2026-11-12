#!/usr/bin/env python
"""Headline figure from the saved result tables (no data downloads).

(a) gamma, the SPY move per 1% TLT move on release mornings, by release type and regime
(b) extra one-day variance SPX options charge on the eve of each release
(c) hedged short-straddle P&L, release eves minus quiet eves, by year

Run after macro_news.py and event_options.py:   python summary_figure.py
Writes figures/summary.png.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGS = ROOT / "data", ROOT / "figures"
C = {"2022-23": "#D55E00", "2024-26": "#0072B2"}


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    g = pd.read_csv(DATA / "macro_gamma.csv")
    g = g[(g.window == "w45") & g.period.isin(C)]
    A = pd.read_csv(DATA / "event_options_A.csv")
    D = pd.read_csv(DATA / "event_options_D.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})

    order = ["CPI", "PPI", "PCE", "JOBS", "GDP", "RETAIL"]
    for j, (reg, col) in enumerate(C.items()):
        for i, k in enumerate(order):
            r = g[(g.kind == k) & (g.period == reg)]
            if r.empty or np.isnan(r.gamma.iloc[0]) or r.hi.iloc[0] - r.lo.iloc[0] > 3:
                ax[0].text(i + (j - 0.5) * 0.3, -0.85, "n/a", ha="center", fontsize=7, color=col)
                continue
            r = r.iloc[0]
            ax[0].errorbar(i + (j - 0.5) * 0.3, r.gamma, yerr=[[r.gamma - r.lo], [r.hi - r.gamma]], fmt="o",
                           color=col, capsize=3, label=reg if i == 0 else None)
    ax[0].axhline(0, c="grey", lw=0.6)
    ax[0].axvline(2.5, c="grey", lw=0.6, ls=":")
    ax[0].text(1, 2.0, "inflation releases", ha="center", fontsize=8, color="#344444")
    ax[0].text(4, 2.0, "activity releases", ha="center", fontsize=8, color="#344444")
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels(order)
    ax[0].set_ylim(-1, 2.2)
    ax[0].set_ylabel("SPY % per 1% TLT, 08:30–08:45")
    ax[0].set_title("(a) Do stocks fall with bonds on the news?", loc="left", fontsize=10, fontweight="bold")
    ax[0].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))

    order_b = ["CPI", "FOMC", "JOBS", "PCE", "GDP", "RETAIL", "PPI", "CLAIMS"]
    for j, (reg, col) in enumerate(C.items()):
        x = A[A.regime == reg].set_index("release").reindex(order_b)
        ax[1].bar(np.arange(len(order_b)) + (j - 0.5) * 0.38, x.priced, 0.38, yerr=1.96 * x.priced_se,
                  capsize=2, color=col, label=reg)
    ax[1].axhline(0, c="grey", lw=0.6)
    ax[1].set_xticks(range(len(order_b)))
    ax[1].set_xticklabels(order_b, rotation=30)
    ax[1].set_ylabel("extra implied variance on the eve (%$^2$)")
    ax[1].set_title("(b) What SPX options charge per release", loc="left", fontsize=10, fontweight="bold")
    ax[1].legend(frameon=False, fontsize=8)

    y = D[(D.sym == "SPY@VIX1D") & D.period.str.fullmatch(r"\d{4}")]
    ax[2].bar(y.period, y["diff"], yerr=1.96 * y.se, capsize=3,
              color=[C["2022-23"] if int(p) <= 2023 else C["2024-26"] for p in y.period])
    ax[2].axhline(0, c="grey", lw=0.6)
    ax[2].set_ylabel("bp of spot, release eves − quiet eves")
    ax[2].set_title("(c) Seller's edge before releases", loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "summary.png", dpi=170)
    print("Saved", FIGS / "summary.png")


if __name__ == "__main__":
    main()
