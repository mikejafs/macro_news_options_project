#!/usr/bin/env python
"""How do stocks respond to macro news? Stock move per unit of bond move on
08:30 ET release mornings, by release type and year, 2022-2026.

Quantity. On a release morning the news moves Treasury prices (b = TLT return)
and stock prices (r = SPY return) in the same 15 minutes. gamma is the SPY
move per 1% TLT move caused by the news:
    gamma > 0   stocks fall when bonds fall: the news acts as discount-rate /
                inflation news ("good news is bad news")
    gamma ~ 0 or < 0   higher rates come with flat or higher stocks: the news
                also carries growth information

Estimator (Rigobon and Sack 2004, identification through heteroskedasticity).
Release mornings have the same background shocks as quiet mornings plus the
news shock, so
    gamma = [Cov_E(r, b) - Cov_Q(r, b)] / [Var_E(b) - Var_Q(b)]
with E = release mornings, Q = quiet mornings of the same period. It needs no
measure of the surprise, so it uses every release, not only those with a
Kalshi market. The OLS slope on release mornings is printed for comparison.
Intervals: percentile bootstrap, resampling E and Q days separately.
Relevance check: Var_E(b)/Var_Q(b) must be well above 1; estimates are not
reported when it is below 1.5.

Pre-specified (11 Nov): bond TLT; window 08:30 -> 08:45 ET (08:35 and 09:00
as robustness); calendar-year and 2022-23 vs 2024-26 splits; a morning is
labelled by its highest-ranked release in the order CPI, JOBS, PCE, GDP,
RETAIL, PPI, CLAIMS. Disclosed: CPI and approximate jobs dates were looked at
with this estimator before the wider calendar was built.

Placebo: the same statistics for 08:05 -> 08:20 on the same mornings, before
the release. Variance ratios there should be about 1.

Run:   python fred_calendar.py      (once)
       python macro_news.py
Writes data/macro_gamma.csv, data/macro_windows.csv, figures/macro_gamma.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette import RED, NAVY, GREY, INK, MUTED, PERIOD  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import narrow_window as nw  # noqa: E402

DATA, FIGS = nw.DATA, nw.ROOT / "figures"
RANK = ["CPI", "JOBS", "PCE", "GDP", "RETAIL", "PPI", "CLAIMS"]
WINDOWS = {"w45": ("08:30", "08:45"), "w35": ("08:30", "08:35"), "w60": ("08:30", "09:00"),
           "placebo": ("08:05", "08:20")}
RNG = np.random.default_rng(0)
MIN_REL = 1.5


def window_returns(sym):
    """Per day, % log return over each window. Price at time T = close of the
    last 1-minute bar starting in [T - 10 min, T - 1 min]."""
    b = nw.load_bars(sym)
    rows = []
    for day, g in b.groupby("day", sort=True):
        row = {"day": day}
        for k, (t0, t1) in WINDOWS.items():
            p0 = nw.last_close(g, nw.minus(t0, 10), nw.minus(t0, 1))
            p1 = nw.last_close(g, nw.minus(t1, 10), nw.minus(t1, 1))
            row[f"{sym}_{k}"] = 100 * np.log(p1 / p0)
        rows.append(row)
    return pd.DataFrame(rows)


def rs(E, Q, w):
    r, b = f"SPY_{w}", f"TLT_{w}"
    den = E[b].var() - Q[b].var()
    return (np.cov(E[r], E[b])[0, 1] - np.cov(Q[r], Q[b])[0, 1]) / den


def estimate(E, Q, w, n_boot=3000):
    r, b = f"SPY_{w}", f"TLT_{w}"
    rel = E[b].var() / Q[b].var()
    out = dict(n=len(E), n_quiet=len(Q), rel=rel, corr=E[[r, b]].corr().iloc[0, 1],
               ols=np.polyfit(E[b], E[r], 1)[0], gamma=np.nan, lo=np.nan, hi=np.nan,
               sd_spy=E[r].std(), sd_tlt=E[b].std())
    if len(E) >= 8 and rel >= MIN_REL:
        Ev, Qv = E[[r, b]].values, Q[[r, b]].values
        g = []
        for _ in range(n_boot):
            e = Ev[RNG.integers(0, len(Ev), len(Ev))]
            q = Qv[RNG.integers(0, len(Qv), len(Qv))]
            den = e[:, 1].var(ddof=1) - q[:, 1].var(ddof=1)
            g.append((np.cov(e.T)[0, 1] - np.cov(q.T)[0, 1]) / den)
        out.update(gamma=rs(E, Q, w), lo=np.percentile(g, 2.5), hi=np.percentile(g, 97.5))
    return out


def load_days(calendar):
    d = window_returns("SPY").merge(window_returns("TLT"), on="day")
    cal = pd.read_csv(calendar, parse_dates=["date"])
    lab = cal.groupby("date")["release"].agg(
        lambda x: min(x, key=RANK.index)).rename("kind")
    nrel = cal.groupby("date")["release"].nunique().rename("n_rel")
    d = d.merge(lab, left_on="day", right_index=True, how="left").merge(
        nrel, left_on="day", right_index=True, how="left")
    d["kind"] = d["kind"].fillna("QUIET")
    d["year"] = d["day"].dt.year
    d["regime"] = np.where(d["year"] <= 2023, "2022-23", "2024-26")
    return d


def main(calendar=DATA / "macro_calendar.csv"):
    if not Path(calendar).exists():
        raise SystemExit("data/macro_calendar.csv not found: run fred_calendar.py first.")
    d = load_days(calendar)
    d.to_csv(DATA / "macro_windows.csv", index=False)
    ok = d.dropna(subset=["SPY_w45", "TLT_w45"])
    print("Mornings by label:", ok.kind.value_counts().to_dict())
    q = ok[ok.kind == "QUIET"]
    print("\nPlacebo, 08:05 -> 08:20 (before the release): TLT variance, release / quiet mornings")
    pl = d.dropna(subset=["SPY_placebo", "TLT_placebo"])
    print("  " + "  ".join(f"{k} {pl[pl.kind == k].TLT_placebo.var() / pl[pl.kind == 'QUIET'].TLT_placebo.var():.2f}"
                           for k in RANK if (pl.kind == k).sum() >= 8))

    rows = []
    groups = [(k, ok.kind == k) for k in RANK] + [("TIER1 (all but claims)", ok.kind.isin(RANK[:-1]))]
    for w in ["w45", "w35", "w60"]:
        x = d.dropna(subset=[f"SPY_{w}", f"TLT_{w}"])
        for kind, _ in groups:
            sel = x.kind.isin(RANK[:-1]) if kind.startswith("TIER1") else x.kind == kind
            for per, m in ([(p, x.regime == p) for p in ["2022-23", "2024-26"]]
                           + [(str(y), x.year == y) for y in sorted(x.year.unique())]):
                E, Q = x[sel & m], x[(x.kind == "QUIET") & m]
                if len(E) < 5 or len(Q) < 30:
                    continue
                rows.append(dict(window=w, kind=kind, period=per, **estimate(E, Q, w)))
    t = pd.DataFrame(rows)
    t.to_csv(DATA / "macro_gamma.csv", index=False)

    def show(w, periods, title):
        print(f"\n{title}")
        print(f"{'release':24s}{'period':>9s}{'n':>5s}{'var ratio':>10s}{'sd SPY':>8s}{'sd TLT':>8s}"
              f"{'corr':>7s}{'OLS':>7s}{'gamma':>8s}{'95% CI':>17s}")
        for _, r in t[(t.window == w) & t.period.isin(periods)].iterrows():
            ci = f"[{r.lo:5.2f},{r.hi:5.2f}]" if not np.isnan(r.gamma) else "   weak / small n"
            g = f"{r.gamma:8.2f}" if not np.isnan(r.gamma) else "       ."
            print(f"{r.kind:24s}{r.period:>9s}{r.n:5.0f}{r.rel:10.1f}{r.sd_spy:8.2f}{r.sd_tlt:8.2f}"
                  f"{r['corr']:7.2f}{r.ols:7.2f}{g}{ci:>17s}")

    show("w45", ["2022-23", "2024-26"], "gamma = SPY % per 1% TLT, 08:30 -> 08:45, by regime")
    show("w45", [str(y) for y in range(2022, 2027)], "By calendar year, 08:30 -> 08:45")
    show("w35", ["2022-23", "2024-26"], "Robustness: 08:30 -> 08:35")
    show("w60", ["2022-23", "2024-26"], "Robustness: 08:30 -> 09:00")

    # figure: gamma by year for the release types with enough signal
    yrs = [str(y) for y in range(2022, 2027)]
    kinds = [k for k in ["CPI", "JOBS", "PCE", "GDP", "RETAIL", "PPI", "TIER1 (all but claims)"]
             if t[(t.window == "w45") & (t.kind == k) & t.period.isin(yrs)].gamma.notna().sum() >= 3]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.1 * len(kinds), 3.6), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], kinds):
        g = t[(t.window == "w45") & (t.kind == k) & t.period.isin(yrs)].dropna(subset=["gamma"])
        xx = g.period.astype(int)
        ax.errorbar(xx, g.gamma, yerr=[g.gamma - g.lo.clip(lower=-3), g.hi.clip(upper=4) - g.gamma],
                    fmt="o-", capsize=3, color=NAVY)
        ax.axhline(0, c="grey", lw=0.6)
        ax.set_title(k.split(" ")[0], fontsize=10)
        ax.set_xticks(range(2022, 2027))
        ax.tick_params(axis="x", labelrotation=45)
    axes[0][0].set_ylabel("gamma: SPY % per 1% TLT")
    axes[0][0].set_ylim(-2, 3.5)
    fig.suptitle("Stock move per unit of bond move on 08:30 release mornings (95% bootstrap intervals)",
                 fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "macro_gamma.png", dpi=160)
    print("\nSaved data/macro_gamma.csv, data/macro_windows.csv, figures/macro_gamma.png")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
