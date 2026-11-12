#!/usr/bin/env python
"""Placebo for the Kalshi CPI decomposition: move every CPI release to a nearby
non-event day and rerun the same statistics.

For each shift k in +-3..+-10 trading days, release i keeps its Kalshi
surprise s_i and variance sigma_K,i^2 but is placed on the trading day k
sessions away (skipped if that day is a CPI, FOMC or approximate jobs-report
day). On the placebo day, with the same conventions as the real panel:
    r_on        overnight SPY return (previous close -> 09:30 open, minute bars)
    EV          V_imp(eve) - median V_imp over non-event eves within +-5 sessions
    beta_hat    OLS slope of r_on on s
    |beta_impl| sqrt(max(mean EV, 0) / mean sigma_K^2)
If the pipeline is sound, placebo beta_hat and EV are near zero. The real
values are printed for comparison; the spread over the 16 shifts is the null
range.

Run:   python placebo.py
Reads data/decomposition_panel.csv, data/cache_cboe/VIX1D.csv, data/cache_alpaca.
Writes data/placebo.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narrow_window import load_bars, nfp_dates  # noqa: E402
from straddle import FOMC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SHIFTS = [k for k in range(-10, 11) if abs(k) >= 3]
W = 5


def ols(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xc = x - x.mean()
    b = xc @ (y - y.mean()) / (xc @ xc)
    e = y - y.mean() - b * xc
    se = np.sqrt(np.sum(xc ** 2 * e ** 2) / (xc @ xc) ** 2 * len(x) / (len(x) - 2))  # HC1
    return b, se


def stats(d):
    b, se = ols(d.s, d.r_on)
    return dict(n=len(d), beta_hat=b, beta_se=se, ev_mean=d.ev.mean(),
                beta_impl=np.sqrt(max(d.ev.mean(), 0) / d.sk2.mean()))


def main():
    panel = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date"])
    vix = pd.read_csv(DATA / "cache_cboe" / "VIX1D.csv")
    vix.columns = [c.strip().lower() for c in vix.columns]
    vix = pd.Series(vix["close"].values, index=pd.to_datetime(vix["date"]).dt.normalize())

    b = load_bars()
    b = b[(b.hm >= "09:30") & (b.hm <= "15:59")]
    o = b[b.hm == "09:30"].groupby("day")["open"].first()
    c = b.groupby("day")["close"].last()
    days = pd.DatetimeIndex(sorted(set(o.index) & set(c.index)))
    r_on = pd.Series(100 * np.log(o[days].values / c[days].shift(1).values), index=days)

    events = set(panel.date) | set(FOMC) | nfp_dates(pd.Series(days))
    pos = {d: i for i, d in enumerate(days)}
    v_imp = (vix ** 2 / 252).reindex(days)          # implied variance at the close of each day
    ev_eve = pd.Series([days[i + 1] in events if i + 1 < len(days) else True
                        for i in range(len(days))], index=days)

    def event_var(day):
        i = pos[day]
        eve = days[i - 1]
        lo, hi = max(0, i - 1 - W), min(len(days), i + W)
        nb = v_imp.iloc[lo:hi][~ev_eve.iloc[lo:hi]].drop(eve, errors="ignore")
        return v_imp[eve] - nb.median()

    real = panel[["date", "period", "s", "sk2", "r_on", "ev_med"]].rename(columns={"ev_med": "ev"})
    rows = []
    for p, g in [("all", real)] + list(real.groupby("period")):
        rows.append(dict(k=0, period=p, **stats(g.dropna())))
    for k in SHIFTS:
        pl = []
        for _, x in real.iterrows():
            if x.date not in pos:
                continue
            j = pos[x.date] + k
            if j < 1 or j >= len(days):
                continue
            d = days[j]
            if d in events or np.isnan(v_imp[days[j - 1]]):
                continue
            pl.append(dict(period=x.period, s=x.s, sk2=x.sk2, r_on=r_on[d], ev=event_var(d)))
        pl = pd.DataFrame(pl).dropna()
        for p, g in [("all", pl)] + list(pl.groupby("period")):
            rows.append(dict(k=k, period=p, **stats(g)))
    t = pd.DataFrame(rows)
    t.to_csv(DATA / "placebo.csv", index=False)

    print("Real (shift 0) vs placebo distribution over shifts +-3..+-10 trading days")
    print(f"{'period':9s}{'':8s}{'n':>5s}{'beta_hat':>10s}{'|t|':>6s}{'EV mean':>9s}{'|b_impl|':>10s}")
    for p in ["2021-23", "2024-26", "all"]:
        r = t[(t.k == 0) & (t.period == p)].iloc[0]
        print(f"{p:9s}{'real':8s}{r.n:5.0f}{r.beta_hat:10.2f}{abs(r.beta_hat / r.beta_se):6.2f}"
              f"{r.ev_mean:9.3f}{r.beta_impl:10.2f}")
        q = t[(t.k != 0) & (t.period == p)]
        for lab, f in [("median", np.median), ("max abs", lambda v: np.max(np.abs(v)))]:
            print(f"{'':9s}{lab:8s}{q.n.median():5.0f}{f(q.beta_hat):10.2f}"
                  f"{f(np.abs(q.beta_hat / q.beta_se)):6.2f}{f(q.ev_mean):9.3f}{f(q.beta_impl):10.2f}")
        print(f"{'':9s}placebo shifts with |b_impl| >= real: {(q.beta_impl >= r.beta_impl).sum()} of {len(q)}")
    print("\nSaved data/placebo.csv")


if __name__ == "__main__":
    main()
