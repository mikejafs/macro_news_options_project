#!/usr/bin/env python
"""Narrow-window SPY returns around the 08:30 ET CPI release, from minute bars.

Pre-specified (10 Nov, before seeing any window returns):
    pre   = close of the last bar starting in [08:20, 08:29] ET (price at 08:30:00)
    post  = close of the last bar starting in [H-10min, H-1min]  (price at H)
    r_win = 100 * log(post / pre), in %
    Primary H = 08:45. Robustness H = 08:35 and 09:00.

Noise variance (what the window return looks like with no CPI news):
    for each CPI date, v_win = mean of r_win^2 over non-event days within
    +-20 trading days, each r_win^2 capped at the global non-event 99th
    percentile (limits single-day outliers without the downward bias of
    dropping the largest value).
    Non-event days exclude CPI dates and approximate NFP dates (first Friday).
    Other 08:30 releases (claims, PPI, retail sales) remain in the noise
    sample on purpose: they are the non-CPI news a CPI-day window also sees.

Also reports the global ratio rho = mean r_win^2 / mean r_on^2 on non-event
days (the injection-recovery run assumed rho ~ 0.05 * 1/phi), and checks the
bar-based overnight return against the yfinance r_on in the panel.

Run:   python narrow_window.py
Reads data/cache_alpaca/*.csv.gz and data/decomposition_panel.csv;
writes data/narrow_window_panel.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache_alpaca"
HORIZONS = {"35": "08:35", "45": "08:45", "60": "09:00"}
PRIMARY = "45"
HALF = 20


def load_bars(sym="SPY"):
    files = sorted(CACHE.glob(f"{sym}_*.csv.gz"))
    if not files:
        raise SystemExit(f"No {sym} bars in data/cache_alpaca. Run alpaca_bars.py --sym {sym} first.")
    b = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    b["ts"] = pd.to_datetime(b["ts"], utc=True).dt.tz_convert("America/New_York")
    b["day"] = b["ts"].dt.tz_localize(None).dt.normalize()
    b["hm"] = b["ts"].dt.strftime("%H:%M")
    return b


def last_close(g, lo, hi):
    """Close of the last bar whose start time is in [lo, hi] (HH:MM strings)."""
    x = g[(g.hm >= lo) & (g.hm <= hi)]
    return x["close"].iloc[-1] if len(x) else np.nan


def minus(hm, minutes):
    t = pd.Timestamp("2000-01-01 " + hm) - pd.Timedelta(minutes=minutes)
    return t.strftime("%H:%M")


def daily_table(b):
    rows = []
    for day, g in b.groupby("day", sort=True):
        pre = last_close(g, "08:20", "08:29")
        row = {"day": day, "pre": pre,
               "open": g.loc[g.hm == "09:30", "open"].iloc[0] if (g.hm == "09:30").any() else np.nan,
               "close": last_close(g, "15:50", "15:59")}
        for k, H in HORIZONS.items():
            post = last_close(g, minus(H, 10), minus(H, 1))
            row[f"r_win{k}"] = 100 * np.log(post / pre)
        rows.append(row)
    t = pd.DataFrame(rows)
    t = t[t["close"].notna() & t["open"].notna()].reset_index(drop=True)
    t["r_on_bars"] = 100 * np.log(t["open"] / t["close"].shift(1))
    return t


def nfp_dates(days):
    months = pd.period_range(days.min(), days.max(), freq="M").to_timestamp()
    return {d + pd.offsets.Week(weekday=4) if d.weekday() != 4 else d for d in months}


def main():
    panel = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date"])
    t = daily_table(load_bars())
    cpi = set(panel["date"])
    t["event"] = t["day"].isin(cpi) | t["day"].isin(nfp_dates(t["day"]))
    quiet = t[~t["event"]]

    r2 = quiet[f"r_win{PRIMARY}"] ** 2
    on2 = quiet["r_on_bars"] ** 2
    keep = (r2 < r2.quantile(0.99)) & (on2 < on2.quantile(0.99))
    rho = r2[keep].mean() / on2[keep].mean()
    print(f"non-event days: {len(quiet)};  rho = mean r_win{PRIMARY}^2 / mean r_on^2 "
          f"(ex top 1%) = {rho:.3f}")

    caps = {k: (quiet[f"r_win{k}"] ** 2).quantile(0.99) for k in HORIZONS}
    out = []
    idx = {d: i for i, d in enumerate(t["day"])}
    for d in panel["date"]:
        if d not in idx:
            out.append({"date": d})
            continue
        i = idx[d]
        row = {"date": d, **{c: t.at[i, c] for c in t.columns if c.startswith("r_")}}
        nb = t.iloc[max(0, i - HALF): i + HALF + 1]
        nb = nb[~nb["event"]]
        for k in HORIZONS:
            x = (nb[f"r_win{k}"] ** 2).dropna().clip(upper=caps[k])
            row[f"v_win{k}"] = x.mean() if len(x) else np.nan
        out.append(row)
    w = pd.DataFrame(out)
    m = panel[["date", "r_on"]].merge(w, on="date")
    ok = m[["r_on", "r_on_bars"]].dropna()
    print(f"bar-based vs yfinance r_on on CPI days: corr = {ok.corr().iloc[0, 1]:.3f} "
          f"(n = {len(ok)}); should be > 0.95")
    print(f"CPI dates with a window return: {w[f'r_win{PRIMARY}'].notna().sum()} / {len(w)}")
    w.to_csv(DATA / "narrow_window_panel.csv", index=False)
    print("Saved", DATA / "narrow_window_panel.csv")


if __name__ == "__main__":
    main()
