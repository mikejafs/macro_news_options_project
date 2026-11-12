#!/usr/bin/env python
"""Implied CPI event variance from Cboe VIX1D, joined to the go/no-go panel.

Window convention
-----------------
VIX1D annualizes in business time (business minutes over a business year),
not calendar time. At the 16:00 ET close the 0DTE strip has expired and the
index equals the volatility of the 1DTE SPXW strip, which has one business
session left. That option settles at the next session's 16:00 close, so its
price covers the full close-to-close window, overnight gap included.
Weekends and holidays add no business time. In %^2:

    V_imp(eve) = VIX1D_close(eve)^2 / BUS_DAYS

This is the implied variance of the release day's close-to-close return r_cc.
BUS_DAYS = 252 is assumed; confirm it against the Cboe VIX1D methodology
document. The calibration diagnostic printed below checks the scaling
empirically (realized/implied by eve weekday).

Event variance
--------------
    V_bg = median (and mean) V_imp over non-event eves within +-W trading days
    EV   = V_imp(eve) - V_bg

Premiums (primary pre-specified: close-to-close, where the windows match)
    prem_cc    = V_imp - r_cc^2      primary
    prem_ev_on = EV    - r_on^2      secondary; assumes CPI news is in by the open

Run:   python event_variance.py
       python event_variance.py --bg-window 10 --macro-dates macro_dates.csv
Needs: requests pandas numpy scipy matplotlib yfinance
Reads data/go_no_go_panel_pre.csv; writes data/event_variance_panel.csv and
figures/event_variance.png. Cboe files are cached in data/cache_cboe.
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Project layout: kalshi_vs_spx/{experiments,data,figures}. Paths are anchored
# to this file, so the script runs correctly from any working directory.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
CACHE = DATA / "cache_cboe"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
BUS_DAYS = 252
VIX1D_LIVE = pd.Timestamp("2023-04-24")   # earlier values are Cboe's backfill
REGIME_SPLIT = pd.Timestamp("2024-01-01")

# FOMC decision days (statement at 14:00 ET). Verify against federalreserve.gov.
FOMC = """2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02
2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01
2023-12-13 2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07
2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29
2025-12-10 2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28
2026-12-09""".split()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def cboe_series(sym, refresh=False):
    """Daily close of a Cboe index from its history CSV (yfinance fallback)."""
    CACHE.mkdir(exist_ok=True)
    fn = CACHE / f"{sym}.csv"
    if refresh or not fn.exists():
        try:
            r = requests.get(CBOE_URL.format(sym), timeout=30,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            fn.write_text(r.text)
        except Exception as exc:
            print(f"  Cboe download failed for {sym} ({exc}); trying yfinance ^{sym}")
            import yfinance as yf
            px = yf.download(f"^{sym}", start="2010-01-01", auto_adjust=False, progress=False)
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            px[["Close"]].rename_axis("DATE").to_csv(fn)
    df = pd.read_csv(fn)
    df.columns = [c.strip().lower() for c in df.columns]
    dcol = next(c for c in df.columns if "date" in c)
    vcol = "close" if "close" in df.columns else df.columns[-1]
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol]).dt.normalize(), name=sym.lower())
    return s.dropna().sort_index()


def spy_daily(start):
    import yfinance as yf
    px = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return pd.DataFrame({"r_on": 100 * (px["Open"] / px["Close"].shift(1) - 1),
                         "r_cc": 100 * (px["Close"] / px["Close"].shift(1) - 1)})


def event_calendar(cpi_dates, macro_file):
    """Macro event days that must not enter the background sample.

    Default: CPI (from the panel), FOMC (table above), NFP approximated as the
    first Friday of each month. A CSV with columns date,type replaces the
    approximation and adds e.g. PPI.
    """
    ev = [pd.DataFrame({"date": pd.to_datetime(cpi_dates), "type": "CPI"}),
          pd.DataFrame({"date": pd.to_datetime(FOMC), "type": "FOMC"})]
    if macro_file:
        m = pd.read_csv(macro_file)
        m["date"] = pd.to_datetime(m["date"])
        ev.append(m[["date", "type"]])
    else:
        months = pd.date_range("2022-01-01", "2026-12-01", freq="MS")
        nfp = [d + pd.offsets.Week(weekday=4) if d.weekday() != 4 else d for d in months]
        ev.append(pd.DataFrame({"date": nfp, "type": "NFP~"}))
    return pd.concat(ev).drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------
def implied_table(vix1d, events):
    """One row per trading day: V_imp at its close and the session it prices."""
    t = vix1d.to_frame()
    t["v_imp"] = t["vix1d"] ** 2 / BUS_DAYS
    t["covers"] = pd.Series(t.index[1:].tolist() + [pd.NaT], index=t.index)
    t["event_eve"] = t["covers"].isin(set(events["date"]))
    t["weekday"] = t.index.day_of_week
    return t


def event_rows(panel, t, events, W):
    days = t.index
    same = events.groupby("date")["type"].apply(lambda s: ",".join(sorted(set(s) - {"CPI"})))
    rows = []
    for d in panel["date"]:
        if d not in days:
            continue
        pos = days.get_loc(d)
        if pos == 0:
            continue
        eve = days[pos - 1]
        lo, hi = max(0, pos - 1 - W), min(len(days), pos + W)
        nb = t.iloc[lo:hi].drop(index=eve)
        nb = nb[~nb["event_eve"] & nb["covers"].notna()]
        v_bg_med, v_bg_mean = nb["v_imp"].median(), nb["v_imp"].mean()
        v = t.at[eve, "v_imp"]
        rows.append({"date": d, "eve": eve, "vix1d_eve": t.at[eve, "vix1d"],
                     "v_imp": v, "v_bg_med": v_bg_med, "v_bg_mean": v_bg_mean,
                     "n_bg": len(nb), "ev_med": v - v_bg_med, "ev_mean": v - v_bg_mean,
                     "vix1d_live": eve >= VIX1D_LIVE,
                     "same_day_events": same.get(d, "")})
    out = pd.DataFrame(rows)
    out["sd_ev_med"] = np.sqrt(out["ev_med"].clip(lower=0))   # implied event move, %
    return out


def mean_t(x):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def calibration(t, spy):
    """Realized/implied on non-event eves, by eve weekday; returns phi.

    phi = mean(r_on^2) / mean(r_cc^2) over non-event sessions: the share of a
    normal day's variance realized in the overnight gap.
    """
    d = t[~t["event_eve"] & t["covers"].notna()].copy()
    d["r2"] = spy["r_cc"].reindex(d["covers"]).to_numpy() ** 2
    d["r2_on"] = spy["r_on"].reindex(d["covers"]).to_numpy() ** 2
    d = d.dropna(subset=["r2", "r2_on"])
    cut = d["r2"].quantile(0.99)
    print("\nCalibration on non-event eves: mean(r_cc^2 next session) / mean(V_imp)")
    print(f"  {'':10s} {'ratio':>6s} {'ex top 1%':>10s} {'n':>5s}")
    groups = [("all", d)] + [(["Mon", "Tue", "Wed", "Thu", "Fri"][wd], g)
                             for wd, g in d.groupby("weekday")]
    for name, g in groups:
        tr = g[g["r2"] <= cut]
        print(f"  eve {name:6s} {g['r2'].mean() / g['v_imp'].mean():6.2f} "
              f"{tr['r2'].mean() / tr['v_imp'].mean():10.2f} {len(g):5d}")
    top = d.nlargest(3, "r2")
    print("  largest sessions: " + ", ".join(f"{c:%Y-%m-%d} (r_cc^2={r:.0f})"
                                             for c, r in zip(top["covers"], top["r2"])))
    phi = d["r2_on"].mean() / d["r2"].mean()
    phi_tr = d.loc[d["r2"] <= cut, "r2_on"].mean() / d.loc[d["r2"] <= cut, "r2"].mean()
    print(f"  phi (overnight share of normal-day variance): {phi:.2f}  (ex top 1%: {phi_tr:.2f})")
    return phi_tr


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DATA / "go_no_go_panel_pre.csv"))
    ap.add_argument("--bg-window", type=int, default=5)
    ap.add_argument("--macro-dates", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-spy", action="store_true", help="skip calibration diagnostic")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel, parse_dates=["date"])
    vix1d = cboe_series("VIX1D", args.refresh)
    vix9d = cboe_series("VIX9D", args.refresh)
    vix = cboe_series("VIX", args.refresh)
    print(f"VIX1D {vix1d.index.min():%Y-%m-%d} to {vix1d.index.max():%Y-%m-%d}, n={len(vix1d)}")

    events = event_calendar(panel["date"], args.macro_dates)
    t = implied_table(vix1d, events)
    ev = event_rows(panel, t, events, args.bg_window)
    ev["vix9d_eve"] = vix9d.reindex(ev["eve"]).to_numpy()
    ev["vix_eve"] = vix.reindex(ev["eve"]).to_numpy()

    out = panel.merge(ev, on="date", how="left")
    out["r_on2"], out["r_cc2"] = out["r_on"] ** 2, out["r_cc"] ** 2
    out["prem_cc"] = out["v_imp"] - out["r_cc2"]
    out["prem_ev_on"] = out["ev_med"] - out["r_on2"]

    phi = np.nan
    if not args.no_spy:
        try:
            phi = calibration(t, spy_daily(str(vix1d.index.min().date())))
        except Exception as exc:
            print(f"\nCalibration skipped: {exc}")
    # Overnight implied variance = CPI event + a normal night's share of background.
    out["phi"] = phi
    out["v_on_imp"] = out["ev_med"] + phi * out["v_bg_med"]
    out["prem_on"] = out["v_on_imp"] - out["r_on2"]
    out.to_csv(DATA / "event_variance_panel.csv", index=False)
    k = out.dropna(subset=["v_imp"])
    print(f"Releases with VIX1D eve: {len(k)} of {len(out)}; "
          f"backfilled eves (pre {VIX1D_LIVE:%Y-%m-%d}): {(~k['vix1d_live'].astype(bool)).sum()}; "
          f"same-day other events: {k.loc[k['same_day_events'] != '', 'date'].dt.date.tolist()}")
    print(f"Background sample size per release: min {k['n_bg'].min()}, median {k['n_bg'].median():.0f}")

    print("\nPer-release means (%^2), t = mean / SE")
    print("  close-to-close (primary)                  overnight")
    hdr = f"  {'period':8s} {'n':>3s} {'V_imp':>6s} {'r_cc2':>6s} {'prem_cc':>8s} {'t':>5s}   " \
          f"{'EV':>5s} {'V_on':>5s} {'r_on2':>6s} {'prem_on':>8s} {'t':>5s} {'EV<0':>5s}"
    print(hdr)
    for per, g in [("all", k), *k.groupby("period")]:
        m1, t1, n = mean_t(g["prem_cc"])
        m2, t2, _ = mean_t(g["prem_on"])
        print(f"  {per:8s} {n:3d} {g['v_imp'].mean():6.2f} {g['r_cc2'].mean():6.2f} "
              f"{m1:8.2f} {t1:5.2f}   {g['ev_med'].mean():5.2f} {g['v_on_imp'].mean():5.2f} "
              f"{g['r_on2'].mean():6.2f} {m2:8.2f} {t2:5.2f} {(g['ev_med'] < 0).mean():5.0%}")
    print("  V_on = EV + phi*V_bg (CPI event plus a normal night). prem_ev_on = EV - r_on2\n"
          "  omits the normal night and is kept in the CSV for reference only.")
    print("  Squared returns are heavy-tailed; treat these t-stats as indicative and\n"
          "  rely on the power analysis and robustness sweeps for inference.")

    print("\nRank correlations (Spearman)")
    for a, b in [("ev_med", "r_on2"), ("v_imp", "r_cc2"), ("ev_med", "k_sd_core")]:
        d = k[[a, b]].dropna()
        if len(d) >= 6:
            rho, p = stats.spearmanr(d[a], d[b])
            print(f"  {a:7s} vs {b:9s}: rho={rho:5.2f}, p={p:.3f}, n={len(d)}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(t.index, t["v_imp"], color="0.75", lw=0.6, label="V_imp, all eves")
    ax[0].scatter(k["eve"], k["v_imp"], s=14, color="tab:red", label="V_imp, CPI eve", zorder=3)
    ax[0].scatter(k["eve"], k["v_bg_med"], s=14, color="tab:blue", label="background (median)", zorder=3)
    ax[0].set(yscale="log", ylabel="implied variance of next session (%²)",
              title="VIX1D-implied variance: CPI eves vs background")
    ax[0].legend()
    for (per, g), col in zip(k.groupby("period"), ["tab:red", "tab:blue"]):
        ax[1].scatter(g["ev_med"], g["r_on2"], color=col, alpha=0.8, label=per)
    lim = np.nanmax([k["ev_med"].max(), 1.0])
    ax[1].plot([0, lim], [0, lim], color="grey", lw=0.8, ls="--", label="r_on² = EV")
    ax[1].axvline(0, color="grey", lw=0.5)
    ax[1].set(xlabel="implied event variance EV (%²)", ylabel="realized r_on² (%²)",
              title="Implied event variance vs realized overnight move")
    ax[1].legend()
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "event_variance.png", dpi=150)
    print(f"\nSaved {DATA / 'event_variance_panel.csv'} and {FIGS / 'event_variance.png'}")


if __name__ == "__main__":
    main()