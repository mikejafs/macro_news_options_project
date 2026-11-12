#!/usr/bin/env python
"""Which options respond to which macro news, and do they charge the right amount?

For every 08:30 ET release type (FRED calendar) plus FOMC days, three numbers:
    priced    extra variance options charge on the eve          (%^2)
    realized  extra variance the underlying delivers that day   (%^2)
    earned    delta-hedged short-straddle P&L, minus the same trade on quiet eves (bp)
"Extra" is always relative to quiet days (no listed release, no FOMC).

Part A  SPY, 2022-26, by release type and regime, options priced at VIX1D.
        priced   = V1(eve) - median V1 over quiet eves within +-5 sessions,
                   V1 = VIX1D^2 / 252
        realized = mean r_cc^2 on release days - mean r_cc^2 on quiet days
        earned   = BS straddle sold at VIX1D, hedged open + close (straddle.py
                   engine). VIX1D overstates ATM vol by about 6%, the same on
                   all eves, so only the difference from quiet eves is reported.
Part B  Maturity: implied variance that leaves the curve when the event passes,
        TV_m(eve close) - TV_m(event-day close), minus the same change on quiet
        days, for 1-day (VIX1D^2/252), 9-day (VIX9D^2 * 9/365) and 30-day
        (VIX^2 * 30/365) total variance. A one-day event should remove about the
        same amount of total variance from each maturity.
Part C  Underlying, from Feb 2024, real traded option prices (straddle_real.py
        output for SPY, QQQ, IWM): priced from ATM vol, realized from the ETF's
        own returns, earned at traded prices.

Part D  Pooled: every release eve against quiet eves, by regime and year
        (SPY at VIX1D) and by underlying (traded prices).

Excluded by default (post hoc, disclosed): 3-11 April 2025, the tariff
announcement and pause. It contains a jobs report and a CPI release, moved SPY
by up to 10% a day for unrelated reasons, and by itself changes the 2024-26
CPI "priced" cell from 0.38 to 0.14. Run with --all to keep it.

Days are labelled by their highest-ranked event in the order
CPI, JOBS, FOMC, PCE, GDP, RETAIL, PPI, CLAIMS. Standard errors treat release
days as independent; quiet-day means use Newey-West (5 lags).

Run:   python event_options.py
Reads data/macro_calendar.csv, Cboe caches, minute bars, straddle_real_oc*.csv.
Writes data/event_options_{A,B,C}.csv and figures/event_options.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import narrow_window as nw  # noqa: E402
import ss_model as sm  # noqa: E402
from straddle import FOMC, YEAR, hedge_pnl, session_table  # noqa: E402

DATA, FIGS = nw.DATA, nw.ROOT / "figures"
RANK = ["CPI", "JOBS", "FOMC", "PCE", "GDP", "RETAIL", "PPI", "CLAIMS"]
W = 5
TARIFF = (pd.Timestamp("2025-04-03"), pd.Timestamp("2025-04-11"))
KEEP_ALL = "--all" in sys.argv


def cboe(sym):
    v = pd.read_csv(DATA / "cache_cboe" / f"{sym}.csv")
    v.columns = [c.strip().lower() for c in v.columns]
    return pd.Series(v["close"].values, index=pd.to_datetime(v["date"]).dt.normalize())


def labels(calendar):
    cal = pd.read_csv(calendar, parse_dates=["date"])
    cal = pd.concat([cal, pd.DataFrame({"date": FOMC, "release": "FOMC"})])
    return cal.groupby("date")["release"].agg(lambda x: min(x, key=RANK.index))


def local_excess(x, quiet_mask, w=W):
    """x_t minus the median of x over quiet observations within +-w rows (excluding t)."""
    out = np.full(len(x), np.nan)
    xv, qm = x.values, quiet_mask.values
    for i in range(len(x)):
        lo, hi = max(0, i - w), min(len(x), i + w + 1)
        nb = [xv[j] for j in range(lo, hi) if j != i and qm[j] and not np.isnan(xv[j])]
        if len(nb) >= 3:
            out[i] = xv[i] - np.median(nb)
    return out


def mean_se(x, lags=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    return sm.newey_west_mean(x, lags)


def diff(E, Q):
    """Mean of E minus mean of Q with a Welch standard error (Q with Newey-West)."""
    me, se = mean_se(E)
    mq, sq = mean_se(Q, 5)
    return me - mq, np.sqrt(se ** 2 + sq ** 2)


def fmt(m, s, w=7, p=2):
    return f"{m:{w}.{p}f} ({m / s:5.1f})" if s and not np.isnan(s) and s > 0 else f"{'.':>{w}s}        "


def main(calendar=DATA / "macro_calendar.csv"):
    calendar = DATA / "macro_calendar.csv" if str(calendar).startswith("--") else calendar
    lab = labels(calendar)
    phi = pd.read_csv(DATA / "decomposition_panel.csv")["phi"].iloc[0]

    # ------------------------------------------------ daily SPY frame, 2022-26
    sess = session_table(nw.load_bars("SPY"), 390)
    days = pd.DatetimeIndex(sorted(sess))
    d = pd.DataFrame({"day": days, "open": [sess[x][0] for x in days], "close": [sess[x][-1] for x in days]})
    d["eve"] = d["day"].shift(1)
    d["S_eve"] = d["close"].shift(1)
    d["r2"] = (100 * np.log(d["close"] / d["S_eve"])) ** 2
    d["kind"] = d["day"].map(lab).fillna("QUIET")
    d["regime"] = np.where(d["day"] < "2024-01-01", "2022-23", "2024-26")
    quiet = d["kind"] == "QUIET"
    tv = {"1d": cboe("VIX1D") ** 2 / YEAR, "9d": cboe("VIX9D") ** 2 * 9 / 365, "30d": cboe("VIX") ** 2 * 30 / 365}
    for m, s in tv.items():
        d[f"tv{m}_eve"] = d["eve"].map(s)
        d[f"tv{m}_day"] = d["day"].map(s)
        d[f"rel{m}"] = d[f"tv{m}_eve"] - d[f"tv{m}_day"]          # variance released when the day passes
    d["priced"] = local_excess(d["tv1d_eve"], quiet)
    ok = d["tv1d_eve"].notna() & d["S_eve"].notna()
    sig = np.sqrt(d.loc[ok, "tv1d_eve"] * YEAR) / 100
    grid = d.loc[ok, ["open", "close"]].values
    prem, on, intra = hedge_pnl(d.loc[ok, "S_eve"].values, grid, sig.values,
                                (1 - phi) * d.loc[ok, "tv1d_eve"].values / 1e4)
    d.loc[ok, "pnl"] = 1e4 * (on + intra) / d.loc[ok, "S_eve"].values
    d = d[ok].reset_index(drop=True)
    if not KEEP_ALL:
        d = d[~d["day"].between(*TARIFF)].reset_index(drop=True)
    print("Sample:", "all days" if KEEP_ALL else "excluding 3-11 April 2025 (see header)")

    gam = None
    if (DATA / "macro_gamma.csv").exists():
        g = pd.read_csv(DATA / "macro_gamma.csv")
        gam = g[g.window == "w45"].set_index(["kind", "period"])["gamma"]

    rowsA, rowsB = [], []
    print("Part A. SPY options priced at VIX1D, by release type and regime (t statistics in brackets)")
    print(f"{'release':8s}{'regime':>9s}{'n':>5s}{'priced %^2':>16s}{'realized %^2':>16s}"
          f"{'earned vs quiet bp':>20s}{'gamma':>8s}")
    for k in RANK:
        for reg in ["2022-23", "2024-26"]:
            E = d[(d.kind == k) & (d.regime == reg)]
            Q = d[(d.kind == "QUIET") & (d.regime == reg)]
            if len(E) < 6:
                continue
            pm, ps = mean_se(E.priced)
            rm, rs_ = diff(E.r2, Q.r2)
            em, es = diff(E.pnl, Q.pnl)
            gv = gam.get((k, reg), np.nan) if gam is not None else np.nan
            rowsA.append(dict(release=k, regime=reg, n=len(E), priced=pm, priced_se=ps, realized=rm,
                              realized_se=rs_, earned_vs_quiet=em, earned_se=es, gamma=gv))
            print(f"{k:8s}{reg:>9s}{len(E):5d}{fmt(pm, ps, 8):>16s}{fmt(rm, rs_, 8):>16s}"
                  f"{fmt(em, es, 9, 1):>20s}" + (f"{gv:8.2f}" if not np.isnan(gv) else f"{'.':>8s}"))
            rb = dict(release=k, regime=reg, n=len(E))
            for m in tv:
                rb[f"released_{m}"], rb[f"se_{m}"] = diff(E[f"rel{m}"], Q[f"rel{m}"])
            rowsB.append(rb)
    for reg in ["2022-23", "2024-26"]:
        Q = d[(d.kind == "QUIET") & (d.regime == reg)]
        print(f"{'QUIET':8s}{reg:>9s}{len(Q):5d}   mean r_cc^2 {Q.r2.mean():.2f}, mean V1 {Q.tv1d_eve.mean():.2f}, "
              f"VIX1D-priced straddle {Q.pnl.mean():.1f} bp")
    A = pd.DataFrame(rowsA)
    A.to_csv(DATA / "event_options_A.csv", index=False)

    print("\nPart B. Implied variance released when the event passes, minus quiet days (%^2; t in brackets)")
    print(f"{'release':8s}{'regime':>9s}{'n':>5s}{'1-day':>16s}{'9-day':>16s}{'30-day':>16s}")
    B = pd.DataFrame(rowsB)
    for _, r in B.iterrows():
        print(f"{r.release:8s}{r.regime:>9s}{r.n:5d}" + "".join(
            f"{fmt(r[f'released_{m}'], r[f'se_{m}'], 8):>16s}" for m in tv))
    B.to_csv(DATA / "event_options_B.csv", index=False)

    # ------------------------------------------------ Part C: underlyings, real prices
    rowsC, real = [], {}
    files = {"SPY": "straddle_real_oc.csv", "QQQ": "straddle_real_oc_QQQ.csv", "IWM": "straddle_real_oc_IWM.csv"}
    print("\nPart C. Real option prices from Feb 2024, by underlying and release type (t in brackets)")
    print(f"{'sym':5s}{'release':8s}{'n':>5s}{'ATM vol':>9s}{'priced %^2':>16s}{'realized %^2':>16s}"
          f"{'earned bp':>16s}{'vs quiet bp':>16s}")
    for sym, fn in files.items():
        if not (DATA / fn).exists():
            print(f"{sym}: data/{fn} not found (run straddle_real.py --sym {sym})")
            continue
        x = pd.read_csv(DATA / fn, parse_dates=["date", "eve"]).sort_values("date").reset_index(drop=True)
        t = nw.daily_table(nw.load_bars(sym))
        cc = pd.Series(100 * np.log(t["close"] / t["close"].shift(1)).values, index=t["day"])
        x["r2"] = x["date"].map(cc) ** 2
        x["kind"] = x["date"].map(lab).fillna("QUIET")
        x["priced"] = local_excess(x["v_atm"], x["kind"] == "QUIET")
        if not KEEP_ALL:
            x = x[~x["date"].between(*TARIFF)]
        real[sym] = x
        Q = x[x.kind == "QUIET"]
        for k in RANK:
            E = x[x.kind == k]
            if len(E) < 6:
                continue
            pm, ps = mean_se(E.priced)
            rm, rs_ = diff(E.r2, Q.r2)
            em, es = mean_se(E.pnl_bp)
            dm, ds = diff(E.pnl_bp, Q.pnl_bp)
            rowsC.append(dict(sym=sym, release=k, n=len(E), atm_vol=E.atm_vol.mean(), priced=pm, priced_se=ps,
                              realized=rm, realized_se=rs_, earned=em, earned_se=es, vs_quiet=dm, vs_quiet_se=ds))
            print(f"{sym:5s}{k:8s}{len(E):5d}{E.atm_vol.mean():9.1f}{fmt(pm, ps, 8):>16s}{fmt(rm, rs_, 8):>16s}"
                  f"{fmt(em, es, 8, 1):>16s}{fmt(dm, ds, 8, 1):>16s}")
        qm, qs = mean_se(Q.pnl_bp, 5)
        print(f"{sym:5s}{'QUIET':8s}{len(Q):5d}{Q.atm_vol.mean():9.1f}{'':16s}   mean r_cc^2 {Q.r2.mean():.2f}"
              f"{fmt(qm, qs, 12, 1):>20s}")
    C = pd.DataFrame(rowsC)
    C.to_csv(DATA / "event_options_C.csv", index=False)

    # ------------------------------------------------ Part D: pooled release eves vs quiet eves
    print("\nPart D. Short hedged straddle, all release eves minus quiet eves (bp; t in brackets)")
    print("SPY priced at VIX1D:")
    rowsD = []
    for per, m in ([(p, d.regime == p) for p in ["2022-23", "2024-26"]]
                   + [(str(y), d.day.dt.year == y) for y in sorted(d.day.dt.year.unique())]):
        E, Q = d[m & (d.kind != "QUIET")], d[m & (d.kind == "QUIET")]
        dm, ds = diff(E.pnl, Q.pnl)
        rowsD.append(dict(sym="SPY@VIX1D", period=per, n_release=len(E), n_quiet=len(Q), release=E.pnl.mean(),
                          quiet=Q.pnl.mean(), diff=dm, se=ds, rv_iv_release=E.r2.mean() / E.tv1d_eve.mean(),
                          rv_iv_quiet=Q.r2.mean() / Q.tv1d_eve.mean()))
        print(f"  {per:8s} release n {len(E):3d} mean {E.pnl.mean():6.1f} | quiet n {len(Q):3d} mean {Q.pnl.mean():6.1f} | "
              f"difference {fmt(dm, ds, 6, 1)} | realized/implied variance: release {rowsD[-1]['rv_iv_release']:.2f}, "
              f"quiet {rowsD[-1]['rv_iv_quiet']:.2f}")
    print("Traded prices, from Feb 2024:")
    groups = {"all releases": RANK, "inflation (CPI, PPI, PCE)": ["CPI", "PPI", "PCE"],
              "activity (JOBS, RETAIL, GDP, CLAIMS)": ["JOBS", "RETAIL", "GDP", "CLAIMS"], "FOMC": ["FOMC"]}
    for sym, x in real.items():
        Q = x[x.kind == "QUIET"]
        for nm, ks in groups.items():
            E = x[x.kind.isin(ks)]
            dm, ds = diff(E.pnl_bp, Q.pnl_bp)
            rowsD.append(dict(sym=sym, period=nm, n_release=len(E), n_quiet=len(Q), release=E.pnl_bp.mean(),
                              quiet=Q.pnl_bp.mean(), diff=dm, se=ds))
            print(f"  {sym} {nm:38s} n {len(E):3d} mean {E.pnl_bp.mean():6.1f} | quiet {Q.pnl_bp.mean():5.1f} | "
                  f"difference {fmt(dm, ds, 6, 1)}")
    pd.DataFrame(rowsD).to_csv(DATA / "event_options_D.csv", index=False)

    # ------------------------------------------------ figure: priced vs realized, SPY, by regime
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, reg in zip(axes, ["2022-23", "2024-26"]):
        g = A[A.regime == reg]
        xx = np.arange(len(g))
        ax.bar(xx - 0.2, g.priced, 0.4, yerr=1.96 * g.priced_se, capsize=2, color="#0072B2", label="priced (VIX1D)")
        ax.bar(xx + 0.2, g.realized, 0.4, yerr=1.96 * g.realized_se, capsize=2, color="#D55E00", label="realized")
        ax.set_xticks(xx)
        ax.set_xticklabels(g.release, rotation=30)
        ax.axhline(0, c="grey", lw=0.6)
        ax.set_title(reg, fontsize=10)
    axes[0].set_ylabel("extra one-day variance vs quiet days (%$^2$)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("What SPX options charge for each release, and what stocks delivered", fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "event_options.png", dpi=160)
    print("\nSaved data/event_options_A.csv, _B.csv, _C.csv, _D.csv and figures/event_options.png")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
