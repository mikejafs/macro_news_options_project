#/usr/bin/env python
"""Delta-hedged short straddle with REAL option prices (SPY 1-day options;
--sym QQQ or IWM runs the same trade on those ETFs).

straddle.py prices the straddle by Black-Scholes at VIX1D, which sits above
at-the-money implied volatility and so overstates the premium. Here the
premium is the traded price of the SPY call and put expiring the next trading
day, at the strike nearest the eve close.

Data: Alpaca historical option bars (free plan), available only from
February 2024, so this covers the 2024-26 regime. Bars are built from trades,
so there are no bid/ask quotes: each leg's price is the close of its last
1-minute bar in 15:45-15:59 ET on the eve. Eves whose next session is a SPY
ex-dividend date (third Friday of Mar/Jun/Sep/Dec) are dropped because the
no-dividend Black-Scholes delta is wrong there.

Trade: sell call + put at strike K (nearest to S_eve among K0-1, K0, K0+1 with
both legs traded); sigma_atm solves BS straddle(S_eve, K, sigma, 1/252) =
premium; hedge with the BS straddle delta at sigma_atm in SPY; rebalance at
the open and then every --every minutes; payoff |S_close - K|. The open mark
uses the same phi split as straddle.py. P&L in bp of S_eve.

Reported for 2024-26 (CPI eves vs non-event eves):
    sigma_atm / VIX1D          size of the VIX1D bias
    P&L with real premium      the tradeable number, before bid-ask costs
    P&L of the same trade priced at VIX1D (straddle.py convention)
    event variance from real ATM vol: sigma_atm^2/252 minus the median over
    non-event eves within +-5 sessions, against the VIX1D-based EV

Run:   export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
       python straddle_real.py            # open + close
       python straddle_real.py --every 5
Writes data/straddle_real_<tag>.csv; option bars cached in data/cache_alpaca_opt.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_bs import straddle, straddle_delta  # noqa: E402
from narrow_window import load_bars, nfp_dates  # noqa: E402
from straddle import FOMC, YEAR, R, session_table  # noqa: E402
import ss_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE = ROOT / "data", ROOT / "data" / "cache_alpaca_opt"
URL = "https://data.alpaca.markets/v1beta1/options/bars"
START = pd.Timestamp("2024-02-01")
T0 = 1.0 / YEAR


def occ(sym, expiry, cp, strike):
    return f"{sym}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


def is_exdiv(day, sym="SPY"):
    """SPY goes ex-dividend on the third Friday of Mar/Jun/Sep/Dec; those eves are
    dropped. QQQ and IWM ex-dates are irregular and their dividends are small
    (about 0.15-0.3% a quarter against a one-day sd above 1%), and the dividend is
    in both the option prices and the payoff, so no eves are dropped for them."""
    return sym == "SPY" and day.month in (3, 6, 9, 12) and day.weekday() == 4 and 15 <= day.day <= 21


def fetch(session, sym, eve, expiry, strikes):
    """Last traded price in 15:45-15:59 ET on the eve for each call/put; cached per eve."""
    fn = CACHE / f"{sym}_{eve:%Y-%m-%d}.json"
    if fn.exists():
        return json.loads(fn.read_text())
    syms = {occ(sym, expiry, cp, k): (cp, k) for k in strikes for cp in "CP"}
    t0 = pd.Timestamp(f"{eve:%Y-%m-%d} 15:45", tz="America/New_York")
    params = dict(symbols=",".join(syms), timeframe="1Min", start=t0.isoformat(),
                  end=(t0 + pd.Timedelta(minutes=15)).isoformat(), limit=10000, sort="asc")
    bars = {}
    while True:
        for k in range(8):
            try:
                r = session.get(URL, params=params, timeout=60)
            except requests.RequestException:
                time.sleep(min(60, 2 ** k))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** k))
                continue
            break
        if r.status_code != 200:
            raise SystemExit(f"Alpaca options request failed: HTTP {r.status_code} {r.text[:300]}")
        js = r.json()
        for s, b in (js.get("bars") or {}).items():
            bars.setdefault(s, []).extend(b)
        if not js.get("next_page_token"):
            break
        params["page_token"] = js["next_page_token"]
    out = {}
    for s, (cp, k) in syms.items():
        b = [x for x in bars.get(s, []) if pd.Timestamp(x["t"]) < t0 + pd.Timedelta(minutes=15)]
        if b:
            out[f"{cp}{k}"] = dict(price=b[-1]["c"], t=b[-1]["t"], volume=sum(x["v"] for x in b))
    fn.write_text(json.dumps(out))
    time.sleep(0.35)
    return out


def implied_vol(S, K, premium):
    f = lambda sig: straddle(S, K, sig, T0, R) - premium
    if premium <= abs(S - K) + 1e-6:
        return np.nan
    return brentq(f, 1e-4, 5.0)


def pnl(S_eve, K, premium, sigma, grid, w_open):
    """Short straddle sold at `premium`, hedged with BS deltas at vol `sigma`."""
    m = len(grid) - 1
    S_open = grid[0]
    sig_d = np.sqrt(w_open * YEAR)
    tau = (1.0 - np.arange(m) / m) / YEAR
    mark_open = straddle(S_open, K, sig_d, T0, R)
    deltas = straddle_delta(np.array(grid[:m]), K, sig_d, tau, R)
    overnight = premium - mark_open + straddle_delta(S_eve, K, sigma, T0, R) * (S_open - S_eve)
    intraday = mark_open - abs(grid[-1] - K) + np.sum(deltas * np.diff(grid))
    return overnight, intraday


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=390)
    ap.add_argument("--sym", default="SPY", help="underlying with daily expirations: SPY, QQQ, IWM")
    ap.add_argument("--offline", action="store_true", help="use cached option bars only")
    a = ap.parse_args()
    sym = a.sym
    key, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    CACHE.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    if key and sec:
        s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})

    panel = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date"])
    phi = panel["phi"].iloc[0]
    share = (panel.set_index("date")["v_bg_med"] / panel.set_index("date")["v_imp"]).clip(0.05, 1)
    vix = pd.read_csv(DATA / "cache_cboe" / "VIX1D.csv")
    vix.columns = [c.strip().lower() for c in vix.columns]
    vix = pd.Series(vix["close"].values, index=pd.to_datetime(vix["date"]).dt.normalize())
    sess = session_table(load_bars(sym), a.every)
    days = sorted(sess)
    cpi = set(panel["date"])
    skip = cpi | set(FOMC) | nfp_dates(pd.Series(days))

    rows, missing, n_new = [], 0, 0
    for eve, day in zip(days[:-1], days[1:]):
        if eve < START or eve not in vix.index or is_exdiv(day, sym):
            continue
        # FOMC and jobs-report eves are kept in the file (kind "other_event") so that
        # event_options.py can label every eve from the release calendar
        kind = "CPI" if day in cpi else ("other_event" if day in skip else "non_event")
        S_eve = sess[eve][-1]
        k0 = round(S_eve)
        if a.offline and not (CACHE / f"{sym}_{eve:%Y-%m-%d}.json").exists():
            continue
        if not (CACHE / f"{sym}_{eve:%Y-%m-%d}.json").exists() and not (key and sec):
            raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY first.")
        if not (CACHE / f"{sym}_{eve:%Y-%m-%d}.json").exists():
            if a.offline:
                continue
            n_new += 1
            if n_new % 25 == 1:
                print(f"  downloading {sym} option bars: {eve:%Y-%m-%d} (about 690 eves in total, "
                      f"cached after the first run)", flush=True)
        q = fetch(s, sym, eve, day, [k0 - 1, k0, k0 + 1])
        ks = [k for k in (k0, k0 - 1, k0 + 1) if f"C{k}" in q and f"P{k}" in q]
        if not ks:
            missing += 1
            continue
        K = min(ks, key=lambda k: abs(k - S_eve))
        prem = q[f"C{K}"]["price"] + q[f"P{K}"]["price"]
        sig = implied_vol(S_eve, K, prem)
        if np.isnan(sig):
            missing += 1
            continue
        sh = share.get(day, 1.0) if (kind == "CPI" and sym == "SPY") else 1.0
        grid = sess[day]
        on, intra = pnl(S_eve, K, prem, sig, grid, (1 - phi) * sh * sig ** 2 / YEAR)
        sv = vix[eve] / 100
        prem_v = straddle(S_eve, K, sv, T0, R)
        on_v, in_v = pnl(S_eve, K, prem_v, sv, grid, (1 - phi) * sh * sv ** 2 / YEAR)
        rows.append(dict(date=day, eve=eve, kind=kind, S_eve=S_eve, K=K, vix1d=vix[eve],
                         atm_vol=100 * sig, ratio=100 * sig / vix[eve],
                         premium_bp=1e4 * prem / S_eve, premium_vix_bp=1e4 * prem_v / S_eve,
                         pnl_on_bp=1e4 * on / S_eve, pnl_intra_bp=1e4 * intra / S_eve,
                         pnl_bp=1e4 * (on + intra) / S_eve, pnl_vix_bp=1e4 * (on_v + in_v) / S_eve,
                         v_atm=(100 * sig) ** 2 / YEAR))
    df = pd.DataFrame(rows)
    print(f"Trades: {len(df)} ({(df.kind == 'CPI').sum()} CPI eves); eves without both legs: {missing}")
    print(f"Rebalancing: {'open + close' if a.every >= 390 else f'every {a.every} min'}; bp of S_eve; "
          f"sample from {df.date.min():%Y-%m-%d}")
    print(f"{'sample':11s}{'n':>5s}{'ATM/VIX1D':>10s}{'prem':>7s}{'prem@VIX':>9s} | {'P&L real':>9s}{'se':>6s}{'t':>6s}"
          f"{'overnight':>10s}{'intraday':>9s} | {'P&L@VIX':>8s}{'t':>6s}")
    for k, g in df[df.kind != "other_event"].groupby("kind"):
        lags = 0 if k == "CPI" else 5
        mu, se = sm.newey_west_mean(g.pnl_bp.values, lags)
        mv, sev = sm.newey_west_mean(g.pnl_vix_bp.values, lags)
        print(f"{k:11s}{len(g):5d}{g.ratio.mean():10.3f}{g.premium_bp.mean():7.1f}{g.premium_vix_bp.mean():9.1f} | "
              f"{mu:9.1f}{se:6.1f}{mu / se:6.2f}{g.pnl_on_bp.mean():10.1f}{g.pnl_intra_bp.mean():9.1f} | "
              f"{mv:8.1f}{mv / sev:6.2f}")
    x, y = df[df.kind == "CPI"].pnl_bp, df[df.kind == "non_event"].pnl_bp
    dd, se = x.mean() - y.mean(), np.sqrt(x.var() / len(x) + y.var() / len(y))
    print(f"CPI minus non-event, real premium: {dd:.1f} bp (se {se:.1f}, t {dd / se:.2f})")

    # event variance from real ATM vol, same background rule as event_variance.py
    ne = df[df.kind == "non_event"].set_index("eve")["v_atm"]
    pos = {d: i for i, d in enumerate(days)}
    ev = []
    for _, r_ in df[df.kind == "CPI"].iterrows():
        i = pos[r_.eve]
        nb = ne.reindex(days[max(0, i - 5): i + 6]).dropna()
        ev.append(r_.v_atm - nb.median() if len(nb) >= 3 else np.nan)
    c = df[df.kind == "CPI"].assign(ev_atm=ev).merge(panel[["date", "ev_med", "sk2"]], on="date")
    mu, se = sm.newey_west_mean(c.ev_atm.values, 0)
    print(f"\nEvent variance on CPI eves (n = {c.ev_atm.notna().sum()}): real ATM {mu:.3f} %^2 (se {se:.3f}); "
          f"VIX1D-based on the same releases {c.ev_med.mean():.3f}")
    print(f"Implied |beta| from real ATM vol: {np.sqrt(max(mu, 0) / c.sk2.mean()):.2f} "
          f"(VIX1D-based: {np.sqrt(max(c.ev_med.mean(), 0) / c.sk2.mean()):.2f}; realized window beta 3.8)")
    tag = ("oc" if a.every >= 390 else str(a.every)) + ("" if sym == "SPY" else f"_{sym}")
    df.insert(0, "sym", sym)
    df.to_csv(DATA / f"straddle_real_{tag}.csv", index=False)
    print(f"\nSaved data/straddle_real_{tag}.csv")


if __name__ == "__main__":
    main()
