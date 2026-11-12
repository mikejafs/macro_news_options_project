#!/usr/bin/env python
"""Go/no-go test for the Kalshi CPI x SPX event-variance project.

Question: does SPY react to the CPI surprise measured against Kalshi's
pre-release mean?  If not, the project's premise fails.

Steps
  1. For every settled Kalshi CPI event (headline KXCPI, core KXCPICORE):
     pull the strike ladder, take each strike's price just before the
     release, build the implied distribution, compute mean and sd.
  2. Read the actual print from Kalshi's settlement.
  3. Pull SPY daily prices; release-day overnight return
     (previous close -> open, the 8:30 ET release falls in this window).
  4. Regress the reaction on the surprise (actual - Kalshi mean), split into
     2021-2023 (high-inflation regime) and 2024-2026; permutation tests.
  5. Print a GO / SWITCH verdict, save the panel CSV and a figure.

Run:  python go_nogo.py              (snapshot just before the release)
      python go_nogo.py --snapshot eve   (prior-day 16:00 ET close)
Needs: requests pandas numpy scipy statsmodels matplotlib yfinance
Raw API responses are cached in data/cache_kalshi, so an interrupted run
resumes where it stopped. Writes data/go_no_go_panel_<snapshot>.csv and
figures/go_no_go_<snapshot>.png. Runs from any working directory.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
NY = "America/New_York"
# Project layout: {scripts,data,figures}, anchored to this file.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
CACHE = DATA / "cache_kalshi"
SESSION = requests.Session()   # reuses the TLS connection across requests
SERIES = {"KXCPI": "headline", "KXCPICORE": "core"}
REGIME_SPLIT = pd.Timestamp("2024-01-01")

# Known CPI release dates, used only to validate the inferred dates.
KNOWN_RELEASES = ["2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13",
                  "2022-10-13", "2022-11-10", "2022-12-13"]


# --------------------------------------------------------------------------
# Kalshi API helpers
# --------------------------------------------------------------------------
def get(path, params=None, tries=8):
    """GET with on-disk cache and exponential backoff.

    Retries on HTTP 429, HTTP 5xx and network errors (timeouts, dropped
    connections). A 4xx other than 429 means the resource does not exist and
    returns None. If every retry fails the run stops with an error rather than
    silently dropping a strike; rerunning resumes from the cache.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = path + json.dumps(params or {}, sort_keys=True)
    fn = CACHE / (hashlib.md5(key.encode()).hexdigest() + ".json")
    if fn.exists():
        return json.loads(fn.read_text())
    wait = 1.0
    for _ in range(tries):
        try:
            r = SESSION.get(KALSHI + path, params=params, timeout=(10, 30))
        except requests.RequestException as exc:
            print(f"    network error ({type(exc).__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue
        if r.status_code != 200:
            return None
        js = r.json()
        fn.write_text(json.dumps(js))
        time.sleep(0.1)
        return js
    raise RuntimeError(f"Kalshi request failed after {tries} tries: {path} {params}. "
                       "Rerun; cached responses are kept.")


def list_events(series):
    events, cursor = [], None
    while True:
        params = {"series_ticker": series, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        js = get("/events", params)
        if not js:
            return events
        events += js.get("events", [])
        cursor = js.get("cursor")
        if not cursor:
            return events


def markets_for(event_ticker):
    """Live listing first, then the historical database for settled events."""
    for path, source in (("/markets", "live"), ("/historical/markets", "historical")):
        js = get(path, {"event_ticker": event_ticker, "limit": 100})
        mk = (js or {}).get("markets", [])
        if mk:
            return mk, source
    return [], None


def candles(series, ticker, source, start_ts, end_ts):
    params = {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": 60}
    paths = [f"/series/{series}/markets/{ticker}/candlesticks",
             f"/historical/markets/{ticker}/candlesticks"]
    if source == "historical":
        paths.reverse()
    for p in paths:
        js = get(p, params)
        c = (js or {}).get("candlesticks", [])
        if c:
            return c
    return []


def num(d, key):
    """Price field as a probability in [0, 1]. Handles cents ints and *_dollars strings."""
    if not isinstance(d, dict):
        return None
    if d.get(key + "_dollars") not in (None, ""):
        return float(d[key + "_dollars"])
    v = d.get(key)
    if v in (None, ""):
        return None
    if isinstance(v, int):
        return v / 100.0
    v = float(v)
    return v / 100.0 if v > 1.0 else v


def price_at(cands, ts, max_spread=0.25):
    """Yes-price at time ts: last quote mid if the spread is tight, else last trade."""
    past = sorted((c for c in cands if c.get("end_period_ts", 0) <= ts),
                  key=lambda c: c["end_period_ts"])
    if not past:
        return np.nan
    last = past[-1]
    bid, ask = num(last.get("yes_bid"), "close"), num(last.get("yes_ask"), "close")
    if bid is not None and ask is not None and ask > 0 and 0 <= ask - bid <= max_spread:
        return 0.5 * (bid + ask)
    for c in reversed(past):
        p = num(c.get("price"), "close")
        if p is not None:
            return p
    return np.nan


def strike_of(m):
    if m.get("floor_strike") is not None:
        return float(m["floor_strike"])
    mt = re.search(r"-T(-?\d+(?:\.\d+)?)$", m.get("ticker", ""))
    return float(mt.group(1)) if mt else np.nan


# --------------------------------------------------------------------------
# Distribution and settlement
# --------------------------------------------------------------------------
def distribution(strikes, surv, inclusive):
    """Mean and sd of the print from a ladder of survival probabilities.

    surv[i] = P(X > k_i) (or P(X >= k_i) if inclusive). Survival is forced
    non-increasing in the strike; bins are represented by grid points.
    """
    order = np.argsort(strikes)
    k = np.asarray(strikes, float)[order]
    s = np.minimum.accumulate(np.clip(np.asarray(surv, float)[order], 0, 1))
    step = np.median(np.diff(k)) if len(k) > 1 else 0.1
    probs = np.r_[1 - s[0], s[:-1] - s[1:], s[-1]]
    if inclusive:
        vals = np.r_[k[0] - step, k]
    else:
        vals = np.r_[k[0], k[1:], k[-1] + step]
    mean = float(np.sum(probs * vals))
    sd = float(np.sqrt(np.sum(probs * (vals - mean) ** 2)))
    return mean, sd


def actual_value(markets, inclusive):
    """Settled print: expiration_value if given, else inferred from yes/no results."""
    for m in markets:
        try:
            return float(m.get("expiration_value")), False
        except (TypeError, ValueError):
            pass
    ys = sorted(strike_of(m) for m in markets if m.get("result") == "yes")
    ns = sorted(strike_of(m) for m in markets if m.get("result") == "no")
    if not ys and not ns:
        return np.nan, True
    ks = sorted(strike_of(m) for m in markets)
    step = np.median(np.diff(ks)) if len(ks) > 1 else 0.1
    if inclusive:
        if not ys:
            return ns[0] - step, True
        x = ys[-1]
        approx = bool(ns) and ns[0] - ys[-1] > 1.5 * step
    else:
        if not ns:
            return ys[-1] + step, True
        x = ns[0]
        approx = bool(ys) and ns[0] - ys[-1] > 1.5 * step
    return x, approx


def release_times(markets):
    """Release (8:30 ET) and market close, from the markets' close_time."""
    t = pd.to_datetime([m.get("close_time") for m in markets if m.get("close_time")],
                       utc=True, errors="coerce").dropna()
    if len(t) == 0:
        return None, None
    close = t.min().tz_convert(NY)
    day = close.normalize()
    if close.hour >= 12:  # closed the evening before -> release next business day
        day = (day.tz_localize(None) + pd.offsets.BDay(1)).tz_localize(NY)
    return day + pd.Timedelta(hours=8, minutes=30), close


# --------------------------------------------------------------------------
# Build the panel
# --------------------------------------------------------------------------
def build_kalshi(snapshot):
    rows = []
    for series, label in SERIES.items():
        events = list_events(series)
        print(f"{series}: {len(events)} events")
        for e in events:
            et = e.get("event_ticker")
            markets, source = markets_for(et)
            if not markets or not any(m.get("result") in ("yes", "no") for m in markets):
                continue  # not yet settled
            release, close = release_times(markets)
            if release is None:
                continue
            inclusive = any(m.get("strike_type") == "greater_or_equal" for m in markets)
            if snapshot == "pre":
                snap = min(close, release) - pd.Timedelta(minutes=1)
            else:
                eve = (release.tz_localize(None).normalize() - pd.offsets.BDay(1))
                snap = (eve + pd.Timedelta(hours=16)).tz_localize(NY)
            start = snap - pd.Timedelta(days=4)

            strikes, surv = [], []
            for m in markets:
                k = strike_of(m)
                if np.isnan(k):
                    continue
                p = price_at(candles(series, m["ticker"], source, start.timestamp(),
                                     close.timestamp()), snap.timestamp())
                if not np.isnan(p):
                    strikes.append(k)
                    surv.append(p)
            actual, approx = actual_value(markets, inclusive)
            row = {"series": label, "event": et, "release": release.tz_localize(None),
                   "date": release.tz_localize(None).normalize(), "close_ny": close,
                   "n_markets": len(markets), "n_priced": len(strikes),
                   "actual": actual, "actual_approx": approx,
                   "k_mean": np.nan, "k_sd": np.nan}
            if len(strikes) >= 3:
                row["k_mean"], row["k_sd"] = distribution(strikes, surv, inclusive)
            rows.append(row)
            print(f"  {et:22s} release {release:%Y-%m-%d %H:%M}  priced {len(strikes):2d}/"
                  f"{len(markets):2d}  mean {row['k_mean']:.3f}  actual {actual}")
    df = pd.DataFrame(rows)
    df["surprise"] = df["actual"] - df["k_mean"]
    return df


def spy_returns(start):
    import yfinance as yf
    px = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    out = pd.DataFrame(index=px.index)
    out["r_on"] = 100 * (px["Open"] / px["Close"].shift(1) - 1)   # prev close -> open
    out["r_cc"] = 100 * (px["Close"] / px["Close"].shift(1) - 1)  # close -> close
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def fit(d, y, x, n_perm=10000):
    d = d[[y, x]].dropna()
    if len(d) < 6:
        return None
    res = sm.OLS(d[y], sm.add_constant(d[x])).fit(cov_type="HC1")
    xs, ys = d[x].to_numpy(), d[y].to_numpy()
    obs = np.corrcoef(xs, ys)[0, 1]
    rng = np.random.default_rng(0)
    null = np.array([np.corrcoef(rng.permutation(xs), ys)[0, 1] for _ in range(n_perm)])
    p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"n": len(d), "slope": res.params[x], "t": res.tvalues[x],
            "r2": res.rsquared, "p_perm": p_perm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", choices=["pre", "eve"], default="pre")
    args = ap.parse_args()

    k = build_kalshi(args.snapshot)
    found = set(k["date"].dt.strftime("%Y-%m-%d"))
    missing = [d for d in KNOWN_RELEASES if d not in found]
    print(f"\nRelease-date check against known 2022 dates: "
          f"{len(KNOWN_RELEASES) - len(missing)}/{len(KNOWN_RELEASES)} matched"
          + (f"; MISSING {missing} (inspect close_ny in the CSV)" if missing else ""))

    wide = k.pivot_table(index="date", columns="series",
                         values=["surprise", "k_sd", "actual", "k_mean"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    sur_cols = [c for c in ("surprise_headline", "surprise_core") if c in wide]
    wide["surprise_avg"] = wide[sur_cols].mean(axis=1)

    spy = spy_returns(k["date"].min() - pd.Timedelta(days=10))
    panel = wide.merge(spy, left_on="date", right_index=True, how="left")
    panel["period"] = np.where(panel["date"] < REGIME_SPLIT, "2021-23", "2024-26")
    panel.to_csv(DATA / f"go_no_go_panel_{args.snapshot}.csv", index=False)
    print(f"Panel: {len(panel)} release dates, "
          f"{panel['r_on'].notna().sum()} with SPY returns")

    print("\nReaction regressions: r_on (%) on surprise (pp), HC1 t, permutation p")
    results = {}
    for x in ["surprise_core", "surprise_headline", "surprise_avg"]:
        if x not in panel:
            continue
        for per, d in [("all", panel), *panel.groupby("period")]:
            r = fit(d, "r_on", x)
            if r:
                results[(x, per)] = r
                print(f"  {x:18s} {per:8s} n={r['n']:3d} slope={r['slope']:7.2f} "
                      f"t={r['t']:6.2f} R2={r['r2']:.2f} p_perm={r['p_perm']:.3f}")

    print("\nMagnitude check (Spearman):")
    main_x = "surprise_core" if "surprise_core" in panel else "surprise_headline"
    sd_col = main_x.replace("surprise", "k_sd")
    d = panel.dropna(subset=["r_on", main_x])
    rho_s, p_s = stats.spearmanr(d["r_on"].abs(), d[main_x].abs())
    print(f"  |r_on| vs |{main_x}|: rho={rho_s:.2f}, p={p_s:.3f}")
    d2 = panel.dropna(subset=["r_on", sd_col])
    rho_k, p_k = stats.spearmanr(d2["r_on"].abs(), d2[sd_col])
    print(f"  |r_on| vs Kalshi sd ({sd_col}): rho={rho_k:.2f}, p={p_k:.3f}")

    # Verdict: primary measure is core CPI in the 2021-23 regime.
    r_main = results.get((main_x, "2021-23"))
    react = r_main is not None and abs(r_main["t"]) > 2 and r_main["p_perm"] < 0.05
    size = rho_s > 0 and p_s < 0.10
    print("\n" + "=" * 70)
    print(f"Reaction to surprise in 2021-23 ({main_x}): {'YES' if react else 'NO'}")
    print(f"Bigger surprises -> bigger moves:            {'YES' if size else 'NO'}")
    if react and size:
        print("VERDICT: GO. The mechanism is visible in the data.")
    elif react or size:
        print("VERDICT: BORDERLINE. Inspect the figure; check eve snapshot and r_cc.")
    else:
        print("VERDICT: SWITCH. No measurable reaction to the Kalshi surprise.")
    print("Note: three surprise measures are reported; the verdict uses only the primary one.")
    print("=" * 70)

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for (per, g), col in zip(panel.groupby("period"), ["tab:red", "tab:blue"]):
        g = g.dropna(subset=[main_x, "r_on"])
        ax[0].scatter(g[main_x], g["r_on"], color=col, label=per, alpha=0.8)
        if len(g) >= 3:
            b = np.polyfit(g[main_x], g["r_on"], 1)
            xx = np.linspace(g[main_x].min(), g[main_x].max(), 10)
            ax[0].plot(xx, np.polyval(b, xx), color=col)
        ax[1].scatter(g[main_x].abs(), g["r_on"].abs(), color=col, label=per, alpha=0.8)
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].axvline(0, color="grey", lw=0.5)
    ax[0].set(xlabel=f"{main_x} (actual - Kalshi mean, pp)",
              ylabel="SPY overnight return on release day (%)",
              title="Reaction vs Kalshi surprise")
    ax[1].set(xlabel=f"|{main_x}| (pp)", ylabel="|SPY overnight return| (%)",
              title="Size of move vs size of surprise")
    for a in ax:
        a.legend()
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"go_no_go_{args.snapshot}.png", dpi=150)
    print(f"Saved {FIGS / f'go_no_go_{args.snapshot}.png'} and "
          f"{DATA / f'go_no_go_panel_{args.snapshot}.csv'}")


if __name__ == "__main__":
    main()