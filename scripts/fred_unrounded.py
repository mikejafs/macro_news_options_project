#!/usr/bin/env python
"""Unrounded first-release CPI month-over-month changes from FRED/ALFRED.

Kalshi settles on the BLS figure rounded to one decimal (0.2, 0.3, ...), but
the market sees the index level and reacts to the unrounded change (0.24 vs
0.26). For every release date D in the panel this asks ALFRED for the series
as it was published on D (realtime_start = realtime_end = D), takes the last
two monthly index levels in that vintage and computes
    u = 100 * (L_m / L_{m-1} - 1).
Using the same vintage for both months avoids mixing in later seasonal
revisions.

Series: CPILFESL (core, SA), CPIAUCSL (headline, SA).
Needs a free FRED API key (https://fredaccount.stlouisfed.org/apikeys):
    export FRED_API_KEY=...
Run:   python fred_unrounded.py
Writes data/cpi_unrounded.csv; responses cached in data/cache_fred.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE = ROOT / "data", ROOT / "data" / "cache_fred"
URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES = {"core": "CPILFESL", "headline": "CPIAUCSL"}


def vintage(series, day, key):
    params = dict(series_id=series, file_type="json", realtime_start=day, realtime_end=day,
                  observation_start=(pd.Timestamp(day) - pd.Timedelta(days=150)).strftime("%Y-%m-%d"),
                  sort_order="asc")
    fn = CACHE / (hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest() + ".json")
    if fn.exists():
        return json.loads(fn.read_text())
    for k in range(6):
        r = requests.get(URL, params={**params, "api_key": key}, timeout=30)
        if r.status_code == 200:
            fn.write_text(r.text)
            time.sleep(0.6)
            return r.json()
        if r.status_code in (400, 404):
            print(f"  {series} {day}: HTTP {r.status_code} {r.text[:120]}")
            return None
        time.sleep(2 ** k)
    raise RuntimeError(f"FRED request failed: {series} {day}")


def mom(js):
    if not js:
        return np.nan, None
    obs = [(pd.Timestamp(o["date"]), float(o["value"])) for o in js.get("observations", [])
           if o["value"] not in (".", "")]
    if len(obs) < 2:
        return np.nan, None
    (d0, l0), (d1, l1) = obs[-2], obs[-1]
    if (d1.year - d0.year) * 12 + d1.month - d0.month != 1:
        return np.nan, d1          # gap (e.g. a month never published)
    return 100 * (l1 / l0 - 1), d1


def main():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("Set FRED_API_KEY first (free: https://fredaccount.stlouisfed.org/apikeys).")
    CACHE.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(DATA / "go_no_go_panel_eve.csv", parse_dates=["date"])
    rows = []
    for d in panel["date"]:
        row = {"date": d}
        for lab, sid in SERIES.items():
            row[f"{lab}_u"], row[f"{lab}_month"] = mom(vintage(sid, d.strftime("%Y-%m-%d"), key))
        rows.append(row)
    u = pd.DataFrame(rows)
    u.to_csv(DATA / "cpi_unrounded.csv", index=False)
    m = panel.merge(u, on="date")
    for lab in SERIES:
        x = m.dropna(subset=[f"{lab}_u", f"actual_{lab}"])
        agree = (x[f"{lab}_u"].round(1) == x[f"actual_{lab}"].round(1)).mean()
        print(f"{lab}: {len(x)} releases; unrounded rounds to the Kalshi settlement value on "
              f"{100 * agree:.0f}% (expect > 90%); sd of rounding error "
              f"{(x[f'{lab}_u'] - x[f'actual_{lab}']).std():.3f} pp (uniform rounding: 0.029)")
    print("Saved", DATA / "cpi_unrounded.csv")


if __name__ == "__main__":
    main()
