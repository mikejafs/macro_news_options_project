#!/usr/bin/env python
"""Exact release dates of the main 08:30 ET US macro releases, from FRED.

For each release below this calls fred/release (to print FRED's own name for
the id, so a wrong id is visible) and fred/release/dates, and writes one row
per (date, release) to data/macro_calendar.csv.

All seven are published at 08:30 ET. Release ids are FRED's; check the printed
names against the labels.

Run:   export FRED_API_KEY=...
       python fred_calendar.py
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
API = "https://api.stlouisfed.org/fred/"
RELEASES = {          # label: FRED release id
    "CPI": 10,        # Consumer Price Index
    "JOBS": 50,       # Employment Situation
    "PPI": 46,        # Producer Price Index
    "RETAIL": 9,      # Advance Monthly Sales for Retail and Food Services
    "GDP": 53,        # Gross Domestic Product
    "PCE": 54,        # Personal Income and Outlays
    "CLAIMS": 180,    # Unemployment Insurance Weekly Claims Report
}
START, END = "2022-05-01", "2026-10-31"


def get(path, key, **params):
    for k in range(6):
        r = requests.get(API + path, params={**params, "api_key": key, "file_type": "json"}, timeout=30)
        if r.status_code == 200:
            time.sleep(0.6)
            return r.json()
        print(f"  {path} {params}: HTTP {r.status_code} {r.text[:120]}", flush=True)
        if r.status_code in (400, 404):
            return None
        time.sleep(2 ** k)
    raise RuntimeError("FRED request failed")


def main():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("Set FRED_API_KEY first.")
    rows = []
    for lab, rid in RELEASES.items():
        info = get("release", key, release_id=rid)
        name = info["releases"][0]["name"] if info else "NOT FOUND"
        js = get("release/dates", key, release_id=rid, realtime_start=START, realtime_end=END,
                 include_release_dates_with_no_data="false", limit=10000, sort_order="asc")
        dates = [x["date"] for x in (js or {}).get("release_dates", []) if START <= x["date"] <= END]
        print(f"{lab:7s} id {rid:4d}  FRED name: {name:60s} {len(dates):4d} dates", flush=True)
        rows += [{"date": d, "release": lab} for d in dates]
    cal = pd.DataFrame(rows).drop_duplicates().sort_values(["date", "release"])
    cal.to_csv(DATA / "macro_calendar.csv", index=False)
    kalshi = pd.read_csv(DATA / "go_no_go_panel_eve.csv", parse_dates=["date"])
    k = set(kalshi.loc[kalshi.date >= START, "date"].dt.strftime("%Y-%m-%d"))
    f = set(cal.loc[cal.release == "CPI", "date"])
    print(f"\nCheck: {len(k & f)} of {len(k)} Kalshi-inferred CPI dates are FRED CPI release dates "
          f"(expect nearly all); FRED-only CPI dates: {sorted(f - k)[:6]}")
    print("Expected counts over 4.5 years: monthly releases about 54, GDP about 54 (three estimates "
          "per quarter), claims about 234.")
    print("Saved", DATA / "macro_calendar.csv")


if __name__ == "__main__":
    main()
