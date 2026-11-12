#!/usr/bin/env python
"""Download SPY 1-minute bars (pre-market included) from Alpaca Market Data.

Needs a free Alpaca account and an API key pair, set as environment variables:
    export APCA_API_KEY_ID=...
    export APCA_API_SECRET_KEY=...

Bars are cached one month per file in data/cache_alpaca/SPY_YYYY-MM.csv.gz,
so reruns only fetch missing months. Timestamps are bar START times in UTC
(a bar stamped 13:29Z covers trades in [13:29:00, 13:30:00)).

Feed: "sip" (consolidated tape). The free plan allows SIP history older than
15 minutes. If the request is refused, rerun with --feed iex (IEX only,
thinner pre-market) and note it as a limitation.

Run:   python alpaca_bars.py                     # 2022-05 .. 2026-10
       python alpaca_bars.py --start 2022-05 --end 2026-10 --feed sip
Needs: requests pandas
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache_alpaca"
URL = "https://data.alpaca.markets/v2/stocks/{sym}/bars"


def get(session, url, params, tries=8):
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if k == tries - 1 or (isinstance(e, requests.HTTPError) and "HTTP 4" in str(e)
                                  and "429" not in str(e)):
                raise
            time.sleep(min(60, 2 ** k))


def month_bars(session, sym, month, feed):
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    params = dict(timeframe="1Min", start=start.isoformat(), end=end.isoformat(),
                  limit=10000, adjustment="raw", feed=feed, sort="asc")
    rows = []
    while True:
        js = get(session, URL.format(sym=sym), params)
        rows += js.get("bars") or []
        tok = js.get("next_page_token")
        if not tok:
            break
        params["page_token"] = tok
    df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                                            "c": "close", "v": "volume", "n": "trades",
                                            "vw": "vwap"})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="SPY")
    ap.add_argument("--start", default="2022-05")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--feed", default="sip")
    a = ap.parse_args()
    key, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY first.")
    CACHE.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    for m in pd.period_range(a.start, a.end, freq="M").astype(str):
        fn = CACHE / f"{a.sym}_{m}.csv.gz"
        if fn.exists():
            continue
        df = month_bars(s, a.sym, m, a.feed)
        df.to_csv(fn, index=False)
        print(f"{m}: {len(df):6d} bars")
    print("done:", CACHE)


if __name__ == "__main__":
    main()
