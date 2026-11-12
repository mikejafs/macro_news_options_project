#!/usr/bin/env python
"""Robustness: does a stricter Kalshi quote filter change sigma_K and beta_impl?

go_nogo.price_at uses the quote mid when the yes bid-ask spread is at most
max_spread (default 0.25) and the last trade otherwise. Wide quotes on tail
strikes pull survival probabilities toward 0.5 and can inflate sigma_K.
This rebuilds the eve-snapshot Kalshi core-CPI distribution with max_spread =
0.25, 0.10 and 0.05 from the cached API responses and recomputes, by period:
    uncertainty factor  mean sigma_K^2 / mean s^2      (1 = Kalshi calibrated)
    |beta_impl|         sqrt(mean EV / mean sigma_K^2)
If sigma_K were inflated by wide quotes, the factor would fall toward 1 and
|beta_impl| would rise as the filter tightens.

Run:   python spread_filter.py [--offline]
       --offline uses cached Kalshi responses only (a cache miss is treated as
       no data, which is what go_nogo received for requests that returned 4xx).
Reads data/cache_kalshi and data/decomposition_panel.csv.
"""
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import go_nogo as g  # noqa: E402

DATA = g.DATA
_PRICE_AT = g.price_at
g.SERIES = {"KXCPICORE": "core"}   # core is the primary measure


def _cached_only(path, params=None, tries=8):
    key = path + json.dumps(params or {}, sort_keys=True)
    fn = g.CACHE / (hashlib.md5(key.encode()).hexdigest() + ".json")
    return json.loads(fn.read_text()) if fn.exists() else None


if "--offline" in sys.argv:
    g.get = _cached_only


def kalshi_core(max_spread):
    g.price_at = functools.partial(_PRICE_AT, max_spread=max_spread)
    k = g.build_kalshi("eve")
    k = k[k.series == "core"][["date", "k_mean", "k_sd", "actual", "n_priced"]]
    return k.rename(columns={"k_mean": "k_mean_core", "k_sd": "k_sd_core"})


def summarize(panel, k, label):
    d = panel[["date", "period", "ev_med", "contaminated"]].merge(k, on="date", how="inner")
    d["s"] = d["actual"] - d["k_mean_core"]
    rows = []
    for p, x in [("all", d)] + list(d.groupby("period")):
        x = x.dropna(subset=["s", "k_sd_core", "ev_med"])
        sk2, s2 = (x.k_sd_core ** 2).mean(), (x.s ** 2).mean()
        rows.append(dict(spread=label, period=p, n=len(x), sd_K=np.sqrt(sk2), rms_s=np.sqrt(s2),
                         unc_factor=sk2 / s2, beta_impl=np.sqrt(max(x.ev_med.mean(), 0) / sk2),
                         priced=x.n_priced.mean()))
    return rows


def main():
    panel = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date"])
    out = []
    for ms in (0.25, 0.10, 0.05):
        out += summarize(panel, kalshi_core(ms), f"{ms:.2f}")
    t = pd.DataFrame(out)
    print("\nKalshi core CPI, eve snapshot, by quote-spread filter")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    t.to_csv(DATA / "spread_filter.csv", index=False)
    print("\nSaved data/spread_filter.csv")


if __name__ == "__main__":
    main()
