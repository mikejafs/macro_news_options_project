#!/usr/bin/env python
"""Rates channel: split the CPI sensitivity of stocks into two links.

    bond reaction     b_t = delta * s_t + e_t      (Treasury ETF return on the surprise)
    stock reaction    r_t = beta  * s_t + e_t      (SPY, from narrow_window.py)
    beta = gamma * delta,  gamma = beta / delta    (stock move per unit bond move
                                                    caused by the CPI surprise)

gamma = beta/delta is the instrumental-variable (Wald) estimate of the slope of
SPY on the bond with the Kalshi surprise as instrument. It is not biased by
noise in the bond return, unlike the OLS slope of r on b, which is printed for
comparison together with the same OLS slope on non-event mornings.

Pre-specified (11 Nov, before any bond returns were computed): bond = TLT
(most liquid pre-market), window 08:30 -> 08:45 ET, IEF and the 08:35 / 09:00
horizons as robustness, regime split 2024-01-01.

Units: bond and stock returns in %, s in pp. delta > 0 would mean bonds rally
on a high print; the expected sign is delta < 0 (yields up, prices down) and
gamma > 0 (stocks fall with bonds) in an inflation regime.

Run:   python alpaca_bars.py --sym TLT ; python alpaca_bars.py --sym IEF
       python rates_channel.py
Writes data/rates_channel_panel.csv and figures/rates_channel.png.
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

DATA, FIGS = nw.DATA, nw.ROOT / "figures"
BONDS, PRIMARY, H = ["TLT", "IEF"], "TLT", ["35", "45", "60"]
RNG = np.random.default_rng(0)


def ols(x, y):
    """Slope, HC1 se, R^2 of y on x (with intercept)."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    xc, yc = x - x.mean(), y - y.mean()
    b = xc @ yc / (xc @ xc)
    e = yc - b * xc
    se = np.sqrt(np.sum(xc ** 2 * e ** 2) / (xc @ xc) ** 2 * len(x) / (len(x) - 2))
    return b, se, 1 - e @ e / (yc @ yc), len(x)


def wald(s, b, r, n_boot=4000):
    """gamma = beta/delta with a percentile bootstrap interval over releases."""
    ok = ~(np.isnan(s) | np.isnan(b) | np.isnan(r))
    s, b, r = s[ok], b[ok], r[ok]
    g = ols(s, r)[0] / ols(s, b)[0]
    idx = RNG.integers(0, len(s), (n_boot, len(s)))
    gs = np.array([ols(s[i], r[i])[0] / ols(s[i], b[i])[0] for i in idx])
    return g, np.percentile(gs, 2.5), np.percentile(gs, 97.5)


def main():
    panel = pd.read_csv(DATA / "decomposition_panel.csv", parse_dates=["date"])
    spy = pd.read_csv(DATA / "narrow_window_panel.csv", parse_dates=["date"])
    d = panel[["date", "period", "s", "contaminated"]].merge(
        spy[["date"] + [f"r_win{h}" for h in H]], on="date")
    d = d.rename(columns={f"r_win{h}": f"spy{h}" for h in H})
    events = set(panel["date"])
    quiet = {}
    for sym in BONDS:
        t = nw.daily_table(nw.load_bars(sym))
        t["event"] = t["day"].isin(events) | t["day"].isin(nw.nfp_dates(t["day"]))
        d = d.merge(t[["day"] + [f"r_win{h}" for h in H]].rename(
            columns={"day": "date", **{f"r_win{h}": f"{sym}{h}" for h in H}}), on="date", how="left")
        quiet[sym] = t[~t["event"]]
        print(f"{sym}: window return available on {d[f'{sym}45'].notna().sum()} / {len(d)} CPI days")
    d.to_csv(DATA / "rates_channel_panel.csv", index=False)

    samples = [("2021-23", d[d.period == "2021-23"]), ("2024-26", d[d.period == "2024-26"]),
               ("2024-26 ex-contam", d[(d.period == "2024-26") & d.contaminated.isna()])]
    for sym in BONDS:
        for h in H:
            if sym != PRIMARY and h != "45":
                continue
            print(f"\n{sym}, window 08:30 -> {nw.HORIZONS[h]}")
            print(f"{'sample':18s}{'n':>4s} | {'delta':>7s}{'t':>6s}{'R2':>6s} | {'beta':>7s}{'t':>6s}{'R2':>6s}"
                  f" | {'gamma=b/d':>10s}{'95% CI':>16s} | {'OLS r~b':>8s}")
            for lab, g in samples:
                s, b, r = g["s"].values, g[f"{sym}{h}"].values, g[f"spy{h}"].values
                dl, dse, dr2, n = ols(s, b)
                be, bse, br2, _ = ols(s, r)
                gm, lo, hi = wald(s, b, r)
                print(f"{lab:18s}{n:4d} | {dl:7.2f}{dl / dse:6.1f}{dr2:6.2f} | {be:7.2f}{be / bse:6.1f}{br2:6.2f}"
                      f" | {gm:10.2f}   [{lo:5.2f},{hi:5.2f}] | {ols(b, r)[0]:8.2f}")
    # stock-bond comovement on ordinary mornings, same window
    sq = nw.daily_table(nw.load_bars("SPY"))
    q = quiet[PRIMARY][["day", "r_win45"]].merge(sq[["day", "r_win45"]], on="day", suffixes=("_b", "_s"))
    print(f"\nNon-event mornings, OLS slope of SPY on {PRIMARY} (08:30 -> 08:45):")
    for lab, g in [("2021-23", q[q.day < "2024-01-01"]), ("2024-26", q[q.day >= "2024-01-01"])]:
        b, se, r2, n = ols(g.r_win45_b.values, g.r_win45_s.values)
        print(f"  {lab}: {b:5.2f} (t {b / se:5.1f}, R2 {r2:.2f}, n {n})")

    # time-varying delta_t and beta_t with the same Kalman filter and the same
    # local noise model as ss_fit.py (quiet mornings within +-20 trading days)
    def local_v(t, col):
        cap = (t.loc[~t.event, col] ** 2).quantile(0.99)
        pos = {day: i for i, day in enumerate(t["day"])}
        out = []
        for day in d["date"]:
            i = pos.get(day)
            if i is None:
                out.append(np.nan)
                continue
            nb = t.iloc[max(0, i - nw.HALF): i + nw.HALF + 1]
            out.append((nb.loc[~nb.event, col] ** 2).clip(upper=cap).mean())
        out = np.array(out)
        return np.where(np.isnan(out), np.nanmedian(out), out)

    sq["event"] = sq["day"].isin(events) | sq["day"].isin(nw.nfp_dates(sq["day"]))
    bt = nw.daily_table(nw.load_bars(PRIMARY))
    bt["event"] = bt["day"].isin(events) | bt["day"].isin(nw.nfp_dates(bt["day"]))
    out = {}
    for name, col, v in [("delta", f"{PRIMARY}45", local_v(bt, "r_win45")),
                         ("beta", "spy45", local_v(sq, "r_win45"))]:
        f = sm.fit("returns", dict(s=d["s"].values, r=d[col].values, v=v))
        out[name] = (f.m_smooth[:, 0], np.sqrt(f.P_smooth[:, 0, 0]))
        print(f"Kalman {name}_t: q {f.params['q_b']:.3g}, c {f.params['c']:.3g}, loglik {f.loglik:.2f}")
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for a, name, lab in [(ax[0], "delta", f"delta_t: {PRIMARY} % per pp"), (ax[1], "beta", "beta_t: SPY % per pp")]:
        m, sd = out[name]
        a.fill_between(d.date, m - 1.96 * sd, m + 1.96 * sd, alpha=0.2)
        a.plot(d.date, m, lw=2)
        a.axhline(0, c="grey", lw=0.5)
        a.set_ylabel(lab)
    ax[2].plot(d.date, out["beta"][0] / out["delta"][0], lw=2, c="k")
    ax[2].axhline(0, c="grey", lw=0.5)
    ax[2].set_ylabel("gamma_t = beta_t / delta_t")
    ax[0].set_title("Rates channel of the CPI reaction (Kalman-smoothed, 08:30 -> 08:45)")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "rates_channel.png", dpi=150)
    print("\nSaved data/rates_channel_panel.csv and figures/rates_channel.png")


if __name__ == "__main__":
    main()
