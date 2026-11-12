"""Injection-recovery for the beta_t / pi_t state-space estimators.

Simulated panels match the real core-CPI panel: T = 52 releases, regime
break at release 18 (2021-23 -> 2024-26), surprise sd 0.124 -> 0.085 pp,
Kalshi sd = 1.27 x surprise sd, overnight noise phi * V_bg with phi = 0.44,
VIX1D ~ 25 -> 14.

True beta: -10 before the break, -2 after, plus a small random walk.

Scenarios for what options price (EV_t):
  fair    EV = beta_t^2 sK2 + pi + u,  pi = 0.15 %^2 (flat premium)
  sticky  EV = beta_opt^2 sK2 + u,     beta_opt = beta before the break,
                                       -6 after (options keep pricing |beta| = 6)

Questions answered:
  1. Does the returns-only filter recover beta_t, with honest 95% bands?
  2. Does excess = EV - fair recover the truth in each scenario, and does
     its slope on sK2 distinguish a flat premium (slope 0) from mispriced
     sensitivity (slope beta_opt^2 - beta^2 = 32)?
  3. Does the joint EKF get contaminated when options misprice beta?

Run:  python ss_injection.py [n_sims] [ratio]
      ratio = window noise variance / overnight noise variance (default 1 =
      overnight returns). Use the rho printed by narrow_window.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ss_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
T, BREAK = 52, 18
PHI = 0.44
RATIO = 1.0   # noise variance relative to the overnight window


def simulate(rng, scenario):
    t = np.arange(T)
    pre = t < BREAK
    sd_s = np.where(pre, 0.124, 0.085)
    s = rng.normal(0, sd_s)
    sk2 = (1.27 * sd_s * np.exp(rng.normal(0, 0.25, T))) ** 2
    vix1d = np.where(pre, 25.0, 14.0) * np.exp(rng.normal(0, 0.2, T))
    v = RATIO * PHI * vix1d ** 2 / 252.0
    beta = np.where(pre, -10.0, -2.0) + np.cumsum(rng.normal(0, 0.3, T))
    r = beta * s + rng.normal(0, np.sqrt(v))
    u = rng.normal(0, 0.35, T)
    if scenario == "fair":
        pi_true = np.full(T, 0.15)
        ev = beta ** 2 * sk2 + pi_true + u
        excess_true = pi_true
    elif scenario == "sticky":
        beta_opt = np.where(pre, beta, -6.0)
        ev = beta_opt ** 2 * sk2 + u
        excess_true = (beta_opt ** 2 - beta ** 2) * sk2
    else:
        raise ValueError(scenario)
    return dict(s=s, r=r, v=v, ev=ev, sk2=sk2), beta, excess_true


def slope(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xc = x - x.mean()
    return xc @ (y - y.mean()) / (xc @ xc)


def run(n_sims=200, seed=1):
    rng = np.random.default_rng(seed)
    post = np.arange(T) >= BREAK
    pre = ~post
    rows = {}
    for scen in ["fair", "sticky"]:
        R = {k: [] for k in ["rmse_pre", "rmse_post", "cover", "exc_mean", "exc_true",
                             "exc_slope", "exc_tstat", "joint_rmse_post",
                             "joint_pi_post", "q_b"]}
        for _ in range(n_sims):
            data, beta, exc_true = simulate(rng, scen)
            fr = sm.fit("returns", data, se=False)
            b_f = fr.filt.m_filt[:, 0]
            sd_f = np.sqrt(fr.filt.P_filt[:, 0, 0])
            R["rmse_pre"].append(np.sqrt(np.mean((b_f - beta)[pre][3:] ** 2)))
            R["rmse_post"].append(np.sqrt(np.mean((b_f - beta)[post] ** 2)))
            R["cover"].append(np.mean(np.abs(b_f - beta) < 1.96 * sd_f))
            exc = data["ev"] - sm.fair_event_variance(fr, data["sk2"])
            mu, se = sm.newey_west_mean(exc[post])
            R["exc_mean"].append(mu)
            R["exc_tstat"].append(mu / se)
            R["exc_true"].append(exc_true[post].mean())
            R["exc_slope"].append(slope(data["sk2"][post], exc[post]))
            R["q_b"].append(fr.params["q_b"])
            fj = sm.fit("joint", data, se=False)
            R["joint_rmse_post"].append(np.sqrt(np.mean((fj.m_smooth[post, 0] - beta[post]) ** 2)))
            R["joint_pi_post"].append(fj.m_smooth[post, 1].mean())
        rows[scen] = {k: np.array(v) for k, v in R.items()}
    return rows


def report(rows):
    print(f"{'':28s}{'fair':>18s}{'sticky':>18s}")

    def line(label, key, f=np.median, fmt="{:8.3f}"):
        vals = [fmt.format(f(rows[s][key])) for s in ["fair", "sticky"]]
        print(f"{label:28s}" + "".join(f"{v:>18s}" for v in vals))

    print("returns-only filter (beta_t)")
    line("  RMSE beta, pre (ex 3)", "rmse_pre")
    line("  RMSE beta, post", "rmse_post")
    line("  95% band coverage", "cover", np.mean)
    line("  q_b (median)", "q_b")
    print("excess EV, post-break")
    line("  mean estimate (median)", "exc_mean")
    line("  truth (median)", "exc_true")
    line("  slope on sK2 (median)", "exc_slope", fmt="{:8.1f}")
    line("  P(NW t > 1.96)", "exc_tstat", lambda x: np.mean(x > 1.96))
    print("joint EKF")
    line("  RMSE beta, post (smoothed)", "joint_rmse_post")
    line("  mean pi, post (smoothed)", "joint_pi_post")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    RATIO = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    print(f"n_sims = {n}, noise ratio = {RATIO}")
    report(run(n))
