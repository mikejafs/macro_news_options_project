"""State-space model for the CPI sensitivity beta_t and the event premium pi_t.

One time step = one CPI release. Units: returns in %, surprises in pp,
variances in %^2, beta in %/pp.

State (random walks):
    beta_t = beta_{t-1} + w_b,   w_b ~ N(0, q_b)
    pi_t   = pi_{t-1}   + w_p,   w_p ~ N(0, q_p)          (joint model only)

Observations:
    r_t  = beta_t * s_t + e_t,                  e_t ~ N(0, c * v_t)
    EV_t = beta_t^2 * sK2_t + pi_t + u_t,        u_t ~ N(0, su2)  (joint only)

    r_t   overnight SPY return on release day t
    s_t   surprise = actual print - Kalshi eve mean
    v_t   known baseline overnight noise variance (phi * V_bg,t); c scales it
    EV_t  implied event variance from VIX1D (eve)
    sK2_t Kalshi eve variance of the print

Two estimators:
    "returns": beta_t filtered from returns only (linear Gaussian, exact
               Kalman). Option prices never enter beta_t. The fair event
               variance uses the one-step-ahead prediction,
               E_{t-1}[beta_t^2] sK2_t = (m_{t|t-1}^2 + P_{t|t-1}) sK2_t,
               and excess_t = EV_t - fair_t (no look-ahead).
    "joint":   (beta_t, pi_t) filtered from both observations by an
               iterated extended Kalman filter (EV is quadratic in beta).

Parameters are estimated by maximising the prediction-error-decomposition
log likelihood (exact for "returns", quasi-likelihood for "joint").
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

LOG2PI = np.log(2.0 * np.pi)


@dataclass
class FilterResult:
    m_pred: np.ndarray   # (T, d) one-step-ahead state means  m_{t|t-1}
    P_pred: np.ndarray   # (T, d, d)                           P_{t|t-1}
    m_filt: np.ndarray   # (T, d) filtered means                m_{t|t}
    P_filt: np.ndarray   # (T, d, d)
    loglik: float


def ekf(y, R, h, Q, m0, P0, n_iter=5):
    """Iterated extended Kalman filter for a random-walk state.

    y   (T, k) observations, NaN = missing
    R   (T, k) observation noise variances (diagonal)
    h   callable h(x, t) -> (yhat (k,), H (k, d))
    Q   (d, d) state innovation covariance
    For linear h the iteration converges in one step and the filter is exact.
    """
    T, k = y.shape
    d = len(m0)
    m, P = np.asarray(m0, float).copy(), np.asarray(P0, float).copy()
    out = FilterResult(np.zeros((T, d)), np.zeros((T, d, d)),
                       np.zeros((T, d)), np.zeros((T, d, d)), 0.0)
    ll = 0.0
    I = np.eye(d)
    for t in range(T):
        mp, Pp = m, P + Q
        out.m_pred[t], out.P_pred[t] = mp, Pp
        obs = ~np.isnan(y[t])
        if not obs.any():
            m, P = mp, Pp
        else:
            x = mp.copy()
            for _ in range(n_iter):
                yhat, H = h(x, t)
                yhat, H = yhat[obs], H[obs]
                v = y[t, obs] - yhat - H @ (mp - x)
                S = H @ Pp @ H.T + np.diag(R[t, obs])
                K = Pp @ H.T @ np.linalg.inv(S)
                x_new = mp + K @ v
                if np.max(np.abs(x_new - x)) < 1e-10:
                    x = x_new
                    break
                x = x_new
            A = I - K @ H
            m = x
            P = A @ Pp @ A.T + K @ np.diag(R[t, obs]) @ K.T   # Joseph form
            sign, logdet = np.linalg.slogdet(S)
            ll += -0.5 * (obs.sum() * LOG2PI + logdet + v @ np.linalg.solve(S, v))
        out.m_filt[t], out.P_filt[t] = m, P
    out.loglik = ll
    return out


def rts_smoother(f: FilterResult):
    """Rauch-Tung-Striebel smoother for a random-walk transition (F = I)."""
    T = len(f.m_filt)
    ms, Ps = f.m_filt.copy(), f.P_filt.copy()
    for t in range(T - 2, -1, -1):
        G = f.P_filt[t] @ np.linalg.inv(f.P_pred[t + 1])
        ms[t] = f.m_filt[t] + G @ (ms[t + 1] - f.m_pred[t + 1])
        Ps[t] = f.P_filt[t] + G @ (Ps[t + 1] - f.P_pred[t + 1]) @ G.T
    return ms, Ps


# ---------------------------------------------------------------- model specs

def _returns_spec(theta, data):
    q_b, c = np.exp(theta)
    s, r, v = data["s"], data["r"], data["v"]
    y = r[:, None]
    R = (c * v)[:, None]

    def h(x, t):
        return np.array([x[0] * s[t]]), np.array([[s[t]]])

    return y, R, h, np.array([[q_b]]), np.array([0.0]), np.array([[100.0]])


def _joint_spec(theta, data):
    q_b, q_p, c, su2 = np.exp(theta)
    s, r, v, ev, sk2 = data["s"], data["r"], data["v"], data["ev"], data["sk2"]
    y = np.column_stack([r, ev])
    R = np.column_stack([c * v, np.full(len(r), su2)])

    def h(x, t):
        b, p = x
        yhat = np.array([b * s[t], b * b * sk2[t] + p])
        H = np.array([[s[t], 0.0], [2.0 * b * sk2[t], 1.0]])
        return yhat, H

    Q = np.diag([q_b, q_p])
    return y, R, h, Q, np.array([0.0, 0.0]), np.diag([100.0, 4.0])


SPECS = {
    "returns": (_returns_spec, np.log([0.5, 1.0]), ["q_b", "c"]),
    "joint": (_joint_spec, np.log([0.5, 0.01, 1.0, 0.1]), ["q_b", "q_p", "c", "su2"]),
}
BOUNDS = (-12.0, 4.0)   # bounds on log-parameters


def negloglik(theta, model, data):
    spec, _, _ = SPECS[model]
    f = ekf(*spec(theta, data))
    return -f.loglik


def numerical_hessian(fun, x, eps=1e-4):
    n = len(x)
    Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i, e_j = np.eye(n)[i] * eps, np.eye(n)[j] * eps
            Hm[i, j] = Hm[j, i] = (fun(x + e_i + e_j) - fun(x + e_i - e_j)
                                   - fun(x - e_i + e_j) + fun(x - e_i - e_j)) / (4 * eps * eps)
    return Hm


@dataclass
class FitResult:
    model: str
    params: dict          # natural-scale parameter estimates
    se_log: dict          # standard errors of log-parameters (inverse Hessian)
    loglik: float
    filt: FilterResult
    m_smooth: np.ndarray
    P_smooth: np.ndarray


def fit(model, data, theta0=None, se=True):
    """QML fit. data: dict of 1-D arrays s, r, v (+ ev, sk2 for joint)."""
    spec, default0, names = SPECS[model]
    theta0 = default0 if theta0 is None else theta0
    obj = lambda th: negloglik(th, model, data)
    res = minimize(obj, theta0, method="L-BFGS-B", bounds=[BOUNDS] * len(theta0))
    th = res.x
    se_log = {n: np.nan for n in names}
    if se:
        try:
            cov = np.linalg.inv(numerical_hessian(obj, th))
            se_log = dict(zip(names, np.sqrt(np.clip(np.diag(cov), 0, None))))
        except np.linalg.LinAlgError:
            pass
    f = ekf(*spec(th, data))
    ms, Ps = rts_smoother(f)
    return FitResult(model, dict(zip(names, np.exp(th))), se_log, -res.fun, f, ms, Ps)


def fair_event_variance(fit_returns: FitResult, sk2):
    """E_{t-1}[beta_t^2] * sK2_t from the returns-only filter (no look-ahead)."""
    m = fit_returns.filt.m_pred[:, 0]
    P = fit_returns.filt.P_pred[:, 0, 0]
    return (m * m + P) * sk2


def newey_west_mean(x, lags=3):
    """Mean of x and its Newey-West standard error (Bartlett kernel)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    u = x - x.mean()
    g = u @ u / n
    for L in range(1, lags + 1):
        g += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return x.mean(), np.sqrt(g / n)
