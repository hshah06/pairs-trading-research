"""
Step 3b — Letting the hedge ratio move: a Kalman filter
=======================================================

The problem with a fixed hedge ratio
------------------------------------
The OLS hedge ratio beta is ONE number fitted to 2012–2020 and then frozen.
But the relationship between two companies drifts. KO/PEP is the example from
our first run: after 2024, PepsiCo's snack business pulled it away from
Coca-Cola. With beta frozen, that drift shows up as a "mispricing" that never
corrects, and the strategy keeps betting against a trend.

The fix is to let beta change over time, learning from each new day of data.
A Kalman filter is the standard tool for that.

The model
---------
We treat the regression coefficients as hidden quantities that wander slowly:

    observation:  log Y_t = alpha_t + beta_t * xc_t + eps_t,    eps_t ~ N(0, R)
    state:        alpha_t = alpha_{t-1} + w_alpha,               w ~ N(0, Q)
                  beta_t  = beta_{t-1}  + w_beta

* The observation equation is the same regression as before, except that
  alpha and beta get a time subscript.
* The state equation says that alpha and beta follow slow random walks. We
  never see them directly; we only see the prices they produce.
* ``xc_t = log X_t - c`` is log X CENTRED on its in-sample mean ``c``. This
  small detail matters. log X sits around 4 to 5, so without centring, a tiny
  change in beta shifts the fitted level by 4–5× as much, and the filter can't
  tell "beta moved" from "alpha moved". Centring decouples them. It doesn't
  change what beta means: log Y = (alpha - beta*c) + beta * log X.

The filter, one day at a time
-----------------------------
Every day the Kalman filter does two steps:

1. **Predict.** Before seeing today's prices, the best guess for today's
   (alpha, beta) is yesterday's. Our uncertainty P grows by Q, because the
   state might have wandered overnight.
2. **Update.** We see log Y_t and compare it with the prediction:

        e_t = log Y_t - (alpha_pred + beta_pred * xc_t)       ← the "innovation"
        S_t = H P H' + R                                       ← its variance
        K_t = P H' / S_t                                       ← the Kalman gain

   and then nudge the state by K_t * e_t. The gain K decides how much to
   believe today's surprise. A large Q relative to R means a large K and a
   jumpy beta; a small Q means a small K and a sluggish beta.

The trading spread is the innovation e_t
----------------------------------------
e_t is today's log Y minus what yesterday's (alpha, beta) predicts. It is the
dynamic-beta version of the OLS residual, and it uses only information
available before today's close, so there is no look-ahead.

The key tuning choice: how fast should beta adapt?
--------------------------------------------------
This is a real dilemma, not a detail.

* If beta adapts too quickly, the filter "explains away" the mispricing we
  want to trade. When Y gets cheap, the filter decides the relationship
  changed, the spread snaps back to zero instantly, and there is nothing left
  to trade.
* If beta adapts too slowly, we're back to a frozen OLS beta.

We parameterise this as a **memory** of N trading days. For a slowly moving
level, the steady-state Kalman gain is about sqrt(Q/R), so the filter averages
over roughly 1/K = sqrt(R/Q) days. Setting

        Q_alpha = R / N^2            Q_beta = R / (N^2 * var(xc))

gives both coefficients an effective memory of about N days. (Q_beta is scaled
by var(xc) so that a typical beta move shifts the fitted value about as much as
a typical alpha move.)

We choose N = 250 days (one year) BEFORE looking at any results. That is
several times longer than the spreads' half-lives (30–60 days), so a
mispricing lasting weeks is mostly left alone, while a drift lasting years gets
absorbed. R is the variance of the in-sample OLS residuals. Both N and R are
decided using in-sample information only.

The run also prints a sensitivity table over N. Its purpose is to check that
the conclusion doesn't hinge on one lucky value. It is NOT a menu to pick the
best out-of-sample N from: doing that would turn the OOS period into
in-sample data.

Caveat: the filter assumes eps_t is white noise. Our "noise" is the
mean-reverting spread itself, which is strongly autocorrelated. The filter
still works as a smoother of beta, but its probabilistic outputs (e.g.
S_t as a true forecast variance) shouldn't be taken literally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanResult:
    beta: pd.Series        # beta_{t|t}: estimate after seeing day t (used for position sizing)
    alpha: pd.Series       # alpha_{t|t}
    beta_prior: pd.Series  # beta_{t|t-1}: estimate before seeing day t
    spread: pd.Series      # innovation e_t (the trading spread)
    spread_var: pd.Series  # S_t, the filter's innovation variance
    memory: float
    R: float
    center: float


def kalman_hedge(
    log_y: pd.Series,
    log_x: pd.Series,
    R: float,
    center: float,
    var_xc: float,
    memory: float = 250.0,
    init_beta: float = 0.0,
) -> KalmanResult:
    """Run the Kalman filter over the WHOLE history (IS then OOS, uninterrupted).

    Running straight through is legitimate: at each date the filter has only
    seen data up to that date. What we must NOT do is change R, memory or
    center using OOS data, and we don't. All three come from in-sample.
    """
    ly = log_y.values
    xc = (log_x - center).values
    n = len(ly)

    Q = np.diag([R / memory**2, R / (memory**2 * var_xc)])

    # Start with a loose prior: alpha at the first observation, beta at
    # ``init_beta``, and a large uncertainty so the first few hundred days of
    # data quickly overwhelm the guess.
    state = np.array([ly[0], init_beta])
    P = np.eye(2)

    out = np.full((n, 6), np.nan)  # alpha_post, beta_post, beta_prior, e, S, alpha_prior
    for t in range(n):
        # ---- 1. Predict: state carries over, uncertainty grows by Q ----------
        P = P + Q
        H = np.array([1.0, xc[t]])

        # ---- 2. Update with today's observation ----------------------------
        y_hat = H @ state
        e = ly[t] - y_hat                  # innovation = the trading spread
        S = H @ P @ H + R                  # how big we expect e to be
        K = P @ H / S                      # Kalman gain (a 2-vector)

        out[t, 2] = state[1]               # beta before seeing today (prior)
        out[t, 5] = state[0]
        state = state + K * e
        P = P - np.outer(K, H) @ P

        out[t, 0], out[t, 1] = state
        out[t, 3], out[t, 4] = e, S

    idx = log_y.index
    return KalmanResult(
        alpha=pd.Series(out[:, 0], idx, name="alpha"),
        beta=pd.Series(out[:, 1], idx, name="beta"),
        beta_prior=pd.Series(out[:, 2], idx, name="beta_prior"),
        spread=pd.Series(out[:, 3], idx, name="kalman_spread"),
        spread_var=pd.Series(out[:, 4], idx, name="S"),
        memory=memory, R=R, center=center,
    )


def kalman_from_insample(prices: pd.DataFrame, y: str, x: str, split: pd.Timestamp,
                         ols_resid: pd.Series, memory: float = 250.0,
                         init_beta: float = 0.0) -> KalmanResult:
    """Set up the filter's constants from IN-SAMPLE data, then filter all dates."""
    log_y, log_x = np.log(prices[y]), np.log(prices[x])
    is_lx = log_x.loc[: split - pd.Timedelta(days=1)]
    return kalman_hedge(
        log_y, log_x,
        R=float(ols_resid.var()),     # size of the "noise" around the fitted line
        center=float(is_lx.mean()),
        var_xc=float(is_lx.var()),
        memory=memory,
        init_beta=init_beta,
    )
