"""
Step 3 — How FAST does the spread mean-revert? (Ornstein–Uhlenbeck)
===================================================================

Cointegration tells you *whether* the spread reverts. It doesn't tell you how
quickly — and speed is what turns a statistical curiosity into a trade. A
spread that reverts with a half-life of 5 years is "cointegrated" but useless:
you'd tie up capital (and pay borrow fees on the short) for years.

The Ornstein–Uhlenbeck process
------------------------------
The OU process is the continuous-time model of a mean-reverting quantity:

        dS_t = theta * (mu - S_t) dt + sigma dW_t

* mu     — the long-run level the spread is pulled toward
* theta  — the speed of the pull (per day here). Bigger = faster reversion.
* sigma  — the size of the random shocks

The term theta*(mu - S) is a spring: the further S is from mu, the harder it's
pulled back.

Fitting it with a simple regression
-----------------------------------
Sampled at fixed intervals dt, an OU process is exactly an AR(1):

        S_t = c + phi * S_{t-1} + eps_t,     with  phi = exp(-theta*dt)

So we regress today's spread on yesterday's and back out the parameters:

        theta = -ln(phi) / dt
        mu    = c / (1 - phi)
        sigma = std(eps) * sqrt(2*theta / (1 - phi^2))

Half-life
---------
The expected gap from mu decays like exp(-theta * t). The time for the gap to
halve is

        half_life = ln(2) / theta

This single number is the most practical output of the whole analysis. It
tells you (a) roughly how long a trade will last, (b) what lookback window to
use for the rolling z-score (a few half-lives), and (c) whether the pair is
worth trading at all. If phi >= 1 there's no mean reversion and the half-life
is infinite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class OUParams:
    theta: float      # mean-reversion speed, per trading day
    mu: float         # long-run mean of the spread
    sigma: float      # OU diffusion volatility (per sqrt(day))
    phi: float        # AR(1) coefficient
    half_life: float  # in trading days (np.inf if not mean-reverting)


def fit_ou(spread: pd.Series, dt: float = 1.0) -> OUParams:
    """Fit an OU process to ``spread`` via the exact AR(1) discretisation."""
    s = spread.dropna()
    s_lag = s.shift(1).iloc[1:]
    s_now = s.iloc[1:]

    model = sm.OLS(s_now.values, sm.add_constant(s_lag.values)).fit()
    c, phi = model.params
    resid_std = np.std(model.resid, ddof=2)

    if not (0 < phi < 1):
        # phi >= 1: random walk or explosive → no mean reversion.
        # phi <= 0: oscillates every step; not an OU process in any useful sense.
        return OUParams(theta=0.0, mu=float(s.mean()), sigma=float(resid_std),
                        phi=float(phi), half_life=np.inf)

    theta = -np.log(phi) / dt
    mu = c / (1 - phi)
    sigma = resid_std * np.sqrt(2 * theta / (1 - phi ** 2))
    half_life = np.log(2) / theta
    return OUParams(theta=float(theta), mu=float(mu), sigma=float(sigma),
                    phi=float(phi), half_life=float(half_life))
