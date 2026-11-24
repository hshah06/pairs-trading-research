"""
Step 2 — Is there a relationship worth trading? (Cointegration)
===============================================================

Correlation is NOT what we want
-------------------------------
Two stocks can be 90% correlated in daily *returns* and still drift apart
forever in *price*. Correlation says "they wiggle together today"; it says
nothing about whether a gap that opens up will ever close. A pairs trade bets
precisely that the gap closes, so we need a stronger property: cointegration.

What cointegration means
------------------------
Individually, stock prices behave like random walks: they are *non-stationary*
(technically "integrated of order 1", I(1)). A random walk has no anchor — its
variance grows with time and it has no tendency to come back anywhere.

Two I(1) series Y and X are *cointegrated* if some linear combination

        spread_t = Y_t - beta * X_t

is *stationary* (I(0)): it has a constant mean and variance and keeps getting
pulled back toward that mean. That pull is the entire source of profit.

We work with LOG prices, so beta is roughly an elasticity ("a 1% move in X goes
with a beta% move in Y") and the spread is in percentage terms. This keeps the
relationship stable when prices double over a decade.

The Engle–Granger two-step procedure
------------------------------------
1. Regress log(Y) on log(X) with OLS:  log Y = alpha + beta * log X + e.
   The fitted beta is the *hedge ratio*; the residuals e are the spread.
2. Run an Augmented Dickey–Fuller (ADF) test on the residuals. If we can reject
   "the residuals have a unit root" the spread is stationary → cointegrated.

Two traps to know about:

* **Critical values.** Because beta was *estimated* to make the residuals look
  as stationary as possible, ordinary ADF critical values are too lenient. The
  correct ones are MacKinnon's Engle–Granger critical values, which is what
  ``statsmodels.tsa.stattools.coint`` uses. We show both so you can see the
  difference — the "naive" ADF p-value on residuals is always too optimistic.

* **Direction matters.** Regressing Y on X and X on Y give different betas and
  can give different test results. We test both directions and keep the one
  with the stronger evidence, but report both.

Before any of this we also confirm, with a plain ADF test, that each individual
leg is non-stationary. If one leg were already stationary, "cointegration" would
be meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class CointResult:
    y: str                 # dependent leg (the one we "buy one unit of")
    x: str                 # hedge leg
    alpha: float           # regression intercept
    beta: float            # hedge ratio on log prices
    eg_pvalue: float       # Engle–Granger p-value (correct critical values)
    eg_stat: float         # Engle–Granger test statistic
    naive_adf_pvalue: float  # ADF on residuals with *standard* critical values (too optimistic)
    adf_y_pvalue: float    # ADF p-value on log(Y) alone; we want this LARGE (non-stationary)
    adf_x_pvalue: float    # ADF p-value on log(X) alone; we want this LARGE
    reverse_eg_pvalue: float  # Engle–Granger p-value with X and Y swapped
    spread: pd.Series      # in-sample residuals log(Y) - alpha - beta*log(X)


def hedge_regression(log_y: pd.Series, log_x: pd.Series) -> tuple[float, float, pd.Series]:
    """OLS of log_y on a constant and log_x. Returns (alpha, beta, residuals)."""
    X = sm.add_constant(log_x)
    model = sm.OLS(log_y, X).fit()
    alpha, beta = model.params.iloc[0], model.params.iloc[1]
    resid = log_y - alpha - beta * log_x
    return float(alpha), float(beta), resid


def engle_granger(prices: pd.DataFrame, y: str, x: str) -> CointResult:
    """Run the full Engle–Granger analysis for one ordering (Y on X)."""
    log_y = np.log(prices[y])
    log_x = np.log(prices[x])

    # --- Pre-check: each leg on its own should look like a random walk -------
    # ADF null hypothesis = "has a unit root" (non-stationary). A HIGH p-value
    # means we can't reject that, which is what we expect for a price series.
    adf_y = adfuller(log_y, autolag="AIC", result_object=False)[1]
    adf_x = adfuller(log_x, autolag="AIC", result_object=False)[1]

    # --- Step 1: estimate the hedge ratio ------------------------------------
    alpha, beta, spread = hedge_regression(log_y, log_x)

    # --- Step 2a: naive ADF on the residuals (for teaching only) -------------
    naive_p = adfuller(spread, autolag="AIC", result_object=False)[1]

    # --- Step 2b: proper Engle–Granger test with MacKinnon critical values ---
    eg_stat, eg_p, _ = coint(log_y, log_x, trend="c", autolag="AIC")

    # Reverse direction, so we can see if the conclusion is robust.
    _, rev_p, _ = coint(log_x, log_y, trend="c", autolag="AIC")

    return CointResult(
        y=y, x=x, alpha=alpha, beta=beta,
        eg_pvalue=float(eg_p), eg_stat=float(eg_stat),
        naive_adf_pvalue=float(naive_p),
        adf_y_pvalue=float(adf_y), adf_x_pvalue=float(adf_x),
        reverse_eg_pvalue=float(rev_p),
        spread=spread,
    )


def best_direction(prices: pd.DataFrame, a: str, b: str) -> CointResult:
    """Test A~B and B~A and return whichever has the smaller EG p-value."""
    r1 = engle_granger(prices, a, b)
    r2 = engle_granger(prices, b, a)
    return r1 if r1.eg_pvalue <= r2.eg_pvalue else r2
