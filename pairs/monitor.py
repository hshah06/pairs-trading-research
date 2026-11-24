"""
Step 3c — Is the relationship STILL there? (Rolling cointegration monitor)
==========================================================================

The lesson from Steps 5 and 6
-----------------------------
KO/PEP was cointegrated from 2012 to 2020 and then stopped being cointegrated.
A fixed hedge ratio kept betting on reversion; a Kalman hedge ratio tried to
"adapt" and got it wrong. Neither one can answer the question that actually
matters: **is this pair still a pair?**

The monitor
-----------
Every week we re-run the Engle–Granger test on the TRAILING two years of data
(504 trading days), using only prices up to that day. That gives a p-value
series, p_t. Read it like a smoke detector: a low p_t means the last two years
still look cointegrated; a high p_t means they don't.

We turn p_t into an on/off switch with HYSTERESIS, i.e. two thresholds:

* trading is switched OFF when p_t rises above 0.10
* trading is switched back ON only when p_t falls below 0.05

Why two thresholds instead of one? A p-value computed on overlapping windows
wanders around noisily. With a single cutoff at 0.05, a p-value hovering
around 0.05 would flip the switch on and off every week, and every flip costs
money. The gap between 0.05 and 0.10 is a "dead band". Once the switch is on,
it takes clear evidence against cointegration to turn it off, and once it's
off, it takes clear evidence for cointegration to turn it back on. A
thermostat works the same way.

What the switch does
--------------------
* While OFF: no new trades.
* At the moment it turns OFF: any open trade is closed. The relationship we
  were betting on is now in doubt, and "hold and hope" is exactly what lost
  money on KO/PEP in 2025–2026.
* The switch starts OFF and needs a first reading below 0.05 to turn on, so
  trading requires positive evidence.

Things to be honest about
-------------------------
* **Low power.** Two years is a short window for a cointegration test. A pair
  whose spread has a 50-day half-life only completes a handful of full cycles
  in 504 days, so even a genuinely cointegrated pair will often show p > 0.10.
  Expect the monitor to switch trading off a lot, including during good periods.
* **It reacts late.** It needs months of misbehaviour in the window before p
  rises. It can limit the damage from a broken pair, but it can't avoid it.
* **No look-ahead.** The p-value for day t uses prices up to day t, and the
  test re-estimates its own hedge ratio inside each window. Re-testing weekly
  instead of daily is a practical choice; between tests the last value holds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import coint


def rolling_eg_pvalue(log_y: pd.Series, log_x: pd.Series, window: int = 504,
                      step: int = 5) -> pd.Series:
    """Engle–Granger p-value on the trailing ``window`` days, re-tested every ``step`` days.

    Returns a Series on the same index as the inputs. Days before the first
    full window are NaN. Between re-tests the latest p-value is carried forward,
    which is what a trader re-checking weekly would see.
    """
    n = len(log_y)
    p = np.full(n, np.nan)
    ly, lx = log_y.values, log_x.values
    for end in range(window, n + 1, step):
        # Window [end-window, end) ends at day end-1: data up to that close only.
        with warnings.catch_warnings():
            # IVV/SPY track the same index, so their log prices are almost
            # perfectly collinear and statsmodels warns. That's expected here.
            warnings.simplefilter("ignore", CollinearityWarning)
            _, pval, _ = coint(ly[end - window:end], lx[end - window:end],
                               trend="c", autolag="AIC")
        p[end - 1] = pval
    return pd.Series(p, index=log_y.index, name="rolling_eg_p").ffill()


def hysteresis_gate(pvalues: pd.Series, off_above: float = 0.10,
                    on_below: float = 0.05) -> pd.Series:
    """Turn a p-value series into a True/False 'trading allowed' series.

    Starts OFF. Turns ON when p < on_below; turns OFF when p > off_above;
    otherwise keeps its previous state (that's the hysteresis).
    """
    allowed = np.zeros(len(pvalues), dtype=bool)
    on = False
    for i, p in enumerate(pvalues.values):
        if np.isnan(p):
            on = False
        elif on and p > off_above:
            on = False
        elif not on and p < on_below:
            on = True
        allowed[i] = on
    return pd.Series(allowed, index=pvalues.index, name="trading_allowed")


def count_switches(allowed: pd.Series) -> int:
    """Number of ON→OFF transitions (how often the monitor pulled the plug)."""
    a = allowed.astype(int)
    return int(((a.shift(1) == 1) & (a == 0)).sum())
