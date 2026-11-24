"""
Step 1 — Getting the data
=========================

Everything in pairs trading starts with a clean, aligned price history for both
legs of each pair. Three things matter more than they first appear:

1. **Adjusted prices.** Dividends and splits create artificial jumps in raw
   prices. KO and XOM pay big dividends; on every ex-dividend date the raw price
   drops by the dividend amount. If we used raw prices, the spread would show
   fake "mean-reverting" jumps that no trader could actually capture. Using
   ``auto_adjust=True`` gives a total-return series instead.

2. **Alignment.** The two legs must share the same dates. We drop any date where
   either leg is missing, rather than forward-filling, because a forward-filled
   price is a price you could not actually have traded at.

3. **Caching.** Re-downloading on every run is slow and makes results drift as
   Yahoo revises history. We cache to CSV so reruns are reproducible; delete the
   file to refresh.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of daily adjusted close prices, one column per ticker.

    If ``cache_path`` exists and already contains every requested ticker, it is
    read instead of hitting the network.
    """
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Reuse the cache only if it has every ticker AND reaches back far
        # enough (within a week of ``start``, allowing for holidays).
        if (set(tickers).issubset(cached.columns)
                and cached.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7)):
            print(f"[data] Loaded cached prices from {cache_path.parent.name}/{cache_path.name}")
            return cached[tickers]

    print(f"[data] Downloading {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # dividend- and split-adjusted (see module docstring)
        progress=False,
    )
    # yfinance returns a column MultiIndex like ('Close', 'KO'). We only need
    # closing prices: our strategy decides at the close and trades at the close.
    prices = raw["Close"][tickers].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        print(f"[data] Cached prices to {cache_path.parent.name}/{cache_path.name}")
    return prices


def get_pair(prices: pd.DataFrame, y: str, x: str) -> pd.DataFrame:
    """Return the two legs of a pair with any date where either is missing removed.

    Some ETFs (e.g. GDX) started later than others; ``dropna`` trims the history
    to the period where *both* actually traded.
    """
    return prices[[y, x]].dropna()
