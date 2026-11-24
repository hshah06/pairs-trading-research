"""
Step 4 — Turning the spread into a trading strategy (z-score backtest)
======================================================================

The z-score
-----------
The spread is in "log-price units", which isn't intuitive. We standardise it:

        z_t = (spread_t - rolling_mean_t) / rolling_std_t

z = +2 means "the spread is two standard deviations ABOVE its recent average":
Y is expensive relative to X. We bet on reversion by SHORTING the spread
(short Y, long beta*X). z = -2 is the mirror image: LONG the spread.

Why ROLLING mean and std, instead of the in-sample values? Because on any given
day a real trader only knows the past. Using the full-sample mean would leak
future information into the signal (look-ahead bias) — one of the most common
ways backtests lie. The window length is set from the half-life (a couple of
half-lives) so the "recent average" is measured on the timescale the spread
actually reverts over.

Trading rules (a small state machine)
-------------------------------------
* Flat  → enter LONG  spread when z <= -entry_z
* Flat  → enter SHORT spread when z >= +entry_z
* Long  → exit when z rises back to -exit_z (default 0: "back to the mean")
* Short → exit when z falls back to +exit_z
* Stop-loss: if |z| blows out past stop_z, the relationship may have broken
  (e.g. a takeover, an earnings shock). Get out, and don't re-enter until z has
  come back inside the entry band — otherwise we'd re-enter immediately.
* Time stop (optional, ``max_hold_days``): close any trade that has been open
  longer than this many trading days, whatever z is doing. The idea: the OU
  half-life says how long a mispricing *should* take to fade. A trade still
  open after ~3 half-lives is evidence that the spread isn't behaving the way
  our model says, and that evidence arrives far sooner than a 2-year
  cointegration test could provide it. Like the stop-loss, we then don't
  re-enter until z has come back inside the entry band, otherwise we'd
  simply re-open the same stale trade the next day.

Position sizing
---------------
Holding "one unit of spread" log(Y) - beta*log(X) means $1 long Y for every
$beta short X. We scale so that GROSS exposure is $1:

        w_Y = pos * 1 / (1 + |beta|)
        w_X = -pos * beta / (1 + |beta|)

Returns are therefore "per dollar of gross capital deployed", which is a
conservative (unlevered) convention.

Timing — avoiding look-ahead
----------------------------
The signal uses the close on day t. We assume we trade AT that close and then
earn day t+1's return. In code: ``pnl_t = weights_{t-1} * returns_t``. If you
instead wrote ``weights_t * returns_t``, you'd be using today's close both to
decide AND to profit — a classic bug that makes almost any strategy look great.

Realistic costs
---------------
Pairs strategies trade often and capture small moves, so costs decide whether
they work at all. We charge:

* ``cost_bps`` per dollar traded, per leg, per side. This bundles commission,
  half the bid–ask spread, and market impact. 5 bps is reasonable for liquid
  US large caps and ETFs for a small account; GDX/EWA-type ETFs can be wider.
  A round trip on both legs therefore costs ~4 * cost_bps * (gross weight).
* ``borrow_bps_annual`` on the short leg, accrued daily. Shorting isn't free;
  easy-to-borrow names cost ~25–50 bps/year, hard-to-borrow ones far more.

We ignore the fact that fixed dollar weights would need small daily
rebalancing as prices drift — a real implementation would hold fixed SHARE
counts instead, and the difference is minor over typical holding periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class StrategyConfig:
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0
    lookback: int = 60          # rolling window for the z-score, in days
    cost_bps: float = 5.0       # per leg, per side, per $ traded
    borrow_bps_annual: float = 50.0
    max_hold_days: int | None = None  # time stop in trading days (None = off)


@dataclass
class BacktestResult:
    frame: pd.DataFrame                      # daily: z, position, returns, equity...
    trades: pd.DataFrame                     # one row per round trip
    metrics: dict = field(default_factory=dict)


def rolling_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Z-score using ONLY data up to and including each day (no look-ahead)."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std


def generate_positions(z: pd.Series, cfg: StrategyConfig,
                       allowed: pd.Series | None = None) -> tuple[pd.Series, list[dict]]:
    """Walk through z day by day and apply the trading rules.

    A plain loop is clearer than a clever vectorised version, and with ~4,000
    days it runs instantly. Returns the position series (+1 long spread,
    -1 short spread, 0 flat) and a list of trade events for plotting.

    ``allowed`` is an optional True/False series from the cointegration
    monitor (pairs/monitor.py). On days it is False we open nothing, and any
    open trade is closed that day (event kind "monitor_off").
    """
    ok = (allowed.reindex(z.index).fillna(False).values.astype(bool)
          if allowed is not None else np.ones(len(z), dtype=bool))
    pos = 0
    entry_i = 0            # index of the current trade's entry day (for the time stop)
    blocked = False        # True after a stop-loss / time stop, until z re-enters the entry band
    positions = np.zeros(len(z))
    events: list[dict] = []
    dates = z.index
    zv = z.values

    for i, zt in enumerate(zv):
        # The monitor overrides everything: relationship in doubt → be flat.
        if not ok[i]:
            if pos != 0:
                events.append(dict(date=dates[i], kind="monitor_off", side=pos, z=zt))
                pos = 0
            positions[i] = 0
            continue

        if np.isnan(zt):
            positions[i] = pos
            continue

        if blocked and abs(zt) < cfg.entry_z:
            blocked = False

        # Time stop: the trade has outlived what the half-life says is normal.
        if pos != 0 and cfg.max_hold_days and i - entry_i >= cfg.max_hold_days:
            events.append(dict(date=dates[i], kind="time_stop", side=pos, z=zt))
            pos, blocked = 0, True
            positions[i] = 0
            continue

        # Only enter inside the band entry_z <= |z| < stop_z. Opening a trade
        # with |z| already past the stop would mean entering a position our
        # own rules say is broken.
        if pos == 0 and not blocked and abs(zt) < cfg.stop_z:
            if zt <= -cfg.entry_z:
                pos, entry_i = 1, i
                events.append(dict(date=dates[i], kind="entry", side=1, z=zt))
            elif zt >= cfg.entry_z:
                pos, entry_i = -1, i
                events.append(dict(date=dates[i], kind="entry", side=-1, z=zt))

        elif pos == 1:
            if zt <= -cfg.stop_z:
                pos, blocked = 0, True
                events.append(dict(date=dates[i], kind="stop", side=1, z=zt))
            elif zt >= -cfg.exit_z:
                pos = 0
                events.append(dict(date=dates[i], kind="exit", side=1, z=zt))

        elif pos == -1:
            if zt >= cfg.stop_z:
                pos, blocked = 0, True
                events.append(dict(date=dates[i], kind="stop", side=-1, z=zt))
            elif zt <= cfg.exit_z:
                pos = 0
                events.append(dict(date=dates[i], kind="exit", side=-1, z=zt))

        positions[i] = pos

    # Close any open trade on the last day so each period is evaluated on
    # completed trades only and the next period starts flat.
    if pos != 0:
        positions[-1] = 0
        events.append(dict(date=dates[-1], kind="period_end", side=pos, z=zv[-1]))

    return pd.Series(positions, index=z.index, name="position"), events


def compute_metrics(net: pd.Series, trades: pd.DataFrame, position: pd.Series) -> dict:
    """Standard performance statistics on a daily net-return series."""
    equity = (1 + net).cumprod()
    years = len(net) / TRADING_DAYS
    vol = net.std() * np.sqrt(TRADING_DAYS)
    # Sharpe ratio: annualised mean / annualised std of daily returns. We use a
    # zero risk-free rate for simplicity; a market-neutral strategy is usually
    # compared to cash, so subtracting T-bill yields would lower these numbers
    # (noticeably in 2023–2025 when cash paid ~5%).
    sharpe = net.mean() / net.std() * np.sqrt(TRADING_DAYS) if net.std() > 0 else 0.0
    # Max drawdown: the worst peak-to-trough fall of the equity curve. This is
    # the number that tells you how much pain you'd have had to sit through.
    drawdown = equity / equity.cummax() - 1
    n_trades = len(trades)
    return {
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "n_trades": n_trades,
        "win_rate": (trades["net_return"] > 0).mean() if n_trades else np.nan,
        "avg_hold_days": trades["hold_days"].mean() if n_trades else np.nan,
        "pct_time_in_market": (position != 0).mean(),
    }


def run_backtest(
    prices: pd.DataFrame,
    y: str,
    x: str,
    beta: float | pd.Series,
    cfg: StrategyConfig,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    spread: pd.Series | None = None,
    allowed: pd.Series | None = None,
) -> BacktestResult:
    """Backtest the z-score strategy on [start, end].

    ``prices`` should contain history BEFORE ``start`` too: it's used only to
    warm up the rolling z-score window, so the first day of the test period
    already has a valid signal. No trades happen before ``start``.

    Two ways to call it:

    * **Static hedge (default):** ``beta`` is a single number fitted in-sample
      and the spread is log(Y) - beta*log(X).
    * **Dynamic hedge (Kalman):** ``beta`` is a Series (the hedge ratio known at
      each day's close) and ``spread`` is supplied (the Kalman innovation).
      Weights then change slightly EVERY day as beta moves, even with no new
      signal. We rebalance daily to the current beta and pay costs on that
      turnover too. "Adaptive" is not free.
    """
    if spread is None:
        # The spread uses the hedge ratio estimated IN-SAMPLE. In the out-of-sample
        # period we keep it fixed — re-estimating it with OOS data would be cheating.
        spread = np.log(prices[y]) - beta * np.log(prices[x])
    spread_full = spread
    z_full = rolling_zscore(spread_full, cfg.lookback)

    window = slice(pd.Timestamp(start), pd.Timestamp(end))
    z = z_full.loc[window]
    spread = spread_full.loc[window]
    rets = prices[[y, x]].pct_change().loc[window]

    position, events = generate_positions(z, cfg, allowed)

    # Dollar weights per unit of gross capital. With a Series beta these are
    # aligned by date, so each day uses the hedge ratio known at that close.
    if isinstance(beta, pd.Series):
        beta = beta.loc[window]
    scale = 1.0 + abs(beta)
    w_y = position / scale
    w_x = -position * beta / scale

    # Yesterday's weights earn today's return (see "Timing" in the docstring).
    gross = w_y.shift(1).fillna(0) * rets[y] + w_x.shift(1).fillna(0) * rets[x]
    gross = gross.fillna(0)

    # Transaction costs: proportional to how much of each leg we traded today.
    turnover = w_y.diff().abs().fillna(w_y.abs()) + w_x.diff().abs().fillna(w_x.abs())
    tcost = turnover * cfg.cost_bps / 1e4

    # Borrow fee on whichever leg is short, on yesterday's holdings.
    short_exposure = (-w_y.shift(1).clip(upper=0) - w_x.shift(1).clip(upper=0)).fillna(0)
    borrow = short_exposure * cfg.borrow_bps_annual / 1e4 / TRADING_DAYS

    net = gross - tcost - borrow

    frame = pd.DataFrame({
        "spread": spread, "z": z, "position": position,
        "gross_return": gross, "cost": tcost + borrow, "net_return": net,
    })
    frame["equity"] = (1 + net).cumprod()

    trades = _build_trade_log(frame, events)
    metrics = compute_metrics(net, trades, position)
    metrics["gross_sharpe"] = (gross.mean() / gross.std() * np.sqrt(TRADING_DAYS)
                               if gross.std() > 0 else 0.0)
    metrics["total_costs"] = frame["cost"].sum()
    metrics["turnover"] = turnover.sum() / (len(turnover) / TRADING_DAYS)  # per year
    metrics["pct_time_allowed"] = (float(allowed.reindex(z.index).fillna(False).mean())
                                   if allowed is not None else 1.0)
    metrics["monitor_closes"] = sum(e["kind"] == "monitor_off" for e in events)
    metrics["time_stops"] = sum(e["kind"] == "time_stop" for e in events)
    return BacktestResult(frame=frame, trades=trades, metrics=metrics)


def _build_trade_log(frame: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Pair each entry with its exit and compute the trade's net return."""
    rows = []
    open_evt = None
    for e in events:
        if e["kind"] == "entry":
            open_evt = e
        elif open_evt is not None:
            # P&L of a trade = compounded net returns from the entry day (which
            # carries only the entry cost, since we held nothing the night
            # before) through the exit day (last overnight return + exit cost).
            seg = frame.loc[open_evt["date"]:e["date"], "net_return"]
            trade_ret = (1 + seg).prod() - 1
            rows.append(dict(
                entry_date=open_evt["date"], exit_date=e["date"],
                side="long spread" if open_evt["side"] == 1 else "short spread",
                entry_z=open_evt["z"], exit_z=e["z"], exit_reason=e["kind"],
                hold_days=len(seg) - 1, net_return=trade_ret,
            ))
            open_evt = None
    return pd.DataFrame(rows, columns=["entry_date", "exit_date", "side", "entry_z",
                                       "exit_z", "exit_reason", "hold_days", "net_return"])


def events_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Flatten the trade log into entry/exit markers for plotting."""
    rows = []
    for t in trades.itertuples():
        rows.append(dict(date=t.entry_date, kind="entry", side=t.side, z=t.entry_z))
        rows.append(dict(date=t.exit_date, kind=t.exit_reason, side=t.side, z=t.exit_z))
    return pd.DataFrame(rows)
