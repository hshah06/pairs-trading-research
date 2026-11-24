"""
Pairs trading research pipeline
===============================

Run:   .venv/bin/python run_research.py

What this script does, in order:

  1. Downloads daily adjusted prices for every candidate pair.
  2. Splits history into IN-SAMPLE (IS) and OUT-OF-SAMPLE (OOS) periods.
  3. On IS data ONLY: tests each pair for cointegration (Engle–Granger + ADF),
     estimates the hedge ratio, and fits an Ornstein–Uhlenbeck model to get
     the half-life.
  4. Backtests the same z-score strategy on every pair, IS and OOS, with costs.
  5. Repeats step 4 with a DYNAMIC hedge ratio from a Kalman filter (see
     pairs/kalman.py) and compares it with the fixed OLS hedge ratio.
  5b. Repeats both with a rolling 2-year cointegration monitor that switches
     trading off when the relationship looks broken (see pairs/monitor.py).
  5c. Repeats static and Kalman with a TIME STOP at 3× the in-sample half-life.
  6. Picks the "best" pair using IS information only, then plots it and runs
     transaction-cost and Kalman-memory sensitivity checks.
  7. Writes everything to ./output/.

The single most important discipline in this file: every DECISION (which pair,
which hedge ratio, which lookback) is made using in-sample data. The OOS period
is touched only to measure what would have happened. If you find yourself
tweaking parameters after looking at OOS results, the OOS period has quietly
become in-sample and its numbers no longer mean anything.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pairs.backtest import StrategyConfig, events_from_trades, run_backtest
from pairs.cointegration import best_direction, engle_granger
from pairs.data import download_prices, get_pair
from pairs.kalman import kalman_from_insample
from pairs.monitor import count_switches, hysteresis_gate, rolling_eg_pvalue
from pairs.ou import fit_ou
from pairs.plotting import (plot_beta, plot_equity, plot_equity_compare, plot_monitor,
                            plot_monitor_grid, plot_spread, plot_zscore)
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).parent
OUT = ROOT / "output"
DATA_CACHE = ROOT / "data" / "prices.csv"

# --------------------------------------------------------------------------
# Candidate pairs. Each has an economic story — that matters! Searching
# thousands of random pairs for cointegration WILL find "significant" ones by
# pure chance (at a 5% level, 1 in 20 unrelated pairs passes). An economic
# reason for the link is your first defence against data-mining.
# --------------------------------------------------------------------------
PAIRS = [
    ("KO", "PEP", "Cola duopoly: same products, same consumers"),
    ("XOM", "CVX", "US oil majors: both driven by crude prices"),
    ("GLD", "GDX", "Gold vs gold miners: miners are leveraged gold"),
    ("EWA", "EWC", "Australia vs Canada ETFs: commodity-exporter economies"),
    ("GLD", "SLV", "Gold vs silver: both precious metals"),
    ("XLE", "XOP", "Energy sector vs oil & gas E&P"),
    ("TLT", "IEF", "Long vs intermediate Treasuries: same issuer, different duration"),
    ("SPY", "IVV", "Two S&P 500 trackers: a sanity check (should be ~perfectly linked)"),
]

START = "2012-01-01"
# Extra history used ONLY to warm up the 2-year cointegration monitor, so it
# has a reading on the first trading day. Nothing else sees these two years.
DATA_START = "2010-01-01"
SPLIT = pd.Timestamp("2021-01-01")   # IS = 2012–2020 (9y), OOS = 2021–today
END = None                           # None = up to the latest available day

BASE_CFG = StrategyConfig(entry_z=2.0, exit_z=0.0, stop_z=4.0,
                          cost_bps=5.0, borrow_bps_annual=50.0)

# Kalman filter memory, in trading days: roughly how far back the filter
# "remembers" when estimating today's hedge ratio. Chosen BEFORE seeing any
# results: one year, several times the spreads' 30–60 day half-lives (see
# pairs/kalman.py for why it must be much slower than the half-life).
KALMAN_MEMORY = 250
KALMAN_MEMORY_GRID = (60, 125, 250, 500, 1000)   # robustness check only
KALMAN_BURN_IN = 60   # skip the filter's first days (diffuse prior) in diagnostics
METHODS = ("static OLS", "Kalman", "static OLS + monitor", "Kalman + monitor",
           "static OLS + time stop", "Kalman + time stop")

# Time stop: close any trade held longer than TIME_STOP_MULT × the pair's
# in-sample OU half-life. 3× was specified up front; others are robustness only.
TIME_STOP_MULT = 3.0
TIME_STOP_GRID = (2.0, 3.0, 4.0, 5.0)


def max_hold_from_half_life(hl: float, mult: float) -> int | None:
    """Time-stop length in trading days, or None if the half-life is infinite."""
    return int(np.ceil(mult * hl)) if np.isfinite(hl) else None

# Rolling cointegration monitor (pairs/monitor.py).
MONITOR_WINDOW = 504          # trailing 2 years of trading days
MONITOR_STEP = 5              # re-test weekly
MONITOR_OFF_ABOVE = 0.10      # stop trading when p rises above this
MONITOR_ON_BELOW = 0.05       # resume only when p falls back below this
MONITOR_WINDOW_GRID = (252, 504, 756)   # robustness check only

# Selection filters for the "best" pair (applied to IS stats only).
MAX_EG_PVALUE = 0.05
HALF_LIFE_RANGE = (2.0, 60.0)   # faster than 2d is mostly noise/bid-ask bounce;
                                # slower than ~3 months ties up capital too long


def lookback_from_half_life(hl: float) -> int:
    """Rolling z-score window ≈ 2 half-lives, clipped to [20, 120] trading days.

    Too short and the rolling mean chases the spread (z can never get large);
    too long and it reacts slowly to genuine level shifts. A couple of
    half-lives is a common, defensible rule of thumb.
    """
    if not np.isfinite(hl):
        return 120
    return int(np.clip(round(2 * hl), 20, 120))


def fmt_pct(x):
    return f"{x:.1%}" if pd.notna(x) else "—"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tickers = sorted({t for a, b, _ in PAIRS for t in (a, b)})
    prices_full = download_prices(tickers, start=DATA_START, end=END, cache_path=DATA_CACHE)
    prices = prices_full.loc[START:]   # everything except the monitor uses 2012+
    print(f"[data] {prices.index[0].date()} → {prices.index[-1].date()}, "
          f"{len(prices)} trading days\n")

    stat_rows, bt_rows, details = [], [], {}

    for a, b, story in PAIRS:
        pair = get_pair(prices, a, b)
        is_px = pair.loc[: SPLIT - pd.Timedelta(days=1)]
        oos_px = pair.loc[SPLIT:]

        # ---- Cointegration on IN-SAMPLE data ------------------------------
        res = best_direction(is_px, a, b)
        ou = fit_ou(res.spread)
        lookback = lookback_from_half_life(ou.half_life)

        # ---- Diagnostic only: does the relationship still hold OOS? -------
        # This is HINDSIGHT. We never use it to choose anything; it helps
        # explain afterwards why a pair did or didn't work.
        oos_res = engle_granger(oos_px, res.y, res.x)
        oos_ou = fit_ou(oos_res.spread)

        name = f"{res.y}/{res.x}"
        stat_rows.append({
            "pair": name, "story": story,
            "beta (IS)": res.beta,
            "ADF p Y": res.adf_y_pvalue, "ADF p X": res.adf_x_pvalue,
            "naive ADF p": res.naive_adf_pvalue,
            "EG p (IS)": res.eg_pvalue, "EG p reverse": res.reverse_eg_pvalue,
            "half-life IS (d)": ou.half_life,
            "lookback": lookback,
            "EG p (OOS, hindsight)": oos_res.eg_pvalue,
            "half-life OOS (d, hindsight)": oos_ou.half_life,
            "OOS beta (hindsight)": oos_res.beta,
        })

        # ---- Backtest IS and OOS with the IS-chosen parameters ------------
        cfg = replace(BASE_CFG, lookback=lookback)
        bt_is = run_backtest(pair, res.y, res.x, res.beta, cfg,
                             is_px.index[0], is_px.index[-1])
        bt_oos = run_backtest(pair, res.y, res.x, res.beta, cfg,
                              oos_px.index[0], oos_px.index[-1])
        # ---- Same strategy, dynamic hedge ratio (Kalman) ------------------
        # Everything else is IDENTICAL: same pair direction, same lookback,
        # same entry/exit/stop, same costs. Only the hedge ratio (and hence the
        # spread) changes, so any difference in results comes from it alone.
        kal = kalman_from_insample(pair, res.y, res.x, SPLIT, res.spread,
                                   memory=KALMAN_MEMORY)
        kbt_is = run_backtest(pair, res.y, res.x, kal.beta, cfg,
                              is_px.index[0], is_px.index[-1], spread=kal.spread)
        kbt_oos = run_backtest(pair, res.y, res.x, kal.beta, cfg,
                               oos_px.index[0], oos_px.index[-1], spread=kal.spread)

        for method, label, bt in (("static OLS", "IS", bt_is), ("static OLS", "OOS", bt_oos),
                                  ("Kalman", "IS", kbt_is), ("Kalman", "OOS", kbt_oos)):
            bt_rows.append({"pair": name, "method": method, "period": label, **bt.metrics})

        # Kalman spread diagnostics. The ADF p-value here is only a rough
        # indicator: the spread wasn't produced by one OLS fit, so neither the
        # standard nor the Engle–Granger critical values strictly apply.
        k_is = kal.spread.loc[: SPLIT - pd.Timedelta(days=1)].iloc[KALMAN_BURN_IN:]
        k_oos = kal.spread.loc[SPLIT:]
        stat_rows[-1].update({
            "Kalman ADF p (IS)": adfuller(k_is, autolag="AIC", result_object=False)[1],
            "Kalman ADF p (OOS)": adfuller(k_oos, autolag="AIC", result_object=False)[1],
            "Kalman half-life IS (d)": fit_ou(k_is).half_life,
            "Kalman half-life OOS (d)": fit_ou(k_oos).half_life,
            "Kalman beta at split": kal.beta.loc[: SPLIT].iloc[-1],
            "Kalman beta at end": kal.beta.iloc[-1],
        })

        # ---- Rolling cointegration monitor -----------------------------------
        # Computed on prices only (it re-fits its own hedge inside each 2-year
        # window), so ONE monitor serves both hedge methods. Same pair
        # direction as the in-sample test. Warm-up history comes from 2010.
        pair_full = get_pair(prices_full, a, b)
        pvals = rolling_eg_pvalue(np.log(pair_full[res.y]), np.log(pair_full[res.x]),
                                  window=MONITOR_WINDOW, step=MONITOR_STEP)
        allowed = hysteresis_gate(pvals, MONITOR_OFF_ABOVE, MONITOR_ON_BELOW)

        mbt_is = run_backtest(pair, res.y, res.x, res.beta, cfg,
                              is_px.index[0], is_px.index[-1], allowed=allowed)
        mbt_oos = run_backtest(pair, res.y, res.x, res.beta, cfg,
                               oos_px.index[0], oos_px.index[-1], allowed=allowed)
        mkbt_is = run_backtest(pair, res.y, res.x, kal.beta, cfg, is_px.index[0],
                               is_px.index[-1], spread=kal.spread, allowed=allowed)
        mkbt_oos = run_backtest(pair, res.y, res.x, kal.beta, cfg, oos_px.index[0],
                                oos_px.index[-1], spread=kal.spread, allowed=allowed)
        for method, label, bt in (("static OLS + monitor", "IS", mbt_is),
                                  ("static OLS + monitor", "OOS", mbt_oos),
                                  ("Kalman + monitor", "IS", mkbt_is),
                                  ("Kalman + monitor", "OOS", mkbt_oos)):
            bt_rows.append({"pair": name, "method": method, "period": label, **bt.metrics})

        is_mask = allowed.index < SPLIT
        oos_mask = allowed.index >= SPLIT
        stat_rows[-1].update({
            "monitor % on (IS)": allowed[is_mask & (allowed.index >= pd.Timestamp(START))].mean(),
            "monitor % on (OOS)": allowed[oos_mask].mean(),
            "monitor switch-offs (IS)": count_switches(allowed[is_mask & (allowed.index >= pd.Timestamp(START))]),
            "monitor switch-offs (OOS)": count_switches(allowed[oos_mask]),
            "first OOS switch-off": (allowed[oos_mask][~allowed[oos_mask]].index[0].date()
                                     if (~allowed[oos_mask]).any() else None),
        })

        # ---- Time stop: same as static / Kalman, plus a maximum holding time --
        cfg_ts = replace(cfg, max_hold_days=max_hold_from_half_life(ou.half_life,
                                                                     TIME_STOP_MULT))
        tbt_is = run_backtest(pair, res.y, res.x, res.beta, cfg_ts,
                              is_px.index[0], is_px.index[-1])
        tbt_oos = run_backtest(pair, res.y, res.x, res.beta, cfg_ts,
                               oos_px.index[0], oos_px.index[-1])
        tkbt_is = run_backtest(pair, res.y, res.x, kal.beta, cfg_ts, is_px.index[0],
                               is_px.index[-1], spread=kal.spread)
        tkbt_oos = run_backtest(pair, res.y, res.x, kal.beta, cfg_ts, oos_px.index[0],
                                oos_px.index[-1], spread=kal.spread)
        for method, label, bt in (("static OLS + time stop", "IS", tbt_is),
                                  ("static OLS + time stop", "OOS", tbt_oos),
                                  ("Kalman + time stop", "IS", tkbt_is),
                                  ("Kalman + time stop", "OOS", tkbt_oos)):
            bt_rows.append({"pair": name, "method": method, "period": label, **bt.metrics})
        stat_rows[-1]["time stop (days)"] = cfg_ts.max_hold_days

        details[name] = dict(res=res, ou=ou, cfg=cfg, cfg_ts=cfg_ts,
                             tbt_is=tbt_is, tbt_oos=tbt_oos,
                             tkbt_is=tkbt_is, tkbt_oos=tkbt_oos,
                             pair=pair, pair_full=pair_full,
                             kal=kal, pvals=pvals, allowed=allowed,
                             bt_is=bt_is, bt_oos=bt_oos, kbt_is=kbt_is, kbt_oos=kbt_oos,
                             mbt_is=mbt_is, mbt_oos=mbt_oos,
                             mkbt_is=mkbt_is, mkbt_oos=mkbt_oos)

    stats = pd.DataFrame(stat_rows).set_index("pair")
    bts_all = pd.DataFrame(bt_rows).set_index(["pair", "method", "period"])
    bts = bts_all.xs("static OLS", level="method")    # the original strategy
    bts_k = bts_all.xs("Kalman", level="method")
    stats.to_csv(OUT / "cointegration_ou_stats.csv")
    bts_all.to_csv(OUT / "backtest_all_pairs.csv")

    # ---- Print the cointegration / OU table -------------------------------
    pd.set_option("display.width", 200)
    print("=" * 100)
    print("COINTEGRATION & ORNSTEIN–UHLENBECK (in-sample 2012–2020)")
    print("=" * 100)
    show = stats[["beta (IS)", "ADF p Y", "ADF p X", "naive ADF p", "EG p (IS)",
                  "EG p reverse", "half-life IS (d)", "EG p (OOS, hindsight)",
                  "half-life OOS (d, hindsight)"]]
    print(show.round(3).to_string())
    print("\nRead it like this: ADF p on each leg should be LARGE (prices are random walks);"
          "\nEG p should be SMALL (< 0.05 → the spread is stationary). Note naive ADF p is"
          "\nalways smaller than EG p — that's why the naive test is misleading.\n")

    # ---- Print the backtest table -----------------------------------------
    print("=" * 100)
    print(f"BACKTEST, ALL PAIRS (entry ±{BASE_CFG.entry_z}, exit {BASE_CFG.exit_z}, "
          f"stop ±{BASE_CFG.stop_z}, {BASE_CFG.cost_bps} bps/leg/side, "
          f"{BASE_CFG.borrow_bps_annual} bps/yr borrow)")
    print("=" * 100)
    tbl = bts[["sharpe", "gross_sharpe", "cagr", "max_drawdown", "n_trades",
               "win_rate", "avg_hold_days", "pct_time_in_market"]].copy()
    print(tbl.round(3).to_string())
    print()

    # ---- Kalman: spread diagnostics and backtest -----------------------------
    print("=" * 100)
    print(f"KALMAN DYNAMIC HEDGE RATIO (memory ≈ {KALMAN_MEMORY} days) — spread diagnostics")
    print("=" * 100)
    print(stats[["beta (IS)", "Kalman beta at split", "Kalman beta at end",
                 "EG p (IS)", "Kalman ADF p (IS)", "Kalman ADF p (OOS)",
                 "half-life IS (d)", "Kalman half-life IS (d)",
                 "Kalman half-life OOS (d)"]].round(3).to_string())
    print()
    print("=" * 100)
    print("BACKTEST, ALL PAIRS — KALMAN HEDGE (same rules, same costs)")
    print("=" * 100)
    print(bts_k[["sharpe", "gross_sharpe", "cagr", "max_drawdown", "n_trades",
                 "win_rate", "avg_hold_days", "turnover"]].round(3).to_string())
    print()

    # ---- Head-to-head comparison --------------------------------------------
    two = bts_all[bts_all.index.get_level_values("method").isin(["static OLS", "Kalman"])]
    cmp = two[["sharpe", "max_drawdown", "n_trades"]].unstack(["method", "period"])
    cmp = cmp.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
    cmp.to_csv(OUT / "hedge_comparison.csv")
    print("=" * 100)
    print("STATIC OLS vs KALMAN — net Sharpe / max drawdown / trades")
    print("=" * 100)
    print(cmp.round(3).to_string())
    delta = (bts_k["sharpe"] - bts["sharpe"]).unstack("period")
    print("\nKalman minus static, net Sharpe:")
    print(delta.round(3).to_string())
    print(f"Pairs where Kalman improved OOS Sharpe: {int((delta['OOS'] > 0).sum())} of {len(delta)}")
    print()

    # ---- Rolling cointegration monitor: diagnostics and comparison -----------
    print("=" * 100)
    print(f"ROLLING COINTEGRATION MONITOR ({MONITOR_WINDOW}-day window, re-tested every "
          f"{MONITOR_STEP} days; OFF above p={MONITOR_OFF_ABOVE}, ON below p={MONITOR_ON_BELOW})")
    print("=" * 100)
    print(stats[["EG p (IS)", "EG p (OOS, hindsight)", "monitor % on (IS)",
                 "monitor % on (OOS)", "monitor switch-offs (IS)",
                 "monitor switch-offs (OOS)", "first OOS switch-off"]].round(3).to_string())
    print()
    mon = bts_all[["sharpe", "max_drawdown", "n_trades"]].unstack(["method", "period"])
    mon = mon.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
    mon.to_csv(OUT / "monitor_comparison.csv")
    for period in ("IS", "OOS"):
        print(f"{period}: net Sharpe / max drawdown / trades by method")
        sub = bts_all.xs(period, level="period")[["sharpe", "max_drawdown", "n_trades"]]
        print(sub.unstack("method").reorder_levels([1, 0], axis=1)
              .reindex(columns=list(METHODS), level=0).round(3).to_string())
        print()
    real = [p_ for p_ in bts_all.index.get_level_values("pair").unique() if p_ != "IVV/SPY"]
    avg = (bts_all.loc[real].groupby(["method", "period"])
           [["sharpe", "max_drawdown", "n_trades", "pct_time_in_market"]].mean())
    print("Averages over the 7 non-trivial pairs (excl. IVV/SPY):")
    print(avg.reindex(list(METHODS), level=0).round(3).to_string())
    print()

    # ---- Choose the best pair using IN-SAMPLE information only ------------
    is_sharpe = bts.xs("IS", level="period")["sharpe"]
    eligible = stats[(stats["EG p (IS)"] < MAX_EG_PVALUE)
                     & stats["half-life IS (d)"].between(*HALF_LIFE_RANGE)].index
    if len(eligible):
        best = is_sharpe.loc[eligible].idxmax()
        why = (f"cointegrated in-sample (EG p < {MAX_EG_PVALUE}), half-life within "
               f"{HALF_LIFE_RANGE} days, and highest in-sample net Sharpe among those")
    else:
        best = stats["EG p (IS)"].idxmin()
        why = "no pair passed all filters; falling back to the lowest EG p-value"
    print(f"BEST PAIR: {best}  ({why})")
    print(f"Eligible pairs: {list(eligible)}")
    if len(eligible):
        k_best = bts_k.xs("IS", level="period")["sharpe"].loc[eligible].idxmax()
        print(f"(Applying the same rule to Kalman IS Sharpe would pick: {k_best})")
    print()

    d = details[best]
    res, ou, cfg = d["res"], d["ou"], d["cfg"]
    print(f"  hedge ratio β = {res.beta:.3f}  →  for $1 of {res.y}, trade ${res.beta:.3f} of {res.x}")
    print(f"  OU: θ = {ou.theta:.4f}/day, μ = {ou.mu:.4f}, σ = {ou.sigma:.4f}, "
          f"half-life = {ou.half_life:.1f} days, z lookback = {cfg.lookback} days")

    # ---- Plots for the best pair ------------------------------------------
    safe = best.replace("/", "_")
    trades = pd.concat([d["bt_is"].trades, d["bt_oos"].trades], ignore_index=True)
    trades.to_csv(OUT / f"trades_{safe}.csv", index=False)
    z = pd.concat([d["bt_is"].frame["z"], d["bt_oos"].frame["z"]])
    plot_zscore(z, events_from_trades(trades), SPLIT,
                f"{best} spread z-score — entries, exits and stops "
                f"(β={res.beta:.2f}, lookback={cfg.lookback}d)",
                cfg.entry_z, OUT / f"zscore_{safe}.png")
    plot_equity({"IS": d["bt_is"].frame, "OOS": d["bt_oos"].frame}, SPLIT,
                f"{best} strategy equity, net of costs", OUT / f"equity_{safe}.png")
    spread = np.log(d["pair"][res.y]) - res.beta * np.log(d["pair"][res.x])
    plot_spread(spread, SPLIT, f"{best} log-price spread (in-sample β held fixed)",
                OUT / f"spread_{safe}.png")

    # ---- Kalman plots for the best pair --------------------------------------
    kal = d["kal"]
    k_trades = pd.concat([d["kbt_is"].trades, d["kbt_oos"].trades], ignore_index=True)
    k_trades.to_csv(OUT / f"trades_{safe}_kalman.csv", index=False)
    kz = pd.concat([d["kbt_is"].frame["z"], d["kbt_oos"].frame["z"]])
    plot_zscore(kz, events_from_trades(k_trades), SPLIT,
                f"{best} Kalman-hedged spread z-score — entries, exits and stops "
                f"(memory≈{KALMAN_MEMORY}d, lookback={cfg.lookback}d)",
                cfg.entry_z, OUT / f"zscore_{safe}_kalman.png")
    # A rolling 1-year OLS beta, as a simple past-only reference for the filter.
    ly, lx = np.log(d["pair"][res.y]), np.log(d["pair"][res.x])
    rolling_beta = (ly.rolling(250).cov(lx) / lx.rolling(250).var()).rename("rolling")
    plot_beta(kal.beta.iloc[KALMAN_BURN_IN:], res.beta, rolling_beta, SPLIT,
              f"{best} hedge ratio over time: fixed OLS vs Kalman vs rolling 1-year OLS",
              OUT / f"beta_{safe}.png")
    plot_equity_compare(
        {"Static OLS hedge": pd.concat([d["bt_is"].frame["net_return"],
                                        d["bt_oos"].frame["net_return"]]),
         "Kalman hedge": pd.concat([d["kbt_is"].frame["net_return"],
                                    d["kbt_oos"].frame["net_return"]])},
        SPLIT, f"{best} equity, net of costs: static vs Kalman hedge",
        OUT / f"equity_{safe}_compare.png")

    is_rng = (d["bt_is"].frame.index[0], d["bt_is"].frame.index[-1])
    oos_rng = (d["bt_oos"].frame.index[0], d["bt_oos"].frame.index[-1])

    def both_periods(beta, c, spread=None):
        return {lab: run_backtest(d["pair"], res.y, res.x, beta, c, s_, e_,
                                  spread=spread).metrics
                for lab, (s_, e_) in (("IS", is_rng), ("OOS", oos_rng))}

    # ---- Transaction-cost sensitivity ---------------------------------------
    # The honest question isn't "is it profitable?" but "how much cost can it
    # absorb before it isn't?" If the edge disappears between 5 and 10 bps,
    # it's fragile: your real costs are uncertain. The Kalman version trades
    # a little every day (rebalancing to the new beta), so it is more
    # cost-sensitive by construction.
    sens = []
    for bps in (0, 2, 5, 10, 20):
        c = replace(cfg, cost_bps=bps)
        for method, beta, spread in (("static OLS", res.beta, None),
                                     ("Kalman", kal.beta, kal.spread)):
            for lab, m in both_periods(beta, c, spread).items():
                sens.append({"cost_bps": bps, "method": method, "period": lab,
                             "sharpe": m["sharpe"]})
    sens = pd.DataFrame(sens).pivot(index="cost_bps", columns=["method", "period"],
                                    values="sharpe")
    sens.to_csv(OUT / f"cost_sensitivity_{safe}.csv")

    # ---- Kalman memory sensitivity --------------------------------------------
    # A ROBUSTNESS check, not a tuning step. If only one memory value looks
    # good, that's a warning sign, and picking it after seeing OOS results
    # would be exactly the overfitting this project is trying to avoid.
    mem_rows = []
    for mem in KALMAN_MEMORY_GRID:
        k = kalman_from_insample(d["pair"], res.y, res.x, SPLIT, res.spread, memory=mem)
        for lab, m in both_periods(k.beta, cfg, k.spread).items():
            mem_rows.append({"memory_days": mem, "period": lab, "sharpe": m["sharpe"],
                             "max_drawdown": m["max_drawdown"], "n_trades": m["n_trades"],
                             "turnover": m["turnover"]})
    mem_sens = pd.DataFrame(mem_rows).pivot(index="memory_days", columns="period")
    mem_sens.to_csv(OUT / f"kalman_memory_sensitivity_{safe}.csv")

    print("\n" + "=" * 100)
    print(f"{best}: IN-SAMPLE vs OUT-OF-SAMPLE, static vs Kalman")
    print("=" * 100)
    cols = ["sharpe", "gross_sharpe", "cagr", "ann_vol", "max_drawdown", "n_trades",
            "win_rate", "avg_hold_days", "turnover", "total_costs"]
    summary = bts_all.loc[best][cols]
    print(summary.T.round(3).to_string())
    print("\nCost sensitivity (net Sharpe by cost per leg per side):")
    print(sens.round(3).to_string())
    print(f"\nKalman memory sensitivity (robustness only; memory={KALMAN_MEMORY} was fixed in advance):")
    print(mem_sens.round(3).to_string())
    # ---- Memory robustness across ALL pairs (OOS net Sharpe) -----------------
    # Does "Kalman beat static" on a given pair survive a different memory
    # setting? If the sign of the improvement flips with memory, the
    # improvement was luck rather than evidence.
    grid = {}
    for name, dd in details.items():
        r = dd["res"]
        row = {"static OLS": bts.loc[(name, "OOS"), "sharpe"]}
        for mem in KALMAN_MEMORY_GRID:
            k = kalman_from_insample(dd["pair"], r.y, r.x, SPLIT, r.spread, memory=mem)
            row[f"K{mem}"] = run_backtest(dd["pair"], r.y, r.x, k.beta, dd["cfg"],
                                          dd["bt_oos"].frame.index[0],
                                          dd["bt_oos"].frame.index[-1],
                                          spread=k.spread).metrics["sharpe"]
        grid[name] = row
    grid = pd.DataFrame(grid).T
    grid.to_csv(OUT / "kalman_memory_grid_oos_all_pairs.csv")
    print("\nOOS net Sharpe by Kalman memory (days), all pairs:")
    print(grid.round(3).to_string())
    beats = grid.filter(like="K").gt(grid["static OLS"], axis=0).sum(axis=1)
    print(f"Memory settings (of {len(KALMAN_MEMORY_GRID)}) where Kalman beat static OOS:")
    print(beats.to_string())

    # ---- Monitor plots ----------------------------------------------------------
    allowed_bt = d["allowed"].loc[START:]
    plot_monitor(d["pvals"].loc[START:], allowed_bt, SPLIT, MONITOR_OFF_ABOVE,
                 MONITOR_ON_BELOW,
                 f"{best} rolling 2-year cointegration p-value (shaded = trading switched off)",
                 OUT / f"monitor_{safe}.png")
    plot_monitor_grid({n: (dd["pvals"].loc[START:], dd["allowed"].loc[START:])
                       for n, dd in details.items()}, SPLIT, MONITOR_OFF_ABOVE,
                      MONITOR_ON_BELOW, OUT / "monitor_all_pairs.png")
    m_trades = pd.concat([d["mbt_is"].trades, d["mbt_oos"].trades], ignore_index=True)
    m_trades.to_csv(OUT / f"trades_{safe}_monitor.csv", index=False)
    mz = pd.concat([d["mbt_is"].frame["z"], d["mbt_oos"].frame["z"]])
    plot_zscore(mz, events_from_trades(m_trades), SPLIT,
                f"{best} z-score with cointegration monitor (static OLS hedge; "
                f"shaded = trading off)", cfg.entry_z,
                OUT / f"zscore_{safe}_monitor.png", allowed=allowed_bt)

    def chain(key):
        return pd.concat([d[f"{key}_is"].frame["net_return"],
                          d[f"{key}_oos"].frame["net_return"]])
    plot_equity_compare({"Static OLS": chain("bt"), "Kalman": chain("kbt"),
                         "Static OLS + monitor": chain("mbt"),
                         "Kalman + monitor": chain("mkbt")},
                        SPLIT, f"{best} equity, net of costs: with and without the monitor",
                        OUT / f"equity_{safe}_monitor.png")

    # ---- Monitor window robustness across ALL pairs (OOS net Sharpe) --------
    # Same idea as the Kalman memory grid: the 2-year window was specified up
    # front; 1- and 3-year windows show whether the conclusion depends on it.
    wrows = {}
    for name, dd in details.items():
        r = dd["res"]
        oos_s, oos_e = dd["bt_oos"].frame.index[0], dd["bt_oos"].frame.index[-1]
        row = {"static (no monitor)": bts.loc[(name, "OOS"), "sharpe"],
               "Kalman (no monitor)": bts_k.loc[(name, "OOS"), "sharpe"]}
        for w in MONITOR_WINDOW_GRID:
            if w == MONITOR_WINDOW:
                al = dd["allowed"]
            else:
                pv = rolling_eg_pvalue(np.log(dd["pair_full"][r.y]),
                                       np.log(dd["pair_full"][r.x]),
                                       window=w, step=MONITOR_STEP)
                al = hysteresis_gate(pv, MONITOR_OFF_ABOVE, MONITOR_ON_BELOW)
            row[f"static+mon {w}d"] = run_backtest(
                dd["pair"], r.y, r.x, r.beta, dd["cfg"], oos_s, oos_e,
                allowed=al).metrics["sharpe"]
            row[f"Kalman+mon {w}d"] = run_backtest(
                dd["pair"], r.y, r.x, dd["kal"].beta, dd["cfg"], oos_s, oos_e,
                spread=dd["kal"].spread, allowed=al).metrics["sharpe"]
        wrows[name] = row
    wgrid = pd.DataFrame(wrows).T
    wgrid.to_csv(OUT / "monitor_window_grid_oos_all_pairs.csv")
    print("\nOOS net Sharpe by monitor window (days), all pairs:")
    print(wgrid.round(3).to_string())
    print("\nMean over the 7 non-trivial pairs:")
    print(wgrid.drop(index="IVV/SPY").mean().round(3).to_string())

    # ---- Does the monitor separate good trades from bad ones? --------------
    # Sharpe ratios built on 1–3 trades are mostly noise. A sharper test of the
    # monitor's value: take every trade from the UN-monitored backtests and
    # ask whether trades entered while the monitor said "on" did better than
    # those entered while it said "off". Several hundred trades, not a handful.
    att = []
    for name, dd in details.items():
        if name == "IVV/SPY":
            continue
        for method, keys in (("static OLS", ("bt_is", "bt_oos")),
                             ("Kalman", ("kbt_is", "kbt_oos"))):
            for key, period in zip(keys, ("IS", "OOS")):
                t = dd[key].trades
                if t.empty:
                    continue
                on = dd["allowed"].reindex(t["entry_date"]).fillna(False).values
                att.append(t.assign(pair=name, method=method, period=period,
                                    monitor_at_entry=np.where(on, "on", "off")))
    att = pd.concat(att, ignore_index=True)
    att.to_csv(OUT / "monitor_trade_attribution.csv", index=False)
    summ = (att.groupby(["method", "period", "monitor_at_entry"])["net_return"]
            .agg(n_trades="count", mean_return="mean", median_return="median",
                 win_rate=lambda r: (r > 0).mean(), worst="min"))
    print("\nTrades from the UN-monitored backtests, split by monitor state at entry "
          "(7 pairs, excl. IVV/SPY):")
    print(summ.round(4).to_string())
    allp = (att.groupby("monitor_at_entry")["net_return"]
            .agg(n_trades="count", mean_return="mean", win_rate=lambda r: (r > 0).mean(),
                 worst="min"))
    print("\nAll periods and both methods pooled:")
    print(allp.round(4).to_string())
    # Welch t-test on the pooled difference in means: is "on" really better?
    from scipy import stats as sps
    on_r = att.loc[att["monitor_at_entry"] == "on", "net_return"]
    off_r = att.loc[att["monitor_at_entry"] == "off", "net_return"]
    tt = sps.ttest_ind(on_r, off_r, equal_var=False)
    print(f"Welch t-test, mean(on) - mean(off) = {on_r.mean() - off_r.mean():.4f}, "
          f"t = {tt.statistic:.2f}, p = {tt.pvalue:.3f}")

    # ---- Time stop: per-pair results ------------------------------------------
    print("\n" + "=" * 100)
    print(f"TIME STOP at {TIME_STOP_MULT:g}× in-sample half-life")
    print("=" * 100)
    ts_tbl = pd.DataFrame({
        "half-life IS (d)": stats["half-life IS (d)"],
        "time stop (d)": stats["time stop (days)"],
    })
    for method in ("static OLS", "static OLS + time stop", "Kalman", "Kalman + time stop"):
        for period in ("IS", "OOS"):
            sub = bts_all.xs((method, period), level=("method", "period"))
            ts_tbl[f"{method} | {period} Sharpe"] = sub["sharpe"]
            ts_tbl[f"{method} | {period} MaxDD"] = sub["max_drawdown"]
            if "time stop" in method:
                ts_tbl[f"{method} | {period} # time stops"] = sub["time_stops"]
    ts_tbl.to_csv(OUT / "time_stop_comparison.csv")
    print(ts_tbl.T.round(3).to_string())

    # ---- Counterfactual: what would each time-stopped trade have done? -----
    # Up to the moment of the time stop, the stopped and un-stopped backtests
    # are identical, so each time-stopped trade has a twin in the baseline run
    # with the SAME entry date. Comparing the two isolates the time stop's
    # effect on exactly the trades it touched.
    cf = []
    for name, dd in details.items():
        if name == "IVV/SPY":
            continue
        for method, base_keys, ts_keys in (("static OLS", ("bt_is", "bt_oos"), ("tbt_is", "tbt_oos")),
                                           ("Kalman", ("kbt_is", "kbt_oos"), ("tkbt_is", "tkbt_oos"))):
            for bk, tk, period in zip(base_keys, ts_keys, ("IS", "OOS")):
                tt = dd[tk].trades
                tt = tt[tt["exit_reason"] == "time_stop"]
                base = dd[bk].trades.set_index("entry_date")
                for t in tt.itertuples():
                    if t.entry_date not in base.index:
                        continue
                    b_ = base.loc[t.entry_date]
                    cf.append(dict(pair=name, method=method, period=period,
                                   entry_date=t.entry_date, stopped_after=t.hold_days,
                                   stopped_return=t.net_return,
                                   baseline_hold=b_["hold_days"],
                                   baseline_return=b_["net_return"],
                                   baseline_exit=b_["exit_reason"]))
    cf = pd.DataFrame(cf)
    cf["saved"] = cf["stopped_return"] - cf["baseline_return"]
    cf.to_csv(OUT / "time_stop_counterfactual.csv", index=False)
    print("\nTime-stopped trades vs the same trade without a time stop (7 pairs):")
    print(cf.groupby(["method", "period"]).agg(
        n=("saved", "size"), stopped_mean=("stopped_return", "mean"),
        baseline_mean=("baseline_return", "mean"), mean_saved=("saved", "mean"),
        helped_share=("saved", lambda x: (x > 0).mean()),
        baseline_worst=("baseline_return", "min"), stopped_worst=("stopped_return", "min"),
    ).round(4).to_string())
    from scipy import stats as sps2
    tt_ = sps2.ttest_1samp(cf["saved"], 0.0)
    print(f"Pooled: n = {len(cf)}, mean saved per stopped trade = {cf['saved'].mean():.4f}, "
          f"t = {tt_.statistic:.2f}, p = {tt_.pvalue:.3f}")

    # ---- Why the time stop rarely fires: how long do trades actually last? ---
    # A rolling z-score whose window is ~2 half-lives drags its own mean toward
    # the spread, so z drifts back to 0 within roughly one window even if the
    # spread never reverts. The z-exit therefore already acts as a time stop.
    att_d = att.drop_duplicates(["pair", "method", "period", "entry_date"]).copy()
    att_d["hold_x_hl"] = att_d["hold_days"] / att_d["pair"].map(stats["half-life IS (d)"])
    losers = att_d[att_d["net_return"] < 0]
    print(f"\nTrade durations (un-monitored, no time stop, 7 pairs, {len(att_d)} trades):")
    print(f"  median hold = {att_d['hold_x_hl'].median():.2f}× half-life; "
          f"90th pct = {att_d['hold_x_hl'].quantile(0.9):.2f}×; "
          f"share > 2× = {(att_d['hold_x_hl'] > 2).mean():.1%}; "
          f"share > 3× = {(att_d['hold_x_hl'] > 3).mean():.1%}")
    print(f"  losing trades: {len(losers)}; exit reasons: "
          f"{losers['exit_reason'].value_counts().to_dict()}")

    # ---- Multiplier robustness (OOS mean Sharpe over 7 pairs) ---------------
    mrows = {}
    for name, dd in details.items():
        r = dd["res"]
        row = {}
        for mult in TIME_STOP_GRID:
            c = replace(dd["cfg"], max_hold_days=max_hold_from_half_life(dd["ou"].half_life, mult))
            for period, key in (("IS", "bt_is"), ("OOS", "bt_oos")):
                s_, e_ = dd[key].frame.index[0], dd[key].frame.index[-1]
                row[(f"{mult:g}x", "static", period)] = run_backtest(
                    dd["pair"], r.y, r.x, r.beta, c, s_, e_).metrics["sharpe"]
                row[(f"{mult:g}x", "Kalman", period)] = run_backtest(
                    dd["pair"], r.y, r.x, dd["kal"].beta, c, s_, e_,
                    spread=dd["kal"].spread).metrics["sharpe"]
        mrows[name] = row
    mgrid = pd.DataFrame(mrows).T
    mgrid.to_csv(OUT / "time_stop_multiplier_grid.csv")
    print("\nMean net Sharpe over the 7 non-trivial pairs, by time-stop multiplier:")
    mean_grid = mgrid.drop(index="IVV/SPY").mean().unstack(0)
    base_row = {("static", "IS"): bts.xs("IS", level="period")["sharpe"].drop("IVV/SPY").mean(),
                ("static", "OOS"): bts.xs("OOS", level="period")["sharpe"].drop("IVV/SPY").mean(),
                ("Kalman", "IS"): bts_k.xs("IS", level="period")["sharpe"].drop("IVV/SPY").mean(),
                ("Kalman", "OOS"): bts_k.xs("OOS", level="period")["sharpe"].drop("IVV/SPY").mean()}
    mean_grid.insert(0, "none", pd.Series(base_row))
    print(mean_grid.round(3).to_string())

    # ---- Time-stop plots for the best pair ------------------------------------
    t_trades = pd.concat([d["tbt_is"].trades, d["tbt_oos"].trades], ignore_index=True)
    t_trades.to_csv(OUT / f"trades_{safe}_timestop.csv", index=False)
    tz = pd.concat([d["tbt_is"].frame["z"], d["tbt_oos"].frame["z"]])
    plot_zscore(tz, events_from_trades(t_trades), SPLIT,
                f"{best} z-score with time stop at {TIME_STOP_MULT:g}× half-life "
                f"({d['cfg_ts'].max_hold_days} days, static OLS hedge)",
                cfg.entry_z, OUT / f"zscore_{safe}_timestop.png")
    plot_equity_compare({"Static OLS": chain("bt"), "Kalman": chain("kbt"),
                         "Static OLS + time stop": chain("tbt"),
                         "Kalman + time stop": chain("tkbt")},
                        SPLIT, f"{best} equity, net of costs: with and without the time stop",
                        OUT / f"equity_{safe}_timestop.png")
    print(f"\nTrade log for {best} (static OLS + time stop):")
    print(t_trades.assign(net_return=t_trades["net_return"].map(fmt_pct)).to_string(index=False))

    print(f"\nTrade log for {best} (static OLS + monitor):")
    print(m_trades.assign(net_return=m_trades["net_return"].map(fmt_pct)).to_string(index=False))

    print(f"\nTrade log for {best} (static OLS):")
    print(trades.assign(net_return=trades["net_return"].map(fmt_pct)).to_string(index=False))
    print(f"\nTrade log for {best} (Kalman):")
    print(k_trades.assign(net_return=k_trades["net_return"].map(fmt_pct)).to_string(index=False))
    print(f"\nOutputs written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
