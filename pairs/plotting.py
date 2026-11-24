"""
Step 5 — Looking at it
======================

Never trust a backtest number you haven't looked at. The z-score chart answers
questions statistics can't: Does the spread actually oscillate, or did one
lucky episode make all the money? Did the behaviour change after a certain
date? Are stops clustered in one regime?

Markers encode the trade type by SHAPE as well as colour, so the chart still
reads in greyscale or for colour-blind readers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# A small, colour-blind-checked palette.
BLUE = "#2a78d6"     # the z-score / equity line
AQUA = "#1baf7a"     # long-spread entries
ORANGE = "#eb6834"   # short-spread entries
RED = "#d03b3b"      # stop-losses
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _shade_off(ax, allowed: pd.Series | None, label: bool = True) -> None:
    """Grey background wherever the cointegration monitor blocked trading."""
    if allowed is None:
        return
    off = ~allowed.astype(bool)
    # Find contiguous runs of "off" days and shade each one.
    runs = (off != off.shift()).cumsum()[off]
    first = True
    for _, seg in runs.groupby(runs):
        ax.axvspan(seg.index[0], seg.index[-1], color=GRID, alpha=0.7, linewidth=0,
                   label="Monitor: trading OFF" if (label and first) else None)
        first = False


def plot_zscore(z: pd.Series, events: pd.DataFrame, split_date: pd.Timestamp,
                title: str, entry_z: float, path, allowed: pd.Series | None = None) -> None:
    """Z-score with ±1/±2 bands, trade entries/exits, and the IS/OOS boundary."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=SURFACE)
    _style(ax)
    _shade_off(ax, allowed.reindex(z.index) if allowed is not None else None)

    ax.plot(z.index, z.values, color=BLUE, linewidth=1.0, label="z-score")

    # Bands: ±1 is "normal noise", ±2 is our entry threshold.
    ax.axhline(0, color=INK, linewidth=0.8)
    for lvl, style in ((1, ":"), (2, "--")):
        ax.axhline(lvl, color=MUTED, linestyle=style, linewidth=1.0)
        ax.axhline(-lvl, color=MUTED, linestyle=style, linewidth=1.0)
        ax.text(z.index[0], lvl, f" +{lvl}σ", va="bottom", fontsize=8, color=MUTED)
        ax.text(z.index[0], -lvl, f" −{lvl}σ", va="top", fontsize=8, color=MUTED)

    if not events.empty:
        def pick(kind, side=None):
            m = events["kind"].eq(kind)
            if side is not None:
                m &= events["side"].eq(side)
            return events[m]

        e = pick("entry", "long spread")
        ax.scatter(e["date"], e["z"], marker="^", s=70, color=AQUA, edgecolor=SURFACE,
                   linewidth=1, zorder=5, label="Enter long spread (z ≤ −2)")
        e = pick("entry", "short spread")
        ax.scatter(e["date"], e["z"], marker="v", s=70, color=ORANGE, edgecolor=SURFACE,
                   linewidth=1, zorder=5, label="Enter short spread (z ≥ +2)")
        e = events[events["kind"].isin(["exit", "period_end"])]
        ax.scatter(e["date"], e["z"], marker="o", s=40, facecolor=SURFACE, edgecolor=INK,
                   linewidth=1.2, zorder=5, label="Exit (z back to 0)")
        e = pick("time_stop")
        if not e.empty:
            ax.scatter(e["date"], e["z"], marker="D", s=50, color=RED, edgecolor=SURFACE,
                       linewidth=1, zorder=6, label="Time stop (held > 3× half-life)")
        e = pick("monitor_off")
        if not e.empty:
            ax.scatter(e["date"], e["z"], marker="s", s=55, color=INK, edgecolor=SURFACE,
                       linewidth=1, zorder=6, label="Closed by monitor (p > 0.10)")
        e = pick("stop")
        if not e.empty:
            ax.scatter(e["date"], e["z"], marker="X", s=80, color=RED, edgecolor=SURFACE,
                       linewidth=1, zorder=6, label="Stop-loss (|z| ≥ 4)")

    ax.axvline(split_date, color=INK, linewidth=1.2, linestyle="-.")
    ymax = max(4.5, z.abs().max() * 1.05)
    ax.set_ylim(-ymax, ymax)
    ax.text(split_date, ymax * 0.95, "  out-of-sample →", fontsize=10, color=INK, va="top")
    ax.text(split_date, ymax * 0.95, "← in-sample  ", fontsize=10, color=INK,
            va="top", ha="right")

    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.set_ylabel("z-score of spread (σ from rolling mean)", color="#52514e")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.06), fontsize=9, frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity(frames: dict[str, pd.DataFrame], split_date, title: str, path) -> None:
    """Net equity curve (after costs), with the in-/out-of-sample boundary.

    The OOS curve is chained onto the end of the IS curve so you can see the
    whole history as one account.
    """
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=SURFACE)
    _style(ax)
    chained = pd.concat([f["net_return"] for f in frames.values()])
    equity = (1 + chained).cumprod()
    ax.plot(equity.index, equity.values, color=BLUE, linewidth=1.6)
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.axvline(split_date, color=INK, linewidth=1.2, linestyle="-.")
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.set_ylabel("Growth of $1 (net of costs)", color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spread(spread: pd.Series, split_date, title: str, path) -> None:
    """Raw log-price spread. Useful to see slow drift that the rolling z hides."""
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=SURFACE)
    _style(ax)
    ax.plot(spread.index, spread.values, color=BLUE, linewidth=1.0)
    ax.axvline(split_date, color=INK, linewidth=1.2, linestyle="-.")
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.set_ylabel("log(Y) − β·log(X)", color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_beta(kalman_beta: pd.Series, static_beta: float, rolling_beta: pd.Series,
              split_date, title: str, path) -> None:
    """Hedge ratio over time: fixed OLS, Kalman, and a rolling-OLS reference.

    If the Kalman line wanders far from the fixed line, the fixed hedge was
    holding the wrong mix of the two legs during that time.
    """
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.plot(rolling_beta.index, rolling_beta.values, color=MUTED, linewidth=1.0,
            linestyle=":", label="Rolling 1-year OLS β")
    ax.plot(kalman_beta.index, kalman_beta.values, color=BLUE, linewidth=1.6,
            label="Kalman β")
    ax.axhline(static_beta, color=ORANGE, linewidth=1.6, linestyle="--",
               label=f"Fixed in-sample OLS β = {static_beta:.2f}")
    ax.axvline(split_date, color=INK, linewidth=1.2, linestyle="-.")
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.set_ylabel("hedge ratio β", color="#52514e")
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity_compare(returns: dict[str, pd.Series], split_date, title: str, path) -> None:
    """Two equity curves on one axis (same units, growth of $1), direct-labelled."""
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=SURFACE)
    _style(ax)
    styles = ((ORANGE, "--"), (BLUE, "-"), (AQUA, "--"), (INK, "-"))
    for (label, r), (color, ls) in zip(returns.items(), styles):
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq.values, color=color, linewidth=1.6, linestyle=ls, label=label)
        ax.annotate(f" {label}: {eq.iloc[-1]:.2f}", (eq.index[-1], eq.iloc[-1]),
                    fontsize=9, color=INK, va="center")
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.axvline(split_date, color=INK, linewidth=1.2, linestyle="-.")
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.set_ylabel("Growth of $1 (net of costs)", color="#52514e")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    # Leave room on the right for the end-of-line labels only.
    idx = next(iter(returns.values())).index
    ax.set_xlim(idx[0] - pd.Timedelta(days=60), idx[-1] + (idx[-1] - idx[0]) * 0.13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_monitor(pvalues: pd.Series, allowed: pd.Series, split_date, off_above: float,
                 on_below: float, title: str, path) -> None:
    """Rolling Engle–Granger p-value with the on/off thresholds and OFF periods shaded."""
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=SURFACE)
    _style(ax)
    _shade_off(ax, allowed)
    ax.plot(pvalues.index, pvalues.values, color=BLUE, linewidth=1.2,
            label="Rolling 2-year Engle–Granger p-value")
    ax.axhline(off_above, color=RED, linestyle="--", linewidth=1.2,
               label=f"Switch OFF above {off_above:.2f}")
    ax.axhline(on_below, color=AQUA, linestyle="--", linewidth=1.2,
               label=f"Switch ON below {on_below:.2f}")
    ax.axvline(split_date, color=INK, linewidth=1.2, linestyle="-.")
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.set_ylabel("p-value", color="#52514e")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=9, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_monitor_grid(series: dict[str, tuple[pd.Series, pd.Series]], split_date,
                      off_above: float, on_below: float, path) -> None:
    """Small multiples: one rolling-p-value panel per pair, same y-axis everywhere."""
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.55 * n), sharex=True, facecolor=SURFACE)
    for ax, (name, (pv, allowed)) in zip(axes, series.items()):
        _style(ax)
        _shade_off(ax, allowed, label=False)
        ax.plot(pv.index, pv.values, color=BLUE, linewidth=1.0)
        ax.axhline(off_above, color=RED, linestyle="--", linewidth=0.9)
        ax.axhline(on_below, color=AQUA, linestyle="--", linewidth=0.9)
        ax.axvline(split_date, color=INK, linewidth=1.0, linestyle="-.")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.text(0.005, 0.92, f"{name}   on {allowed.mean():.0%} of days",
                transform=ax.transAxes, fontsize=9.5, color=INK, va="top",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    axes[0].set_title("Rolling 2-year Engle–Granger p-value by pair "
                      f"(red: off above {off_above:.2f}, green: on below {on_below:.2f}, "
                      "shaded: trading off)", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
