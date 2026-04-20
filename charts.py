"""
charts.py
─────────
All chart rendering functions.  Returns matplotlib Figure objects.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── palette ──────────────────────────────────────────────────────────────────
BG     = "#06090f"
PANEL  = "#0b1220"
CARD   = "#0f1c2e"
GRID   = "#1a2535"
TEXT   = "#94a3b8"
BLUE   = "#38bdf8"
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
PURPLE = "#a78bfa"
TEAL   = "#2dd4bf"


def _t(fig, axes=None):
    """Apply dark theme to figure."""
    fig.patch.set_facecolor(PANEL)
    for ax in (axes or fig.get_axes()):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8.5)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(BLUE)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linestyle="--", lw=0.5, alpha=0.55)


def _fv(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


# ─────────────────────────────────────────────────────────────────────────────
# MARKET OVERVIEW CHART
# ─────────────────────────────────────────────────────────────────────────────

def chart_market_heatmap(market_df: pd.DataFrame) -> plt.Figure:
    """Bubble scatter: x=pct_change, y=volume_ratio, size=turnover, colour=sector."""
    df = market_df.dropna(subset=["pct_change", "volume_ratio"]).copy()
    sectors = df["sector"].unique() if "sector" in df.columns else ["ALL"]
    cmap    = plt.cm.get_cmap("tab20", len(sectors))
    sec_col = {s: cmap(i) for i, s in enumerate(sectors)}

    fig, ax = plt.subplots(figsize=(13, 5))
    _t(fig, [ax])

    for _, row in df.iterrows():
        sz  = max(30, min(1200, np.log1p(row.get("turnover", 1e6)) * 12))
        col = sec_col.get(row.get("sector", ""), TEXT)
        ax.scatter(row["pct_change"], row["volume_ratio"],
                   s=sz, color=col, alpha=0.65, edgecolors=GRID, lw=0.4, zorder=3)
        if abs(row["pct_change"]) > 2.5 or row["volume_ratio"] > 2.5:
            ax.annotate(row["symbol"],
                        (row["pct_change"], row["volume_ratio"]),
                        fontsize=7, color=TEXT, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

    ax.axvline(0,  color=GRID,  lw=1)
    ax.axhline(1,  color=AMBER, lw=0.8, ls="--", alpha=0.6, label="Avg Volume")
    ax.axvspan(-0.5, 0.5, alpha=0.04, color=TEXT)
    ax.set_xlabel("Price Change %"); ax.set_ylabel("Volume / 30-Day Avg")
    ax.set_title("Market Overview — Activity Bubble Map", fontsize=12, pad=10)

    # Legend for top sectors
    handles = [mpatches.Patch(color=sec_col[s], label=s) for s in list(sectors)[:8]]
    ax.legend(handles=handles, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig


def chart_scanner_bar(scanner_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of scanner scores, coloured by signal."""
    df = scanner_df.copy().sort_values("scanner_score", ascending=True)
    colors = [GREEN if s == "BULLISH" else (RED if s == "BEARISH" else TEXT)
              for s in df.get("signal", ["NEUTRAL"] * len(df))]

    fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.42)))
    _t(fig, [ax])
    bars = ax.barh(df["symbol"].astype(str), df["scanner_score"],
                   color=colors, alpha=0.8, zorder=3)
    for bar, v in zip(bars, df["scanner_score"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=8, color=TEXT)
    ax.set_xlabel("Scanner Score (composite)"); ax.set_title("Top Scanned Stocks", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def chart_pct_change(market_df: pd.DataFrame, symbols: list[str]) -> plt.Figure:
    """Bar chart of % price change for selected symbols."""
    df = market_df[market_df["symbol"].isin(symbols)].copy()
    df = df.sort_values("pct_change", ascending=False)
    colors = [GREEN if v > 0 else RED for v in df["pct_change"]]

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.7), 4))
    _t(fig, [ax])
    ax.bar(df["symbol"].astype(str), df["pct_change"], color=colors, alpha=0.82, zorder=3)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_ylabel("Price Change %"); ax.set_title("Price Movement — Selected Stocks", fontsize=12, pad=10)
    for i, (sym, val) in enumerate(zip(df["symbol"], df["pct_change"])):
        ax.text(i, val + (0.05 if val >= 0 else -0.12),
                f"{val:+.2f}%", ha="center", fontsize=8, color=TEXT)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SMART MONEY SCORE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def chart_score_matrix(results: dict[str, dict]) -> plt.Figure:
    """
    Single figure showing Smart Money Score for every analysed symbol.
    Horizontal bars, green/red, with score labels.
    """
    items = [(sym, r["score"]["score"]) for sym, r in results.items() if "score" in r]
    items.sort(key=lambda x: x[1], reverse=True)
    syms   = [i[0] for i in items]
    scores = [i[1] for i in items]
    colors = [GREEN if s >= 2 else (RED if s <= -2 else TEXT) for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(3, len(syms) * 0.52)))
    _t(fig, [ax])
    bars = ax.barh(syms, scores, color=colors, alpha=0.82, zorder=3)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlim(-11, 11)
    ax.set_xlabel("Smart Money Score (−10 to +10)")
    ax.set_title("Smart Money Score — All Analysed Symbols", fontsize=12, pad=10)
    for bar, v in zip(bars, scores):
        off = 0.2 if v >= 0 else -0.5
        ax.text(v + off, bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}", va="center", ha="left", fontsize=9,
                color=TEXT, fontfamily="monospace")
    fig.tight_layout()
    return fig


def chart_gauge(score: float, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 2.6), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL); ax.axis("off")
    segs = [(-10, -5, RED), (-5, -2, AMBER), (-2, 2, TEXT), (2, 5, "#86efac"), (5, 10, GREEN)]
    for lo, hi, c in segs:
        t1 = 180 - (lo + 10) / 20 * 180
        t2 = 180 - (hi + 10) / 20 * 180
        ax.add_patch(mpatches.Wedge((0, 0), 1.0, t2, t1,
                     width=0.28, facecolor=c, alpha=0.27, edgecolor=BG, lw=0.5))
    ang = np.radians(180 - (score + 10) / 20 * 180)
    ax.annotate("", xy=(0.70 * np.cos(ang), 0.70 * np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
    ax.plot(0, 0, "o", color=color, ms=5)
    ax.text(0, -0.12, f"{score:+.1f}", ha="center", va="center",
            fontsize=20, color=color, fontfamily="monospace", fontweight="bold")
    ax.text(-1.1, 0, "−10", ha="center", va="center", fontsize=7, color=RED)
    ax.text(1.1,  0, "+10", ha="center", va="center", fontsize=7, color=GREEN)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.35, 1.2)
    fig.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DETAIL CHARTS  (for single symbol drill-down)
# ─────────────────────────────────────────────────────────────────────────────

def chart_broker_activity(broker_df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    top = broker_df.head(top_n).copy()
    x, w = np.arange(len(top)), 0.35
    fig, ax = plt.subplots(figsize=(12, 4.5))
    _t(fig, [ax])
    ax.bar(x - w / 2, top["buy_qty"],  w, color=GREEN, alpha=0.8, label="Buy",  zorder=3)
    ax.bar(x + w / 2, top["sell_qty"], w, color=RED,   alpha=0.8, label="Sell", zorder=3)
    for i, (b, s) in enumerate(zip(top["buy_qty"], top["sell_qty"])):
        net = b - s
        ax.annotate("▲" if net > 0 else "▼",
                    (x[i], max(b, s) * 1.02), ha="center", va="bottom",
                    fontsize=8, color=GREEN if net > 0 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(top["broker"].astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Top {top_n} Brokers — Buy vs Sell", fontsize=12, pad=10)
    ax.set_ylabel("Quantity"); _fv(ax)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return fig


def chart_net_positions(broker_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    accs  = broker_df[broker_df["net_qty"] > 0].nlargest(top_n // 2 + 1, "net_qty")
    dists = broker_df[broker_df["net_qty"] < 0].nsmallest(top_n // 2 + 1, "net_qty")
    top   = pd.concat([accs, dists]).sort_values("net_qty", ascending=False)
    colors = [GREEN if v > 0 else RED for v in top["net_qty"]]
    fig, ax = plt.subplots(figsize=(11, max(3, len(top) * 0.4)))
    _t(fig, [ax])
    ax.barh(top["broker"].astype(str), top["net_qty"], color=colors, alpha=0.8)
    ax.axvline(0, color=TEXT, lw=0.8)
    ax.set_title("Net Position per Broker (Buy − Sell)", fontsize=12, pad=10)
    ax.set_xlabel("Net Quantity")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    return fig


def chart_acf(acf_d: dict) -> plt.Figure:
    lags, vals, ci = acf_d["lags"], acf_d["acf"], acf_d["ci"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    _t(fig, [ax])
    colors = [GREEN if v > 0 else RED for v in vals]
    ax.bar(lags, vals, color=colors, alpha=0.75, width=0.6, zorder=3)
    ax.axhline(ci,  color=AMBER, ls="--", lw=1.1, label=f"95% CI ±{ci:.3f}", alpha=0.8)
    ax.axhline(-ci, color=AMBER, ls="--", lw=1.1, alpha=0.8)
    ax.axhline(0,   color=GRID,  lw=0.8)
    ax.set_title("Order Flow Autocorrelation", fontsize=12, pad=10)
    ax.set_xlabel("Lag (trades)"); ax.set_ylabel("ACF")
    ax.set_xlim(0.5, max(lags) + 0.5)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return fig


def chart_size_dist(sz_d: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    _t(fig, [ax1, ax2])
    qty, thr = sz_d["qty"], sz_d["threshold"]
    n, bins, patches = ax1.hist(qty, bins=min(50, max(5, len(qty) // 5 + 1)),
                                 color=BLUE, alpha=0.7, edgecolor=BG, lw=0.3)
    for p, left in zip(patches, bins[:-1]):
        if left >= thr:
            p.set_facecolor(RED); p.set_alpha(0.85)
    ax1.axvline(sz_d["mean"],   color=AMBER, ls="--", lw=1.4, label=f"Mean {sz_d['mean']:,.0f}")
    ax1.axvline(sz_d["median"], color=GREEN, ls=":",  lw=1.4, label=f"Median {sz_d['median']:,.0f}")
    ax1.axvline(thr,            color=RED,   ls="-",  lw=1.4, label=f"Large ≥{thr:,.0f}")
    ax1.set_title("Quantity Distribution", fontsize=12)
    ax1.set_xlabel("Quantity"); ax1.set_ylabel("Frequency")
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax2.boxplot(qty, patch_artist=True, vert=True,
                boxprops=dict(facecolor=BLUE, color=GRID, alpha=0.5),
                medianprops=dict(color=AMBER, lw=2),
                whiskerprops=dict(color=TEXT),
                capprops=dict(color=TEXT),
                flierprops=dict(marker="o", color=RED, alpha=0.4, ms=3))
    ax2.set_title("Qty Box Plot", fontsize=12)
    ax2.set_ylabel("Quantity")
    ax2.set_xticks([1]); ax2.set_xticklabels(["All Trades"])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    return fig


def chart_volume_profile(vbp: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _t(fig, [ax1, ax2])
    ax1.barh(vbp["rate"].astype(str), vbp["total_qty"], color=BLUE, alpha=0.7, height=0.6)
    ax1.set_title("Volume Profile", fontsize=12)
    ax1.set_xlabel("Quantity"); ax1.set_ylabel("Price")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    sz = (vbp["total_qty"] / vbp["total_qty"].max() * 600).clip(15)
    sc = ax2.scatter(vbp["rate"], vbp["trades"], s=sz,
                     c=vbp["total_qty"], cmap="plasma", alpha=0.7,
                     edgecolors=GRID, lw=0.4)
    ax2.set_title("Price vs Trade Count (bubble=vol)", fontsize=12)
    ax2.set_xlabel("Price (Rate)"); ax2.set_ylabel("# Trades")
    cb = plt.colorbar(sc, ax=ax2)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=7)
    cb.set_label("Total Qty", color=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def chart_intraday_flow(df: pd.DataFrame) -> plt.Figure:
    """
    Cumulative order imbalance chart (Order Flow Imbalance / Delta).
    Positive = net buying pressure, negative = net selling.
    """
    df = df.copy().reset_index(drop=True)
    # Proxy: if buyer < seller (lower broker ID = more institutional) → buy signal
    df["imb"] = df.apply(
        lambda r: r["qty"] if str(r["buyer"]) < str(r["seller"]) else -r["qty"], axis=1
    )
    df["cum_delta"] = df["imb"].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    _t(fig, [ax1, ax2])

    ax1.fill_between(range(len(df)), df["rate"], alpha=0.6,
                     color=BLUE, linewidth=0)
    ax1.plot(range(len(df)), df["rate"], color=BLUE, lw=0.8)
    ax1.set_title("Intraday Price & Order Flow Delta", fontsize=12, pad=8)
    ax1.set_ylabel("Price")
    _fv(ax1)

    pos = df["cum_delta"].clip(lower=0)
    neg = df["cum_delta"].clip(upper=0)
    ax2.fill_between(range(len(df)), pos, 0, alpha=0.6, color=GREEN)
    ax2.fill_between(range(len(df)), neg, 0, alpha=0.6, color=RED)
    ax2.plot(range(len(df)), df["cum_delta"], color=TEXT, lw=0.6)
    ax2.axhline(0, color=GRID, lw=0.8)
    ax2.set_ylabel("Cum. Delta"); ax2.set_xlabel("Trade Sequence")
    _fv(ax2)

    fig.tight_layout()
    return fig


def chart_sector_flow(results: dict[str, dict], market_df: pd.DataFrame) -> plt.Figure:
    """Stacked bar: net flow by sector."""
    rows = []
    for sym, r in results.items():
        if "score" not in r:
            continue
        sector_row = market_df[market_df["symbol"] == sym]
        sector = sector_row.iloc[0]["sector"] if len(sector_row) > 0 and "sector" in sector_row.columns else "Other"
        rows.append({"symbol": sym, "sector": sector, "score": r["score"]["score"]})
    if not rows:
        return plt.Figure()

    df = pd.DataFrame(rows)
    sector_scores = df.groupby("sector")["score"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    _t(fig, [ax])
    colors = [GREEN if v > 0 else RED for v in sector_scores]
    ax.bar(sector_scores.index, sector_scores.values, color=colors, alpha=0.8, zorder=3)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_title("Avg Smart Money Score by Sector", fontsize=12, pad=10)
    ax.set_ylabel("Avg Score"); ax.set_ylim(-11, 11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    return fig
