"""
analysis.py
───────────
Full institutional analysis pipeline.
All functions are pure (no Streamlit imports) and can be unit-tested independently.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# BROKER AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def broker_analysis(df: pd.DataFrame) -> pd.DataFrame:
    buy  = df.groupby("buyer")["qty"].sum().reset_index().rename(
        columns={"buyer": "broker", "qty": "buy_qty"})
    sell = df.groupby("seller")["qty"].sum().reset_index().rename(
        columns={"seller": "broker", "qty": "sell_qty"})
    m = pd.merge(buy, sell, on="broker", how="outer").fillna(0)
    m["net_qty"]   = m["buy_qty"] - m["sell_qty"]
    m["total_qty"] = m["buy_qty"] + m["sell_qty"]
    m["net_pct"]   = (m["net_qty"] / m["total_qty"].replace(0, np.nan) * 100).round(2)
    return m.sort_values("total_qty", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# ORDER-FLOW AUTOCORRELATION
# ──────────────────────────────────────────────────────────────────────────────

def acf_analysis(df: pd.DataFrame, max_lags: int = 20) -> dict:
    signs = df["trade_sign"].values.astype(float)
    n = len(signs)
    mu, var = signs.mean(), signs.var()
    vals = []
    for lag in range(1, min(max_lags + 1, n)):
        cov = np.mean((signs[: n - lag] - mu) * (signs[lag:] - mu))
        vals.append(float(cov / var) if var > 0 else 0.0)
    lags = list(range(1, len(vals) + 1))
    ci   = 1.96 / np.sqrt(n)
    sig  = [l for l, v in zip(lags, vals) if abs(v) > ci]
    pos5 = sum(1 for v in vals[:5] if v > ci)
    return {
        "acf":   vals,
        "lags":  lags,
        "ci":    ci,
        "sig":   sig,
        "pos5":  pos5,
        "mean5": float(np.mean(vals[:5])) if vals else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METAORDER DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_metaorders(df: pd.DataFrame, min_run: int = 3) -> pd.DataFrame:
    ref = "sn" if "sn" in df.columns else None
    rows = []

    def _scan(grp: pd.DataFrame, direction: str, broker: str):
        g = grp.reset_index(drop=True)
        run, vol, prices = 1, g.loc[0, "qty"], [g.loc[0, "rate"]]
        for i in range(1, len(g)):
            gap = 1
            if ref:
                a, b = g.loc[i - 1, ref], g.loc[i, ref]
                if pd.notna(a) and pd.notna(b):
                    gap = abs(b - a)
            if gap <= 15:
                run += 1
                vol += g.loc[i, "qty"]
                prices.append(g.loc[i, "rate"])
            else:
                if run >= min_run:
                    cv = np.std(prices) / np.mean(prices) * 100 if np.mean(prices) else 0
                    rows.append({"broker": broker, "direction": direction, "run_len": run,
                                 "total_qty": vol, "avg_rate": round(np.mean(prices), 2),
                                 "price_cv": round(cv, 2),
                                 "suspicion": "HIGH" if run >= 5 and cv < 1 else "MODERATE"})
                run, vol, prices = 1, g.loc[i, "qty"], [g.loc[i, "rate"]]
        if run >= min_run:
            cv = np.std(prices) / np.mean(prices) * 100 if np.mean(prices) else 0
            rows.append({"broker": broker, "direction": direction, "run_len": run,
                         "total_qty": vol, "avg_rate": round(np.mean(prices), 2),
                         "price_cv": round(cv, 2),
                         "suspicion": "HIGH" if run >= 5 and cv < 1 else "MODERATE"})

    for broker, grp in df.groupby("buyer"):
        g = grp.sort_values(ref) if ref else grp
        _scan(g, "BUY", str(broker))
    for broker, grp in df.groupby("seller"):
        g = grp.sort_values(ref) if ref else grp
        _scan(g, "SELL", str(broker))

    cols = ["broker", "direction", "run_len", "total_qty", "avg_rate", "price_cv", "suspicion"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("total_qty", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# TRADE SIZE STATS
# ──────────────────────────────────────────────────────────────────────────────

def size_analysis(df: pd.DataFrame) -> dict:
    qty = df["qty"].dropna()
    q25, q75 = qty.quantile(0.25), qty.quantile(0.75)
    iqr = q75 - q25
    threshold = max(q75 + 1.5 * iqr, qty.mean() + 2.5 * qty.std())
    large = df[df["qty"] > threshold]
    return {
        "mean":      float(qty.mean()),
        "std":       float(qty.std()),
        "median":    float(qty.median()),
        "q25":       float(q25),
        "q75":       float(q75),
        "threshold": float(threshold),
        "large":     large,
        "large_count": len(large),
        "large_vol": float(large["qty"].sum()),
        "qty":       qty,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VOLUME PROFILE
# ──────────────────────────────────────────────────────────────────────────────

def volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    vbp = (
        df.groupby("rate")
        .agg(total_qty=("qty", "sum"), trades=("qty", "count"))
        .reset_index()
        .sort_values("rate")
    )
    vbp["pct"]     = vbp["total_qty"] / vbp["total_qty"].sum() * 100
    vbp["cum_qty"] = vbp["total_qty"].cumsum()
    return vbp


# ──────────────────────────────────────────────────────────────────────────────
# PRESSURE
# ──────────────────────────────────────────────────────────────────────────────

def pressure_stats(df: pd.DataFrame, broker_df: pd.DataFrame) -> dict:
    net_buy  = float(broker_df[broker_df["net_qty"] > 0]["net_qty"].sum())
    net_sell = float(abs(broker_df[broker_df["net_qty"] < 0]["net_qty"].sum()))
    total    = net_buy + net_sell
    return {
        "total_qty":   float(df["qty"].sum()),
        "trade_count": len(df),
        "buyers":      df["buyer"].nunique(),
        "sellers":     df["seller"].nunique(),
        "net_buy":     net_buy,
        "net_sell":    net_sell,
        "buy_ratio":   round(net_buy / total * 100, 2) if total > 0 else 50.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SMART MONEY SCORE
# ──────────────────────────────────────────────────────────────────────────────

GREEN = "#22c55e"
RED   = "#ef4444"
AMBER = "#f59e0b"
CYAN  = "#38bdf8"
MUTED = "#94a3b8"


def smart_money_score(
    broker_df: pd.DataFrame,
    acf_d:     dict,
    sz_d:      dict,
    meta_df:   pd.DataFrame,
    pr:        dict,
) -> dict:
    total = pr["total_qty"]

    # 1. Broker net flow ±4
    net_total = pr["net_buy"] + pr["net_sell"]
    c1 = float(np.clip((pr["net_buy"] / net_total - 0.5) * 8, -4, 4)) if net_total else 0.0

    # 2. ACF persistence ±2
    c2 = float(np.clip(acf_d["mean5"] * 20, -2, 2))

    # 3. Trade size anomaly ±2
    lv_pct  = sz_d["large_vol"] / (total + 1e-9)
    med_net = float(broker_df["net_pct"].median()) if len(broker_df) > 0 else 0.0
    c3 = float(np.clip((med_net / 100) * 2 * (1 + lv_pct), -2, 2))

    # 4. Metaorder bias ±2
    c4 = 0.0
    if len(meta_df) > 0:
        bv = float(meta_df[meta_df["direction"] == "BUY"]["total_qty"].sum())
        sv = float(meta_df[meta_df["direction"] == "SELL"]["total_qty"].sum())
        mt = bv + sv
        c4 = float(np.clip((bv - sv) / mt * 2, -2, 2)) if mt > 0 else 0.0

    score = round(float(np.clip(c1 + c2 + c3 + c4, -10, 10)), 2)
    comps = {
        "Broker Net Flow":   round(c1, 2),
        "ACF Persistence":   round(c2, 2),
        "Size Anomaly":      round(c3, 2),
        "Metaorder Bias":    round(c4, 2),
    }

    # Confidence
    signs = [int(np.sign(v)) for v in comps.values() if v != 0]
    conf  = round(abs(sum(signs)) / len(signs) * 100) if signs else 50

    # Interpretation + colour
    if   score >= 6:   interp, col = "Strong Bullish — Accumulation", GREEN
    elif score >= 3:   interp, col = "Weak Bullish — Mild Accumulation", "#86efac"
    elif score >= -2:  interp, col = "Neutral — No Clear Bias", MUTED
    elif score >= -5:  interp, col = "Weak Bearish — Mild Distribution", AMBER
    else:              interp, col = "Strong Bearish — Distribution", RED

    icon = "🟢" if score >= 3 else ("🔴" if score <= -3 else "⚪")
    return {
        "score":  score,
        "interp": interp,
        "icon":   icon,
        "color":  col,
        "comps":  comps,
        "conf":   conf,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE  (one call per symbol)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(df_floor: pd.DataFrame, symbol: str, max_lags: int = 20, min_run: int = 3) -> dict[str, Any]:
    """
    Run the complete analysis pipeline for a single symbol.
    df_floor must already be filtered to that symbol and have columns:
      sn, contract, symbol, buyer, seller, qty, rate, amount
    Returns a dict of all analysis results.
    """
    df = df_floor.copy()
    df["trade_sign"] = 1  # all trades buyer-initiated by definition in floorsheet

    if len(df) < 5:
        return {"error": f"Insufficient data ({len(df)} trades) for {symbol}"}

    bd   = broker_analysis(df)
    ad   = acf_analysis(df, max_lags)
    sd   = size_analysis(df)
    md   = detect_metaorders(df, min_run)
    vbp  = volume_profile(df)
    pr   = pressure_stats(df, bd)
    sc   = smart_money_score(bd, ad, sd, md, pr)

    # Point of control
    poc = float(vbp.loc[vbp["total_qty"].idxmax(), "rate"]) if len(vbp) > 0 else 0.0

    return {
        "symbol":     symbol,
        "df":         df,
        "broker_df":  bd,
        "acf":        ad,
        "size":       sd,
        "meta":       md,
        "vbp":        vbp,
        "pressure":   pr,
        "score":      sc,
        "poc":        poc,
    }
