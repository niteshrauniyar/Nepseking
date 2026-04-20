"""
app.py — NEPSE FloorSheet Intelligence  (Automated Edition)
────────────────────────────────────────────────────────────
Fully automated pipeline:
  1. Fetch market data (live → fallback simulation)
  2. Scan & rank top stocks
  3. Generate / load floorsheet
  4. Run analysis for every stock
  5. Dashboard + click-through drill-down
"""

from __future__ import annotations

import io
import csv
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

# ── local modules ─────────────────────────────────────────────────────────────
from data_engine import (
    fetch_market_data,
    scan_top_stocks,
    generate_floorsheet,
    market_summary,
)
from analysis import run_pipeline
import charts as ch

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEPSE Intelligence",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #06090f;
    color: #cdd6e0;
}
.stApp {
    background: radial-gradient(ellipse at 15% 0%, #0a1f38 0%, #06090f 55%);
}
section[data-testid="stSidebar"] {
    background: #080d18;
    border-right: 1px solid #111d2e;
}
h1,h2,h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #38bdf8 !important;
}
h4,h5,h6 { color: #94a3b8 !important; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg,#0d1a2e,#101824);
    border: 1px solid #1a2e4a;
    border-top: 2px solid #38bdf8;
    border-radius: 10px;
    padding: 12px 14px;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #38bdf8 !important;
    font-size: 1.65rem !important;
}
[data-testid="stMetricLabel"] { color: #475569 !important; font-size:.76rem !important; }
[data-testid="stMetricDelta"]  { font-family: 'IBM Plex Mono', monospace !important; }

.stButton > button {
    background: linear-gradient(135deg,#0ea5e9,#0284c7);
    color: #fff; font-weight: 600; border: none;
    border-radius: 7px; padding: 9px 22px;
    font-family: 'IBM Plex Mono', monospace;
    transition: all .18s;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#38bdf8,#0ea5e9);
    box-shadow: 0 0 18px rgba(56,189,248,.3);
}
.stTabs [data-baseweb="tab-list"] { background:#080d18; border-radius:8px; padding:3px; }
.stTabs [data-baseweb="tab"] {
    color:#475569; font-family:'IBM Plex Mono',monospace; font-size:.8rem;
}
.stTabs [aria-selected="true"] {
    background:#0d1a2e !important; color:#38bdf8 !important; border-radius:6px;
}
.card {
    background:linear-gradient(135deg,#0d1a2e,#101824);
    border:1px solid #1a2e4a; border-radius:10px;
    padding:16px 18px; margin:5px 0;
}
.score-card {
    background:linear-gradient(135deg,#0d1a2e,#0c2240);
    border:2px solid #1a2e4a; border-radius:12px;
    padding:20px; text-align:center;
    box-shadow: 0 6px 28px rgba(0,0,0,.55);
}
.stock-btn {
    background:linear-gradient(135deg,#0d1a2e,#112038);
    border:1px solid #1a2e4a; border-radius:10px;
    padding:14px 16px; margin:4px 0;
    cursor:pointer; transition:all .15s;
}
.stock-btn:hover { border-color:#38bdf8; box-shadow:0 0 14px rgba(56,189,248,.2); }
.bull { border-left:4px solid #22c55e !important; }
.bear { border-left:4px solid #ef4444 !important; }
.neu  { border-left:4px solid #94a3b8 !important; }
.insight {
    background:#080f1a; border:1px solid #1a2e4a;
    border-left:4px solid #38bdf8; border-radius:8px;
    padding:12px 14px; margin:5px 0; font-size:.87rem; line-height:1.65;
}
.flag {
    background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.22);
    border-left:4px solid #ef4444; border-radius:8px;
    padding:12px 14px; margin:5px 0; font-size:.87rem;
}
.ok {
    background:rgba(34,197,94,.06); border:1px solid rgba(34,197,94,.22);
    border-left:4px solid #22c55e; border-radius:8px;
    padding:12px 14px; margin:5px 0; font-size:.87rem;
}
.sec {
    font-family:'IBM Plex Mono',monospace; font-size:.68rem;
    letter-spacing:2px; color:#1e3a5f; text-transform:uppercase;
    margin:20px 0 7px; padding-bottom:5px; border-bottom:1px solid #0d1a2e;
}
.badge-bull { color:#22c55e; font-weight:700; font-family:monospace; }
.badge-bear { color:#ef4444; font-weight:700; font-family:monospace; }
.badge-neu  { color:#94a3b8; font-family:monospace; }
.ticker {
    font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
    color:#38bdf8; font-weight:600;
}
.source-tag {
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:.7rem; font-family:monospace; letter-spacing:.5px;
    background:#0d1a2e; border:1px solid #1a2e4a; color:#475569;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "market_df":    None,
        "data_source":  "",
        "scanner_df":   None,
        "floor_df":     None,
        "results":      {},           # symbol → pipeline output
        "selected_sym": None,
        "last_run":     None,
        "n_stocks":     8,
        "min_run":      3,
        "max_lags":     20,
        "top_n_broker": 12,
        "use_upload":   False,
        "uploaded_floor": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏔️ NEPSE Intelligence\n**Automated Analysis System**")
    st.markdown("---")

    st.markdown("### 🔄 Data Mode")
    data_mode = st.radio(
        "Floorsheet source",
        ["Auto-generate (recommended)", "Upload my own floorsheet"],
        index=0,
    )
    st.session_state.use_upload = (data_mode == "Upload my own floorsheet")

    if st.session_state.use_upload:
        up = st.file_uploader("Upload floorsheet (.xlsx)", type=["xlsx"])
        if up:
            st.session_state.uploaded_floor = up.read()
            st.success("✅ File ready")

    st.markdown("---")
    st.markdown("### ⚙️ Scanner Settings")
    st.session_state.n_stocks     = st.slider("Stocks to analyse",       3, 20, 8)
    st.session_state.top_n_broker = st.slider("Top N brokers in charts", 5, 20, 12)
    st.session_state.min_run      = st.slider("Metaorder min run",       2, 8,  3)
    st.session_state.max_lags     = st.slider("ACF max lags",            5, 40, 20)

    force_sim = st.checkbox("Force simulation mode", value=False,
                            help="Skip live fetch; always use simulated data")

    st.markdown("---")
    run_btn = st.button("🚀 Run Full Analysis", use_container_width=True)

    if st.session_state.last_run:
        st.caption(f"Last run: {st.session_state.last_run}")

    if st.session_state.data_source:
        src = st.session_state.data_source
        icon = "🌐" if "simulation" not in src else "🔬"
        st.markdown(f"{icon} <span class='source-tag'>{src}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem;color:#1e3a5f;line-height:1.9'>
    <b>Pipeline Modules</b><br>
    ✅ Live data fetch (3 sources)<br>
    ✅ Simulation fallback<br>
    ✅ Activity scanner<br>
    ✅ Floorsheet generation<br>
    ✅ Broker-level analysis<br>
    ✅ Order flow ACF<br>
    ✅ Metaorder detection<br>
    ✅ Smart Money Score<br>
    ✅ Sector aggregation<br>
    ✅ Click-through drill-down<br>
    ✅ CSV export
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(force_sim: bool = False):
    progress = st.progress(0, text="Fetching market data...")
    status   = st.empty()

    # 1. Market data
    market_df, source = fetch_market_data(force_simulate=force_sim)
    st.session_state.market_df   = market_df
    st.session_state.data_source = source
    progress.progress(20, text=f"Market data loaded ({source})")

    # 2. Scanner
    scanner_df = scan_top_stocks(market_df, n=st.session_state.n_stocks)
    st.session_state.scanner_df = scanner_df
    symbols = scanner_df["symbol"].tolist()
    progress.progress(35, text=f"Scanned {len(symbols)} top stocks")

    # 3. Floorsheet
    if st.session_state.use_upload and st.session_state.uploaded_floor:
        status.info("📂 Loading uploaded floorsheet...")
        raw = pd.read_excel(io.BytesIO(st.session_state.uploaded_floor), engine="openpyxl")
        # Auto-normalise columns
        col_map = {}
        for col in raw.columns:
            low = col.strip().lower()
            if "symbol" in low or "scrip" in low or "stock" in low:
                col_map[col] = "Stock Symbol"
            elif "buyer" in low:
                col_map[col] = "Buyer Broker No"
            elif "seller" in low:
                col_map[col] = "Seller Broker No"
            elif "quantity" in low or "qty" in low or "shares" in low:
                col_map[col] = "Quantity"
            elif low in ("rate", "price", "ltp"):
                col_map[col] = "Rate"
            elif "amount" in low or "turnover" in low or "value" in low:
                col_map[col] = "Amount"
            elif low in ("sn", "s.n", "s.n.", "serial"):
                col_map[col] = "SN"
            elif "contract" in low:
                col_map[col] = "Contract No"
        raw = raw.rename(columns=col_map)
        floor_df = raw
        # Filter to scanned symbols if possible
        if "Stock Symbol" in floor_df.columns:
            floor_df["Stock Symbol"] = floor_df["Stock Symbol"].astype(str).str.upper().str.strip()
            available = floor_df["Stock Symbol"].unique().tolist()
            symbols = [s for s in symbols if s in available] or available[:st.session_state.n_stocks]
            floor_df = floor_df[floor_df["Stock Symbol"].isin(symbols)]
    else:
        status.info("⚙️ Generating floorsheet data...")
        floor_df = generate_floorsheet(market_df, symbols,
                                       trades_per_stock=250)

    st.session_state.floor_df = floor_df
    progress.progress(55, text="Floorsheet ready — running analysis pipeline...")

    # 4. Prepare normalised column names
    col_rename = {
        "SN": "sn", "Contract No": "contract",
        "Stock Symbol": "symbol",
        "Buyer Broker No": "buyer", "Seller Broker No": "seller",
        "Quantity": "qty", "Rate": "rate", "Amount": "amount",
    }
    floor_norm = floor_df.rename(columns=col_rename)
    for c in ["qty", "rate", "amount"]:
        if c in floor_norm.columns:
            floor_norm[c] = pd.to_numeric(floor_norm[c], errors="coerce")
    if "buyer" in floor_norm.columns:
        floor_norm["buyer"] = floor_norm["buyer"].astype(str).str.strip()
    if "seller" in floor_norm.columns:
        floor_norm["seller"] = floor_norm["seller"].astype(str).str.strip()
    floor_norm = floor_norm.dropna(subset=["qty", "rate"])
    floor_norm = floor_norm[floor_norm["qty"] > 0]

    # 5. Per-symbol analysis
    results = {}
    for i, sym in enumerate(symbols):
        pct_done = 55 + int((i + 1) / len(symbols) * 40)
        progress.progress(pct_done, text=f"Analysing {sym}... ({i+1}/{len(symbols)})")
        sub = floor_norm[floor_norm["symbol"] == sym].reset_index(drop=True)
        res = run_pipeline(sub, sym,
                           max_lags=st.session_state.max_lags,
                           min_run=st.session_state.min_run)
        results[sym] = res

    st.session_state.results  = results
    st.session_state.last_run = datetime.now().strftime("%H:%M:%S")
    progress.progress(100, text="✅ Analysis complete!")
    time.sleep(0.4)
    progress.empty()
    status.empty()


# ── Auto-run on first load ────────────────────────────────────────────────────
if st.session_state.results == {} and not run_btn:
    run_full_pipeline(force_sim=False)

if run_btn:
    run_full_pipeline(force_sim=force_sim)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _score_color(s):
    if s >= 3:   return "#22c55e"
    if s <= -3:  return "#ef4444"
    return "#94a3b8"

def _score_icon(s):
    if s >= 3:  return "🟢"
    if s <= -3: return "🔴"
    return "⚪"

def _fmt_large(n):
    if n >= 1e9:  return f"Rs {n/1e9:.2f}B"
    if n >= 1e6:  return f"Rs {n/1e6:.2f}M"
    if n >= 1e3:  return f"Rs {n/1e3:.1f}K"
    return f"Rs {n:.0f}"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🏔️ NEPSE FloorSheet Intelligence")
st.markdown("##### Fully Automated Institutional Order Flow Detection · Real-time Market Scanning")

if st.session_state.data_source:
    src = st.session_state.data_source
    icon = "🌐 Live" if "simulation" not in src else "🔬 Simulated"
    st.markdown(
        f"<span class='source-tag'>{icon} — {src}</span> &nbsp;"
        f"<span class='source-tag'>📅 {date.today().strftime('%d %b %Y')}</span> &nbsp;"
        f"<span class='source-tag'>🕐 {st.session_state.last_run or 'not yet run'}</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

if not st.session_state.results:
    st.info("Click **Run Full Analysis** in the sidebar to start.")
    st.stop()

market_df  = st.session_state.market_df
scanner_df = st.session_state.scanner_df
results    = st.session_state.results

# ─────────────────────────────────────────────────────────────────────────────
# MARKET SUMMARY ROW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="sec">◆ Market Overview</p>', unsafe_allow_html=True)
msum = market_summary(market_df)

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Advances 📈",  f"{msum['advances']}",
          delta=f"{msum['breadth']:.1f}x breadth")
m2.metric("Declines 📉",  f"{msum['declines']}")
m3.metric("Unchanged ⚪", f"{msum['unchanged']}")
m4.metric("Avg Move",     f"{msum['avg_pct_change']:+.2f}%")
m5.metric("Total Volume", f"{msum['total_volume']:,.0f}")
m6.metric("Turnover",     _fmt_large(msum['total_turnover']))

# Market bubble map
with st.expander("📊 Market Activity Map — click to expand", expanded=False):
    st.pyplot(ch.chart_market_heatmap(market_df), use_container_width=True)
    if msum["top_gainer"] is not None:
        tg = msum["top_gainer"]
        tl = msum["top_loser"]
        ta = msum["top_active"]
        c1,c2,c3 = st.columns(3)
        c1.metric(f"Top Gainer: {tg['symbol']}",
                  f"{tg.get('close',0):,.2f}",
                  delta=f"{tg.get('pct_change',0):+.2f}%")
        c2.metric(f"Top Loser: {tl['symbol']}",
                  f"{tl.get('close',0):,.2f}",
                  delta=f"{tl.get('pct_change',0):+.2f}%")
        c3.metric(f"Most Active: {ta['symbol']}",
                  f"{ta.get('volume',0):,.0f} shares")

# ─────────────────────────────────────────────────────────────────────────────
# SCANNER RESULTS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="sec">◆ Stock Scanner — Top Selected Stocks</p>', unsafe_allow_html=True)

col_scan_chart, col_scan_table = st.columns([1.3, 1])
with col_scan_chart:
    st.pyplot(ch.chart_scanner_bar(scanner_df), use_container_width=True)
with col_scan_table:
    disp_scan = scanner_df[["symbol","close","pct_change","volume_ratio",
                             "turnover","scanner_score","signal"]].copy()
    disp_scan.columns = ["Symbol","Close","Chg%","Vol Ratio","Turnover","Score","Signal"]
    disp_scan["Close"]    = disp_scan["Close"].map("Rs {:,.2f}".format)
    disp_scan["Chg%"]     = disp_scan["Chg%"].map("{:+.2f}%".format)
    disp_scan["Vol Ratio"]= disp_scan["Vol Ratio"].map("{:.2f}x".format)
    disp_scan["Turnover"] = disp_scan["Turnover"].apply(_fmt_large)
    disp_scan["Score"]    = disp_scan["Score"].map("{:.3f}".format)
    st.dataframe(disp_scan, use_container_width=True, hide_index=True)

# Price movement bar
st.pyplot(ch.chart_pct_change(market_df, scanner_df["symbol"].tolist()),
          use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SMART MONEY SCORE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="sec">◆ Smart Money Score — All Analysed Symbols</p>', unsafe_allow_html=True)
st.pyplot(ch.chart_score_matrix(results), use_container_width=True)

# Sector aggregation
with st.expander("📊 Smart Money by Sector", expanded=False):
    st.pyplot(ch.chart_sector_flow(results, market_df), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# BULLISH / BEARISH PANELS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="sec">◆ Top Bullish & Bearish Stocks</p>', unsafe_allow_html=True)

scored = [(sym, r["score"]) for sym, r in results.items() if "score" in r]
scored.sort(key=lambda x: x[1]["score"], reverse=True)

bullish = [(s, sc) for s, sc in scored if sc["score"] >= 0]
bearish = [(s, sc) for s, sc in scored if sc["score"] < 0]

bull_col, bear_col = st.columns(2)

with bull_col:
    st.markdown("#### 🟢 Bullish Stocks")
    if not bullish:
        st.info("No bullish stocks detected.")
    for sym, sc in bullish[:5]:
        mrow = market_df[market_df["symbol"] == sym]
        close_price = mrow.iloc[0]["close"] if len(mrow) > 0 else 0
        pct   = mrow.iloc[0]["pct_change"]  if len(mrow) > 0 else 0
        color = _score_color(sc["score"])
        conf  = sc["conf"]
        if st.button(f"🟢 {sym}  |  Score: {sc['score']:+.1f}  |  {pct:+.2f}%  |  Rs {close_price:,.2f}  |  Conf: {conf}%",
                     key=f"btn_bull_{sym}", use_container_width=True):
            st.session_state.selected_sym = sym
        st.markdown(
            f"<div style='margin:-8px 0 4px 0;padding:3px 10px;"
            f"font-size:.78rem;color:{color};font-family:monospace'>"
            f"{sc['interp']}</div>",
            unsafe_allow_html=True,
        )

with bear_col:
    st.markdown("#### 🔴 Bearish Stocks")
    if not bearish:
        st.info("No bearish stocks detected.")
    for sym, sc in reversed(bearish[-5:]):
        mrow = market_df[market_df["symbol"] == sym]
        close_price = mrow.iloc[0]["close"] if len(mrow) > 0 else 0
        pct   = mrow.iloc[0]["pct_change"]  if len(mrow) > 0 else 0
        if st.button(f"🔴 {sym}  |  Score: {sc['score']:+.1f}  |  {pct:+.2f}%  |  Rs {close_price:,.2f}  |  Conf: {sc['conf']}%",
                     key=f"btn_bear_{sym}", use_container_width=True):
            st.session_state.selected_sym = sym
        st.markdown(
            f"<div style='margin:-8px 0 4px 0;padding:3px 10px;"
            f"font-size:.78rem;color:#ef4444;font-family:monospace'>"
            f"{sc['interp']}</div>",
            unsafe_allow_html=True,
        )

# Quick-select from all analysed stocks
st.markdown('<p class="sec">◆ Select Any Stock for Drill-Down</p>', unsafe_allow_html=True)
all_syms    = list(results.keys())
pick_cols   = st.columns(min(len(all_syms), 8))
for i, sym in enumerate(all_syms):
    sc = results[sym].get("score", {}).get("score", 0)
    col_idx = i % len(pick_cols)
    with pick_cols[col_idx]:
        c = _score_color(sc)
        if st.button(f"{_score_icon(sc)} {sym}\n{sc:+.1f}",
                     key=f"pick_{sym}", use_container_width=True):
            st.session_state.selected_sym = sym

# ══════════════════════════════════════════════════════════════════════════════
# DRILL-DOWN — DETAILED VIEW
# ══════════════════════════════════════════════════════════════════════════════

selected = st.session_state.selected_sym
if selected and selected in results:
    r = results[selected]
    if "error" in r:
        st.error(r["error"])
    else:
        st.markdown("---")
        st.markdown(f"## 🔎 Deep Analysis — <span style='color:#38bdf8'>{selected}</span>",
                    unsafe_allow_html=True)

        # Market data for this stock
        mrow = market_df[market_df["symbol"] == selected]
        if len(mrow) > 0:
            mrow = mrow.iloc[0]
            d1,d2,d3,d4,d5,d6 = st.columns(6)
            d1.metric("Close",       f"Rs {mrow.get('close',0):,.2f}",
                      delta=f"{mrow.get('pct_change',0):+.2f}%")
            d2.metric("Open",        f"Rs {mrow.get('open',0):,.2f}")
            d3.metric("High",        f"Rs {mrow.get('high',0):,.2f}")
            d4.metric("Low",         f"Rs {mrow.get('low',0):,.2f}")
            d5.metric("Volume",      f"{mrow.get('volume',0):,.0f}")
            d6.metric("Turnover",    _fmt_large(mrow.get("turnover",0)))

        # Score + gauge
        st.markdown('<p class="sec">◆ Smart Money Score</p>', unsafe_allow_html=True)
        sc_d = r["score"]
        ga, gb, gc = st.columns([0.9, 1, 1.5])
        with ga:
            st.pyplot(ch.chart_gauge(sc_d["score"], sc_d["color"]), use_container_width=True)
        with gb:
            c = sc_d["color"]
            st.markdown(f"""
            <div class='score-card'>
                <div style='font-size:.68rem;letter-spacing:2px;color:#1e3a5f;text-transform:uppercase'>Smart Money Score</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:3rem;font-weight:700;color:{c};line-height:1'>{sc_d['score']:+.1f}</div>
                <div style='font-size:.9rem;color:{c};margin:8px 0'>{sc_d['icon']} {sc_d['interp']}</div>
                <div style='font-size:.76rem;color:#475569'>Confidence: <b style='color:{c}'>{sc_d['conf']}%</b></div>
            </div>""", unsafe_allow_html=True)
        with gc:
            st.markdown("**Score Components**")
            comp_rows = [
                {"Component": k, "Points": v,
                 "Direction": "▲ Bullish" if v > 0 else ("▼ Bearish" if v < 0 else "— Neutral")}
                for k, v in sc_d["comps"].items()
            ]
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
            pr = r["pressure"]
            st.metric("Buy Pressure", f"{pr['buy_ratio']:.1f}%",
                      delta=f"{pr['buy_ratio']-50:+.1f}% vs neutral")

        # Summary metrics
        st.markdown('<p class="sec">◆ Flow Metrics</p>', unsafe_allow_html=True)
        pr = r["pressure"]
        fm1,fm2,fm3,fm4,fm5 = st.columns(5)
        fm1.metric("Total Trades",    f"{pr['trade_count']:,}")
        fm2.metric("Total Quantity",  f"{pr['total_qty']:,.0f}")
        fm3.metric("Active Brokers",  f"{max(pr['buyers'],pr['sellers'])}")
        fm4.metric("Large Trades",    f"{r['size']['large_count']}")
        fm5.metric("POC Price",       f"Rs {r['poc']:,.2f}")

        # Detail tabs
        dtabs = st.tabs(["🏦 Brokers","📈 ACF","🔍 Metaorders",
                         "📐 Trade Size","📊 Volume Profile",
                         "📉 Intraday Flow","💡 Insights","📁 Export"])

        # ── Brokers ──────────────────────────────────────────────────────────
        with dtabs[0]:
            st.pyplot(ch.chart_broker_activity(r["broker_df"], st.session_state.top_n_broker),
                      use_container_width=True)
            st.pyplot(ch.chart_net_positions(r["broker_df"], st.session_state.top_n_broker * 2),
                      use_container_width=True)
            ba, bb = st.columns(2)
            with ba:
                st.markdown("#### 🟢 Top Accumulators")
                acc = r["broker_df"][r["broker_df"]["net_qty"] > 0].nlargest(8, "net_qty")[
                    ["broker","buy_qty","sell_qty","net_qty","net_pct"]].copy()
                acc.columns = ["Broker","Buy","Sell","Net","Net%"]
                for c in ["Buy","Sell","Net"]:
                    acc[c] = acc[c].map("{:,.0f}".format)
                acc["Net%"] = acc["Net%"].map("{:.1f}%".format)
                st.dataframe(acc, use_container_width=True, hide_index=True)
            with bb:
                st.markdown("#### 🔴 Top Distributors")
                dis = r["broker_df"][r["broker_df"]["net_qty"] < 0].nsmallest(8, "net_qty")[
                    ["broker","buy_qty","sell_qty","net_qty","net_pct"]].copy()
                dis.columns = ["Broker","Buy","Sell","Net","Net%"]
                for c in ["Buy","Sell","Net"]:
                    dis[c] = dis[c].map("{:,.0f}".format)
                dis["Net%"] = dis["Net%"].map("{:.1f}%".format)
                st.dataframe(dis, use_container_width=True, hide_index=True)

        # ── ACF ──────────────────────────────────────────────────────────────
        with dtabs[1]:
            st.pyplot(ch.chart_acf(r["acf"]), use_container_width=True)
            a1,a2,a3 = st.columns(3)
            a1.metric("Mean ACF (5 lags)", f"{r['acf']['mean5']:.4f}")
            a2.metric("Sig. Lags",         str(len(r['acf']['sig'])))
            a3.metric("Pos. Persist ≤5",   str(r['acf']['pos5']))
            if r["acf"]["pos5"] >= 3:
                st.markdown("<div class='flag'>⚠️ <b>Persistent buy-side flow</b> — autocorrelation suggests institutional order-splitting or momentum execution.</div>",
                            unsafe_allow_html=True)
            elif r["acf"]["mean5"] < -0.05:
                st.markdown("<div class='insight'>📊 <b>Negative ACF</b> — alternating direction consistent with market-maker activity.</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='insight'>📊 <b>Near-zero ACF</b> — no persistent directional bias; normal two-sided market.</div>",
                            unsafe_allow_html=True)
            if r["acf"]["sig"]:
                st.info(f"Significant lags: {r['acf']['sig']}")

        # ── Metaorders ────────────────────────────────────────────────────────
        with dtabs[2]:
            meta = r["meta"]
            if len(meta) == 0:
                st.info(f"No metaorders detected (min run = {st.session_state.min_run}).")
            else:
                high = meta[meta["suspicion"] == "HIGH"]
                mc1,mc2,mc3 = st.columns(3)
                mc1.metric("Total Metaorders",   len(meta))
                mc2.metric("HIGH Suspicion",     len(high))
                mc3.metric("MODERATE",           len(meta[meta["suspicion"]=="MODERATE"]))
                if len(high) > 0:
                    st.markdown(
                        f"<div class='flag'>🚨 <b>HIGH-suspicion institutional activity</b> "
                        f"from brokers: <b>{list(high['broker'].unique())}</b> — "
                        f"consistent direction, tight price range, repeated execution.</div>",
                        unsafe_allow_html=True)
                dm = meta.copy()
                dm["total_qty"] = dm["total_qty"].map("{:,.0f}".format)
                dm["avg_rate"]  = dm["avg_rate"].map("{:.2f}".format)
                dm["price_cv"]  = dm["price_cv"].map("{:.2f}%".format)
                dm.columns = ["Broker","Dir","Run Len","Total Qty","Avg Rate","Price CV%","Suspicion"]
                st.dataframe(dm, use_container_width=True, hide_index=True)

        # ── Trade Size ────────────────────────────────────────────────────────
        with dtabs[3]:
            st.pyplot(ch.chart_size_dist(r["size"]), use_container_width=True)
            sz = r["size"]
            ts1,ts2,ts3,ts4 = st.columns(4)
            ts1.metric("Mean Qty",          f"{sz['mean']:,.0f}")
            ts2.metric("Median Qty",        f"{sz['median']:,.0f}")
            ts3.metric("Std Dev",           f"{sz['std']:,.0f}")
            ts4.metric("Large Threshold",   f"{sz['threshold']:,.0f}")
            lvp = sz["large_vol"] / (pr["total_qty"] + 1e-9) * 100
            cls = "flag" if lvp > 20 else "insight"
            em  = "⚠️" if lvp > 20 else "ℹ️"
            st.markdown(
                f"<div class='{cls}'>{em} Large trades = <b>{sz['large_count']}</b> "
                f"({lvp:.1f}% of total qty) {'— possible block/institutional.' if lvp>20 else ''}",
                unsafe_allow_html=True)

        # ── Volume Profile ────────────────────────────────────────────────────
        with dtabs[4]:
            st.pyplot(ch.chart_volume_profile(r["vbp"]), use_container_width=True)
            vbp = r["vbp"]
            if len(vbp) > 0:
                cum = vbp["cum_qty"].max()
                poc_row = vbp.loc[vbp["total_qty"].idxmax()]
                vah_row = vbp[vbp["cum_qty"] >= cum * 0.70].iloc[0]
                val_row = vbp[vbp["cum_qty"] >= cum * 0.30].iloc[0]
                vp1,vp2,vp3 = st.columns(3)
                vp1.metric("POC", f"Rs {poc_row['rate']:,.2f}")
                vp2.metric("VAH", f"Rs {vah_row['rate']:,.2f}")
                vp3.metric("VAL", f"Rs {val_row['rate']:,.2f}")
                for _, vrow in vbp.nlargest(3, "total_qty").iterrows():
                    pct = vrow["total_qty"] / vbp["total_qty"].sum() * 100
                    st.markdown(
                        f"<div class='insight'>🎯 Price <b>Rs {vrow['rate']:,.2f}</b> — "
                        f"Qty: <b>{vrow['total_qty']:,.0f}</b> ({pct:.1f}%) across "
                        f"<b>{vrow['trades']:,}</b> trades.</div>",
                        unsafe_allow_html=True)

        # ── Intraday Flow ─────────────────────────────────────────────────────
        with dtabs[5]:
            st.markdown("**Intraday Price + Cumulative Order Flow Delta**")
            st.pyplot(ch.chart_intraday_flow(r["df"]), use_container_width=True)
            st.caption(
                "Delta = cumulative difference in qty between lower-ID (institutional) brokers buying vs selling. "
                "Sustained positive delta → stealth accumulation.")

        # ── Insights ─────────────────────────────────────────────────────────
        with dtabs[6]:
            st.markdown("### 🧠 Automated Insights")

            # Key broker findings
            bd = r["broker_df"]
            acc_all = bd[bd["net_qty"] > 0]
            dis_all = bd[bd["net_qty"] < 0]
            if len(acc_all) > 0:
                top_a = acc_all.iloc[0]
                st.markdown(
                    f"<div class='insight'>🟢 <b>Primary Accumulator — Broker {top_a['broker']}</b><br>"
                    f"Net buy: <b>{top_a['net_qty']:,.0f}</b> shares ({top_a['net_pct']:.1f}% bias). "
                    f"Buy: {top_a['buy_qty']:,.0f} | Sell: {top_a['sell_qty']:,.0f}</div>",
                    unsafe_allow_html=True)
            if len(dis_all) > 0:
                top_d = dis_all.sort_values("net_qty").iloc[0]
                st.markdown(
                    f"<div class='insight'>🔴 <b>Primary Distributor — Broker {top_d['broker']}</b><br>"
                    f"Net sell: <b>{abs(top_d['net_qty']):,.0f}</b> shares ({abs(top_d['net_pct']):.1f}% bias). "
                    f"Buy: {top_d['buy_qty']:,.0f} | Sell: {top_d['sell_qty']:,.0f}</div>",
                    unsafe_allow_html=True)

            # Flags
            st.markdown("### 🔍 Institutional Activity Flags")
            flags = []
            if r["acf"]["pos5"] >= 3:
                flags.append(("⚠️ Order Flow Persistence",
                               f"Positive ACF at {r['acf']['pos5']}/5 early lags. "
                               "Consistent with institutional order-splitting.", False))
            lvp = r["size"]["large_vol"] / (pr["total_qty"] + 1e-9) * 100
            if lvp > 15:
                flags.append(("⚠️ Abnormal Large Trade Volume",
                               f"{r['size']['large_count']} large trades = {lvp:.1f}% of volume.", False))
            high_meta = r["meta"][r["meta"]["suspicion"] == "HIGH"] if len(r["meta"]) > 0 else pd.DataFrame()
            if len(high_meta) > 0:
                flags.append(("🚨 HIGH-Suspicion Metaorders",
                               f"Brokers {list(high_meta['broker'].unique())} flagged for institutional execution.", True))
            top_share = bd.iloc[0]["total_qty"] / bd["total_qty"].sum() * 100 if len(bd) > 0 else 0
            if top_share > 25:
                flags.append(("⚠️ Broker Concentration",
                               f"Broker {bd.iloc[0]['broker']} holds {top_share:.1f}% of total volume.", False))
            if pr["buy_ratio"] > 70:
                flags.append(("📈 Strong Buy Imbalance", f"Buy ratio {pr['buy_ratio']:.1f}%.", False))
            elif pr["buy_ratio"] < 30:
                flags.append(("📉 Strong Sell Imbalance", f"Buy ratio {pr['buy_ratio']:.1f}%.", False))

            if flags:
                for title, desc, critical in flags:
                    cls = "flag" if critical else "insight"
                    st.markdown(f"<div class='{cls}'><b>{title}</b><br>{desc}</div>",
                                unsafe_allow_html=True)
            else:
                st.markdown("<div class='ok'>✅ No significant institutional activity flags. Market appears normally balanced.</div>",
                            unsafe_allow_html=True)

            # Bias summary
            st.markdown("### 📊 Signal Summary")
            bias_items = [
                ("Buy/Sell Pressure",   f"{pr['buy_ratio']:.1f}% buy",    pr["buy_ratio"] > 55, pr["buy_ratio"] < 45),
                ("Broker Sentiment",    f"{len(acc_all)} acc vs {len(dis_all)} dist", len(acc_all)>len(dis_all), len(dis_all)>len(acc_all)),
                ("ACF Momentum",        f"mean5 = {r['acf']['mean5']:.3f}", r["acf"]["mean5"]>0.05, r["acf"]["mean5"]<-0.05),
                ("Metaorder Bias",      f"Buy {len(r['meta'][r['meta']['direction']=='BUY']) if len(r['meta'])>0 else 0} "
                                         f"| Sell {len(r['meta'][r['meta']['direction']=='SELL']) if len(r['meta'])>0 else 0}",
                 (len(r["meta"][r["meta"]["direction"]=="BUY"]) if len(r["meta"])>0 else 0) >
                 (len(r["meta"][r["meta"]["direction"]=="SELL"]) if len(r["meta"])>0 else 0), False),
                ("Large Trade Impact",  f"{lvp:.1f}% of volume", lvp < 10, lvp > 25),
            ]
            for label, value, is_pos, is_neg in bias_items:
                icon2  = "🟢" if is_pos else ("🔴" if is_neg else "⚪")
                color2 = "#22c55e" if is_pos else ("#ef4444" if is_neg else "#475569")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:9px 14px;"
                    f"background:#080d18;border:1px solid #111d2e;border-radius:6px;margin:3px 0'>"
                    f"<span>{icon2} <b style='color:{color2}'>{label}</b></span>"
                    f"<span style='color:#334155;font-family:monospace;font-size:.82rem'>{value}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

        # ── Export ────────────────────────────────────────────────────────────
        with dtabs[7]:
            st.markdown("### 📥 Export Analysis for " + selected)

            # Build CSV
            buf = io.StringIO()
            w   = csv.writer(buf)
            sc_d2 = r["score"]
            w.writerow(["NEPSE FloorSheet Intelligence — Detail Report"])
            w.writerow([f"Symbol: {selected}", f"Date: {date.today()}", f"Run: {st.session_state.last_run}"])
            w.writerow(["Source", st.session_state.data_source])
            w.writerow([])
            w.writerow(["=== SMART MONEY SCORE ==="])
            w.writerow(["Score", sc_d2["score"], "Interpretation",
                        sc_d2["interp"], "Confidence", f"{sc_d2['conf']}%"])
            for k2, v2 in sc_d2["comps"].items():
                w.writerow([f"  {k2}", v2])
            w.writerow([])
            w.writerow(["=== FLOW SUMMARY ==="])
            pr2 = r["pressure"]
            w.writerow(["Trades", pr2["trade_count"], "Total Qty", f"{pr2['total_qty']:,.0f}",
                        "Buy Ratio%", f"{pr2['buy_ratio']:.1f}"])
            w.writerow([])
            w.writerow(["=== BROKER TABLE ==="])
            w.writerow(["Broker","Buy Qty","Sell Qty","Net Qty","Total Qty","Net%"])
            for _, row in r["broker_df"].head(30).iterrows():
                w.writerow([row["broker"], f"{row['buy_qty']:,.0f}", f"{row['sell_qty']:,.0f}",
                            f"{row['net_qty']:,.0f}", f"{row['total_qty']:,.0f}", f"{row['net_pct']:.1f}"])
            w.writerow([])
            w.writerow(["=== METAORDERS ==="])
            if len(r["meta"]) > 0:
                w.writerow(list(r["meta"].columns))
                for _, mrow in r["meta"].iterrows():
                    w.writerow(list(mrow.values))
            else:
                w.writerow(["None detected"])

            csv_bytes = buf.getvalue().encode("utf-8")

            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                st.download_button("⬇️ Full Report (CSV)", csv_bytes,
                                   f"NEPSE_{selected}_{date.today()}.csv", "text/csv")
            with col_dl2:
                broker_csv = r["broker_df"].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Broker Summary (CSV)", broker_csv,
                                   f"{selected}_brokers.csv", "text/csv")
            with col_dl3:
                if len(r["meta"]) > 0:
                    st.download_button("⬇️ Metaorders (CSV)",
                                       r["meta"].to_csv(index=False).encode("utf-8"),
                                       f"{selected}_metaorders.csv", "text/csv")

            # Floorsheet for this stock
            st.markdown("#### Trade Data Preview")
            show_df = r["df"].drop(columns=["trade_sign"], errors="ignore")
            st.dataframe(show_df, use_container_width=True)

        # Summary export — all stocks
st.markdown('<p class="sec">◆ Export All Stocks Summary</p>', unsafe_allow_html=True)
summary_rows = []
for sym, r in results.items():
    if "score" not in r:
        continue
    sc_d = r["score"]
    pr   = r["pressure"]
    mrow = market_df[market_df["symbol"] == sym]
    summary_rows.append({
        "Symbol":     sym,
        "Score":      sc_d["score"],
        "Signal":     sc_d["interp"],
        "Confidence": f"{sc_d['conf']}%",
        "Trades":     pr["trade_count"],
        "Buy Ratio%": f"{pr['buy_ratio']:.1f}",
        "Large Trades": r["size"]["large_count"],
        "High Meta":  len(r["meta"][r["meta"]["suspicion"]=="HIGH"]) if len(r["meta"])>0 else 0,
        "Close":      mrow.iloc[0]["close"] if len(mrow)>0 else "",
        "Chg%":       mrow.iloc[0]["pct_change"] if len(mrow)>0 else "",
    })
summary_df = pd.DataFrame(summary_rows).sort_values("Score", ascending=False).reset_index(drop=True)
st.dataframe(summary_df, use_container_width=True, hide_index=True)
all_csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download All-Stock Summary (CSV)", all_csv,
                   f"NEPSE_summary_{date.today()}.csv", "text/csv")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#111d2e;font-size:.74rem;
padding:8px 0;font-family:monospace'>
NEPSE FloorSheet Intelligence &nbsp;|&nbsp; Automated Edition &nbsp;|&nbsp;
For research purposes only — not financial advice
</div>""", unsafe_allow_html=True)
