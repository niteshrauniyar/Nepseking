"""
data_engine.py
──────────────
NEPSE Data Ingestion Engine

Live fetch hierarchy:
  1. nepalstock.com.np  (official NEPSE REST API)
  2. merolagani.com     (popular Nepali financial portal)
  3. sharesansar.com    (scrape-based fallback)
  4. Realistic simulation (deterministic for repeatability, seeded on today's date)

Returns standardised DataFrames consumed by the analysis pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── optional live-fetch deps ──────────────────────────────────────────────────
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://nepalstock.com.np/",
}

TIMEOUT = 10  # seconds per request

# ──────────────────────────────────────────────────────────────────────────────
# NEPSE UNIVERSE  (all sectors)
# ──────────────────────────────────────────────────────────────────────────────

NEPSE_UNIVERSE: list[dict] = [
    # Banking
    {"symbol": "NABIL",  "name": "Nabil Bank",                  "sector": "Banking",       "base_price": 1245, "market_cap": "large"},
    {"symbol": "NICA",   "name": "NIC Asia Bank",               "sector": "Banking",       "base_price":  942, "market_cap": "large"},
    {"symbol": "SCB",    "name": "Standard Chartered Bank",     "sector": "Banking",       "base_price":  820, "market_cap": "large"},
    {"symbol": "EBL",    "name": "Everest Bank",                "sector": "Banking",       "base_price":  932, "market_cap": "large"},
    {"symbol": "SBI",    "name": "Nepal SBI Bank",              "sector": "Banking",       "base_price":  578, "market_cap": "mid"},
    {"symbol": "HBL",    "name": "Himalayan Bank",              "sector": "Banking",       "base_price":  680, "market_cap": "mid"},
    {"symbol": "NMB",    "name": "NMB Bank",                    "sector": "Banking",       "base_price":  472, "market_cap": "mid"},
    {"symbol": "GBIME",  "name": "Global IME Bank",             "sector": "Banking",       "base_price":  388, "market_cap": "mid"},
    {"symbol": "ADBL",   "name": "Agricultural Dev. Bank",      "sector": "Banking",       "base_price":  614, "market_cap": "mid"},
    {"symbol": "PRVU",   "name": "Prabhu Bank",                 "sector": "Banking",       "base_price":  342, "market_cap": "small"},
    {"symbol": "BOKL",   "name": "Bank of Kathmandu",           "sector": "Banking",       "base_price":  295, "market_cap": "small"},
    {"symbol": "MEGA",   "name": "Mega Bank",                   "sector": "Banking",       "base_price":  258, "market_cap": "small"},
    {"symbol": "MBL",    "name": "Machhapuchhre Bank",          "sector": "Banking",       "base_price":  278, "market_cap": "small"},
    {"symbol": "KBL",    "name": "Kumari Bank",                 "sector": "Banking",       "base_price":  245, "market_cap": "small"},
    {"symbol": "NBL",    "name": "Nepal Bank",                  "sector": "Banking",       "base_price":  312, "market_cap": "small"},
    # Development Banks
    {"symbol": "SHINE",  "name": "Shine Resunga Dev Bank",      "sector": "Dev Bank",      "base_price":  178, "market_cap": "small"},
    {"symbol": "MNBBL",  "name": "Muktinath Bikas Bank",        "sector": "Dev Bank",      "base_price":  285, "market_cap": "small"},
    {"symbol": "SAPDBL", "name": "Saptagandaki Dev Bank",       "sector": "Dev Bank",      "base_price":  154, "market_cap": "micro"},
    # Finance
    {"symbol": "ICFC",   "name": "ICFC Finance",                "sector": "Finance",       "base_price":  215, "market_cap": "small"},
    {"symbol": "MFIL",   "name": "Manjushree Finance",          "sector": "Finance",       "base_price":  188, "market_cap": "small"},
    {"symbol": "GUFL",   "name": "Goodwill Finance",            "sector": "Finance",       "base_price":  142, "market_cap": "micro"},
    # Insurance
    {"symbol": "NLIC",   "name": "Nepal Life Insurance",        "sector": "Insurance",     "base_price": 1845, "market_cap": "large"},
    {"symbol": "LICN",   "name": "Life Insurance Corp Nepal",   "sector": "Insurance",     "base_price": 1420, "market_cap": "large"},
    {"symbol": "PRIN",   "name": "Premier Insurance",           "sector": "Insurance",     "base_price":  428, "market_cap": "mid"},
    {"symbol": "SGIC",   "name": "Sagarmatha Insurance",        "sector": "Insurance",     "base_price":  342, "market_cap": "small"},
    {"symbol": "NBI",    "name": "Nepal Bangladesh Insurance",  "sector": "Insurance",     "base_price":  289, "market_cap": "small"},
    # Hydropower
    {"symbol": "HIDCL",  "name": "Hydroelectricity Investment", "sector": "Hydropower",    "base_price":  274, "market_cap": "mid"},
    {"symbol": "NHPC",   "name": "Nepal Hydro Power",           "sector": "Hydropower",    "base_price":  198, "market_cap": "small"},
    {"symbol": "RAISC",  "name": "Rairang Hydro Power",         "sector": "Hydropower",    "base_price":  145, "market_cap": "micro"},
    {"symbol": "AHPC",   "name": "Arun Valley Hydropower",      "sector": "Hydropower",    "base_price":  168, "market_cap": "micro"},
    # Microfinance
    {"symbol": "CBBL",   "name": "Chhimek Bikas Bank",          "sector": "Microfinance",  "base_price": 2240, "market_cap": "large"},
    {"symbol": "SWBBL",  "name": "Swabalamban Bikas Bank",      "sector": "Microfinance",  "base_price": 1945, "market_cap": "large"},
    {"symbol": "SLBSL",  "name": "Sana Lagani Bikas",           "sector": "Microfinance",  "base_price": 1648, "market_cap": "mid"},
    # Manufacturing / Others
    {"symbol": "NRIC",   "name": "Nepal Reinsurance",           "sector": "Insurance",     "base_price": 1842, "market_cap": "large"},
    {"symbol": "SHIVM",  "name": "Shivam Cements",              "sector": "Manufacturing", "base_price":  248, "market_cap": "small"},
    {"symbol": "NKEM",   "name": "Nepal Ko Cements",            "sector": "Manufacturing", "base_price":  172, "market_cap": "micro"},
    # Telecom / Tech
    {"symbol": "NTC",    "name": "Nepal Telecom",               "sector": "Telecom",       "base_price":  878, "market_cap": "large"},
    {"symbol": "NIFRA",  "name": "Nepal Infra Bank",            "sector": "Infrastructure","base_price":  238, "market_cap": "small"},
]

# ──────────────────────────────────────────────────────────────────────────────
# HTTP SESSION
# ──────────────────────────────────────────────────────────────────────────────

def _make_session() -> Any:
    if not _REQUESTS_OK:
        return None
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s

# ──────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"

def _cache_valid(path: Path, max_age_minutes: int = 30) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_minutes * 60

def _cache_read(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)

def _cache_write(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f)

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — nepalstock.com.np  (official NEPSE API)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_nepalstock(session) -> pd.DataFrame | None:
    """Try official NEPSE REST API."""
    try:
        url = "https://nepalstock.com.np/api/nots/nepse-data/today-price?size=500&businessDate="
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get("content", data.get("data", []))
        if not items:
            return None
        df = pd.DataFrame(items)
        # Standardise column names
        col_map = {
            "symbol": "symbol", "securityName": "name",
            "lastTradedPrice": "close", "openPrice": "open",
            "highPrice": "high", "lowPrice": "low",
            "totalTradedQuantity": "volume", "totalTradedValue": "turnover",
            "previousClose": "prev_close", "percentageChange": "pct_change",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for c in ["close", "open", "high", "low", "volume", "turnover", "prev_close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["symbol", "close"])
        df["source"] = "nepalstock.com.np"
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — merolagani.com
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_merolagani(session) -> pd.DataFrame | None:
    """Try merolagani live market table."""
    try:
        url = "https://merolagani.com/LatestMarket.aspx"
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code != 200 or not _BS4_OK:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", {"id": "ctl00_ContentPlaceHolder1_LiveTrading1_gridView"})
        if table is None:
            table = soup.find("table", class_="table-bordered")
        if table is None:
            return None
        rows = []
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(dict(zip(headers, cells)))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        rename = {
            "Symbol": "symbol", "LTP": "close", "Change%": "pct_change",
            "Open": "open", "High": "high", "Low": "low",
            "Vol": "volume", "Prev. Close": "prev_close",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for c in ["close", "open", "high", "low", "volume", "prev_close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(",", "").str.replace("%", ""),
                    errors="coerce",
                )
        df = df.dropna(subset=["symbol", "close"])
        df["source"] = "merolagani.com"
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — sharesansar.com
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_sharesansar(session) -> pd.DataFrame | None:
    """Try sharesansar live market data."""
    try:
        url = "https://www.sharesansar.com/live-trading"
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code != 200 or not _BS4_OK:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", {"id": "headFixed"})
        if not table:
            table = soup.find("table", class_="table-bordered")
        if not table:
            return None
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(dict(zip(headers, cells)))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        rename = {
            "Symbol": "symbol", "LTP": "close",
            "Open": "open", "High": "high", "Low": "low",
            "Volume": "volume", "Previous Close": "prev_close",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for c in ["close", "open", "high", "low", "volume", "prev_close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(",", ""), errors="coerce"
                )
        df = df.dropna(subset=["symbol", "close"])
        df["source"] = "sharesansar.com"
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE  (deterministic, seeded on today's date)
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_market_data(regime: str = "auto") -> pd.DataFrame:
    """
    Generate realistic intraday market data for the full NEPSE universe.
    Seeded on today's date so results are consistent within one trading day.
    Uses genuine NEPSE price bases and realistic volatility profiles.
    """
    today_seed = int(date.today().strftime("%Y%m%d"))
    rng = np.random.default_rng(today_seed)

    # Determine market regime from seed
    if regime == "auto":
        r = (today_seed % 10)
        if r < 3:
            regime = "bull"
        elif r < 6:
            regime = "bear"
        else:
            regime = "mixed"

    regime_drift = {"bull": 0.008, "bear": -0.007, "mixed": 0.001}[regime]

    rows = []
    for stock in NEPSE_UNIVERSE:
        bp = stock["base_price"]
        mc = stock["market_cap"]

        # Volatility and volume parameters by market cap tier
        vol_params = {
            "large": (0.018, 1_200_000),
            "mid":   (0.024, 600_000),
            "small": (0.032, 250_000),
            "micro": (0.042, 80_000),
        }
        sigma, base_vol = vol_params.get(mc, (0.025, 200_000))

        # Stock-specific drift (persistent signal)
        stock_hash = int(hashlib.md5(stock["symbol"].encode()).hexdigest(), 16) % 1000
        stock_drift = (stock_hash / 1000 - 0.5) * 0.02

        total_drift = regime_drift + stock_drift
        pct = float(rng.normal(total_drift, sigma))
        pct = float(np.clip(pct, -0.10, 0.10))

        close = round(bp * (1 + pct), 2)
        open_ = round(bp * (1 + float(rng.normal(0, sigma * 0.4))), 2)
        high  = round(max(close, open_) * (1 + abs(float(rng.normal(0, sigma * 0.3)))), 2)
        low   = round(min(close, open_) * (1 - abs(float(rng.normal(0, sigma * 0.3)))), 2)

        # Volume: spikes on large moves
        vol_mult = 1 + abs(pct) * 15 + float(rng.exponential(0.3))
        volume   = int(base_vol * vol_mult * float(rng.lognormal(0, 0.4)))
        turnover = round(volume * close, 2)

        # 30-day average volume (simulated)
        avg_vol_30 = int(base_vol * float(rng.lognormal(0, 0.2)))

        rows.append({
            "symbol":        stock["symbol"],
            "name":          stock["name"],
            "sector":        stock["sector"],
            "close":         close,
            "open":          open_,
            "high":          high,
            "low":           low,
            "prev_close":    bp,
            "pct_change":    round(pct * 100, 2),
            "volume":        volume,
            "avg_volume_30": avg_vol_30,
            "turnover":      turnover,
            "market_cap":    mc,
            "source":        "simulation",
        })

    df = pd.DataFrame(rows)
    df["volume_ratio"] = (df["volume"] / df["avg_volume_30"].replace(0, 1)).round(2)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC: fetch_market_data
# ──────────────────────────────────────────────────────────────────────────────

def fetch_market_data(force_simulate: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Fetch today's NEPSE market data.

    Returns (df, source_label) where source_label describes how data was obtained.
    The returned DataFrame always contains:
      symbol, name, sector, close, open, high, low, prev_close,
      pct_change, volume, avg_volume_30, turnover, volume_ratio, source
    """
    cache_key = f"market_{date.today().isoformat()}"
    cache_file = _cache_path(cache_key)

    # Return cached data if fresh (within 30 min)
    if not force_simulate and _cache_valid(cache_file, 30):
        try:
            cached = _cache_read(cache_file)
            df = pd.DataFrame(cached["rows"])
            return df, f"cached ({cached.get('source','?')})"
        except Exception:
            pass

    if force_simulate or not _REQUESTS_OK:
        df = _simulate_market_data()
        _cache_write(cache_file, {"rows": df.to_dict("records"), "source": "simulation"})
        return df, "simulation"

    session = _make_session()
    sources_tried = []

    for fetcher, label in [
        (_fetch_nepalstock,  "nepalstock.com.np"),
        (_fetch_merolagani,  "merolagani.com"),
        (_fetch_sharesansar, "sharesansar.com"),
    ]:
        sources_tried.append(label)
        df_live = fetcher(session)
        if df_live is not None and len(df_live) > 10:
            # Enrich with universe metadata
            universe_df = pd.DataFrame(NEPSE_UNIVERSE).rename(
                columns={"base_price": "base_price_ref"}
            )
            df_live = df_live.merge(
                universe_df[["symbol", "name", "sector", "market_cap", "base_price_ref"]],
                on="symbol", how="left"
            )
            # Fill missing computed columns
            if "pct_change" not in df_live.columns and "prev_close" in df_live.columns:
                df_live["pct_change"] = (
                    (df_live["close"] - df_live["prev_close"])
                    / df_live["prev_close"].replace(0, np.nan) * 100
                ).round(2)
            if "avg_volume_30" not in df_live.columns:
                # Estimate from universe base volumes
                df_live["avg_volume_30"] = df_live["volume"] * np.random.uniform(0.7, 1.3, len(df_live))
            df_live["volume_ratio"] = (
                df_live["volume"] / df_live["avg_volume_30"].replace(0, 1)
            ).round(2)
            if "turnover" not in df_live.columns:
                df_live["turnover"] = (df_live["close"] * df_live["volume"]).round(2)

            _cache_write(cache_file, {"rows": df_live.to_dict("records"), "source": label})
            return df_live, label

    # All live sources failed → simulate
    df = _simulate_market_data()
    _cache_write(cache_file, {"rows": df.to_dict("records"), "source": "simulation"})
    return df, f"simulation (tried: {', '.join(sources_tried)})"


# ──────────────────────────────────────────────────────────────────────────────
# STOCK SCANNER
# ──────────────────────────────────────────────────────────────────────────────

def scan_top_stocks(
    market_df: pd.DataFrame,
    n: int = 10,
    min_volume: int = 1000,
    require_both_directions: bool = False,
) -> pd.DataFrame:
    """
    Score and rank every stock by composite activity score.

    Scoring (each 0–1 normalised, then weighted):
      40%  Volume ratio  (today vs 30-day avg)
      30%  |Price move|  (abs % change)
      30%  Turnover rank (log-normalised)
    
    Returns top-n with added columns:
      scanner_score, scanner_rank, signal (BULLISH/BEARISH/NEUTRAL)
    """
    df = market_df.copy()

    # Filter minimum activity
    if "volume" in df.columns:
        df = df[df["volume"] >= min_volume]

    if len(df) == 0:
        return pd.DataFrame()

    # Fill missing columns gracefully
    if "volume_ratio" not in df.columns:
        df["volume_ratio"] = 1.0
    if "pct_change" not in df.columns:
        df["pct_change"] = 0.0
    if "turnover" not in df.columns:
        df["turnover"] = df.get("volume", 0) * df.get("close", 1)

    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    df["_s_vol"]      = _norm(df["volume_ratio"])
    df["_s_move"]     = _norm(df["pct_change"].abs())
    df["_s_turnover"] = _norm(np.log1p(df["turnover"]))

    df["scanner_score"] = (
        0.40 * df["_s_vol"]
        + 0.30 * df["_s_move"]
        + 0.30 * df["_s_turnover"]
    ).round(4)

    df = df.sort_values("scanner_score", ascending=False).reset_index(drop=True)
    df["scanner_rank"] = range(1, len(df) + 1)

    # Directional signal
    df["signal"] = "NEUTRAL"
    df.loc[df["pct_change"] >  1.5, "signal"] = "BULLISH"
    df.loc[df["pct_change"] < -1.5, "signal"] = "BEARISH"

    top = df.drop(columns=["_s_vol", "_s_move", "_s_turnover"]).head(n)
    return top.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# FLOORSHEET GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_floorsheet(
    market_df: pd.DataFrame,
    symbols: list[str],
    trades_per_stock: int = 200,
) -> pd.DataFrame:
    """
    Generate realistic intraday floorsheet data for the given symbols.
    Uses today's market data (price, volume, direction) to create
    statistically consistent trade records.

    The generator deliberately embeds institutional patterns:
      - Large brokers (IDs 1-50)  appear more often in directional runs
      - Retail brokers (IDs 51-150) generate the baseline noise
    """
    today_seed = int(date.today().strftime("%Y%m%d"))
    rng = np.random.default_rng(today_seed + 1)

    INSTITUTIONAL_BROKERS = list(range(1, 51))
    RETAIL_BROKERS        = list(range(51, 151))
    ALL_BROKERS           = INSTITUTIONAL_BROKERS + RETAIL_BROKERS

    records = []
    sn = 1

    for sym in symbols:
        row = market_df[market_df["symbol"] == sym]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        price      = float(row.get("close",  row.get("base_price_ref", 500)))
        prev_close = float(row.get("prev_close", price))
        pct        = float(row.get("pct_change", 0)) / 100
        vol        = int(row.get("volume", trades_per_stock * 300))

        # Calibrate number of trades to volume
        n_trades = max(50, min(trades_per_stock, int(vol / 200)))

        # Pick institutional actors for this stock
        inst_buyer  = int(rng.choice(INSTITUTIONAL_BROKERS[:20]))
        inst_seller = int(rng.choice(INSTITUTIONAL_BROKERS[20:40]))
        while inst_seller == inst_buyer:
            inst_seller = int(rng.choice(INSTITUTIONAL_BROKERS[20:40]))

        # Secondary institutions
        sec_buyers  = rng.choice(INSTITUTIONAL_BROKERS[:30], size=3, replace=False).tolist()
        sec_sellers = rng.choice(INSTITUTIONAL_BROKERS[15:45], size=3, replace=False).tolist()

        # Intraday price walk
        sigma_intra = abs(price * 0.001)
        p = prev_close

        for i in range(n_trades):
            p = float(np.clip(
                p + rng.normal(pct * price / n_trades, sigma_intra),
                price * 0.94, price * 1.06
            ))
            p = round(p, 2)

            rand = float(rng.random())

            # ── Institutional patterns ────────────────────────────────
            if pct > 0.02 and rand < 0.18:
                # Accumulation run: lead inst buyer
                buyer  = inst_buyer
                seller = int(rng.choice(RETAIL_BROKERS))
                qty    = int(rng.normal(vol / n_trades * 1.8, vol / n_trades * 0.2))
            elif pct < -0.02 and rand < 0.18:
                # Distribution run: lead inst seller
                buyer  = int(rng.choice(RETAIL_BROKERS))
                seller = inst_seller
                qty    = int(rng.normal(vol / n_trades * 1.8, vol / n_trades * 0.2))
            elif rand < 0.06:
                # Block trade
                buyer  = int(rng.choice(sec_buyers))
                seller = int(rng.choice(sec_sellers))
                qty    = int(rng.normal(vol / n_trades * 6, vol / n_trades * 1))
            elif rand < 0.12:
                # Secondary institutional
                buyer  = int(rng.choice(sec_buyers if pct >= 0 else RETAIL_BROKERS))
                seller = int(rng.choice(sec_sellers if pct < 0 else RETAIL_BROKERS))
                qty    = int(rng.normal(vol / n_trades * 1.3, vol / n_trades * 0.3))
            else:
                # Retail flow
                buyer  = int(rng.choice(RETAIL_BROKERS))
                seller = int(rng.choice(RETAIL_BROKERS))
                while seller == buyer:
                    seller = int(rng.choice(RETAIL_BROKERS))
                qty = max(1, int(rng.exponential(vol / n_trades * 0.8)))

            qty    = max(1, qty)
            amount = round(qty * p, 2)

            records.append({
                "SN":               sn,
                "Contract No":      f"C{sn:07d}",
                "Stock Symbol":     sym,
                "Buyer Broker No":  str(buyer),
                "Seller Broker No": str(seller),
                "Quantity":         qty,
                "Rate":             p,
                "Amount":           amount,
            })
            sn += 1

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MARKET SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
def market_summary(market_df):
    """High-level market statistics (robust version)."""
    if market_df is None or market_df.empty:
        return {
            "advances": 0,
            "declines": 0,
            "unchanged": 0,
            "total_stocks": 0,
            "total_volume": 0,
            "total_turnover": 0.0,
            "avg_pct_change": 0.0,
            "top_gainer": None,
            "top_loser": None,
            "top_active": None,
            "breadth": 0.0,
        }

    df = market_df.copy()

    # Clean data
    df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce").fillna(0)

    # Market breadth
    advances  = (df["pct_change"] > 0).sum()
    declines  = (df["pct_change"] < 0).sum()
    unchanged = (df["pct_change"] == 0).sum()

    total_vol = df["volume"].sum()
    total_to  = df["turnover"].sum()
    avg_move  = df["pct_change"].mean()

    # Top movers (return clean dict instead of Series)
    def extract(row):
        if row is None:
            return None
        return {
            "symbol": row.get("symbol"),
            "pct_change": round(float(row.get("pct_change", 0)), 2),
            "volume": int(row.get("volume", 0)),
        }

    top_gainer = extract(df.loc[df["pct_change"].idxmax()]) if not df.empty else None
    top_loser  = extract(df.loc[df["pct_change"].idxmin()]) if not df.empty else None
    top_active = extract(df.loc[df["volume"].idxmax()])     if not df.empty else None

    # Better breadth logic
    breadth = round(advances / declines, 2) if declines > 0 else float(advances)

    return {
        "advances": int(advances),
        "declines": int(declines),
        "unchanged": int(unchanged),
        "total_stocks": len(df),
        "total_volume": int(total_vol),
        "total_turnover": float(total_to),
        "avg_pct_change": round(float(avg_move), 2),
        "top_gainer": top_gainer,
        "top_loser": top_loser,
        "top_active": top_active,
        "breadth": breadth,
    }
