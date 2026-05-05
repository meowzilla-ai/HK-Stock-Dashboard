#!/usr/bin/env python3
"""HK stock analysis - technical + fundamental report generator.

Usage:
  python3 analyze.py [STOCK_CODE]

  STOCK_CODE: Futu-format code e.g. HK.00700 (default: HK.09988)

MA strategy:
  Short (daily)  : EMA9, EMA20, EMA50  + SMA200 (institutional reference)
  Medium (weekly): EMA9, EMA20, EMA50  + SMA200 (structural reference)
  Long (weekly)  : SMA20, SMA50, SMA200

Analyst commentary is added post-generation by an AI agent directly into the HTML file.
Run the script once, then ask an AI agent to read the HTML, extract signal data, and
replace the <!-- ANALYST_VIEW:xxx --> placeholders with formatted commentary blocks.
"""

import os
import sys
import time
import math
import json
from html import escape as html_escape
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from futu import *
except ImportError:
    print("futu-api not found")
    sys.exit(1)

def normalize_stock_code(raw):
    """Accept 3690, 03690, HK.03690 or HK:03690 and return Futu HK.03690."""
    code = str(raw).strip().upper()
    if code.startswith("HK:"):
        code = "HK." + code[3:]
    elif code.startswith("HK"):
        code = code[2:].lstrip(".:")

    if code.startswith("HK."):
        digits = code[3:]
    else:
        digits = code

    if not digits.isdigit() or len(digits) > 5:
        raise ValueError(f"Invalid HK stock code: {raw!r}")
    return f"HK.{digits.zfill(5)}"


try:
    STOCK_CODE = normalize_stock_code(sys.argv[1] if len(sys.argv) > 1 else "HK.09988")
except ValueError as e:
    print(e)
    print("Use a Hong Kong stock code like 3690, 03690, HK.03690, or HK:03690.")
    sys.exit(2)
HOST = "127.0.0.1"
PORT = 11111
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Chart display windows
SHORT_CHART_BARS  = 120   # daily sessions shown on short-term charts
MED_CHART_BARS    = 104   # weekly bars shown on medium-term charts (2 years)
LONG_WEEKLY_BARS  = 156   # weekly bars shown on long-term charts (3 years)
TRADING_DAYS_YEAR = 252   # approximate trading days per year (for 52-week range)

# ── Analyst commentary ────────────────────────────────────────────────────────
# Commentary is added by an AI agent directly into the generated HTML file.
# Workflow:
#   1. Run this script → HTML generated with <!-- ANALYST_VIEW:xxx --> placeholders
#   2. Ask an AI agent to read the HTML, extract signal data, draft commentary
#   3. AI edits the HTML file directly, replacing each placeholder with the
#      formatted <div class="analyst-note"> block (see commentary_placeholder())
# No second run of this script is needed.


# ── math helpers ──────────────────────────────────────────────────────────────

def sma(series, n):
    return series.rolling(window=n).mean()


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(series, fast=12, slow=26, sig=9):
    ml = ema(series, fast) - ema(series, slow)
    sl = ema(ml, sig)
    return ml, sl, ml - sl


def bollinger(series, n=20, k=2):
    mid = sma(series, n)
    std = series.rolling(window=n).std()
    return mid + k * std, mid, mid - k * std


def atr(df, n=14):
    """Average True Range — average absolute price movement over n bars."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    """Average Directional Index — measures trend strength (not direction).
    Returns (adx_line, +DI, -DI).
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index
    )

    atr_s    = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_s

    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx_line, plus_di, minus_di


# ── indicator loading ─────────────────────────────────────────────────────────

def add_ema_indicators(df):
    """EMA-based MAs for short/medium term + shared oscillators."""
    c = df["close"]
    df["ma1"]  = ema(c, 9)    # fast
    df["ma2"]  = ema(c, 20)   # mid
    df["ma3"]  = ema(c, 50)   # slow
    df["ma4"]  = sma(c, 200)  # SMA200 kept as institutional reference (daily only)
    df["rsi14"] = rsi(c, 14)
    df["atr14"] = atr(df, 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger(c)
    df["adx14"], df["plus_di"], df["minus_di"] = adx(df)
    df["vol_avg20"] = df["volume"].rolling(window=20).mean()
    return df


def add_sma_indicators(df):
    """SMA-based MAs for long term + shared oscillators."""
    c = df["close"]
    df["ma1"]  = sma(c, 20)
    df["ma2"]  = sma(c, 50)
    df["ma3"]  = sma(c, 200)
    df["ma4"]  = pd.Series([np.nan] * len(df), index=df.index)  # unused
    df["rsi14"] = rsi(c, 14)
    df["atr14"] = atr(df, 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger(c)
    df["adx14"], df["plus_di"], df["minus_di"] = adx(df)
    df["vol_avg20"] = df["volume"].rolling(window=20).mean()
    return df


# ── data fetching ─────────────────────────────────────────────────────────────

def fetch_kline(quote_ctx, code, ktype, count=500, start="2000-01-01"):
    end = datetime.now().strftime("%Y-%m-%d")
    pages = []
    page_key = None
    while True:
        kwargs = dict(ktype=ktype, start=start, end=end,
                      max_count=1000, fields=[KL_FIELD.ALL])
        if page_key:
            kwargs["page_req_key"] = page_key
        ret, df, page_key = quote_ctx.request_history_kline(code, **kwargs)
        if ret != RET_OK:
            raise RuntimeError(f"request_history_kline failed: {df}")
        pages.append(df)
        if page_key is None:
            break
    df = pd.concat(pages, ignore_index=True)
    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df.sort_values("time_key").reset_index(drop=True)
    # keep the most recent `count` bars for daily/weekly to limit memory
    if len(df) > count:
        df = df.tail(count).reset_index(drop=True)
    return df


def fmt_large(v):
    try:
        f = float(v)
        if f >= 1e12: return f"HKD {f/1e12:.2f}T"
        if f >= 1e9:  return f"HKD {f/1e9:.2f}B"
        if f >= 1e6:  return f"HKD {f/1e6:.2f}M"
        return str(f)
    except Exception:
        return str(v)


def fetch_capital_flow(quote_ctx, code):
    """Fetch today's capital distribution (inflow/outflow by order size)."""
    result = {}
    ret, df = quote_ctx.get_capital_distribution(code)
    if ret != RET_OK or df.empty:
        return result
    row = df.iloc[0]
    cats = ["super", "big", "mid", "small"]
    total_in = total_out = inst_net = retail_net = 0
    for cat in cats:
        ci = float(row.get(f"capital_in_{cat}", 0) or 0)
        co = float(row.get(f"capital_out_{cat}", 0) or 0)
        total_in  += ci
        total_out += co
        if cat in ("super", "big"):
            inst_net   += ci - co
        else:
            retail_net += ci - co
    result["total_net"]   = total_in - total_out
    result["inst_net"]    = inst_net
    result["retail_net"]  = retail_net
    result["update_time"] = str(row.get("update_time", ""))
    return result


def fetch_fundamental(quote_ctx, code):
    data = {}
    ret, snap = quote_ctx.get_market_snapshot([code])
    if ret != RET_OK or snap.empty:
        return data
    s = snap.iloc[0]

    def safe(key, fmt=None):
        v = s.get(key, None)
        if v is None: return "—"
        try:
            if isinstance(v, float) and math.isnan(v): return "—"
        except Exception:
            pass
        if fmt == "large": return fmt_large(v)
        if fmt == "pct":
            try: return f"{float(v):.2f}%"
            except Exception: pass
        if fmt == "2f":
            try: return f"{float(v):.2f}"
            except Exception: pass
        return str(v)

    data["Stock Name"]              = safe("name")
    data["Last Price (HKD)"]        = safe("last_price", "2f")
    data["Previous Close (HKD)"]    = safe("prev_close_price", "2f")
    data["52-Week High (HKD)"]      = safe("highest52weeks_price", "2f")
    data["52-Week Low (HKD)"]       = safe("lowest52weeks_price", "2f")
    data["All-Time High (HKD)"]     = safe("highest_history_price", "2f")
    data["Market Cap"]              = safe("total_market_val", "large")
    data["Issued Shares"]           = safe("issued_shares")
    data["P/E Ratio (Static)"]      = safe("pe_ratio", "2f")
    data["P/E Ratio (TTM)"]         = safe("pe_ttm_ratio", "2f")
    data["P/B Ratio"]               = safe("pb_ratio", "2f")
    data["Earnings Yield"]          = safe("ey_ratio", "pct")
    data["EPS (HKD)"]               = safe("earning_per_share", "2f")
    data["Net Asset/Share (HKD)"]   = safe("net_asset_per_share", "2f")
    data["Net Profit (HKD)"]        = safe("net_profit", "large")
    data["Net Assets (HKD)"]        = safe("net_asset", "large")
    data["Dividend Yield (TTM)"]    = safe("dividend_ratio_ttm", "pct")
    data["Dividend Per Share (TTM)"]= safe("dividend_ttm", "2f")
    data["Listing Date"]            = safe("listing_date")
    data["Lot Size"]                = safe("lot_size")
    # Volume fields for stat cards
    data["_volume"]       = s.get("volume", 0)
    data["_volume_ratio"] = s.get("volume_ratio", None)
    data["_turnover"]     = s.get("turnover", 0)
    return data


# ── signal summary ────────────────────────────────────────────────────────────

def macd_momentum_label(hist_series, n_confirm):
    """
    Compute MACD histogram momentum label with directional arrow prefix.

    hist_series : full pandas Series of macd_hist (for 20-bar avg magnitude)
    n_confirm   : 5 = daily (4-of-5 majority rule)
                  3 = weekly (all-3-agree rule)

    Priority order:
      1. Zero-cross detected in window  → crossover label
      2. Direction assessed over window → strengthening / building / weakening
      3. Near-zero magnitude check      → append "· X-over soon"
      4. Marginal magnitude check       → append "(marginal)"
      5. Mixed / unclear                → consolidating
    """
    needed = n_confirm + 1          # +1 boundary bar for zero-cross detection
    if len(hist_series) < needed:
        return "→ Consolidating · insufficient data"

    bars    = hist_series.iloc[-needed:].tolist()
    current = bars[-1]

    # 20-bar average absolute value — self-calibrating magnitude baseline
    avg_abs          = hist_series.iloc[-20:].abs().mean() if len(hist_series) >= 20 else (abs(current) or 1)
    near_zero_thresh = 0.20 * avg_abs   # within 20% of avg → crossover imminent
    marginal_thresh  = 0.10 * avg_abs   # within 10% of avg → noise-level move

    near_zero = abs(current) < near_zero_thresh
    marginal  = abs(current) < marginal_thresh
    hist_str  = f"{current:.2f}"

    def _lbl(arrow, text, marg=False):
        s = f"{arrow} {text} · {hist_str}"
        return s + " (marginal)" if marg else s

    # Step 1: Zero-cross (highest priority) — scan full window including boundary bar
    for i in range(1, len(bars)):
        if bars[i - 1] < 0 and bars[i] >= 0:
            return f"↑ Momentum crossover — bullish · {hist_str}"
        if bars[i - 1] >= 0 and bars[i] < 0:
            return f"↓ Momentum crossover — bearish · {hist_str}"

    # Step 2: Direction over confirmation window
    confirm = bars[-n_confirm:]
    diffs   = [confirm[i] - confirm[i - 1] for i in range(1, len(confirm))]
    n_d     = len(diffs)          # n_confirm - 1 comparisons
    rising  = sum(1 for d in diffs if d > 0)
    falling = n_d - rising

    if n_confirm == 5:
        # Daily: 4-of-5 bars majority → 3-of-4 diffs
        thresh = n_d - 1   # 3 out of 4
        if current > 0:
            if rising == n_d:        return _lbl("↑↑", "Strengthening bullish", marginal)
            if rising >= thresh:     return _lbl("↑",  "Bullish momentum building", marginal)
            if falling >= thresh:
                text = "Weakening bullish · X-over soon" if near_zero else "Weakening bullish"
                return _lbl("↓", text, marginal)
        elif current < 0:
            if falling == n_d:       return _lbl("↓↓", "Strengthening bearish", marginal)
            if falling >= thresh:    return _lbl("↓",  "Bearish momentum building", marginal)
            if rising >= thresh:
                text = "Weakening bearish · X-over soon" if near_zero else "Weakening bearish"
                return _lbl("↑", text, marginal)
    else:
        # Weekly: all-3-agree → both diffs must match
        if current > 0:
            if rising == n_d:        return _lbl("↑↑", "Strengthening bullish", marginal)
            if falling == n_d:
                text = "Weakening bullish · X-over soon" if near_zero else "Weakening bullish"
                return _lbl("↓", text, marginal)
        elif current < 0:
            if falling == n_d:       return _lbl("↓↓", "Strengthening bearish", marginal)
            if rising == n_d:
                text = "Weakening bearish · X-over soon" if near_zero else "Weakening bearish"
                return _lbl("↑", text, marginal)

    return _lbl("→", "Consolidating")


def signal_summary(df, use_ema=True, include_ma4=False, mom_df=None, macd_confirm=3,
                   use_mid_cross=False, include_volume=False):
    """
    df             → source for MA, Bollinger, ADX, volume and close price signals
    mom_df         → optional separate source for RSI/MACD (e.g. weekly df for long-term)
    use_ema        → True: EMA labels (daily/weekly), False: SMA labels (monthly)
    macd_confirm   → confirmation bars for MACD momentum (5=daily, 3=weekly)
    use_mid_cross  → True: crossover uses n2 vs n3 (EMA20/EMA50 for medium-term)
                     False: crossover uses n1 vs n2 (EMA9/EMA20 for short-term)
    include_volume → True: add volume confirmation signal (short + medium term only)
    """
    last, prev = df.iloc[-1], df.iloc[-2]
    c = last["close"]
    sigs = {}

    if use_ema:
        n1, n2, n3 = "EMA9", "EMA20", "EMA50"
    else:
        n1, n2, n3 = "SMA20", "SMA50", "SMA200"

    def _ma_sig(price, ma_val, above_label, below_label, name):
        """Build a signal string for price vs MA with gap %, handling NaN gracefully."""
        try:
            f = float(ma_val)
            if math.isnan(f):
                return f"N/A (insufficient history) · {name}: N/A"
            label = above_label if price > f else below_label
            gap = (price - f) / f * 100
            gap_str = f"+{gap:.1f}%" if gap >= 0 else f"{gap:.1f}%"
            return f"{label} · {name}: {f:.2f} ({gap_str})"
        except Exception:
            return f"N/A · {name}: N/A"

    ma1_v = float(last["ma1"])
    ma2_v = float(last["ma2"])
    ma3_v = float(last["ma3"])
    sigs[f"Price vs {n2}"] = _ma_sig(c, ma2_v, "Above", "Below", n2)
    sigs[f"Price vs {n3}"] = _ma_sig(c, ma3_v, "Above", "Below", n3)
    if include_ma4 and not math.isnan(float(last["ma4"])):
        ma4_v = float(last["ma4"])
        if use_mid_cross:
            # Weekly context: SMA200 = structural bull/bear regime reference
            sigs["Price vs SMA200 (Wkly)"] = _ma_sig(c, ma4_v, "Above — Structural Bull", "Below — Structural Bear", "SMA200")
        else:
            # Daily context: SMA200 = golden zone reference
            sigs["Price vs SMA200"] = _ma_sig(c, ma4_v, "Above (Golden Zone)", "Below (Bearish)", "SMA200")
    if use_mid_cross:
        # Medium-term: EMA20 vs EMA50 — has the multi-week trend shifted?
        if math.isnan(ma2_v) or math.isnan(ma3_v):
            sigs[f"{n2} vs {n3}"] = f"N/A (insufficient history) · {n2}: N/A / {n3}: N/A"
        else:
            cross_lbl = "Bullish crossover" if ma2_v > ma3_v else "Bearish crossover"
            sigs[f"{n2} vs {n3}"] = f"{cross_lbl} · {n2}: {ma2_v:.2f} / {n3}: {ma3_v:.2f}"
    else:
        # Short-term: EMA9 vs EMA20 — short swing momentum
        if math.isnan(ma1_v) or math.isnan(ma2_v):
            sigs[f"{n1} vs {n2}"] = f"N/A (insufficient history) · {n1}: N/A / {n2}: N/A"
        else:
            cross_lbl = "Bullish crossover" if ma1_v > ma2_v else "Bearish crossover"
            sigs[f"{n1} vs {n2}"] = f"{cross_lbl} · {n1}: {ma1_v:.2f} / {n2}: {ma2_v:.2f}"

    # RSI & MACD: use mom_df if provided, otherwise fall back to df
    m_last = mom_df.iloc[-1] if mom_df is not None else last
    m_prev = mom_df.iloc[-2] if mom_df is not None else prev
    mom_label = " (Weekly)" if mom_df is not None else ""

    r = m_last["rsi14"]
    if r > 70:   sigs[f"RSI(14){mom_label}"] = f"{r:.1f} — Overbought"
    elif r < 30: sigs[f"RSI(14){mom_label}"] = f"{r:.1f} — Oversold"
    else:        sigs[f"RSI(14){mom_label}"] = f"{r:.1f} — Neutral"

    ml, ms = m_last["macd_line"], m_last["macd_signal"]
    sigs[f"MACD{mom_label}"] = (f"{'Bullish (MACD > Signal)' if ml > ms else 'Bearish (MACD < Signal)'} · {ml:.2f} / {ms:.2f}")

    mom_hist = mom_df["macd_hist"] if mom_df is not None else df["macd_hist"]
    sigs[f"MACD Momentum{mom_label}"] = macd_momentum_label(mom_hist, macd_confirm)

    # ── ADX (computed before BB so its context is available for BB label) ──────
    adx_v = float(last["adx14"])
    pdi   = float(last["plus_di"])
    mdi   = float(last["minus_di"])
    adx_trending = adx_v > 25
    if adx_v > 25:
        direction = "Bullish trend" if pdi > mdi else "Bearish trend"
        sigs["ADX(14)"] = f"{adx_v:.1f} — {direction} (trending) · +DI: {pdi:.1f} / -DI: {mdi:.1f}"
    elif adx_v > 20:
        direction = "bullish bias" if pdi > mdi else "bearish bias"
        sigs["ADX(14)"] = f"{adx_v:.1f} — Weakening trend ({direction}) · +DI: {pdi:.1f} / -DI: {mdi:.1f}"
    else:
        sigs["ADX(14)"] = f"{adx_v:.1f} — Ranging / no clear trend · +DI: {pdi:.1f} / -DI: {mdi:.1f}"

    # ── Bollinger Bands (ADX-context-aware + squeeze detection) ───────────────
    bu  = float(last["bb_upper"])
    bl  = float(last["bb_lower"])
    bm  = float(last["bb_mid"])
    bw_series = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    bw_cur    = float(bw_series.iloc[-1])
    bw_avg20  = float(bw_series.iloc[-20:].mean()) if len(bw_series) >= 20 else bw_cur

    # Step 1: Squeeze — volatility contraction (highest priority)
    if bw_cur < 0.50 * bw_avg20:
        sigs["Bollinger Bands"] = (f"⚡ Squeeze — volatility contraction · "
                                   f"BW: {bw_cur:.1f}% (avg: {bw_avg20:.1f}%)")
    # Step 2: Price outside bands — interpret using ADX context
    elif c > bu:
        if adx_trending:
            sigs["Bollinger Bands"] = f"Band walking upper — trend continuation (ADX {adx_v:.1f}) · Upper: {bu:.2f}"
        else:
            sigs["Bollinger Bands"] = f"Above upper — overbought in range (ADX {adx_v:.1f}) · Upper: {bu:.2f}"
    elif c < bl:
        if adx_trending:
            sigs["Bollinger Bands"] = f"Band walking lower — trend continuation (ADX {adx_v:.1f}) · Lower: {bl:.2f}"
        else:
            sigs["Bollinger Bands"] = f"Below lower — oversold in range (ADX {adx_v:.1f}) · Lower: {bl:.2f}"
    # Step 3: Within bands — show which half and bandwidth
    else:
        half = "Upper half — mild bullish bias" if c > bm else "Lower half — mild bearish bias"
        sigs["Bollinger Bands"] = f"{half} · BW: {bw_cur:.1f}% · Upper: {bu:.2f} / Lower: {bl:.2f}"

    # ── Volume confirmation (short + medium term only) ─────────────────────────
    if include_volume:
        try:
            v_cur = float(last["volume"])
            v_avg = float(last["vol_avg20"])
            if v_avg > 0 and not math.isnan(v_avg):
                ratio = v_cur / v_avg
                is_up = float(last["close"]) >= float(last["open"])
                if ratio >= 1.2:
                    if is_up:
                        sigs["Volume"] = f"Bullish — up close, high volume (×{ratio:.2f})"
                    else:
                        sigs["Volume"] = f"Bearish — down close, high volume (×{ratio:.2f})"
                else:
                    sigs["Volume"] = f"Neutral — average/low volume (×{ratio:.2f})"
            else:
                sigs["Volume"] = "N/A (insufficient history)"
        except Exception:
            sigs["Volume"] = "N/A"

    return sigs


# ── HTML helpers ──────────────────────────────────────────────────────────────

def h(value):
    return html_escape(str(value), quote=True)


def tag(text):
    t = text.lower()
    # Neutral override first — catches weakening/consolidating/squeeze/unclear states
    if any(w in t for w in ["weakening", "consolidating", "x-over soon", "marginal",
                             "unclear", "n/a", "insufficient", "squeeze"]):
        return f'<span class="tag tag-neut">{h(text)}</span>'
    # Explicit bear before "above" (handles "above upper — overbought")
    if any(w in t for w in ["overbought", "band walking lower"]):
        return f'<span class="tag tag-bear">{h(text)}</span>'
    # Explicit bull before "below" (handles "below lower — oversold in range")
    if any(w in t for w in ["oversold", "band walking upper"]):
        return f'<span class="tag tag-bull">{h(text)}</span>'
    # General bull / bear
    if any(w in t for w in ["bull", "above", "golden", "crossover — bullish"]):
        return f'<span class="tag tag-bull">{h(text)}</span>'
    if any(w in t for w in ["bear", "below", "extended", "crossover — bearish"]):
        return f'<span class="tag tag-bear">{h(text)}</span>'
    return f'<span class="tag tag-neut">{h(text)}</span>'


def signals_html(sigs):
    return "\n".join(
        f'<div class="sig-row"><span class="sig-key">{h(k)}</span>{tag(v)}</div>'
        for k, v in sigs.items()
    )


def overall_verdict(sigs):
    _neutral_words = ["weakening", "consolidating", "x-over soon", "marginal",
                      "unclear", "n/a", "insufficient", "squeeze"]

    def _is_bull(v):
        t = v.lower()
        if any(w in t for w in _neutral_words): return False
        if any(w in t for w in ["overbought", "band walking lower"]): return False
        return any(w in t for w in ["bull", "above", "golden", "oversold",
                                    "band walking upper", "crossover — bullish"])

    def _is_bear(v):
        t = v.lower()
        if any(w in t for w in _neutral_words): return False
        if any(w in t for w in ["oversold", "band walking upper"]): return False
        return any(w in t for w in ["bear", "below", "overbought", "extended",
                                    "band walking lower", "crossover — bearish"])

    bull = sum(1 for v in sigs.values() if _is_bull(v))
    bear = sum(1 for v in sigs.values() if _is_bear(v))
    total = len(sigs)
    score = bull - bear
    if score >= 2:
        cls, label = "verdict-bull", "Overall: Bullish Bias"
        desc = (f"{bull}/{total} indicators lean bullish. Momentum and trend signals suggest "
                "a constructive setup. Watch for confirmation on volume and candle close.")
    elif score <= -2:
        cls, label = "verdict-bear", "Overall: Bearish Bias"
        desc = (f"{bear}/{total} indicators lean bearish. The weight of evidence points to "
                "downside pressure. Consider waiting for oversold extremes or base formation.")
    else:
        cls, label = "verdict-neut", "Overall: Neutral / Mixed"
        desc = (f"Signals are mixed ({bull} bullish, {bear} bearish, {total-bull-bear} neutral). "
                "No clear directional edge. Monitor for a decisive breakout or breakdown.")
    return f'<div class="verdict-box {cls}"><p class="label">{label}</p><p>{desc}</p></div>'


def commentary_placeholder(section_id):
    """
    Emit an HTML comment placeholder for AI post-generation editing.
    The AI agent should replace this comment with:
      <div class="analyst-note">
        <p class="analyst-label">Analyst View</p>
        <p>2–3 sentence commentary here.</p>
      </div>
    section_id values: short | med | long | fundamental
    """
    return f'<!-- ANALYST_VIEW:{section_id} -->'


def stat_card(label, value, color="", sub="", sub_is_html=False):
    """
    label : bottom label text
    value : main large number
    color : optional colour for main number
    sub   : optional second row of small muted text (uses .stat-sub)
    """
    style   = f'color:{color};' if color else ''
    sub_value = sub if sub_is_html else h(sub)
    sub_row = f'<div class="stat-sub">{sub_value}</div>' if sub else ''
    return (f'<div class="card">'
            f'<div class="stat-num" style="{style}">{h(value)}</div>'
            f'{sub_row}'
            f'<div class="stat-label">{h(label)}</div>'
            f'</div>')


def fundamental_table_html(data):
    skip = {"Stock Name", "Lot Size"}
    rows = "".join(
        f"<tr><td>{h(k)}</td><td>{h(v)}</td></tr>"
        for k, v in data.items() if k not in skip and not k.startswith("_")
    )
    return f'<table class="kv-table">{rows}</table>'


def long_fundamental_verdict(data):
    lines = []
    try:
        pe = float(data.get("P/E Ratio (TTM)", "x"))
        if pe <= 0:   lines.append("P/E (TTM) is negative — company is currently loss-making; earnings-based valuation is not meaningful.")
        elif pe < 15: lines.append(f"P/E (TTM) of {pe:.1f}× — stock appears undervalued vs market average.")
        elif pe < 25: lines.append(f"P/E (TTM) of {pe:.1f}× is in a fair value range.")
        else:         lines.append(f"P/E (TTM) of {pe:.1f}× implies growth premium; watch earnings delivery.")
    except Exception: pe = None

    try:
        pb = float(data.get("P/B Ratio", "x"))
        if pb < 1.5:  lines.append(f"P/B of {pb:.2f}× — trades near book value, a value signal.")
        elif pb < 3:  lines.append(f"P/B of {pb:.2f}× is moderate.")
        else:         lines.append(f"P/B of {pb:.2f}× is elevated.")
    except Exception: pb = None

    try:
        dy = float(data.get("Dividend Yield (TTM)", "x").replace("%",""))
        if dy > 2:    lines.append(f"Dividend yield of {dy:.2f}% provides income cushion.")
        elif dy > 0:  lines.append(f"Modest dividend yield of {dy:.2f}%.")
        else:         lines.append("No dividend yield — returns are growth-dependent.")
    except Exception: pass

    # FUNDAMENTAL_VIEW rendered separately as analyst-note below the verdict box

    cls = "verdict-neut"
    if pe is not None:
        if pe <= 0:   cls = "verdict-neut"   # loss-making — not meaningful
        elif pe < 18: cls = "verdict-bull"
        elif pe > 28: cls = "verdict-bear"

    body = " ".join(lines)
    return f'<div class="verdict-box {cls}"><p>{body}</p></div>'


# ── chart data helpers ────────────────────────────────────────────────────────

def df_to_price_chart_data(df, ma_cols, last_n):
    """Prepare OHLC + volume + MA data for Plotly candlestick charts."""
    sub = df.tail(last_n).reset_index(drop=True)

    def sf(v):
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 4)
        except Exception:
            return None

    dates      = [str(t)[:10] for t in sub["time_key"]]
    opens      = [sf(v) for v in sub["open"]]
    highs      = [sf(v) for v in sub["high"]]
    lows       = [sf(v) for v in sub["low"]]
    closes     = [sf(v) for v in sub["close"]]
    volumes    = []
    vol_colors = []
    for _, row in sub.iterrows():
        try:
            v = int(float(row.get("volume", 0) or 0))
            c, o = sf(row["close"]), sf(row["open"])
            volumes.append(v)
            vol_colors.append("#3fb950" if (c is not None and o is not None and c >= o) else "#f85149")
        except Exception:
            volumes.append(0)
            vol_colors.append("#8b949e")

    result = {"dates": dates, "open": opens, "high": highs,
              "low": lows, "close": closes, "volume": volumes, "vol_colors": vol_colors}
    for col in ma_cols:
        result[col] = [sf(v) for v in sub[col]]
    return result


def df_to_js(df, cols, last_n):
    sub = df.tail(last_n)
    dates = [str(t)[:10] for t in sub["time_key"]]
    result = {"dates": dates}
    for col in cols:
        vals = []
        for v in sub[col]:
            try:
                f = float(v)
                vals.append(None if math.isnan(f) else round(f, 4))
            except Exception:
                vals.append(None)
        result[col] = vals
    return result


def js(val):
    """Serialise a Python list to a JavaScript array with proper null (not 'null')."""
    return json.dumps(val)


# ── JS chart builders ─────────────────────────────────────────────────────────

def js_plotly_price_chart(chart_id, data, labels, show_ma4=False):
    """
    Plotly candlestick + volume histogram + MA lines. No watermark, fully open source.
    labels = (ma1_label, ma2_label, ma3_label, ma4_label)
    """
    n1, n2, n3, n4 = labels

    traces = [
        # Candlestick
        {"type": "candlestick",
         "x": data["dates"], "open": data["open"], "high": data["high"],
         "low": data["low"], "close": data["close"],
         "increasing": {"line": {"color": "#3fb950", "width": 1}, "fillcolor": "#3fb950"},
         "decreasing": {"line": {"color": "#f85149", "width": 1}, "fillcolor": "#f85149"},
         "name": "Price", "yaxis": "y", "showlegend": False,
         "hoverinfo": "x+y"},
        # Volume bars
        {"type": "bar",
         "x": data["dates"], "y": data["volume"],
         "marker": {"color": data["vol_colors"]},
         "name": "Vol", "yaxis": "y2", "showlegend": False,
         "opacity": 0.55, "hoverinfo": "x+y"},
    ]

    # MA lines
    ma_defs = [
        ("ma1", n1, "#f0883e", 1.5, None),
        ("ma2", n2, "#79c0ff", 1.5, None),
        ("ma3", n3, "#ffa657", 2.0, None),
    ]
    if show_ma4:
        ma_defs.append(("ma4", n4, "#ff7b72", 2.0, "dash"))

    for col, label, color, width, dash in ma_defs:
        line = {"color": color, "width": width}
        if dash:
            line["dash"] = dash
        traces.append({"type": "scatter", "mode": "lines",
                        "x": data["dates"], "y": data[col],
                        "line": line, "name": label,
                        "yaxis": "y", "connectgaps": False})

    layout = {
        "paper_bgcolor": "#161b22", "plot_bgcolor": "#161b22",
        "font": {"color": "#8b949e", "size": 11},
        "height": 320,
        "margin": {"l": 55, "r": 10, "t": 10, "b": 30},
        "xaxis": {
            "type": "category",
            "gridcolor": "rgba(48,54,61,0.5)", "linecolor": "#30363d",
            "tickfont": {"color": "#8b949e", "size": 10},
            "nticks": 8, "showgrid": True,
            "rangeslider": {"visible": False},
        },
        "yaxis": {
            "domain": [0.22, 1.0],
            "gridcolor": "rgba(48,54,61,0.5)", "linecolor": "#30363d",
            "tickfont": {"color": "#8b949e", "size": 10}, "showgrid": True,
        },
        "yaxis2": {"domain": [0, 0.18], "showgrid": False, "showticklabels": False},
        "legend": {"font": {"size": 10, "color": "#8b949e"},
                   "bgcolor": "rgba(0,0,0,0)", "orientation": "h",
                   "x": 0, "y": 1.02, "xanchor": "left"},
        "dragmode": "pan",
        "hovermode": "x unified",
    }

    config = {"displayModeBar": "hover",
              "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
              "displaylogo": False, "responsive": True}

    return (f"if(document.getElementById({json.dumps(chart_id)}))"
            f" Plotly.newPlot({json.dumps(chart_id)},"
            f"{json.dumps(traces)},{json.dumps(layout)},{json.dumps(config)});")


def js_momentum_chart(chart_id, data):
    hist_vals = data["macd_hist"]
    hist_colors = "[" + ",".join(
        "'rgba(0,0,0,0)'" if v is None else ("'#3fb950'" if v >= 0 else "'#f85149'")
        for v in hist_vals
    ) + "]"

    return f"""mkChart('{chart_id}', {{
  type: 'line',
  data: {{
    labels: {js(data["dates"])},
    datasets: [
      {{label:'RSI(14)', data:{js(data["rsi14"])},       borderColor:'#bc8cff', borderWidth:2,   pointRadius:0, tension:.3, yAxisID:'y'}},
      {{label:'MACD',    data:{js(data["macd_line"])},   borderColor:'#58a6ff', borderWidth:1.5, pointRadius:0, tension:.3, yAxisID:'y2'}},
      {{label:'Signal',  data:{js(data["macd_signal"])}, borderColor:'#f85149', borderWidth:1.5, pointRadius:0, borderDash:[4,4], tension:.3, yAxisID:'y2'}},
      {{label:'Hist',    data:{js(hist_vals)}, type:'bar', backgroundColor:{hist_colors}, yAxisID:'y2'}}
    ]
  }},
  options: {{
    responsive:true, interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{labels:{{color:'#8b949e',boxWidth:12,font:{{size:11}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',maxTicksLimit:8,maxRotation:0}},grid:{{color:'rgba(48,54,61,0.5)'}}}},
      y:{{position:'left',  ticks:{{color:'#bc8cff'}}, grid:{{color:'rgba(48,54,61,0.5)'}}, title:{{display:true,text:'RSI',color:'#bc8cff'}}}},
      y2:{{position:'right', ticks:{{color:'#58a6ff'}}, grid:{{drawOnChartArea:false}}, title:{{display:true,text:'MACD',color:'#58a6ff'}}}}
    }}
  }}
}});"""


def js_bb_chart(chart_id, data):
    return f"""mkChart('{chart_id}', {{
  type: 'line',
  data: {{
    labels: {js(data["dates"])},
    datasets: [
      {{label:'Upper Band', data:{js(data["bb_upper"])}, borderColor:'rgba(248,81,73,0.6)',  borderWidth:1, pointRadius:0, fill:'+2', backgroundColor:'rgba(248,81,73,0.04)'}},
      {{label:'Price',      data:{js(data["close"])},    borderColor:'#58a6ff',              borderWidth:2.5, pointRadius:0, tension:.3, fill:false}},
      {{label:'Mid (SMA20)',data:{js(data["bb_mid"])},   borderColor:'rgba(88,166,255,0.5)', borderWidth:1, pointRadius:0, borderDash:[4,4]}},
      {{label:'Lower Band', data:{js(data["bb_lower"])}, borderColor:'rgba(63,185,80,0.6)',  borderWidth:1, pointRadius:0, fill:'-2', backgroundColor:'rgba(63,185,80,0.04)'}}
    ]
  }},
  options: {{
    responsive:true, interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{labels:{{color:'#8b949e',boxWidth:12,font:{{size:11}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',maxTicksLimit:8,maxRotation:0}},grid:{{color:'rgba(48,54,61,0.5)'}}}},
      y:{{ticks:{{color:'#8b949e'}},grid:{{color:'rgba(48,54,61,0.5)'}}}}
    }}
  }}
}});"""


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{stock_code} {stock_name} — Stock Analysis Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js@2.27.0/dist/plotly-finance.min.js"></script>
<style>
  :root {{
    --bg:#0d1117; --card:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff;
    --green:#3fb950; --red:#f85149; --yellow:#d29922;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:24px}}
  h1{{font-size:1.8rem;margin-bottom:4px}}
  .subtitle{{color:var(--muted);margin-bottom:24px;font-size:.9rem}}
  .grid{{display:grid;gap:16px}}
  .g2{{grid-template-columns:1fr 1fr}}
  .g3{{grid-template-columns:repeat(3,1fr)}}
  .g4{{grid-template-columns:repeat(4,1fr)}}
  .g2 .full{{grid-column:1/-1}}
  @media(max-width:900px){{.g3{{grid-template-columns:1fr 1fr}}}}
  @media(max-width:900px){{.g2,.g4{{grid-template-columns:1fr}}}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px}}
  .card h2{{font-size:.95rem;color:var(--accent);margin-bottom:14px;text-transform:uppercase;letter-spacing:.05em}}
  .card h3{{font-size:.8rem;color:var(--muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:.04em}}
  .kv-table{{width:100%;border-collapse:collapse;font-size:.87rem}}
  .kv-table td{{padding:5px 8px;border-bottom:1px solid var(--border)}}
  .kv-table td:first-child{{color:var(--muted);width:55%}}
  .tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.78rem;font-weight:600}}
  .tag-bull{{background:#1a3a2a;color:var(--green)}}
  .tag-bear{{background:#3a1a1a;color:var(--red)}}
  .tag-neut{{background:#2a2a1a;color:var(--yellow)}}
  .sig-row{{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border);font-size:.87rem}}
  .sig-row:last-child{{border-bottom:none}}
  .sig-key{{color:var(--muted)}}
  .pat-list{{list-style:none;font-size:.87rem}}
  .pat-list li{{padding:5px 0;border-bottom:1px solid var(--border)}}
  .pat-list li:last-child{{border-bottom:none}}
  .verdict{{border-radius:8px;padding:16px;margin-top:16px}}
  .v-bull{{background:#1a3a2a;border:1px solid #3fb950}}
  .v-bear{{background:#3a1a1a;border:1px solid #f85149}}
  .v-neut{{background:#2a2a1a;border:1px solid #d29922}}
  .verdict-box p{{font-size:.84rem;line-height:1.7;color:#8b949e;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.15)}}
  .verdict-box .label{{font-weight:700;font-size:1rem}}
  canvas{{max-height:300px}}
  .stat-num{{font-size:1.5rem;font-weight:700;line-height:1.2}}
  .stat-sub{{font-size:.72rem;color:var(--muted);margin-top:3px}}
  .stat-label{{font-size:.78rem;color:var(--muted);margin-top:4px}}
  .sec{{font-size:1.05rem;font-weight:700;color:var(--accent);margin:24px 0 12px;border-bottom:1px solid var(--border);padding-bottom:6px}}
  .badge{{display:inline-block;padding:1px 7px;border-radius:3px;font-size:.72rem;font-weight:600;margin-left:6px;vertical-align:middle}}
  .b-ema{{background:#1a2a3a;color:#79c0ff}}
  .b-sma{{background:#2a1a3a;color:#d2a8ff}}
  .disclaimer{{font-size:.78rem;color:var(--muted);margin-top:24px;padding:12px;border:1px solid var(--border);border-radius:6px}}
  .analyst-note{{margin-top:14px;padding:12px;border-left:3px solid var(--accent);background:rgba(88,166,255,0.05);border-radius:0 6px 6px 0}}
  .analyst-note .analyst-label{{font-size:.72rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}}
  .analyst-note p:last-child{{font-size:.84rem;line-height:1.7;color:var(--muted)}}
</style>
</head>
<body>

<h1>{stock_code} &nbsp;|&nbsp; {stock_name}</h1>
<p class="subtitle">Generated {generated_at} &nbsp;·&nbsp; Data via Futu OpenD &nbsp;·&nbsp; Charts require internet connection &nbsp;·&nbsp; For informational purposes only</p>

<div class="sec">Key Statistics</div>
<div class="grid g4">{stat_cards_row1}</div>
<div class="grid g4" style="margin-top:16px">{stat_cards_row2}</div>

<!-- SHORT TERM -->
<div class="sec">Short-Term Analysis (Daily) <span class="badge b-ema">EMA 9 / 20 / 50 + SMA 200</span> <span style="font-size:.8rem;color:var(--muted);font-weight:400">&nbsp;·&nbsp; 3–10 day view</span></div>
<div class="grid g2">
  <div class="card">
    <h2>Candlestick &amp; EMA (Daily — last 120 sessions)</h2>
    <div id="cShortMA"></div>
  </div>
  <div class="card">
    <h2>Bollinger Bands — SMA20 ± 2σ (Daily)</h2>
    <canvas id="cBB"></canvas>
  </div>
</div>
<div class="grid g2" style="margin-top:16px">
  <div class="card">
    <h2>RSI(14) &amp; MACD (Daily)</h2>
    <canvas id="cShortMom"></canvas>
  </div>
  <div class="card">
    <h2>Short-Term Signal Summary</h2>
    {short_sigs_html}
    {short_verdict}
    {short_commentary}
  </div>
</div>

<!-- MEDIUM TERM -->
<div class="sec">Medium-Term Analysis (Weekly) <span class="badge b-ema">EMA 9 / 20 / 50</span> <span style="font-size:.8rem;color:var(--muted);font-weight:400">&nbsp;·&nbsp; 3–8 week view</span></div>
<div class="grid g2">
  <div class="card">
    <h2>Candlestick &amp; EMA (Weekly — last 104 weeks)</h2>
    <div id="cMedMA"></div>
  </div>
  <div class="card">
    <h2>Bollinger Bands — SMA20 ± 2σ (Weekly)</h2>
    <canvas id="cMedBB"></canvas>
  </div>
</div>
<div class="grid g2" style="margin-top:16px">
  <div class="card">
    <h2>RSI(14) &amp; MACD (Weekly)</h2>
    <canvas id="cMedMom"></canvas>
  </div>
  <div class="card">
    <h2>Medium-Term Signal Summary</h2>
    {med_sigs_html}
    {med_verdict}
    {med_commentary}
  </div>
</div>

<!-- LONG TERM -->
<div class="sec">Long-Term Analysis (Weekly) + Fundamentals <span class="badge b-sma">SMA 20 / 50 / 200</span> <span style="font-size:.8rem;color:var(--muted);font-weight:400">&nbsp;·&nbsp; 3–6 month view</span></div>
<div class="grid g2">
  <div class="card">
    <h2>Candlestick &amp; SMA (Weekly — last 156 weeks)</h2>
    <div id="cLongMA"></div>
  </div>
  <div class="card">
    <h2>Bollinger Bands — SMA20 ± 2σ (Weekly)</h2>
    <canvas id="cLongBB"></canvas>
  </div>
</div>
<div class="grid g2" style="margin-top:16px">
  <div class="card">
    <h2>RSI(14) &amp; MACD (Weekly — last 156 weeks)</h2>
    <canvas id="cLongMom"></canvas>
  </div>
  <div class="card">
    <h2>Long-Term Signal Summary</h2>
    {long_sigs_html}
    {long_signal_verdict}
    {long_commentary}
  </div>
</div>
<div class="grid g2" style="margin-top:16px">
  <div class="card">
    <h2>Fundamental Data</h2>
    {fund_html}
  </div>
  <div class="card">
    <h2>Long-Term Fundamental View</h2>
    {long_verdict}
    {fundamental_commentary}
  </div>
</div>

<p class="disclaimer">⚠ Disclaimer: This report is generated automatically from market data and is for informational purposes only. It does not constitute investment advice. Past performance is not indicative of future results. Always conduct your own due diligence before making any investment decision.</p>

<script>
function mkChart(id, cfg) {{
  const el = document.getElementById(id);
  if (el) new Chart(el, cfg);
}}

{js_short_ma}
{js_short_mom}
{js_bb}
{js_med_ma}
{js_med_mom}
{js_med_bb}
{js_long_ma}
{js_long_bb}
{js_long_mom}
</script>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    from datetime import timedelta
    today = datetime.now()
    # Smart start dates: fetch only what's needed for warm-up + display window
    # Daily  : SMA200 warm-up (200) + 500 display ≈ 700 bars → ~3.3 years
    # Weekly : SMA200 warm-up (200) + 156 long display ≈ 356 bars → ~8 years
    #          (long-term now uses weekly SMA20/50/200, needs 200-bar warm-up)
    daily_start  = (today - timedelta(days=1200)).strftime("%Y-%m-%d")
    weekly_start = (today - timedelta(days=3000)).strftime("%Y-%m-%d")

    print("Connecting to FutuOpenD …")
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    time.sleep(1)

    try:
        print("Fetching daily kline …")
        df_d = fetch_kline(ctx, STOCK_CODE, KLType.K_DAY,  count=500,  start=daily_start)
        add_ema_indicators(df_d)

        print("Fetching weekly kline …")
        df_w = fetch_kline(ctx, STOCK_CODE, KLType.K_WEEK, count=400,  start=weekly_start)
        add_ema_indicators(df_w)

        # Long-term uses weekly data with SMA20/50/200 (copy to avoid overwriting EMA columns)
        df_wl = df_w.copy()
        add_sma_indicators(df_wl)

        print("Fetching fundamentals …")
        try:
            fund = fetch_fundamental(ctx, STOCK_CODE)
        except Exception as e:
            print(f"  Warning: fundamentals unavailable ({e})")
            fund = {}

        print("Fetching capital flow …")
        try:
            capflow = fetch_capital_flow(ctx, STOCK_CODE)
        except Exception as e:
            print(f"  Warning: capital flow unavailable ({e})")
            capflow = {}

    finally:
        ctx.close()

    print("Computing signals …")

    short_sigs = signal_summary(df_d,  use_ema=True,  include_ma4=True,  macd_confirm=5, include_volume=True)                      # daily:   EMA9/20 cross, SMA200, volume
    med_sigs   = signal_summary(df_w,  use_ema=True,  include_ma4=True,  macd_confirm=3, use_mid_cross=True, include_volume=True)  # weekly:  EMA20/50 cross, SMA200, volume
    long_sigs  = signal_summary(df_wl, use_ema=False, include_ma4=False, macd_confirm=3)                                           # weekly SMA20/50/200 — no volume (too slow)
    long_signal_verdict = overall_verdict(long_sigs)

    # Key stat cards
    ld, pd_ = df_d.iloc[-1], df_d.iloc[-2]
    chg     = ld["close"] - pd_["close"]
    chg_pct = chg / pd_["close"] * 100
    chg_col = "#3fb950" if chg >= 0 else "#f85149"
    arrow   = "▲" if chg >= 0 else "▼"
    hi52    = df_d["close"].tail(TRADING_DAYS_YEAR).max()
    lo52    = df_d["close"].tail(TRADING_DAYS_YEAR).min()

    # Volume card
    vol      = fund.get("_volume", 0)
    vol_ratio= fund.get("_volume_ratio", None)
    try:
        vol_str = f"{int(vol)/1e6:.1f}M"
        ratio_f = float(vol_ratio)
        vol_col = "#3fb950" if ratio_f >= 1.0 else "#8b949e"
        vol_label = f"{vol_str} ({ratio_f:.2f}× avg)"
    except Exception:
        vol_label = f"{int(vol)/1e6:.1f}M"
        vol_col = "#8b949e"

    # Capital flow card
    if capflow:
        net        = capflow["total_net"]
        inst_net   = capflow["inst_net"]
        retail_net = capflow["retail_net"]
        net_col    = "#3fb950" if net >= 0 else "#f85149"
        net_sign   = "+" if net >= 0 else ""
        inst_sign  = "+" if inst_net   >= 0 else ""
        ret_sign   = "+" if retail_net >= 0 else ""
        inst_col   = "#3fb950" if inst_net   >= 0 else "#f85149"
        ret_col    = "#3fb950" if retail_net >= 0 else "#f85149"
        cf_label   = f"{net_sign}{net/1e6:.0f}M"
        cf_sub     = (f"<span style='color:{inst_col}'>Inst: {inst_sign}{inst_net/1e6:.0f}M</span>"
                      f"&nbsp;·&nbsp;"
                      f"<span style='color:{ret_col}'>Retail: {ret_sign}{retail_net/1e6:.0f}M</span>")
        cf_col     = net_col
    else:
        cf_label, cf_sub, cf_col = "—", "", "#8b949e"

    # P/E and dividend
    pe_val = fund.get("P/E Ratio (TTM)", "—")
    pe_col = "#8b949e"
    try:
        pe_f = float(pe_val)
        if pe_f <= 0:
            pe_col = "#8b949e"
            pe_val = "Loss-making"
        else:
            pe_col = "#3fb950" if pe_f < 15 else ("#d29922" if pe_f < 25 else "#f85149")
            pe_val = f"{pe_f:.1f}×"
    except Exception:
        pass

    dy_val = fund.get("Dividend Yield (TTM)", "—")

    # Daily range sub-label for Last Price card
    day_high = float(ld["high"])
    day_low  = float(ld["low"])
    day_range_sub = f"L: {day_low:.2f} &nbsp;·&nbsp; H: {day_high:.2f}"

    # ATR card — volatility/risk context, not directional
    try:
        atr_val = float(ld["atr14"])
        atr_pct = atr_val / float(ld["close"]) * 100
        atr_label = f"HKD {atr_val:.2f} ({atr_pct:.1f}%)"
    except Exception:
        atr_label = "—"

    stat_cards_row1 = "\n".join([
        stat_card("Last Price (HKD)",  f"{ld['close']:.2f}", "#58a6ff", sub=day_range_sub, sub_is_html=True),
        stat_card("Day Change",        f"{arrow} {abs(chg):.2f} ({abs(chg_pct):.2f}%)", chg_col),
        stat_card("Volume",            vol_label, vol_col),
        stat_card(f"Capital Flow ({capflow.get('update_time','')[:10] if capflow else '—'})", cf_label, cf_col, sub=cf_sub, sub_is_html=True),
    ])
    stat_cards_row2 = "\n".join([
        stat_card("52-Week Range",     f"{lo52:.2f} – {hi52:.2f}"),
        stat_card("ATR (14) — Daily",  atr_label),
        stat_card("P/E Ratio (TTM)",   pe_val, pe_col),
        stat_card("Dividend Yield",    dy_val),
    ])

    # Chart data
    # Candlestick + volume + MA data for Plotly price panels
    short_lwc  = df_to_price_chart_data(df_d,  ["ma1","ma2","ma3","ma4"], SHORT_CHART_BARS)
    med_lwc    = df_to_price_chart_data(df_w,  ["ma1","ma2","ma3"],       MED_CHART_BARS)
    long_lwc   = df_to_price_chart_data(df_wl, ["ma1","ma2","ma3"],       LONG_WEEKLY_BARS)

    # Chart.js data for BB and RSI/MACD panels (close price still needed for BB)
    short_d    = df_to_js(df_d,  ["close","rsi14","macd_line","macd_signal","macd_hist","bb_upper","bb_mid","bb_lower"], SHORT_CHART_BARS)
    med_d      = df_to_js(df_w,  ["close","rsi14","macd_line","macd_signal","macd_hist","bb_upper","bb_mid","bb_lower"], MED_CHART_BARS)
    long_d     = df_to_js(df_wl, ["close","bb_upper","bb_mid","bb_lower"], LONG_WEEKLY_BARS)
    long_mom_d = df_to_js(df_wl, ["rsi14","macd_line","macd_signal","macd_hist"], LONG_WEEKLY_BARS)

    # Labels
    ema_labels = ("EMA9", "EMA20", "EMA50", "SMA200")
    sma_labels = ("SMA20", "SMA50", "SMA200", "")

    stock_name = h(fund.get("Stock Name", STOCK_CODE))
    stock_code_display = h(STOCK_CODE.replace("HK.", "HK:"))

    html = HTML.format(
        stock_code     = stock_code_display,
        stat_cards_row1= stat_cards_row1,
        stat_cards_row2= stat_cards_row2,
        stock_name     = stock_name,
        generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M"),
        short_sigs_html= signals_html(short_sigs),
        short_verdict  = overall_verdict(short_sigs),
        short_commentary = commentary_placeholder("short"),
        med_sigs_html  = signals_html(med_sigs),
        med_verdict    = overall_verdict(med_sigs),
        med_commentary = commentary_placeholder("med"),
        fund_html           = fundamental_table_html(fund),
        long_verdict        = long_fundamental_verdict(fund),
        long_sigs_html      = signals_html(long_sigs),
        long_signal_verdict  = long_signal_verdict,
        long_commentary      = commentary_placeholder("long"),
        fundamental_commentary = commentary_placeholder("fundamental"),
        js_short_ma    = js_plotly_price_chart("cShortMA",  short_lwc, ema_labels, show_ma4=True),
        js_short_mom   = js_momentum_chart("cShortMom", short_d),
        js_bb          = js_bb_chart("cBB",             short_d),
        js_med_ma      = js_plotly_price_chart("cMedMA",   med_lwc,   ema_labels, show_ma4=False),
        js_med_mom     = js_momentum_chart("cMedMom",   med_d),
        js_med_bb      = js_bb_chart("cMedBB",          med_d),
        js_long_ma     = js_plotly_price_chart("cLongMA",  long_lwc,  sma_labels, show_ma4=False),
        js_long_bb     = js_bb_chart("cLongBB",         long_d),
        js_long_mom    = js_momentum_chart("cLongMom",  long_mom_d),
    )

    code_digits = STOCK_CODE.replace("HK.", "")
    date_str = datetime.now().strftime("%Y%m%d")
    out = os.path.join(SCRIPT_DIR, f"{code_digits}_analysis_{date_str}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved → {out}")
    return out


if __name__ == "__main__":
    main()
