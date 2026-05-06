# HK Stock Analysis Report Generator

A Python script that generates single-file HTML technical and fundamental analysis reports for Hong Kong stocks, using market data fetched live from [Futu OpenD](https://openapi.futunn.com/).

---

## What it does

For any HK-listed stock, the script:

1. Fetches live OHLCV kline data, market snapshot, capital flow, and HSI benchmark data from FutuOpenD
2. Computes technical indicators across three timeframes
3. Generates a dark-themed single-file HTML report with interactive Chart.js and Plotly charts loaded via CDN
4. Saves the report as `{CODE}_analysis_{YYYYMMDD}.html` in the same directory

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | |
| `futu-api` | `pip install futu-api` |
| `pandas`, `numpy` | `pip install pandas numpy` |
| Futu OpenD running | Desktop app/ CLI must be open and logged in on `127.0.0.1:11111` |

---

## Usage

```bash
python3 analyze.py [STOCK_CODE]
```

`STOCK_CODE` can be a bare HK code or Futu format. Bare codes are normalized to `HK.` + zero-padded 5-digit code.

```bash
python3 analyze.py 3690        # Meituan, normalized to HK.03690
python3 analyze.py HK.00700    # Tencent
python3 analyze.py HK.09988    # Alibaba (default if omitted)
python3 analyze.py HK.00005    # HSBC
python3 analyze.py HK.02800    # Tracker Fund (ETF)
```

The output file is saved in the same directory as the script:
```
00700_analysis_20260504.html
```

---

## Report structure

### Key Statistics (top of report)
Two rows of 4 stat cards:

| Row 1 | Row 2 |
|-------|-------|
| Last Price | 52-Week Range |
| Day Change | ATR (14) — Daily |
| Volume (with avg ratio) | P/E Ratio (TTM) |
| Capital Flow (Inst · Retail) | Dividend Yield |

### Short-Term Analysis — Daily `3–10 day view`
- **Indicators:** EMA 9 / 20 / 50 + SMA200 (institutional reference)
- **Charts:** Price & EMA · Bollinger Bands · RSI(14) & MACD
- **Signals:** Price vs EMA20/50/SMA200 · EMA9/20 crossover · RSI · MACD · BB · ADX · Volume

### Medium-Term Analysis — Weekly `3–8 week view`
- **Indicators:** EMA 9 / 20 / 50 + SMA200 (structural reference)
- **Charts:** Price & EMA · Bollinger Bands · RSI(14) & MACD
- **Signals:** Price vs EMA20/50/SMA200 · EMA20/50 crossover · RSI · MACD · BB · ADX · Volume · RS vs HSI (4W)

### Long-Term Analysis — Weekly SMA `3–6 month view`
- **Indicators:** SMA 20 / 50 / 200 (all weekly)
- **Charts:** Price & SMA · Bollinger Bands · RSI(14) & MACD
- **Signals:** Price vs SMA50/200 · SMA20/50 crossover · RSI · MACD · BB · ADX · RS vs HSI (13W)
- **Fundamental data:** P/E, P/B, EPS, Dividend Yield, Market Cap, Net Profit, etc.
- **Fundamental view:** Quantitative verdict + qualitative commentary

---

## Indicator details

| Indicator | Parameters | Notes |
|-----------|-----------|-------|
| EMA | 9 / 20 / 50 | Short & medium term |
| SMA | 20 / 50 / 200 | Long term (weekly bars) |
| RSI | Wilder's 14 | `ewm(alpha=1/14, adjust=False)` |
| MACD | 12 / 26 / 9 | Gerald Appel standard definition |
| Bollinger Bands | SMA20 ± 2σ | ADX-context labels + squeeze detection + compact range context |
| ADX | 14 | +DI / -DI included |
| ATR | 14 | Daily volatility stat card, not a directional signal |
| Relative Strength vs HSI | 4W / 13W | Stock return minus `HK.800000` HSI return |

### MACD Momentum confirmation windows
| Timeframe | Window | Rule |
|-----------|--------|------|
| Short-term (daily) | 5 bars | 4-of-5 majority |
| Medium-term (weekly) | 3 bars | All-3-agree |
| Long-term (weekly) | 3 bars | All-3-agree |

### Bollinger Bands interpretation
Labels are ADX-context aware:
- **ADX > 25 + price outside band** → "Band walking — trend continuation"
- **ADX < 20 + price outside band** → "Overbought/oversold in range"
- **BW < 50% of 20-bar avg** → "⚡ Squeeze — volatility contraction"
- **Within bands only** → add compact range context: `20D High`, `20D Low`, `In 20D Range`, `13W High`, `13W Low`, or `In 13W Range`
- Range context uses close vs prior range with a 0.5% buffer; squeeze and outside-band labels are unchanged

### Relative Strength vs HSI interpretation
Relative strength compares stock performance against the Hang Seng Index (`HK.800000`):
- **Medium-term:** 4W / 20D return difference; bullish ≥ +3.0pp, bearish ≤ -3.0pp
- **Long-term:** 13W / 60D return difference; bullish ≥ +5.0pp, bearish ≤ -5.0pp
- **Short-term:** intentionally excluded because 5D relative performance is too noisy

### Signal scoring
Signal rows are represented as structured data:

```python
{"text": "Bullish — Upper half · 20D High · BW: 8.4%", "direction": "bull"}
```

`direction` is one of `bull`, `bear`, or `neut`. HTML tag colour and `overall_verdict()` scoring use this explicit direction instead of inferring sentiment from display text.

---

## Data fetching

| Data | Source | Bars fetched |
|------|--------|-------------|
| Daily kline | `request_history_kline` K_DAY | ~500 bars (~2 years) |
| Weekly kline | `request_history_kline` K_WEEK | ~400 bars (~8 years) |
| HSI benchmark kline | `request_history_kline` for `HK.800000` | Daily + weekly |
| Market snapshot | `get_market_snapshot` | Latest |
| Capital distribution | `get_capital_distribution` | Today |

Full history is paginated using `page_req_key`. Only the most recent N bars are kept after fetch. Monthly kline is no longer used — long-term analysis uses weekly SMA20/50/200 for consistency and to ensure SMA200 is computable for all stocks. HSI benchmark data is fetched separately for medium- and long-term relative strength signals.

---

## Qualitative commentary

Commentary is added by an AI agent **directly into the generated HTML file** — no second run of the script is needed.

The script always embeds four HTML comment placeholders in the report:

```html
<!-- ANALYST_VIEW:short -->
<!-- ANALYST_VIEW:med -->
<!-- ANALYST_VIEW:long -->
<!-- ANALYST_VIEW:fundamental -->
```

**Workflow:**

1. Run the script → HTML generated with the above placeholders
2. Ask an AI agent to read the HTML file, extract signal data from each section, and draft commentary
3. The AI replaces each placeholder directly in the HTML with a formatted block:

```html
<div class="analyst-note">
  <p class="analyst-label">Analyst View</p>
  <p>2–3 sentence commentary here.</p>
</div>
```

Commentary guidelines per section:
- `short` — 2–3 sentences based on short-term (3–10 day) signal readings and key statistics
- `med` — 2–3 sentences based on medium-term (3–8 week) signal readings and key statistics
- `long` — 2–3 sentences based on long-term (3–6 month) signal readings and key statistics
- `fundamental` — business overview, key catalysts, key risks, competitive landscape

---

## Configuration

All tunable constants are defined at the top of `analyze.py`:

```python
STOCK_CODE        = sys.argv[1] if len(sys.argv) > 1 else "HK.09988"
HOST              = "127.0.0.1"
PORT              = 11111
SHORT_CHART_BARS  = 120   # daily bars shown on short-term charts
MED_CHART_BARS    = 104   # weekly bars shown on medium-term charts
LONG_WEEKLY_BARS  = 156   # weekly bars shown on long-term charts
TRADING_DAYS_YEAR = 252   # used for 52-week range calculation
```

---

## Output

- **Format:** Single HTML file; Chart.js and Plotly are loaded via CDN at runtime
- **Theme:** Dark (GitHub-style `#0d1117` background)
- **Filename:** `{CODE}_analysis_{YYYYMMDD}.html` e.g. `00700_analysis_20260504.html`
- **Size:** ~125–135 KB per report, excluding CDN-loaded chart libraries
- **Git:** `*.html` is in `.gitignore` — reports are not committed

---

## Known limitations

- Requires Futu OpenD to be running locally; no offline mode
- Capital flow data (`get_capital_distribution`) reflects today's session only — not historical
- Recently listed stocks (< 4 years) may show N/A for weekly SMA200
- Qualitative commentary must be manually drafted each run

---

## Pending improvements

| # | Item | Priority | Notes |
|---|------|----------|-------|
| 1 | Refactor `add_ema_indicators` / `add_sma_indicators` | Low | 80% shared code. Internal housekeeping only, zero user-visible impact. |
| 2 | Split `analyze.py` into focused modules | Low | Refactor-only change with no report-output change. Target modules: `utils.py` (`h()`, `fmt_large()`, `normalize_stock_code()`), `indicators.py` (`sma()`, `ema()`, `rsi()`, `macd()`, `bollinger()`, `atr()`, `adx()`, indicator-enrichment functions), `fetch.py`, `signals.py` including RS signal logic, `charts.py`, and `report.py` for the HTML template/helpers. Keep CLI parsing and constants in `analyze.py`, but move `STOCK_CODE` parsing inside `main()` to avoid import-time side effects. Only `fetch.py` should import Futu; all other modules should remain importable/testable without Futu OpenD. |
| 3 | Landing page (`index.html`) | Low | Auto-generated index of all reports with verdict badges, price, ATR, P/E. Separate `index.py` script, sorted by signal strength. |
