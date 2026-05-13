"""
Stock Scanner — buy/sell signal generator
-----------------------------------------
Scans a watchlist every 20 minutes during market hours.
Generates two signal types per ticker:
  Quick — intraday scalp   (RSI5, real VWAP, volume ratio)
  Long  — swing/position   (RSI14 daily, MA50 pullback, volume ratio)

Outputs:
  • Pushover push notification for any actionable signal
  • Google Sheets log via gspread + service account

Env vars required:
  PUSHOVER_USER       — Pushover user key
  PUSHOVER_TOKEN      — Pushover app token
  GOOGLE_CREDENTIALS  — Google service account JSON (string)
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import time
import datetime
import random
import os
import http.client
from collections import defaultdict
from http.cookiejar import CookieJar

# -------------------------------------------------------------------
# HOLDINGS
# -------------------------------------------------------------------
HOLDINGS = {
    "VOO":  {"shares": 4.33,   "avg": 604.54,  "never_sell_all": True},
    "QQQ":  {"shares": 0.9284, "avg": 592.39,  "never_sell_all": True},
    "VTI":  {"shares": 1.55,   "avg": 323.13,  "never_sell_all": True},
    "SPY":  {"shares": 0.7624, "avg": 655.81,  "never_sell_all": True},
    "VOOG": {"shares": 1.16,   "avg": 429.40,  "never_sell_all": True},
    "SCHD": {"shares": 2.85,   "avg": 27.02,   "never_sell_all": True},
    "DIA":  {"shares": 0.1718, "avg": 464.47,  "never_sell_all": True},
    "GLD":  {"shares": 0.1394, "avg": 365.43,  "never_sell_all": True},
    "SMH":  {"shares": 0.1500, "avg": 333.31,  "never_sell_all": True},
    "JPM":  {"shares": 0.0862, "avg": 304.51,  "never_sell_all": True},
    "XLK":  {"shares": 0.3495, "avg": 143.03,  "never_sell_all": True},
    "F":    {"shares": 3.81,   "avg": 13.23,   "never_sell_all": True},
    "XLV":  {"shares": 0.3469, "avg": 145.54,  "never_sell_all": True},
    "NVDA": {"shares": 0.2455, "avg": 205.65,  "never_sell_all": True},
    "META": {"shares": 0.0791, "avg": 638.41,  "never_sell_all": True},
    "VOOV": {"shares": 0.2473, "avg": 202.18,  "never_sell_all": True},
}

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PUSHOVER_USER      = os.environ.get("PUSHOVER_USER", "")
PUSHOVER_TOKEN     = os.environ.get("PUSHOVER_TOKEN", "")
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS", "")
GOOGLE_SHEET_ID    = "11DeKby2ahmElsZFDNgaytJWCSpB_jgYspd1bTUq2yrg"

DAILY_RISK_BUDGET = 100
daily_spent = 0

BROAD_ETFS = ["VOO", "QQQ", "VTI", "SPY", "VOOG", "SCHD", "DIA", "GLD", "XLK", "XLV", "VOOV"]
MEGA_CAPS  = ["AAPL", "TSLA", "AMZN", "MSFT", "GOOGL", "NVDA", "META"]
HIGH_BETA  = ["SMH", "JPM", "F", "TQQQ", "SOXL"]

STOCKS = list(HOLDINGS.keys()) + [
    "AAPL", "TSLA", "AMZN", "MSFT", "GOOGL", "SCHG", "TQQQ", "SOXL"
]

SHEET_HEADERS = [
    "Timestamp", "Ticker", "Name", "Price",
    "VWAP Diff %", "RSI5", "Vol Ratio", "P&L %",
    "Quick Signal", "Quick Why",
    "RSI14 Daily", "MA50", "SMA200", "MACD", "ATR %",
    "Long Signal", "Long Why",
]

# -------------------------------------------------------------------
# HTTP
# -------------------------------------------------------------------
_UA_POOL = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
]
cookie_jar = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))


def safe_get(url, retries=6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": random.choice(_UA_POOL)})
            with opener.open(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, dict) and data.get("finance", {}).get("error"):
                    print("Yahoo error:", data["finance"]["error"])
                    return None
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (2 ** attempt) * 10 + random.uniform(15, 40)
                print(f"  429 — waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            print(f"  HTTP {e.code} for {url}")
            return None
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Failed after {retries} attempts: {e}")
                return None
            wait = (2 ** attempt) * 4 + random.uniform(5, 12)
            print(f"  Retry {attempt+1}/{retries} in {wait:.1f}s")
            time.sleep(wait)
    return None

# -------------------------------------------------------------------
# INDICATORS
# -------------------------------------------------------------------

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(1, period + 1):
        chg = closes[i] - closes[i - 1]
        if chg > 0:
            gains += chg
        else:
            losses -= chg
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        chg = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(chg,  0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-chg, 0)) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 2)


def calculate_sma(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def calculate_ema(values, period):
    if len(values) < period:
        return None
    k   = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def calculate_macd(closes, fast=12, slow=26):
    if len(closes) < slow:
        return None
    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None
    return round(ema_fast - ema_slow, 6)


def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    atr = sum(trs[-period:]) / period
    return (atr / closes[-1] * 100) if closes[-1] else None


def calculate_vwap(timestamps, highs, lows, closes, volumes):
    """VWAP for today's session only."""
    today   = datetime.datetime.now(datetime.timezone.utc).date()
    cum_tpv = cum_vol = 0.0
    for ts, h, l, c, v in zip(timestamps, highs, lows, closes, volumes):
        if any(x is None for x in (ts, h, l, c, v)):
            continue
        if datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date() != today:
            continue
        cum_tpv += ((h + l + c) / 3) * v
        cum_vol  += v
    return (cum_tpv / cum_vol) if cum_vol > 0 else None


def calculate_vol_ratio(timestamps, volumes):
    """Today's volume vs average of prior days in the dataset."""
    today    = datetime.datetime.now(datetime.timezone.utc).date()
    day_vols = defaultdict(float)
    for ts, v in zip(timestamps, volumes):
        if ts is None or v is None:
            continue
        d = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date()
        day_vols[d] += v
    today_vol  = day_vols.get(today, 0)
    prior_vols = [vol for d, vol in day_vols.items() if d != today]
    if not prior_vols:
        return 1.0
    avg_prior = sum(prior_vols) / len(prior_vols)
    return (today_vol / avg_prior) if avg_prior > 0 else 1.0

# -------------------------------------------------------------------
# DAILY INDICATOR CACHE
# -------------------------------------------------------------------
_daily_cache: dict = {}
DAILY_CACHE_TTL = 3600  # refresh once per hour


def fetch_daily_indicators(ticker: str) -> dict | None:
    """Fetch 2y of daily data, compute indicators, cache for 1 hour."""
    now = time.time()
    if ticker in _daily_cache:
        cached_ts, cached_data = _daily_cache[ticker]
        if now - cached_ts < DAILY_CACHE_TTL:
            return cached_data

    url  = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2y"
    data = safe_get(url)
    if not data or not data.get("chart", {}).get("result"):
        return None

    q      = data["chart"]["result"][0]["indicators"]["quote"][0]
    closes = [x for x in q.get("close", []) if x is not None]
    highs  = [x for x in q.get("high",  []) if x is not None]
    lows   = [x for x in q.get("low",   []) if x is not None]

    if len(closes) < 50:
        return None

    result = {
        "rsi14":   calculate_rsi(closes, 14),
        "ma50":    calculate_sma(closes, 50),
        "sma200":  calculate_sma(closes, 200),
        "macd":    calculate_macd(closes),
        "atr_pct": calculate_atr(highs, lows, closes, 14),
    }
    _daily_cache[ticker] = (now, result)
    return result

# -------------------------------------------------------------------
# NOTIFICATIONS
# -------------------------------------------------------------------

def send_push(title, message):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        return
    try:
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request(
            "POST", "/1/messages.json",
            urllib.parse.urlencode({
                "token":   PUSHOVER_TOKEN,
                "user":    PUSHOVER_USER,
                "title":   title,
                "message": message,
                "sound":   "echo",
            }),
            {"Content-type": "application/x-www-form-urlencoded"},
        )
        conn.getresponse()
    except Exception:
        pass

# -------------------------------------------------------------------
# GOOGLE SHEETS
# -------------------------------------------------------------------

def log_to_sheet(ticker, name, price, vwap_diff, rsi5, vol_ratio, pl_pct,
                 quick_signal, quick_why, rsi14_daily, ma50, sma200,
                 macd_daily, atr_pct, long_signal, long_why):
    if not GOOGLE_CREDENTIALS:
        print("  [SHEETS SKIPPED] GOOGLE_CREDENTIALS not set.")
        return
    try:
        import gspread
        gc    = gspread.service_account_from_dict(json.loads(GOOGLE_CREDENTIALS))
        sheet = gc.open_by_key(GOOGLE_SHEET_ID).sheet1

        if sheet.cell(1, 1).value != "Timestamp":
            sheet.insert_row(SHEET_HEADERS, index=1)

        now_est   = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=-5)))
        timestamp = now_est.strftime("%Y-%m-%d %H:%M EST")

        def fmt(v, d=2):
            return round(v, d) if v is not None else ""

        sheet.append_row([
            timestamp,
            ticker,
            name,
            fmt(price, 4),
            fmt(vwap_diff, 2),
            fmt(rsi5, 1),
            fmt(vol_ratio, 2),
            fmt(pl_pct, 1),
            quick_signal,
            quick_why,
            fmt(rsi14_daily, 1),
            fmt(ma50, 2),
            fmt(sma200, 2),
            str(round(macd_daily, 6)) if macd_daily is not None else "",
            fmt(atr_pct, 2),
            long_signal,
            long_why,
        ], value_input_option="USER_ENTERED")

        print(f"  [SHEETS] {ticker} | Quick: {quick_signal} | Long: {long_signal}")
    except Exception as exc:
        print(f"  [WARNING] Sheets failed: {exc}")

# -------------------------------------------------------------------
# MARKET TIMING
# -------------------------------------------------------------------

def _est_now():
    return datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=-5)))


def is_market_hours():
    now = _est_now()
    if now.weekday() >= 5:
        return False
    return now.replace(hour=9, minute=30, second=0) <= now <= now.replace(hour=16, minute=0, second=0)


def in_quick_window():
    now = _est_now()
    return now.replace(hour=9, minute=40) <= now <= now.replace(hour=15, minute=55)


def get_vwap_thresholds(ticker):
    if ticker in BROAD_ETFS:
        return -0.6
    if ticker in MEGA_CAPS:
        return -0.8
    if ticker in HIGH_BETA:
        return -1.2
    return -1.0

# -------------------------------------------------------------------
# SIGNALS  (logic unchanged from original)
# -------------------------------------------------------------------

def generate_quick_signal(ticker, price, vwap_diff_pct, rsi_5, vol_ratio, pl_pct, atr_pct):
    global daily_spent
    if not in_quick_window() or not is_market_hours():
        return "QUICK HOLD", "Outside quick window"

    if rsi_5 < 20 and vwap_diff_pct < get_vwap_thresholds(ticker) and vol_ratio > 1.5:
        amount = 20 / (2 if atr_pct and atr_pct > 2 else 1)
        if daily_spent + amount > DAILY_RISK_BUDGET:
            return "QUICK HOLD", "Risk budget exceeded"
        daily_spent += amount
        return f"QUICK BUY ${amount:.0f}", "RSI5 oversold + VWAP dip + vol spike"

    if ticker in HOLDINGS:
        if pl_pct > 20:
            pct, reason = 0.15, "20%+ quick profit → trim 15%"
        elif pl_pct > 10:
            pct, reason = 0.10, "10%+ quick profit → trim 10%"
        else:
            return "QUICK HOLD", "No quick edge"
        keep   = 0.05 if HOLDINGS[ticker]["never_sell_all"] else 0
        shares = min(HOLDINGS[ticker]["shares"] * pct, HOLDINGS[ticker]["shares"] - keep)
        dollars = shares * price
        if dollars > 10:
            return f"QUICK SELL ${dollars:,.0f}", reason

    return "QUICK HOLD", "No quick edge"


def generate_long_signal(ticker, price, rsi_14_daily, ma50_pullback_pct,
                         vol_ratio, pl_pct, atr_pct, sma200):
    global daily_spent
    if not is_market_hours():
        return "LONG HOLD", "Outside long window"

    uptrend = (price > sma200) if sma200 else True
    if not (uptrend or (rsi_14_daily and rsi_14_daily < 25)):
        return "LONG HOLD", "Not in uptrend"

    if (30 <= (rsi_14_daily or 50) <= 45
            and -6 <= (ma50_pullback_pct or 0) <= -3
            and vol_ratio > 1.2):
        amount = 100 / (2 if atr_pct and atr_pct > 2 else 1)
        if daily_spent + amount > DAILY_RISK_BUDGET:
            return "LONG HOLD", "Risk budget exceeded"
        daily_spent += amount
        return f"LONG BUY ${amount:.0f}", "RSI14 30-45 + MA50 pullback + vol confirm"

    if ticker in HOLDINGS:
        if pl_pct > 200:
            pct, reason = 0.50, "200%+ profit → taking half"
        elif pl_pct > 120:
            pct, reason = 0.35, "120%+ profit → trimming 35%"
        elif pl_pct > 70:
            pct, reason = 0.25, "70%+ profit → trimming 25%"
        else:
            return "LONG HOLD", "No long edge"
        keep   = 0.05 if HOLDINGS[ticker]["never_sell_all"] else 0
        shares = min(HOLDINGS[ticker]["shares"] * pct, HOLDINGS[ticker]["shares"] - keep)
        dollars = shares * price
        if dollars > 30:
            return f"LONG SELL ${dollars:,.0f}", reason

    return "LONG HOLD", "No long edge"

# -------------------------------------------------------------------
# SCAN
# -------------------------------------------------------------------

def run_scan():
    global daily_spent
    daily_spent = 0
    print(f"\n{'='*60}")
    print(f"  Scan — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    for symbol in STOCKS[:10]:
        try:
            # Name lookup
            quote_data = safe_get(f"https://query2.finance.yahoo.com/v7/finance/quote?symbols={symbol}")
            name = symbol
            if quote_data and quote_data.get("quoteResponse", {}).get("result"):
                name = quote_data["quoteResponse"]["result"][0].get("shortName", symbol)

            # Intraday 5m chart (5 days)
            data = safe_get(f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=5d")
            if not data or not data.get("chart", {}).get("result"):
                time.sleep(8)
                continue

            res    = data["chart"]["result"][0]
            price  = res["meta"].get("regularMarketPrice") or res["meta"].get("previousClose")
            q      = res["indicators"]["quote"][0]
            ts     = res.get("timestamp", [])
            highs  = q.get("high",   [])
            lows   = q.get("low",    [])
            closes = q.get("close",  [])
            vols   = q.get("volume", [])

            c = [x for x in closes if x is not None]

            # Real VWAP
            vwap          = calculate_vwap(ts, highs, lows, closes, vols)
            vwap_diff_pct = ((price - vwap) / vwap * 100) if vwap and price else 0.0

            # Intraday stats
            rsi_5     = calculate_rsi(c[-30:], 5) if len(c) >= 6 else 50.0
            vol_ratio = calculate_vol_ratio(ts, vols)
            pl_pct    = ((price / HOLDINGS.get(symbol, {"avg": price})["avg"]) - 1) * 100 if price else 0

            # Daily indicators (cached, one extra Yahoo call per ticker per hour)
            daily           = fetch_daily_indicators(symbol)
            rsi14_daily     = daily["rsi14"]   if daily else None
            ma50            = daily["ma50"]    if daily else None
            sma200          = daily["sma200"]  if daily else None
            macd_daily      = daily["macd"]    if daily else None
            atr_pct         = daily["atr_pct"] if daily else None
            ma50_pullback   = ((price / ma50) - 1) * 100 if ma50 and price else None

            # Signals
            quick_signal, quick_why = generate_quick_signal(
                symbol, price, vwap_diff_pct, rsi_5, vol_ratio, pl_pct, atr_pct)
            long_signal, long_why   = generate_long_signal(
                symbol, price, rsi14_daily, ma50_pullback, vol_ratio, pl_pct, atr_pct, sma200)

            rsi14_str = f"{rsi14_daily:.1f}" if rsi14_daily is not None else "N/A"
            print(f"  {symbol:<6} ${price:>8.2f}  VWAP diff: {vwap_diff_pct:+.2f}%"
                  f"  RSI5: {rsi_5:.1f}  RSI14: {rsi14_str}"
                  f"  | {quick_signal} / {long_signal}")

            log_to_sheet(symbol, name, price, vwap_diff_pct, rsi_5, vol_ratio, pl_pct,
                         quick_signal, quick_why, rsi14_daily, ma50, sma200,
                         macd_daily, atr_pct, long_signal, long_why)

            if "BUY" in quick_signal or "SELL" in quick_signal:
                send_push(f"{quick_signal} — {symbol}", f"{quick_why}\n${price:,.2f}")
            if "BUY" in long_signal or "SELL" in long_signal:
                send_push(f"{long_signal} — {symbol}", f"{long_why}\n${price:,.2f}")

            time.sleep(8 + random.uniform(4, 10))

        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            time.sleep(10)

    print("  Scan complete.\n")

# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    SCAN_INTERVAL = 1200  # 20 minutes
    while True:
        if is_market_hours():
            run_scan()
        else:
            print("Market closed — sleeping 1 hour.")
            time.sleep(3600)
            continue
        time.sleep(SCAN_INTERVAL)
