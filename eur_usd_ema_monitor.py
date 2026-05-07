"""
EUR/USD EMA Crossover Monitor — Enhanced
-----------------------------------------
Enhancements over the base version:
  1. ADX filter       — only signal when ADX > 25 (real trend exists)
  2. Daily alignment  — only take longs when daily EMA 20 > EMA 50, shorts when below
  3. 2-candle confirm — crossover must hold for 2 closed candles before firing
  4. ATR stop loss    — calculates and includes a suggested stop loss in every notification

Secrets required (GitHub repository secrets):
  PUSHOVER_TOKEN      — Pushover app token
  PUSHOVER_USER_KEY   — Pushover user key
  GOOGLE_CREDENTIALS  — Google service account JSON
"""

import os
import json
import time
import datetime

import pytz
import requests
import yfinance as yf
import pandas as pd

# notify-run optional fallback
try:
    from notify_run import Notify
    notify = Notify()
    NOTIFY_AVAILABLE = True
except Exception:
    notify = None
    NOTIFY_AVAILABLE = False

# ── Credentials ────────────────────────────────────────────────────────────────

PUSHOVER_TOKEN    = os.environ.get("PUSHOVER_TOKEN", "")
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY", "")
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS", "")
GOOGLE_SHEET_ID    = "1-hVtI0klxQfmJtZCRA4U1gnKeMEG6GoMzjg8WdvbqqU"
TEST_NOTIFICATION  = os.environ.get("TEST_NOTIFICATION", "").lower() == "true"

# ── Configuration ──────────────────────────────────────────────────────────────

TICKER            = "EURUSD=X"
EMA_SHORT         = 20
EMA_LONG          = 50
ADX_THRESHOLD     = 25      # minimum ADX to confirm a real trend is in play
ATR_MULTIPLIER    = 1.5     # stop loss = entry ± (ATR × this value)
ADX_PERIOD        = 14      # standard ADX/ATR period
INTERVAL          = "1h"
MARKET_TZ         = pytz.timezone("US/Eastern")

# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_and_prepare(interval="1h", period="60d") -> pd.DataFrame | None:
    """Download EUR/USD candles, flatten columns, drop forming candle."""
    df = yf.download(TICKER, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        print("  [ERROR] Could not fetch price data.")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if interval == "1h":
        df = df.iloc[:-1]   # drop the currently forming candle
    return df


def get_daily_trend() -> bool | None:
    """
    Return True if the daily EMA 20 is above EMA 50 (bullish),
    False if below (bearish), or None if data unavailable.
    """
    df = fetch_and_prepare(interval="1d", period="120d")
    if df is None or len(df) < EMA_LONG:
        return None
    df["EMA_20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=EMA_LONG,  adjust=False).mean()
    latest = df.iloc[-1]
    return float(latest["EMA_20"]) > float(latest["EMA_50"])

# ── Indicator calculation ──────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA_20, EMA_50, ATR, and ADX columns to the DataFrame."""
    df = df.copy()

    df["EMA_20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=EMA_LONG,  adjust=False).mean()

    # True Range
    prev_close = df["Close"].shift(1)
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # ATR (Wilder smoothing)
    df["ATR"] = df["TR"].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    # Directional movement
    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm  > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,  0)

    plus_di  = 100 * plus_dm.ewm( alpha=1/ADX_PERIOD, adjust=False).mean() / df["ATR"]
    minus_di = 100 * minus_dm.ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df["ATR"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    df["ADX"] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    return df

# ── Signal detection ───────────────────────────────────────────────────────────

def detect_crossover(df: pd.DataFrame) -> str | None:
    """
    2-candle confirmed crossover detection.
    Requires the EMA cross to have occurred on bar[-2] AND bar[-1] still holds.
    This filters out single-candle false crosses.
    """
    if len(df) < 3:
        return None

    before  = df.iloc[-3]   # state before crossover
    crossed = df.iloc[-2]   # bar where cross happened
    confirm = df.iloc[-1]   # confirmation bar (must hold the cross)

    before_above  = before["EMA_20"]  > before["EMA_50"]
    crossed_above = crossed["EMA_20"] > crossed["EMA_50"]
    confirm_above = confirm["EMA_20"] > confirm["EMA_50"]

    if not before_above and crossed_above and confirm_above:
        return "buy"
    if before_above and not crossed_above and not confirm_above:
        return "sell"
    return None

# ── Notification helpers ───────────────────────────────────────────────────────

def build_reason(signal: str, ema20: float, ema50: float,
                 adx: float, daily_bullish: bool | None,
                 stop_loss: float | None = None) -> str:
    gap_pips  = abs(ema20 - ema50) * 10000
    daily_str = "bullish" if daily_bullish else ("bearish" if daily_bullish is False else "unknown")

    if signal == "buy":
        base = (f"EMA 20 crossed ABOVE EMA 50 — confirmed over 2 candles. "
                f"Gap: +{gap_pips:.1f} pips. ADX: {adx:.1f} (trend strength). "
                f"Daily trend: {daily_str}.")
    elif signal == "sell":
        base = (f"EMA 20 crossed BELOW EMA 50 — confirmed over 2 candles. "
                f"Gap: -{gap_pips:.1f} pips. ADX: {adx:.1f} (trend strength). "
                f"Daily trend: {daily_str}.")
    else:
        direction = "above" if ema20 > ema50 else "below"
        base = (f"No crossover. EMA 20 is {direction} EMA 50 "
                f"by {gap_pips:.1f} pips — trend continuing. "
                f"ADX: {adx:.1f}. Daily trend: {daily_str}.")

    if stop_loss:
        base += f" Suggested stop loss: {stop_loss:.5f}."

    return base


def send_notification(signal: str, price: float, timestamp: str,
                      ema20: float = 0, ema50: float = 0,
                      adx: float = 0, daily_bullish: bool | None = None,
                      stop_loss: float | None = None,
                      reason: str = "") -> None:
    """Send Pushover notification (or notify-run fallback)."""
    if signal == "buy":
        title    = "EUR/USD BUY Signal"
        priority = 1
    elif signal == "sell":
        title    = "EUR/USD SELL Signal"
        priority = 1
    else:
        title    = "EUR/USD Status Update"
        priority = -1

    gap_pips   = (ema20 - ema50) * 10000
    daily_str  = "Bullish" if daily_bullish else ("Bearish" if daily_bullish is False else "N/A")
    stop_str   = f"\nStop:     {stop_loss:.5f}" if stop_loss else ""

    message = (
        f"{reason}\n\n"
        f"Price:    {price:.5f}\n"
        f"EMA 20:   {ema20:.5f}\n"
        f"EMA 50:   {ema50:.5f}\n"
        f"EMA Gap:  {gap_pips:+.1f} pips\n"
        f"ADX:      {adx:.1f}\n"
        f"Daily:    {daily_str}"
        f"{stop_str}\n"
        f"Time:     {timestamp}"
    )

    if PUSHOVER_TOKEN and PUSHOVER_USER_KEY:
        try:
            resp = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token":    PUSHOVER_TOKEN,
                    "user":     PUSHOVER_USER_KEY,
                    "title":    title,
                    "message":  message,
                    "priority": priority,
                },
                timeout=10,
            )
            print(f"  [PUSHOVER] HTTP {resp.status_code} — {resp.text}")
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [WARNING] Pushover failed: {exc}")
        return

    if not NOTIFY_AVAILABLE:
        print("  [NOTIFICATION SKIPPED] No service configured.")
        return
    try:
        notify.send(message)
    except Exception as exc:
        print(f"  [WARNING] notify-run failed: {exc}")

# ── Google Sheets logging ──────────────────────────────────────────────────────

SHEET_HEADERS = [
    "Timestamp", "Price", "EMA 20", "EMA 50",
    "EMA Gap (pips)", "ADX", "Daily Trend", "Signal", "Stop Loss", "Reason"
]

def log_to_sheets(timestamp: str, price: float, ema20: float, ema50: float,
                  adx: float, daily_bullish: bool | None,
                  signal: str, stop_loss: float | None, reason: str) -> None:
    if not GOOGLE_CREDENTIALS or not GOOGLE_SHEET_ID:
        print("  [SHEETS SKIPPED] Credentials not set.")
        return
    try:
        import gspread
        gc    = gspread.service_account_from_dict(json.loads(GOOGLE_CREDENTIALS))
        sheet = gc.open_by_key(GOOGLE_SHEET_ID).sheet1

        if sheet.cell(1, 1).value != "Timestamp":
            sheet.insert_row(SHEET_HEADERS, index=1)

        daily_str = "Bullish" if daily_bullish else ("Bearish" if daily_bullish is False else "N/A")
        gap_pips  = round((ema20 - ema50) * 10000, 1)

        sheet.append_row([
            timestamp,
            round(price, 5),
            round(ema20, 5),
            round(ema50, 5),
            gap_pips,
            round(adx, 1),
            daily_str,
            signal.upper(),
            round(stop_loss, 5) if stop_loss else "N/A",
            reason,
        ], value_input_option="USER_ENTERED")

        print(f"  [SHEETS] Logged: {timestamp} | {signal.upper()} | ADX {adx:.1f}")
    except Exception as exc:
        print(f"  [WARNING] Sheets failed: {exc}")

# ── Main check ─────────────────────────────────────────────────────────────────

def check_ema_crossover() -> None:
    now_est   = datetime.datetime.now(MARKET_TZ)
    timestamp = now_est.strftime("%Y-%m-%d %H:%M %Z")

    print(f"\n{'='*60}")
    print(f"  Check at {timestamp}")

    if TEST_NOTIFICATION:
        print("  TEST MODE — sending test Pushover notification...")
        send_notification("buy", 1.08000, timestamp + " [TEST]",
                          1.07800, 1.07500, adx=32.5,
                          daily_bullish=True, stop_loss=1.07650,
                          reason="Test notification — not a real signal.")
        return

    # ── Fetch hourly data ──
    df = fetch_and_prepare(interval="1h", period="60d")
    if df is None:
        return

    df = calculate_indicators(df)
    df.dropna(subset=["EMA_20", "EMA_50", "ADX", "ATR"], inplace=True)

    if len(df) < 3:
        print("  [ERROR] Not enough data.")
        return

    latest = df.iloc[-1]
    price  = float(latest["Close"])
    ema20  = float(latest["EMA_20"])
    ema50  = float(latest["EMA_50"])
    adx    = float(latest["ADX"])
    atr    = float(latest["ATR"])

    # ── Fetch daily trend ──
    daily_bullish = get_daily_trend()
    daily_str     = "Bullish" if daily_bullish else ("Bearish" if daily_bullish is False else "N/A")

    # ── Detect 2-candle confirmed crossover ──
    signal = detect_crossover(df)

    # ── Apply ADX filter ──
    if signal and adx < ADX_THRESHOLD:
        print(f"  Signal filtered — ADX {adx:.1f} < {ADX_THRESHOLD} (market not trending)")
        signal = None

    # ── Apply daily trend alignment filter ──
    if signal == "buy"  and daily_bullish is False:
        print(f"  BUY filtered — daily trend is bearish")
        signal = None
    elif signal == "sell" and daily_bullish is True:
        print(f"  SELL filtered — daily trend is bullish")
        signal = None

    # ── Calculate ATR-based stop loss ──
    stop_loss = None
    if signal == "buy":
        stop_loss = price - (ATR_MULTIPLIER * atr)
    elif signal == "sell":
        stop_loss = price + (ATR_MULTIPLIER * atr)

    # ── Print status ──
    if signal:
        label = "BUY (EMA 20 crossed ABOVE EMA 50)" if signal == "buy" else "SELL (EMA 20 crossed BELOW EMA 50)"
        print(f"  *** SIGNAL: {label} ***")
    else:
        trend = "above" if ema20 > ema50 else "below"
        print(f"  No signal — EMA 20 is {trend} EMA 50")

    print(f"  Price    : {price:.5f}")
    print(f"  EMA 20   : {ema20:.5f}")
    print(f"  EMA 50   : {ema50:.5f}")
    print(f"  ADX      : {adx:.1f}  ({'trending' if adx >= ADX_THRESHOLD else 'choppy — signals filtered'})")
    print(f"  Daily    : {daily_str}")
    if stop_loss:
        print(f"  Stop loss: {stop_loss:.5f}")

    reason = build_reason(signal or "status", ema20, ema50, adx, daily_bullish, stop_loss)

    send_notification(signal or "status", price, timestamp,
                      ema20, ema50, adx, daily_bullish, stop_loss, reason)
    log_to_sheets(timestamp, price, ema20, ema50, adx, daily_bullish,
                  signal or "status", stop_loss, reason)

    print(f"{'='*60}")

# ── Scheduler (local) ──────────────────────────────────────────────────────────

def main() -> None:
    import schedule

    print("EUR/USD EMA Monitor started (enhanced).")
    print(f"  Filters: ADX > {ADX_THRESHOLD}, daily alignment, 2-candle confirm, ATR stop")
    print("  Checking every 60 minutes. Press Ctrl+C to stop.\n")

    check_ema_crossover()
    schedule.every(60).minutes.do(check_ema_crossover)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
