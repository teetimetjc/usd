"""
EUR/USD EMA Crossover Monitor
------------------------------
Fetches hourly EUR/USD price data, calculates EMA 20 and EMA 50,
and sends a Pushover notification + logs to Google Sheets on every run.
Crossover signals (buy/sell) are only confirmed on fully closed candles.

Secrets required (set as GitHub repository secrets):
  PUSHOVER_TOKEN      — Pushover app token
  PUSHOVER_USER_KEY   — Pushover user key
  GOOGLE_CREDENTIALS  — Google service account JSON (as a single-line string)
  GOOGLE_SHEET_ID     — ID from your Google Sheet URL
"""

import os
import json
import time
import datetime

import pytz
import requests
import yfinance as yf
import pandas as pd

# notify-run is optional — notifications are skipped if it isn't installed/configured
try:
    from notify_run import Notify
    notify = Notify()
    NOTIFY_AVAILABLE = True
except Exception:
    notify = None
    NOTIFY_AVAILABLE = False

# Pushover credentials — stored as GitHub Secrets
PUSHOVER_TOKEN    = os.environ.get("PUSHOVER_TOKEN", "")
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY", "")

# Google Sheets — credentials JSON stored as a GitHub Secret; sheet ID is hardcoded
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS", "")
GOOGLE_SHEET_ID    = "1-hVtI0klxQfmJtZCRA4U1gnKeMEG6GoMzjg8WdvbqqU"

# Set TEST_NOTIFICATION=true in workflow_dispatch inputs to verify Pushover
TEST_NOTIFICATION = os.environ.get("TEST_NOTIFICATION", "").lower() == "true"

# ── Configuration ──────────────────────────────────────────────────────────────

TICKER            = "EURUSD=X"   # Yahoo Finance symbol for EUR/USD
EMA_SHORT         = 20           # Fast EMA period
EMA_LONG          = 50           # Slow EMA period
INTERVAL          = "1h"         # Hourly candles
MARKET_TZ         = pytz.timezone("US/Eastern")
MARKET_OPEN_HOUR  = 8            # 8 AM EST
MARKET_CLOSE_HOUR = 12           # 12 PM EST (noon)

# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_price_data() -> pd.DataFrame | None:
    """Download recent hourly candles for EUR/USD from Yahoo Finance."""
    df = yf.download(TICKER, period="60d", interval=INTERVAL, progress=False, auto_adjust=True)
    if df is None or df.empty:
        print("  [ERROR] Could not fetch price data.")
        return None
    # Newer yfinance versions return MultiIndex columns like ("Close", "EURUSD=X").
    # Flatten to simple column names so the rest of the code works consistently.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA_20 and EMA_50 columns using pandas exponential weighting."""
    df = df.copy()
    df["EMA_20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=EMA_LONG, adjust=False).mean()
    return df

# ── Signal detection ───────────────────────────────────────────────────────────

def detect_crossover(df: pd.DataFrame) -> str | None:
    """
    Check the last two FULLY CLOSED candles for an EMA crossover.
    The currently forming candle (df.iloc[-1]) is excluded before this
    function is called, so df.iloc[-1] here is already the last closed bar.

    Returns 'buy', 'sell', or None.
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]  # closed candle before last
    curr = df.iloc[-1]  # last fully closed candle

    prev_above = prev["EMA_20"] > prev["EMA_50"]
    curr_above = curr["EMA_20"] > curr["EMA_50"]

    if not prev_above and curr_above:
        return "buy"    # EMA 20 just crossed above EMA 50
    if prev_above and not curr_above:
        return "sell"   # EMA 20 just crossed below EMA 50
    return None

# ── Notifications ──────────────────────────────────────────────────────────────

def send_notification(signal: str, price: float, timestamp: str,
                      ema20: float = 0, ema50: float = 0) -> None:
    """
    Send a Pushover notification.
    signal: 'buy', 'sell', or 'status' (routine update, no crossover).
    Buy/sell use high priority (sound + vibrate); status uses silent priority.
    """
    if signal == "buy":
        title    = "EUR/USD BUY Signal"
        header   = "BUY — EMA 20 crossed ABOVE EMA 50"
        priority = 1
    elif signal == "sell":
        title    = "EUR/USD SELL Signal"
        header   = "SELL — EMA 20 crossed BELOW EMA 50"
        priority = 1
    else:
        title    = "EUR/USD Status Update"
        trend    = "above" if ema20 > ema50 else "below"
        header   = f"No signal — EMA 20 is {trend} EMA 50"
        priority = -1   # silent, no sound for routine updates

    message = (
        f"{header}\n"
        f"Price: {price:.5f}\n"
        f"EMA20: {ema20:.5f}\n"
        f"EMA50: {ema50:.5f}\n"
        f"Time:  {timestamp}"
    )

    # ── Pushover ──
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

    # ── notify-run (local fallback) ──
    if not NOTIFY_AVAILABLE:
        print("  [NOTIFICATION SKIPPED] No notification service configured.")
        return
    try:
        notify.send(message)
        print(f"  [NOTIFICATION SENT via notify-run]")
    except Exception as exc:
        print(f"  [WARNING] notify-run failed: {exc}")

# ── Google Sheets logging ──────────────────────────────────────────────────────

def log_to_sheets(timestamp: str, price: float, ema20: float, ema50: float,
                  signal: str) -> None:
    """
    Append one row to the Google Sheet:
      Timestamp | Price | EMA 20 | EMA 50 | Signal
    Requires GOOGLE_CREDENTIALS (service account JSON) and GOOGLE_SHEET_ID.
    """
    if not GOOGLE_CREDENTIALS or not GOOGLE_SHEET_ID:
        print("  [SHEETS SKIPPED] GOOGLE_CREDENTIALS or GOOGLE_SHEET_ID not set.")
        return

    try:
        import gspread
        creds_dict = json.loads(GOOGLE_CREDENTIALS)
        gc = gspread.service_account_from_dict(creds_dict)
        sheet = gc.open_by_key(GOOGLE_SHEET_ID).sheet1

        # Write headers automatically if the sheet is empty
        if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
            sheet.insert_row(
                ["Timestamp", "Price", "EMA 20", "EMA 50", "Signal"], index=1
            )

        sheet.append_row(
            [timestamp, round(price, 5), round(ema20, 5), round(ema50, 5), signal.upper()],
            value_input_option="USER_ENTERED",
        )
        print(f"  [SHEETS] Row logged: {timestamp} | {price:.5f} | {signal.upper()}")
    except Exception as exc:
        print(f"  [WARNING] Google Sheets logging failed: {exc}")

# ── Main check (runs every hour) ───────────────────────────────────────────────

def check_ema_crossover() -> None:
    """
    Core logic:
      1. Fetch data and strip the currently forming candle.
      2. Calculate EMAs on closed candles only.
      3. Detect crossover, send Pushover notification, log to Google Sheets.
    """
    now_est   = datetime.datetime.now(MARKET_TZ)
    timestamp = now_est.strftime("%Y-%m-%d %H:%M %Z")

    print(f"\n{'='*55}")
    print(f"  Check at {timestamp}")

    # Test mode: verify Pushover credentials without needing a real signal
    if TEST_NOTIFICATION:
        print("  TEST MODE — sending test Pushover notification...")
        send_notification("buy", 1.08000, timestamp + " [TEST]", 1.07800, 1.07500)
        return

    # ── Fetch & calculate ──
    df = fetch_price_data()
    if df is None:
        return

    df = calculate_emas(df)
    df.dropna(subset=["EMA_20", "EMA_50"], inplace=True)

    # Drop the last row — it's the currently forming (not yet closed) candle
    df = df.iloc[:-1]

    if len(df) < 2:
        print("  [ERROR] Not enough closed candles to evaluate crossover.")
        return

    # All metrics are now from the last FULLY CLOSED candle
    latest = df.iloc[-1]
    price = float(latest["Close"])
    ema20 = float(latest["EMA_20"])
    ema50 = float(latest["EMA_50"])

    # ── Detect crossover ──
    signal = detect_crossover(df)

    if signal:
        label = "BUY  (EMA 20 crossed ABOVE EMA 50)" if signal == "buy" else "SELL (EMA 20 crossed BELOW EMA 50)"
        print(f"  *** SIGNAL: {label} ***")
    else:
        trend = "above" if ema20 > ema50 else "below"
        print(f"  No crossover — EMA 20 is {trend} EMA 50")

    print(f"  Price : {price:.5f}")
    print(f"  EMA 20: {ema20:.5f}")
    print(f"  EMA 50: {ema50:.5f}")

    # ── Notify + log ──
    send_notification(signal or "status", price, timestamp, ema20, ema50)
    log_to_sheets(timestamp, price, ema20, ema50, signal or "status")

    print(f"{'='*55}")

# ── Scheduler (local use only) ─────────────────────────────────────────────────

def main() -> None:
    import schedule  # only needed locally; GitHub Actions handles scheduling in CI

    print("EUR/USD EMA Crossover Monitor started.")
    print(f"  Watching : EMA {EMA_SHORT} / EMA {EMA_LONG} crossovers")
    print(f"  Window   : {MARKET_OPEN_HOUR}:00 – {MARKET_CLOSE_HOUR}:00 EST, Mon–Fri")
    print("  Checking every 60 minutes. Press Ctrl+C to stop.\n")

    check_ema_crossover()
    schedule.every(60).minutes.do(check_ema_crossover)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
