"""
EUR/USD EMA Crossover Monitor
------------------------------
Fetches hourly EUR/USD price data, calculates EMA 20 and EMA 50,
and sends a push notification when the EMA 20 crosses above or below
the EMA 50 (buy or sell signal). Runs every hour during market hours
(8 AM – 12 PM EST, Monday–Friday).

Setup:
  1. Install dependencies:
       pip install yfinance pandas schedule pytz notify-run
  2. Register a notify-run channel (one-time, free):
       notify-run register
     Copy the channel URL shown — open it on your phone and tap Subscribe.
  3. Run:
       python eur_usd_ema_monitor.py
"""

import os
import time
import datetime

import pytz
import requests
import schedule
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

# Pushover credentials — stored as GitHub Secrets (PUSHOVER_TOKEN and PUSHOVER_USER_KEY)
PUSHOVER_TOKEN = os.environ.get("PUSHOVER_TOKEN", "")
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY", "")

# ── Configuration ──────────────────────────────────────────────────────────────

TICKER = "EURUSD=X"          # Yahoo Finance symbol for EUR/USD
EMA_SHORT = 20               # Fast EMA period
EMA_LONG = 50                # Slow EMA period
INTERVAL = "1h"              # Hourly candles
# Fetch enough bars so EMA 50 is fully warmed up (2× EMA_LONG as a buffer)
LOOKBACK_BARS = EMA_LONG * 2
MARKET_TZ = pytz.timezone("US/Eastern")
MARKET_OPEN_HOUR = 8         # 8 AM EST
MARKET_CLOSE_HOUR = 12       # 12 PM EST (noon)

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_market_hours() -> bool:
    """Return True if the current time is within the configured market window."""
    now = datetime.datetime.now(MARKET_TZ)
    # Monday = 0, Friday = 4
    if now.weekday() > 4:
        return False
    return MARKET_OPEN_HOUR <= now.hour < MARKET_CLOSE_HOUR


def fetch_price_data() -> pd.DataFrame | None:
    """
    Download recent hourly candles for EUR/USD from Yahoo Finance.
    Returns a DataFrame with at minimum a 'Close' column, or None on failure.
    """
    # 'period' of 60 days gives plenty of hourly bars for the EMA calculation
    df = yf.download(TICKER, period="60d", interval=INTERVAL, progress=False, auto_adjust=True)
    if df is None or df.empty:
        print("  [ERROR] Could not fetch price data.")
        return None
    return df


def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA_20 and EMA_50 columns to the DataFrame.
    pandas ewm() uses the standard exponential-weighting formula.
    """
    df = df.copy()
    df["EMA_20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=EMA_LONG, adjust=False).mean()
    return df


def detect_crossover(df: pd.DataFrame) -> str | None:
    """
    Compare the last two bars to decide if a crossover just occurred.

    Returns:
        'buy'  – EMA 20 crossed ABOVE EMA 50 (bullish)
        'sell' – EMA 20 crossed BELOW EMA 50 (bearish)
        None   – no crossover on the latest bar
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]   # second-to-last bar
    curr = df.iloc[-1]   # most recent bar

    prev_above = prev["EMA_20"] > prev["EMA_50"]
    curr_above = curr["EMA_20"] > curr["EMA_50"]

    if not prev_above and curr_above:
        return "buy"     # EMA 20 just crossed above EMA 50
    if prev_above and not curr_above:
        return "sell"    # EMA 20 just crossed below EMA 50
    return None


def send_notification(signal: str, price: float, timestamp: str) -> None:
    """
    Send a push notification via Pushover (GitHub Actions) or notify-run (local).
    Pushover is used when PUSHOVER_TOKEN and PUSHOVER_USER_KEY env vars are set.
    """
    action = "BUY" if signal == "buy" else "SELL"
    message = (
        f"EUR/USD {action} SIGNAL\n"
        f"Price: {price:.5f}\n"
        f"Time:  {timestamp}"
    )

    # ── Pushover (used in GitHub Actions via repository secrets) ──
    if PUSHOVER_TOKEN and PUSHOVER_USER_KEY:
        try:
            resp = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": PUSHOVER_TOKEN,
                    "user": PUSHOVER_USER_KEY,
                    "title": f"EUR/USD {action} Signal",
                    "message": message,
                    "priority": 1,   # high priority — bypasses quiet hours
                },
                timeout=10,
            )
            resp.raise_for_status()
            print(f"  [NOTIFICATION SENT via Pushover] {message}")
        except Exception as exc:
            print(f"  [WARNING] Pushover notification failed: {exc}")
        return

    # ── notify-run (local fallback) ──
    if not NOTIFY_AVAILABLE:
        print("  [NOTIFICATION SKIPPED] No notification service configured.")
        return
    try:
        notify.send(message)
        print(f"  [NOTIFICATION SENT via notify-run] {message}")
    except Exception as exc:
        print(f"  [WARNING] notify-run notification failed: {exc}")


# ── Main check (runs every hour) ───────────────────────────────────────────────

def check_ema_crossover() -> None:
    """
    Core logic executed each scheduled run:
      1. Verify we are within market hours.
      2. Fetch data and calculate EMAs.
      3. Detect a crossover and notify, or print a status line.
    """
    now_est = datetime.datetime.now(MARKET_TZ)
    timestamp = now_est.strftime("%Y-%m-%d %H:%M %Z")

    print(f"\n{'='*55}")
    print(f"  Check at {timestamp}")

    # Skip outside market hours
    if not is_market_hours():
        print("  Outside market hours — skipping.")
        return

    # ── Fetch & process ──
    df = fetch_price_data()
    if df is None:
        return

    df = calculate_emas(df)

    # Drop rows where EMAs are not yet fully warmed up
    df.dropna(subset=["EMA_20", "EMA_50"], inplace=True)
    if len(df) < 2:
        print("  [ERROR] Not enough data to evaluate crossover.")
        return

    latest = df.iloc[-1]
    current_price = float(latest["Close"].iloc[0]) if hasattr(latest["Close"], "iloc") else float(latest["Close"])
    ema20 = float(latest["EMA_20"].iloc[0]) if hasattr(latest["EMA_20"], "iloc") else float(latest["EMA_20"])
    ema50 = float(latest["EMA_50"].iloc[0]) if hasattr(latest["EMA_50"], "iloc") else float(latest["EMA_50"])

    # ── Crossover detection ──
    signal = detect_crossover(df)

    if signal:
        label = "BUY  (EMA 20 crossed ABOVE EMA 50)" if signal == "buy" else "SELL (EMA 20 crossed BELOW EMA 50)"
        print(f"  *** SIGNAL: {label} ***")
        print(f"  Price : {current_price:.5f}")
        print(f"  EMA 20: {ema20:.5f}")
        print(f"  EMA 50: {ema50:.5f}")
        print(f"  Time  : {timestamp}")
        send_notification(signal, current_price, timestamp)
    else:
        # Status update so you can confirm the script is running
        trend = "above" if ema20 > ema50 else "below"
        print(f"  No crossover.  EMA 20 is {trend} EMA 50.")
        print(f"  Price : {current_price:.5f}")
        print(f"  EMA 20: {ema20:.5f}")
        print(f"  EMA 50: {ema50:.5f}")

    print(f"{'='*55}")


# ── Scheduler ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("EUR/USD EMA Crossover Monitor started.")
    print(f"  Watching : EMA {EMA_SHORT} / EMA {EMA_LONG} crossovers")
    print(f"  Window   : {MARKET_OPEN_HOUR}:00 – {MARKET_CLOSE_HOUR}:00 EST, Mon–Fri")
    print("  Checking every 60 minutes. Press Ctrl+C to stop.\n")

    # Run once immediately so you see output right away
    check_ema_crossover()

    # Then repeat every 60 minutes
    schedule.every(60).minutes.do(check_ema_crossover)

    while True:
        schedule.run_pending()
        time.sleep(30)   # poll the scheduler every 30 s


if __name__ == "__main__":
    main()
