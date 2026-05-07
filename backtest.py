"""
EUR/USD EMA 20/50 Crossover Backtest — last 90 days of hourly data
-------------------------------------------------------------------
Simulates every trade triggered by EMA crossovers and reports
P&L per trade and overall totals.

Two strategies shown:
  Long-only  — buy on BUY signal, exit on SELL signal
  Long+Short — buy on BUY signal, sell/short on SELL signal
"""

import yfinance as yf
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

TICKER    = "EURUSD=X"
EMA_SHORT = 20
EMA_LONG  = 50
CAPITAL   = 1000     # simulated starting capital in USD

# ── Fetch & prepare data ───────────────────────────────────────────────────────

print("Downloading 90 days of EUR/USD hourly data...")
df = yf.download(TICKER, period="90d", interval="1h", progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["EMA_20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=EMA_LONG,  adjust=False).mean()
df.dropna(subset=["EMA_20", "EMA_50"], inplace=True)
df = df.iloc[:-1].copy()  # drop forming candle

print(f"Loaded {len(df)} closed hourly candles.\n")

# ── Detect crossover signals ───────────────────────────────────────────────────

df["above"]  = df["EMA_20"] > df["EMA_50"]
df["signal"] = None

for i in range(1, len(df)):
    if not df["above"].iloc[i-1] and df["above"].iloc[i]:
        df.iloc[i, df.columns.get_loc("signal")] = "buy"
    elif df["above"].iloc[i-1] and not df["above"].iloc[i]:
        df.iloc[i, df.columns.get_loc("signal")] = "sell"

signals = df[df["signal"].notna()].copy()
print(f"Found {len(signals)} crossover signals.\n")

# ── Simulation helper ──────────────────────────────────────────────────────────

def simulate(signals, df, mode="long_only"):
    """
    mode='long_only'  — only take long (buy) trades, skip shorting
    mode='long_short' — go long on BUY, flip to short on SELL
    """
    trades   = []
    position = None   # None, or dict with type/entry_price/entry_time

    for _, row in signals.iterrows():
        sig   = row["signal"]
        price = float(row["Close"])
        time  = row.name

        if sig == "buy":
            if position is None:
                # No position — open long
                position = {"type": "long", "entry_price": price, "entry_time": time}

            elif position["type"] == "short" and mode == "long_short":
                # Close short, open long
                pips = (position["entry_price"] - price) * 10000  # short profits when price falls
                trades.append({
                    "type":        "SHORT",
                    "entry_time":  position["entry_time"],
                    "exit_time":   time,
                    "entry_price": round(position["entry_price"], 5),
                    "exit_price":  round(price, 5),
                    "pips":        round(pips, 1),
                    "pct":         round(-((price - position["entry_price"]) / position["entry_price"]) * 100, 4),
                })
                position = {"type": "long", "entry_price": price, "entry_time": time}

        elif sig == "sell":
            if position and position["type"] == "long":
                # Close long
                pips = (price - position["entry_price"]) * 10000
                trades.append({
                    "type":        "LONG",
                    "entry_time":  position["entry_time"],
                    "exit_time":   time,
                    "entry_price": round(position["entry_price"], 5),
                    "exit_price":  round(price, 5),
                    "pips":        round(pips, 1),
                    "pct":         round(((price - position["entry_price"]) / position["entry_price"]) * 100, 4),
                })
                position = None

            if mode == "long_short":
                # Open short
                position = {"type": "short", "entry_price": price, "entry_time": time}

    # Close any open position at last available price
    if position:
        last_price = float(df.iloc[-1]["Close"])
        last_time  = df.iloc[-1].name
        if position["type"] == "long":
            pips = (last_price - position["entry_price"]) * 10000
            pct  = ((last_price - position["entry_price"]) / position["entry_price"]) * 100
        else:
            pips = (position["entry_price"] - last_price) * 10000
            pct  = -((last_price - position["entry_price"]) / position["entry_price"]) * 100
        trades.append({
            "type":        position["type"].upper() + " (open)",
            "entry_time":  position["entry_time"],
            "exit_time":   last_time,
            "entry_price": round(position["entry_price"], 5),
            "exit_price":  round(last_price, 5),
            "pips":        round(pips, 1),
            "pct":         round(pct, 4),
        })

    return trades


def print_results(trades, label):
    print(f"\n{'═'*100}")
    print(f"  {label}")
    print(f"{'═'*100}")
    print(f"{'Type':<18} {'Entry Time':<22} {'Exit Time':<22} {'Entry':>8} {'Exit':>8} {'Pips':>8} {'%':>8}")
    print(f"{'─'*100}")

    total_pips = 0
    wins = losses = 0

    for t in trades:
        marker = "✓" if t["pips"] > 0 else "✗"
        print(
            f"{t['type']:<18} "
            f"{str(t['entry_time'])[:19]:<22} "
            f"{str(t['exit_time'])[:19]:<22} "
            f"{t['entry_price']:>8.5f} "
            f"{t['exit_price']:>8.5f} "
            f"{t['pips']:>+8.1f} "
            f"{t['pct']:>+8.3f}%  {marker}"
        )
        total_pips += t["pips"]
        if "(open)" not in t["type"]:
            if t["pips"] > 0: wins += 1
            else: losses += 1

    closed = wins + losses
    print(f"{'─'*100}")
    print(f"Closed trades : {closed}  ({wins} wins, {losses} losses)  |  Win rate: {wins/closed*100:.1f}%" if closed else "No closed trades")
    print(f"Total pips    : {total_pips:+.1f}")

    pnl_1x  = CAPITAL * (total_pips / 10000)
    pnl_10x = pnl_1x * 10
    pnl_50x = pnl_1x * 50

    print(f"\n  P&L on ${CAPITAL:,} — no leverage : ${pnl_1x:+.2f}")
    print(f"  P&L on ${CAPITAL:,} — 10:1 leverage : ${pnl_10x:+.2f}")
    print(f"  P&L on ${CAPITAL:,} — 50:1 leverage : ${pnl_50x:+.2f}")
    print(f"\n  * No spread, commission, or slippage included.")


# ── Run both strategies ────────────────────────────────────────────────────────

long_only_trades   = simulate(signals, df, mode="long_only")
long_short_trades  = simulate(signals, df, mode="long_short")

print_results(long_only_trades,  "LONG-ONLY strategy (buy on BUY signal, exit on SELL signal)")
print_results(long_short_trades, "LONG + SHORT strategy (flip direction on every crossover)")
