"""
EUR/USD EMA 20/50 Crossover Backtest — last 90 days of hourly data
-------------------------------------------------------------------
Simulates every trade that would have been triggered by the crossover
strategy and reports P&L per trade and overall totals.
"""

import yfinance as yf
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

TICKER    = "EURUSD=X"
EMA_SHORT = 20
EMA_LONG  = 50
CAPITAL   = 1000     # simulated starting capital in USD

# ── Fetch data ─────────────────────────────────────────────────────────────────

print("Downloading 90 days of EUR/USD hourly data...")
df = yf.download(TICKER, period="90d", interval="1h", progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["EMA_20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=EMA_LONG,  adjust=False).mean()
df.dropna(subset=["EMA_20", "EMA_50"], inplace=True)

# Drop the currently forming candle
df = df.iloc[:-1].copy()

print(f"Loaded {len(df)} closed hourly candles.\n")

# ── Detect all crossovers ──────────────────────────────────────────────────────

df["above"] = df["EMA_20"] > df["EMA_50"]
df["signal"] = None

for i in range(1, len(df)):
    prev_above = df["above"].iloc[i - 1]
    curr_above = df["above"].iloc[i]
    if not prev_above and curr_above:
        df.iloc[i, df.columns.get_loc("signal")] = "buy"
    elif prev_above and not curr_above:
        df.iloc[i, df.columns.get_loc("signal")] = "sell"

signals = df[df["signal"].notna()].copy()
print(f"Found {len(signals)} crossover signals.\n")

# ── Simulate trades ────────────────────────────────────────────────────────────
# Enter on BUY, exit on the next SELL (and vice versa for short trades).

trades = []
position = None  # {'type': 'long'/'short', 'entry_price': x, 'entry_time': t}

for _, row in signals.iterrows():
    if row["signal"] == "buy":
        if position is None:
            position = {
                "type":        "long",
                "entry_price": row["Close"],
                "entry_time":  row.name,
            }
    elif row["signal"] == "sell":
        if position and position["type"] == "long":
            # Close the long trade
            entry = position["entry_price"]
            exit_ = row["Close"]
            pips  = (exit_ - entry) * 10000
            pct   = (exit_ - entry) / entry * 100
            trades.append({
                "type":        "LONG",
                "entry_time":  position["entry_time"],
                "exit_time":   row.name,
                "entry_price": round(entry, 5),
                "exit_price":  round(exit_, 5),
                "pips":        round(pips, 1),
                "pct":         round(pct, 4),
            })
            position = None
        # Open a short trade
        position = {
            "type":        "short",
            "entry_price": row["Close"],
            "entry_time":  row.name,
        }
    # If a buy signal arrives while short, close short and go long
    if row["signal"] == "buy" and position and position["type"] == "short":
        # Already handled above — this case means we had a short open
        pass

# Close any open position at the last available price
if position:
    last_row   = df.iloc[-1]
    entry      = position["entry_price"]
    exit_      = float(last_row["Close"])
    multiplier = 1 if position["type"] == "long" else -1
    pips       = (exit_ - entry) * 10000 * multiplier
    pct        = (exit_ - entry) / entry * 100 * multiplier
    trades.append({
        "type":        position["type"].upper() + " (open)",
        "entry_time":  position["entry_time"],
        "exit_time":   last_row.name,
        "entry_price": round(entry, 5),
        "exit_price":  round(exit_, 5),
        "pips":        round(pips, 1),
        "pct":         round(pct, 4),
    })

# ── Print results ──────────────────────────────────────────────────────────────

print(f"{'─'*95}")
print(f"{'Type':<16} {'Entry Time':<22} {'Exit Time':<22} {'Entry':>8} {'Exit':>8} {'Pips':>8} {'%':>7}")
print(f"{'─'*95}")

total_pips = 0
wins = 0
losses = 0

for t in trades:
    total_pips += t["pips"]
    if t["pips"] > 0:
        wins += 1
    else:
        losses += 1
    print(
        f"{t['type']:<16} "
        f"{str(t['entry_time'])[:19]:<22} "
        f"{str(t['exit_time'])[:19]:<22} "
        f"{t['entry_price']:>8.5f} "
        f"{t['exit_price']:>8.5f} "
        f"{t['pips']:>+8.1f} "
        f"{t['pct']:>+7.3f}%"
    )

print(f"{'─'*95}")
print(f"\nTotal trades : {len(trades)}  ({wins} wins, {losses} losses)")
print(f"Win rate     : {wins/len(trades)*100:.1f}%" if trades else "")
print(f"Total pips   : {total_pips:+.1f}")

# P&L on $1,000 no leverage
pnl_no_lev = CAPITAL * (total_pips / 10000)
# P&L on $1,000 with 10:1 leverage (controlling $10,000)
pnl_10x    = CAPITAL * (total_pips / 10000) * 10
# P&L on $1,000 with 50:1 leverage (controlling $50,000)
pnl_50x    = CAPITAL * (total_pips / 10000) * 50

print(f"\n── Simulated P&L on ${CAPITAL:,} starting capital ──")
print(f"  No leverage (1:1) : ${pnl_no_lev:+.2f}")
print(f"  10:1 leverage     : ${pnl_10x:+.2f}")
print(f"  50:1 leverage     : ${pnl_50x:+.2f}")
print(f"\nNote: no spread, commission, or slippage included — real results will be slightly lower.")
