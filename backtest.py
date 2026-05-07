"""
EUR/USD EMA 20/50 Crossover Backtest — Enhanced
-------------------------------------------------
Applies the same four filters as the live monitor:
  1. ADX filter       — only trade when ADX > 25
  2. Daily alignment  — only longs when daily EMA20 > EMA50, shorts when below
  3. 2-candle confirm — crossover must hold for 2 closed candles
  4. ATR stop loss    — position closed early if price hits the ATR stop

Two strategies shown:
  Long-only  — buy on confirmed BUY signal, exit on SELL signal or stop
  Long+Short — flip direction on every confirmed signal, stop applies to each leg
"""

import yfinance as yf
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

TICKER         = "EURUSD=X"
EMA_SHORT      = 20
EMA_LONG       = 50
ADX_PERIOD     = 14
ADX_THRESHOLD  = 25
ATR_MULTIPLIER = 1.5
CAPITAL        = 1000

# ── Helpers ────────────────────────────────────────────────────────────────────

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_20"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=EMA_LONG,  adjust=False).mean()

    prev_close = df["Close"].shift(1)
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm  > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,  0)
    plus_di  = 100 * plus_dm.ewm( alpha=1/ADX_PERIOD, adjust=False).mean() / df["ATR"]
    minus_di = 100 * minus_dm.ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df["ATR"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    df["ADX"] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    return df

# ── Fetch data ─────────────────────────────────────────────────────────────────

print("Downloading 1 year of EUR/USD hourly data...")
raw = yf.download(TICKER, period="1y", interval="1h", progress=False, auto_adjust=True)
raw = flatten(raw)
raw = raw.iloc[:-1].copy()   # drop forming candle

print("Downloading daily EUR/USD data for trend filter...")
daily_raw = yf.download(TICKER, period="2y", interval="1d", progress=False, auto_adjust=True)
daily_raw = flatten(daily_raw)
daily_raw["EMA_20"] = daily_raw["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
daily_raw["EMA_50"] = daily_raw["Close"].ewm(span=EMA_LONG,  adjust=False).mean()
daily_raw["daily_bullish"] = daily_raw["EMA_20"] > daily_raw["EMA_50"]
daily_raw.index = daily_raw.index.normalize()   # strip time for date-based lookup

df = add_indicators(raw)
df.dropna(subset=["EMA_20", "EMA_50", "ADX", "ATR"], inplace=True)
print(f"Loaded {len(df)} closed hourly candles.\n")


def get_daily_bullish(ts) -> bool | None:
    """Return daily trend (True=bullish) on the date of ts, or None if unavailable."""
    date = ts.normalize() if hasattr(ts, "normalize") else pd.Timestamp(ts).normalize()
    if date in daily_raw.index:
        return bool(daily_raw.loc[date, "daily_bullish"])
    # fall back to the most recent available daily bar before this date
    prior = daily_raw[daily_raw.index <= date]
    if prior.empty:
        return None
    return bool(prior.iloc[-1]["daily_bullish"])

# ── Build confirmed signal list ────────────────────────────────────────────────

records = []
rows    = list(df.itertuples())

for i in range(2, len(rows)):
    before  = rows[i-2]
    crossed = rows[i-1]
    confirm = rows[i]

    before_above  = before.EMA_20  > before.EMA_50
    crossed_above = crossed.EMA_20 > crossed.EMA_50
    confirm_above = confirm.EMA_20 > confirm.EMA_50

    if not before_above and crossed_above and confirm_above:
        raw_signal = "buy"
    elif before_above and not crossed_above and not confirm_above:
        raw_signal = "sell"
    else:
        continue

    adx = confirm.ADX
    if adx < ADX_THRESHOLD:
        continue   # ADX filter

    daily_bullish = get_daily_bullish(confirm.Index)
    if raw_signal == "buy"  and daily_bullish is False:
        continue   # daily alignment filter
    if raw_signal == "sell" and daily_bullish is True:
        continue

    records.append({
        "time":          confirm.Index,
        "signal":        raw_signal,
        "price":         confirm.Close,
        "atr":           confirm.ATR,
        "adx":           adx,
        "daily_bullish": daily_bullish,
    })

signals = pd.DataFrame(records)
print(f"Found {len(signals)} confirmed+filtered signals.\n")

# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate(signals: pd.DataFrame, df: pd.DataFrame, mode="long_only"):
    trades   = []
    position = None

    sig_times = set(signals["time"])

    # Build a fast lookup: timestamp → row data
    df_dict = {row.Index: row for row in df.itertuples()}
    all_times = list(df.index)

    def close_trade(pos, exit_price, exit_time, reason="signal"):
        if pos["type"] == "long":
            pips = (exit_price - pos["entry_price"]) * 10000
            pct  = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
        else:
            pips = (pos["entry_price"] - exit_price) * 10000
            pct  = -((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
        return {
            "type":        pos["type"].upper() + ("" if reason == "signal" else f" ({reason})"),
            "entry_time":  pos["entry_time"],
            "exit_time":   exit_time,
            "entry_price": round(pos["entry_price"], 5),
            "exit_price":  round(exit_price, 5),
            "pips":        round(pips, 1),
            "pct":         round(pct, 4),
        }

    sig_iter = iter(signals.iterrows())
    next_sig = next(sig_iter, None)

    for ts in all_times:
        if ts not in df_dict:
            continue
        row = df_dict[ts]
        low  = float(row.Low)
        high = float(row.High)
        close = float(row.Close)

        # Check stop loss hit on open position (use candle low/high)
        if position is not None:
            stop = position["stop"]
            if position["type"] == "long" and low <= stop:
                trades.append(close_trade(position, stop, ts, "stop"))
                position = None
            elif position["type"] == "short" and high >= stop:
                trades.append(close_trade(position, stop, ts, "stop"))
                position = None

        # Check if this bar has a signal
        if next_sig is not None and next_sig[1]["time"] == ts:
            _, sig_row = next_sig
            sig   = sig_row["signal"]
            price = float(sig_row["price"])
            atr   = float(sig_row["atr"])
            next_sig = next(sig_iter, None)

            if sig == "buy":
                stop = price - ATR_MULTIPLIER * atr
                if position is None:
                    position = {"type": "long", "entry_price": price, "entry_time": ts, "stop": stop}
                elif position["type"] == "short" and mode == "long_short":
                    trades.append(close_trade(position, price, ts))
                    position = {"type": "long", "entry_price": price, "entry_time": ts, "stop": stop}

            elif sig == "sell":
                stop = price + ATR_MULTIPLIER * atr
                if position is not None and position["type"] == "long":
                    trades.append(close_trade(position, price, ts))
                    position = None
                if mode == "long_short" and position is None:
                    position = {"type": "short", "entry_price": price, "entry_time": ts, "stop": stop}

    # Close open position at last price
    if position:
        last = df.iloc[-1]
        last_price = float(last["Close"])
        last_time  = last.name
        trades.append(close_trade(position, last_price, last_time, "open"))

    return trades


def print_results(trades, label):
    print(f"\n{'═'*110}")
    print(f"  {label}")
    print(f"{'═'*110}")
    print(f"{'Type':<22} {'Entry Time':<22} {'Exit Time':<22} {'Entry':>8} {'Exit':>8} {'Pips':>8} {'%':>8}")
    print(f"{'─'*110}")

    total_pips = 0
    wins = losses = stops = 0

    for t in trades:
        marker = "✓" if t["pips"] > 0 else "✗"
        print(
            f"{t['type']:<22} "
            f"{str(t['entry_time'])[:19]:<22} "
            f"{str(t['exit_time'])[:19]:<22} "
            f"{t['entry_price']:>8.5f} "
            f"{t['exit_price']:>8.5f} "
            f"{t['pips']:>+8.1f} "
            f"{t['pct']:>+8.3f}%  {marker}"
        )
        total_pips += t["pips"]
        is_open = "open" in t["type"]
        is_stop = "stop" in t["type"]
        if not is_open:
            if t["pips"] > 0:
                wins += 1
            else:
                losses += 1
            if is_stop:
                stops += 1

    closed = wins + losses
    print(f"{'─'*110}")
    if closed:
        print(f"Closed trades : {closed}  ({wins} wins, {losses} losses)  |  Win rate: {wins/closed*100:.1f}%  |  Stop-outs: {stops}")
    else:
        print("No closed trades")
    print(f"Total pips    : {total_pips:+.1f}")

    pnl_1x  = CAPITAL * (total_pips / 10000)
    pnl_10x = pnl_1x * 10
    pnl_50x = pnl_1x * 50

    print(f"\n  P&L on ${CAPITAL:,} — no leverage   : ${pnl_1x:+.2f}")
    print(f"  P&L on ${CAPITAL:,} — 10:1 leverage : ${pnl_10x:+.2f}")
    print(f"  P&L on ${CAPITAL:,} — 50:1 leverage : ${pnl_50x:+.2f}")
    print(f"\n  * No spread, commission, or slippage included.")
    print(f"  * Filters: ADX > {ADX_THRESHOLD}, daily EMA alignment, 2-candle confirm, ATR {ATR_MULTIPLIER}x stop")


# ── Run both strategies ────────────────────────────────────────────────────────

long_only_trades  = simulate(signals, df, mode="long_only")
long_short_trades = simulate(signals, df, mode="long_short")

print_results(long_only_trades,  "LONG-ONLY strategy  (buy on BUY signal, exit on SELL or stop)")
print_results(long_short_trades, "LONG + SHORT strategy  (flip direction on every signal, stop applies)")
