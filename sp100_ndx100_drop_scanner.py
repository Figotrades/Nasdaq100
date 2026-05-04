"""
S&P 100 + Nasdaq 100 Daily Drop Scanner
=======================================

Scans the combined S&P 100 / Nasdaq 100 universe for any stock whose
close-over-prior-close return was worse than THRESHOLD (default -10%).
Reports hits to the console and writes a CSV.

Usage
-----
    python sp100_ndx100_drop_scanner.py
    python sp100_ndx100_drop_scanner.py --threshold -10 --days 1
    python sp100_ndx100_drop_scanner.py --days 30           # scan last 30 trading days
    python sp100_ndx100_drop_scanner.py --out drops.csv

Dependencies
------------
    pip install yfinance pandas lxml

Scheduling (examples)
---------------------
    # Linux/macOS cron — run every weekday at 22:30 (after US close):
    30 22 * * 1-5 /usr/bin/python3 /path/to/sp100_ndx100_drop_scanner.py

    # Windows Task Scheduler — equivalent: trigger Mon-Fri 22:30, action python.exe + script path
"""

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


# ---------- Constituent fetchers (Wikipedia, refreshed each run) ----------

def get_sp100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    tables = pd.read_html(url)
    for t in tables:
        if any("symbol" in str(c).lower() for c in t.columns):
            sym_col = next(c for c in t.columns if "symbol" in str(c).lower())
            tickers = (
                t[sym_col].astype(str).str.strip()
                 .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B for yfinance
                 .tolist()
            )
            if 90 <= len(tickers) <= 110:
                return tickers
    raise RuntimeError("Could not locate the S&P 100 ticker table on Wikipedia.")


def get_nasdaq100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols_lower = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols_lower):
            sym_col = next(
                c for c in t.columns
                if "ticker" in str(c).lower() or "symbol" in str(c).lower()
            )
            tickers = (
                t[sym_col].astype(str).str.strip()
                 .str.replace(".", "-", regex=False)
                 .tolist()
            )
            if 90 <= len(tickers) <= 110:
                return tickers
    raise RuntimeError("Could not locate the Nasdaq 100 ticker table on Wikipedia.")


# ---------- Scanner ----------

def scan(tickers: list[str], threshold_pct: float, lookback_trading_days: int) -> pd.DataFrame:
    """Return a DataFrame of all (ticker, date) pairs where the daily close-to-close
    return was strictly less than threshold_pct over the lookback window."""

    end = datetime.today()
    # Calendar buffer to ensure we cover the requested trading days across weekends/holidays
    start = end - timedelta(days=int(lookback_trading_days * 1.7) + 7)

    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []
    for tkr in tickers:
        try:
            df = data[tkr] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(subset=["Close"]).copy()
            if len(df) < 2:
                continue
            # Need one extra row to compute pct_change for the first day in window
            df = df.tail(lookback_trading_days + 1)
            df["PctChange"] = df["Close"].pct_change() * 100
            hits = df[df["PctChange"] < threshold_pct]
            for date, row in hits.iterrows():
                prev = row["Close"] / (1 + row["PctChange"] / 100)
                rows.append({
                    "Ticker":    tkr,
                    "Date":      date.strftime("%Y-%m-%d"),
                    "PrevClose": round(float(prev), 2),
                    "Close":     round(float(row["Close"]), 2),
                    "PctChange": round(float(row["PctChange"]), 2),
                    "Volume":    int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                })
        except Exception as e:
            print(f"  ! skipped {tkr}: {e}", file=sys.stderr)

    return pd.DataFrame(rows)


# ---------- CLI entry point ----------

def main() -> None:
    p = argparse.ArgumentParser(description="S&P 100 / Nasdaq 100 daily drop scanner")
    p.add_argument("--threshold", type=float, default=-10.0,
                   help="Drop threshold in percent (default: -10.0). "
                        "Daily returns strictly less than this are flagged.")
    p.add_argument("--days", type=int, default=1,
                   help="Trailing trading days to scan (default: 1 = most recent close only).")
    p.add_argument("--out", type=str, default=None,
                   help="Output CSV path (default: daily_drops_YYYYMMDD.csv)")
    args = p.parse_args()

    print("Fetching index constituents from Wikipedia...")
    sp  = get_sp100_tickers()
    ndx = get_nasdaq100_tickers()
    print(f"  S&P 100:    {len(sp)} tickers")
    print(f"  Nasdaq 100: {len(ndx)} tickers")
    universe = sorted(set(sp) | set(ndx))
    print(f"  Combined universe (deduped): {len(universe)} tickers")

    print(f"\nScanning for daily returns < {args.threshold}% "
          f"over last {args.days} trading day(s)...")
    results = scan(universe, args.threshold, args.days)

    if results.empty:
        print("\nNo qualifying drops found.")
        return

    results["Indices"] = results["Ticker"].apply(
        lambda t: "/".join(name for name, lst in [("SP100", sp), ("NDX", ndx)] if t in lst)
    )
    results = results.sort_values(["Date", "PctChange"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"FOUND {len(results)} drop event(s):")
    print("=" * 90)
    print(results.to_string(index=False))

    out_path = args.out or f"daily_drops_{datetime.today().strftime('%Y%m%d')}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
