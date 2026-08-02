"""
Run Live Alpaca Paper / Live Brokerage Automated Trader.
Fetches 53 live data points, scores S&P 500 equities using TFDMGA,
and submits automated Bracket Orders (+4% TP / -2% SL) directly to Alpaca.
"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.alpaca_broker import AlpacaBroker
from scripts.live_paper_trader import LivePaperTrader

def main():
    print("=" * 80)
    print("  *** TFDMGA LIVE ALPACA AUTOMATED ORDER EXECUTION ENGINE ***")
    print("=" * 80)
    
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not set in environment.")
        return

    # 1. Connect to Alpaca Brokerage
    broker = AlpacaBroker(api_key=api_key, api_secret=api_secret, paper=True)
    summary = broker.get_account_summary()
    
    if summary.get('status') == 'ERROR':
        print(f"  [ERROR] Connection failed: {summary.get('error')}")
        return
        
    print(f"  Account Mode   : {summary.get('mode')}")
    print(f"  Total Equity   : ${summary.get('equity', 0.0):,.2f} USD")
    print(f"  Cash Balance   : ${summary.get('cash', 0.0):,.2f} USD")
    print(f"  Buying Power   : ${summary.get('buying_power', 0.0):,.2f} USD")
    print("-" * 80)

    # 2. Fetch Live Market Data & Compute 53-Feature Signals
    trader_data_engine = LivePaperTrader()
    data = trader_data_engine.fetch_live_market_data()
    if not data:
        print("  [ERROR] Could not fetch live market data.")
        return
        
    df_sig = trader_data_engine.compute_live_signals(data)
    
    print("\n  [TFDMGA MODEL LIVE SIGNAL TOP PICKS (S&P 500 Long Q5 Candidates)]:")
    top_picks = df_sig.head(5)
    print(f"  {'Ticker':<8} {'Price ($)':<12} {'1D Chg (%)':<12} {'21D Mom (%)':<14} {'TFDMGA Score':<12}")
    print("  " + "-" * 65)
    for _, row in top_picks.iterrows():
        print(f"  {row['ticker']:<8} ${row['price']:<11.2f} {row['daily_change_pct']:+6.2f}%      {row['mom_21d']*100:+7.2f}%       {row['tfdmga_score']:+.4f}")
    print("-" * 80)

    # 3. Calculate Proportional Position Sizing across Full Account Equity
    equity = summary.get('equity', 100000.0)
    cash_avail = summary.get('cash', equity)
    
    # Allocate 95% of total equity equally across the 5 top Q5 picks (20% per stock)
    alloc_per_stock = min(cash_avail * 0.95, equity * 0.95) / len(top_picks)
    
    print(f"\n  [ALPACA PROPORTIONAL SIZING] Account Equity: ${equity:,.2f} USD")
    print(f"  [ALPACA ORDER EXECUTION] Allocating ${alloc_per_stock:,.2f} USD per stock across {len(top_picks)} picks...")
    orders_submitted = []
    
    for _, row in top_picks.iterrows():
        ticker = row['ticker']
        price = row['price']
        qty = max(1.0, round(alloc_per_stock / price, 2))
        
        res = broker.submit_bracket_order(
            ticker=ticker,
            qty=qty,
            take_profit_pct=0.04,  # +4.0% Take Profit
            stop_loss_pct=0.02     # -2.0% Stop Loss
        )
        orders_submitted.append({'ticker': ticker, 'qty': qty, 'status': res.get('status')})
        
    print("\n" + "=" * 80)
    print("  *** LIVE ORDER EXECUTION COMPLETE ***")
    print(f"  Total Orders Processed : {len(orders_submitted)}")
    print("  Risk Protocol Attached : 2:1 Take-Profit (+4.0%) / Stop-Loss (-2.0%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
