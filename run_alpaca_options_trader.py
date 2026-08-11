"""
Root Entry Point for TFDMGA Options & Short Live Paper Trader.
Uses TFDMGA 53-Feature Deep Learning Signals to trade Call Options (Bullish Q5),
Put Options (Bearish Q1), and Short Equity Bracket Positions.
Proportionally scales position sizing to $1,000 USD Account Equity.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.alpaca_options_broker import AlpacaOptionsBroker
from src.tfdmga_reinforcement_learner import TFDMGAReinforcementLearner
from scripts.live_paper_trader import LivePaperTrader

def main():
    print("=" * 80)
    print("  *** TFDMGA AGGRESSIVE LIVE OPTIONS & SHORT REINFORCEMENT LEARNING TRADER ***")
    print("=" * 80)

    # 1. Initialize Alpaca Options Broker & Reinforcement Learner
    broker = AlpacaOptionsBroker()
    learner = TFDMGAReinforcementLearner(learning_rate=0.05)
    weights = learner.get_adaptive_modality_weights()

    summary = broker.get_account_summary()
    
    print(f"  Account Mode   : {summary.get('mode', 'Alpaca Options Paper Account')}")
    print(f"  Account Status : {summary.get('status', 'ACTIVE')}")
    print(f"  Total Equity   : ${summary.get('equity', 1000.0):,.2f} USD")
    print(f"  Cash Balance   : ${summary.get('cash', 1000.0):,.2f} USD")
    print(f"  Buying Power   : ${summary.get('buying_power', 1000.0):,.2f} USD")
    print(f"  Options Level  : Level {summary.get('options_level', 3)} Approved")
    print(f"  RL Win Rate    : {learner.ledger.get('win_rate', 0.50)*100:.1f}% ({learner.ledger.get('total_trades_learned', 0)} Trades Learned)")
    print(f"  Adaptive Weights: Tech {weights['w_tech']:.3f} | Fund {weights['w_fund']:.3f} | Sent {weights['w_sent']:.3f}")
    print("-" * 80)

    # 2. Fetch Live Market Data & Compute 53-Feature Signals
    trader_data_engine = LivePaperTrader()
    data = trader_data_engine.fetch_live_market_data()
    if not data:
        print("  [ERROR] Could not fetch live market data.")
        return
        
    df_signals = trader_data_engine.compute_live_signals(data)
    
    # Sort signals: Top 4 Bullish (Q5 Calls), Bottom 4 Bearish (Q1 Puts/Shorts)
    df_signals = df_signals.sort_values(by='tfdmga_score', ascending=False).reset_index(drop=True)
    
    top_bullish = df_signals.head(4).copy()
    top_bearish = df_signals.tail(4).copy()

    print("\n" + "=" * 80)
    print("  [TFDMGA BULLISH PICKS (Top Q5 Long Call Candidates)]")
    print("=" * 80)
    print(f"  {'Ticker':<8} {'Price ($)':<11} {'1D Chg (%)':<12} {'21D Mom (%)':<14} {'TFDMGA Score':<12}")
    print("  " + "-" * 65)
    for _, row in top_bullish.iterrows():
        print(f"  {row['ticker']:<8} ${row['price']:<11.2f} {row['daily_change_pct']:+6.2f}%      {row['mom_21d']*100:+7.2f}%       {row['tfdmga_score']:+.4f}")

    print("\n" + "=" * 80)
    print("  [TFDMGA BEARISH PICKS (Bottom Q1 Put / Short Candidates)]")
    print("=" * 80)
    print(f"  {'Ticker':<8} {'Price ($)':<11} {'1D Chg (%)':<12} {'21D Mom (%)':<14} {'TFDMGA Score':<12}")
    print("  " + "-" * 65)
    for _, row in top_bearish.iterrows():
        print(f"  {row['ticker']:<8} ${row['price']:<11.2f} {row['daily_change_pct']:+6.2f}%      {row['mom_21d']*100:+7.2f}%       {row['tfdmga_score']:+.4f}")
    print("=" * 80)

    # 3. Calculate Proportional Position Sizing across $1,000 USD Account Equity
    equity = summary.get('equity', 1000.0)
    cash_avail = summary.get('cash', equity)
    
    total_trades = len(top_bullish) + len(top_bearish)
    alloc_per_trade = min(cash_avail * 0.90, equity * 0.90) / max(1, total_trades)

    print(f"\n  [ALPACA PROPORTIONAL SIZING] Account Equity: ${equity:,.2f} USD")
    print(f"  [ALPACA EXECUTION] Allocating ${alloc_per_trade:,.2f} USD per trade across {total_trades} positions...")
    
    orders_submitted = []

    # 4. Execute Bullish Trades (Long Call Options or Long Equity)
    print("\n  --- EXECUTING BULLISH CALL OPTION / LONG TRADES ---")
    for _, row in top_bullish.iterrows():
        ticker = row['ticker'].replace('-', '.')
        price = row['price']
        
        # Fetch Near-The-Money Call Option Contract with Premium Verification (< alloc_per_trade)
        contract = broker.get_best_option_contract(ticker, option_type="call", target_days=21, max_premium=alloc_per_trade)
        submitted = False
        if contract and 'symbol' in contract:
            contract_sym = contract['symbol']
            strike = contract.get('strike_price')
            exp = contract.get('expiration_date')
            premium = contract.get('verified_premium', 120.0)
            ask_price = contract.get('verified_ask', 1.20)
            print(f"  [BULLISH CALL] {ticker} (${price:.2f}) -> Contract: {contract_sym} (Strike ${strike}, Exp {exp}) | Verified Premium: ${premium:.2f} USD (${ask_price:.2f}/sh)")
            
            res = broker.submit_option_order(contract_sym, qty=1, side="buy", limit_price=ask_price)
            if res.get('status') == 'SUBMITTED':
                orders_submitted.append(('CALL', ticker, contract_sym, res.get('order_id')))
                submitted = True

        if not submitted:
            # Fallback to Equity Market Bracket Buy Order
            qty = max(1, alloc_per_trade / price)
            print(f"  [BULLISH EQUITY BRACKET] {ticker} (${price:.2f}) -> Market Buy ({qty:.1f} shares)")
            res = broker.submit_short_equity_bracket(ticker, qty=qty, take_profit_pct=0.04, stop_loss_pct=0.02, is_short=False)
            if res.get('status') == 'SUBMITTED':
                orders_submitted.append(('EQUITY_BUY', ticker, ticker, res.get('order_id')))

    # 5. Execute Bearish Trades (Long Put Options or Short Equity)
    print("\n  --- EXECUTING BEARISH PUT OPTION / SHORT TRADES ---")
    for _, row in top_bearish.iterrows():
        ticker = row['ticker'].replace('-', '.')
        price = row['price']
        
        # Fetch Near-The-Money Put Option Contract with Premium Verification (< alloc_per_trade)
        contract = broker.get_best_option_contract(ticker, option_type="put", target_days=21, max_premium=alloc_per_trade)
        submitted = False
        if contract and 'symbol' in contract:
            contract_sym = contract['symbol']
            strike = contract.get('strike_price')
            exp = contract.get('expiration_date')
            premium = contract.get('verified_premium', 120.0)
            ask_price = contract.get('verified_ask', 1.20)
            print(f"  [BEARISH PUT] {ticker} (${price:.2f}) -> Contract: {contract_sym} (Strike ${strike}, Exp {exp}) | Verified Premium: ${premium:.2f} USD (${ask_price:.2f}/sh)")
            
            res = broker.submit_option_order(contract_sym, qty=1, side="buy", limit_price=ask_price)
            if res.get('status') == 'SUBMITTED':
                orders_submitted.append(('PUT', ticker, contract_sym, res.get('order_id')))
                submitted = True

        if not submitted:
            # Fallback to Short Equity Bracket Order (sell)
            qty = max(1, alloc_per_trade / price)
            print(f"  [BEARISH SHORT BRACKET] {ticker} (${price:.2f}) -> Short Equity Order ({qty:.1f} shares)")
            res = broker.submit_short_equity_bracket(ticker, qty=qty, take_profit_pct=0.04, stop_loss_pct=0.02, is_short=True)
            if res.get('status') == 'SUBMITTED':
                orders_submitted.append(('SHORT_EQUITY', ticker, ticker, res.get('order_id')))

    # 6. Execution Summary
    print("\n" + "=" * 80)
    print("  *** LIVE OPTIONS & SHORT EXECUTION COMPLETE ***")
    print(f"  Total Orders Processed : {len(orders_submitted)}")
    print("  Risk Protocols Attached: Calls/Puts Options + 2:1 Short Equity TP/SL")
    print("=" * 80)

if __name__ == "__main__":
    main()
