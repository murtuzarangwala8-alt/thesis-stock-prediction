"""
Alpaca Brokerage API Integration Module.
Provides live automated execution for Alpaca Paper Trading and Live Brokerage Accounts.
Supports order submission, account balance checks, and bracket orders (Take-Profit / Stop-Loss).
"""

import os
import json
from pathlib import Path

# Optional dependency: alpaca-trade-api or requests fallback
try:
    import alpaca_trade_api as tradeapi
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

class AlpacaBroker:
    """
    Connects TFDMGA Trading Signals directly to Alpaca Brokerage API.
    Supports both Paper Trading (demo) and Live Trading modes.
    """
    def __init__(self, api_key: str = None, api_secret: str = None, paper: bool = True):
        self.paper = paper
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "YOUR_ALPACA_KEY_HERE")
        self.api_secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_HERE")
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        
        if HAS_ALPACA and self.api_key != "YOUR_ALPACA_KEY_HERE":
            self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        else:
            self.api = None

    def get_account_summary(self):
        """Fetches live broker account equity, cash, and buying power."""
        if not self.api:
            return {
                'mode': 'Simulated (Add Alpaca Keys to Connect Live Broker)',
                'equity': 1000.0,
                'cash': 1000.0,
                'buying_power': 4000.0,
                'status': 'DISCONNECTED'
            }
        try:
            account = self.api.get_account()
            return {
                'mode': 'Alpaca Paper Account' if self.paper else 'Alpaca Live Brokerage Account',
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'status': account.status
            }
        except Exception as e:
            return {'error': str(e), 'status': 'ERROR'}

    def submit_bracket_order(self, ticker: str, qty: float, take_profit_pct: float = 0.04, stop_loss_pct: float = 0.02):
        """
        Submits a bracket market order to Alpaca with automatic 2:1 Take-Profit and Stop-Loss.
        """
        if not self.api:
            print(f"[ALPACA SIMULATOR] Submit Order: BUY {qty:.2f} {ticker} (TP: +{take_profit_pct*100}%, SL: -{stop_loss_pct*100}%)")
            return {'status': 'SIMULATED'}

        try:
            # Get latest quote to estimate limit prices for bracket order
            last_trade = self.api.get_latest_trade(ticker)
            price = last_trade.price
            tp_price = round(price * (1.0 + take_profit_pct), 2)
            sl_price = round(price * (1.0 - stop_loss_pct), 2)

            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                take_profit={'limit_price': tp_price},
                stop_loss={'stop_price': sl_price}
            )
            print(f"[ALPACA LIVE BROKER] Bracket Order Submitted for {ticker}: ID {order.id}")
            return {'status': 'SUBMITTED', 'order_id': order.id}
        except Exception as e:
            print(f"[ALPACA ERROR] Order failed for {ticker}: {e}")
            return {'error': str(e), 'status': 'FAILED'}

def print_alpaca_instructions():
    """Prints step-by-step instructions to connect Alpaca Live Broker."""
    print("=" * 80)
    print("  [CONNECT] HOW TO CONNECT YOUR THESIS PIPELINE TO A LIVE BROKER (ALPACA)")
    print("=" * 80)
    print("  Step 1: Sign up for a free account at https://alpaca.markets")
    print("  Step 2: Go to your Alpaca Dashboard and click 'Generate API Keys' (Paper Trading)")
    print("  Step 3: Set your environment variables in PowerShell / terminal:")
    print("          $env:ALPACA_API_KEY = 'your_api_key'")
    print("          $env:ALPACA_SECRET_KEY = 'your_secret_key'")
    print("  Step 4: Install official SDK: pip install alpaca-trade-api")
    print("  Step 5: Run your broker execution script:")
    print("          python scripts/alpaca_live_trader.py")
    print("=" * 80)

if __name__ == "__main__":
    print_alpaca_instructions()
    broker = AlpacaBroker()
    summary = broker.get_account_summary()
    print("\n  Account Status:", summary)
