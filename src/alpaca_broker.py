"""
Alpaca Brokerage API Integration Module.
Provides live automated execution for Alpaca Paper Trading and Live Brokerage Accounts.
Uses direct REST API endpoints for zero-dependency reliability and zero version conflicts.
"""

import os
import json
import requests
from pathlib import Path

class AlpacaBroker:
    """
    Connects TFDMGA Trading Signals directly to Alpaca Brokerage REST API.
    Supports both Paper Trading (demo) and Live Trading modes.
    """
    def __init__(self, api_key: str = None, api_secret: str = None, paper: bool = True):
        self.paper = paper
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = "https://paper-api.alpaca.markets/v2" if paper else "https://api.alpaca.markets/v2"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

    def get_account_summary(self):
        """Fetches live broker account equity, cash, and buying power via REST API."""
        if not self.api_key or not self.api_secret:
            return {
                'mode': 'Simulated (Set $env:ALPACA_API_KEY & $env:ALPACA_SECRET_KEY)',
                'equity': 1000.0,
                'cash': 1000.0,
                'buying_power': 4000.0,
                'status': 'DISCONNECTED'
            }
        try:
            url = f"{self.base_url}/account"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'mode': 'Alpaca Paper Account' if self.paper else 'Alpaca Live Account',
                    'equity': float(data.get('equity', 0.0)),
                    'cash': float(data.get('cash', 0.0)),
                    'buying_power': float(data.get('buying_power', 0.0)),
                    'status': data.get('status', 'ACTIVE')
                }
            else:
                return {'error': f"HTTP {resp.status_code}: {resp.text}", 'status': 'ERROR'}
        except Exception as e:
            return {'error': str(e), 'status': 'ERROR'}

    def submit_bracket_order(self, ticker: str, qty: float, take_profit_pct: float = 0.04, stop_loss_pct: float = 0.02):
        """
        Submits a bracket market order to Alpaca with automatic 2:1 Take-Profit and Stop-Loss.
        """
        if not self.api_key or not self.api_secret:
            print(f"[ALPACA SIMULATOR] BUY {qty:.2f} {ticker}")
            return {'status': 'SIMULATED'}

        try:
            # Get latest trade price for bracket limit/stop calculation
            quote_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest"
            q_resp = requests.get(quote_url, headers=self.headers, timeout=5)
            if q_resp.status_code == 200:
                price = float(q_resp.json().get('trade', {}).get('p', 100.0))
            else:
                price = 100.0

            tp_price = round(price * (1.0 + take_profit_pct), 2)
            sl_price = round(price * (1.0 - stop_loss_pct), 2)

            order_payload = {
                "symbol": ticker,
                "qty": str(round(qty, 2)),
                "side": "buy",
                "type": "market",
                "time_in_force": "gtc",
                "order_class": "bracket",
                "take_profit": {"limit_price": tp_price},
                "stop_loss": {"stop_price": sl_price}
            }

            order_url = f"{self.base_url}/orders"
            resp = requests.post(order_url, headers=self.headers, json=order_payload, timeout=10)
            if resp.status_code in [200, 201]:
                data = resp.json()
                print(f"  [ALPACA BROKER] Bracket Order Submitted for {ticker}: ID {data.get('id')}")
                return {'status': 'SUBMITTED', 'order_id': data.get('id')}
            else:
                print(f"  [ALPACA ERROR] Order failed for {ticker}: {resp.text}")
                return {'error': resp.text, 'status': 'FAILED'}
        except Exception as e:
            print(f"  [ALPACA ERROR] Exception for {ticker}: {e}")
            return {'error': str(e), 'status': 'FAILED'}

def print_alpaca_instructions():
    """Prints step-by-step instructions to connect Alpaca Live Broker."""
    print("=" * 80)
    print("  [CONNECT] HOW TO CONNECT YOUR THESIS PIPELINE TO A LIVE BROKER (ALPACA)")
    print("=" * 80)
    print("  Step 1: Navigate to project folder in PowerShell:")
    print("          cd \"C:\\Users\\murta\\Desktop\\thesis final 2.0\"")
    print("  Step 2: Set your Alpaca keys in PowerShell:")
    print("          $env:ALPACA_API_KEY = \"YOUR_API_KEY_HERE\"")
    print("          $env:ALPACA_SECRET_KEY = \"YOUR_SECRET_KEY_HERE\"")
    print("  Step 3: Run your broker execution script:")
    print("          python run_alpaca_trader.py")
    print("=" * 80)

if __name__ == "__main__":
    print_alpaca_instructions()
    broker = AlpacaBroker()
    summary = broker.get_account_summary()
    print("\n  Account Status:", summary)
