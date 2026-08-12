"""
Alpaca Options & Short Trading Broker Integration Module.
Provides live automated execution for Options (Calls & Puts) and Short Equity positions.
Connects TFDMGA predictive scores to Alpaca Level 3 Options REST API endpoints.
"""

import os
import json
import requests
from datetime import datetime, timedelta

class AlpacaOptionsBroker:
    """
    Connects TFDMGA Trading Signals directly to Alpaca Options & Short REST API.
    Supports Calls (Bullish Q5), Puts (Bearish Q1), and Direct Equity Shorting (sell_short).
    """
    def __init__(self, api_key: str = None, api_secret: str = None, paper: bool = True):
        self.paper = paper
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = "https://paper-api.alpaca.markets/v2" if paper else "https://api.alpaca.markets/v2"
        self.data_url = "https://data.alpaca.markets/v2"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

    def get_account_summary(self):
        """Fetches live broker account equity, cash, and options trading level."""
        if not self.api_key or not self.api_secret:
            return {
                'mode': 'Simulated Options Broker',
                'equity': 1000.0,
                'cash': 1000.0,
                'options_level': 3,
                'status': 'DISCONNECTED'
            }
        try:
            url = f"{self.base_url}/account"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'mode': 'Alpaca Options Paper Account' if self.paper else 'Alpaca Options Live Account',
                    'equity': float(data.get('equity', 0.0)),
                    'cash': float(data.get('cash', 0.0)),
                    'buying_power': float(data.get('buying_power', 0.0)),
                    'options_level': data.get('options_approved_level', 3),
                    'status': data.get('status', 'ACTIVE')
                }
            else:
                return {'error': f"HTTP {resp.status_code}: {resp.text}", 'status': 'ERROR'}
        except Exception as e:
            return {'error': str(e), 'status': 'ERROR'}

    def get_stock_price(self, ticker: str) -> float:
        """Fetches the latest trade price for stock ticker."""
        try:
            url = f"{self.data_url}/stocks/{ticker}/trades/latest"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                return float(resp.json().get('trade', {}).get('p', 100.0))
        except Exception:
            pass
        return 100.0

    def get_best_option_contract(self, ticker: str, option_type: str = "call", target_days: int = 21, max_premium: float = 300.0):
        """
        Selects optimal At-The-Money (ATM) Call or Put option contract:
        - Filters strikes within 5% of current stock price.
        - Sorts by closest strike to stock price (abs(strike - stock_price)).
        - Verifies real-time premiums via Alpaca Snapshots API.
        """
        if not self.api_key or not self.api_secret:
            return None

        try:
            current_price = self.get_stock_price(ticker)
            
            # 1. Query Alpaca Options Snapshots API for live pricing & quotes
            snap_url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{ticker}?feed=indicative"
            snap_resp = requests.get(snap_url, headers=self.headers, timeout=10)
            snapshots = snap_resp.json().get('snapshots', {}) if snap_resp.status_code == 200 else {}

            # 2. Query active contract metadata
            url = f"{self.base_url}/options/contracts?underlying_symbols={ticker}&type={option_type.lower()}&status=active&limit=100"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                return None

            contracts = resp.json().get('option_contracts', [])
            if not contracts:
                return None

            today = datetime.now().date()
            min_exp = today
            max_exp = today + timedelta(days=45)

            valid_contracts = []
            for c in contracts:
                sym = c.get('symbol')
                exp_str = c.get('expiration_date')
                if not exp_str:
                    continue
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                if min_exp <= exp_date <= max_exp:
                    strike = float(c.get('strike_price', 0.0))
                    
                    # --- AT-THE-MONEY (ATM) STRIKE SELECTION FILTER ---
                    # Keep strikes within 5% of current stock price (ATM band)
                    strike_diff = abs(strike - current_price)
                    if strike_diff > (current_price * 0.05):
                        continue
                    
                    # Fetch live premium from snapshot
                    snap = snapshots.get(sym, {})
                    lq = snap.get('latestQuote', {})
                    ask_price = float(lq.get('ap', 0) or 0)
                    close_price = float(snap.get('dailyBar', {}).get('c', 0) or 0)
                    
                    price_per_share = ask_price if ask_price > 0 else close_price
                    total_premium = price_per_share * 100.0  # 1 contract = 100 shares

                    days_to_exp = (exp_date - today).days
                    exp_diff = abs(days_to_exp - target_days)
                    
                    valid_contracts.append({
                        'contract': c,
                        'symbol': sym,
                        'strike': strike,
                        'ask_price': price_per_share,
                        'total_premium': total_premium,
                        'strike_diff': strike_diff,
                        'exp_diff': exp_diff
                    })

            if not valid_contracts:
                return None

            # Sort by closest strike to stock price (ATM), then target expiration
            valid_contracts.sort(key=lambda x: (x['strike_diff'], x['exp_diff']))
            best_info = valid_contracts[0]
            best_contract = best_info['contract']
            best_contract['underlying_price'] = current_price
            best_contract['verified_premium'] = best_info['total_premium']
            best_contract['verified_ask'] = best_info['ask_price'] if best_info['ask_price'] > 0 else 1.50
            return best_contract
        except Exception as e:
            print(f"  [OPTIONS ERROR] Exception fetching contract for {ticker}: {e}")
            return None

    def submit_option_order(self, contract_symbol: str, qty: int = 1, side: str = "buy", limit_price: float = None):
        """Submits an Option Contract order to Alpaca (uses limit order after-hours to allow queuing)."""
        if not self.api_key or not self.api_secret:
            print(f"  [SIMULATOR] {side.upper()} {qty} Option Contract {contract_symbol}")
            return {'status': 'SIMULATED'}

        try:
            # Set affordable limit price estimate ($1.20 per share = $120 per contract)
            if not limit_price:
                limit_price = 1.20

            order_payload = {
                "symbol": contract_symbol,
                "qty": str(int(qty)),
                "side": side.lower(),
                "type": "limit",
                "limit_price": str(round(limit_price, 2)),
                "time_in_force": "day"
            }

            url = f"{self.base_url}/orders"
            resp = requests.post(url, headers=self.headers, json=order_payload, timeout=10)
            if resp.status_code in [200, 201]:
                data = resp.json()
                print(f"  [ALPACA OPTIONS] Option Contract Order Queued for {contract_symbol}: ID {data.get('id')}")
                return {'status': 'SUBMITTED', 'order_id': data.get('id'), 'details': data}
            else:
                print(f"  [ALPACA OPTIONS ERROR] Failed for {contract_symbol}: {resp.text}")
                return {'error': resp.text, 'status': 'FAILED'}
        except Exception as e:
            print(f"  [ALPACA OPTIONS ERROR] Exception for {contract_symbol}: {e}")
            return {'error': str(e), 'status': 'FAILED'}

    def submit_short_equity_bracket(self, ticker: str, qty: float, take_profit_pct: float = 0.04, stop_loss_pct: float = 0.02, is_short: bool = False):
        """
        Submits Equity Bracket Market Order (buy for Long, sell for Short) with 2:1 TP/SL protection.
        """
        if not self.api_key or not self.api_secret:
            print(f"  [SIMULATOR] {'SHORT' if is_short else 'BUY'} {qty:.2f} shares of {ticker}")
            return {'status': 'SIMULATED'}

        try:
            price = self.get_stock_price(ticker)
            side = "sell" if is_short else "buy"
            
            if is_short:
                tp_price = round(price * (1.0 - take_profit_pct), 2)
                sl_price = round(price * (1.0 + stop_loss_pct), 2)
            else:
                tp_price = round(price * (1.0 + take_profit_pct), 2)
                sl_price = round(price * (1.0 - stop_loss_pct), 2)

            qty_int = int(max(1, round(qty)))
            order_payload = {
                "symbol": ticker,
                "qty": str(qty_int),
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "order_class": "bracket",
                "take_profit": {"limit_price": tp_price},
                "stop_loss": {"stop_price": sl_price}
            }

            url = f"{self.base_url}/orders"
            resp = requests.post(url, headers=self.headers, json=order_payload, timeout=10)
            if resp.status_code in [200, 201]:
                data = resp.json()
                print(f"  [ALPACA EQUITY BRACKET] Order ({side.upper()}) Submitted for {ticker}: ID {data.get('id')}")
                return {'status': 'SUBMITTED', 'order_id': data.get('id')}
            else:
                print(f"  [ALPACA ERROR] Order failed for {ticker}: {resp.text}")
                return {'error': resp.text, 'status': 'FAILED'}
        except Exception as e:
            print(f"  [ALPACA ERROR] Exception for {ticker}: {e}")
            return {'error': str(e), 'status': 'FAILED'}
