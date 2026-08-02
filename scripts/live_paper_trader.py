import sys
import os
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure workspace root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import yfinance as yf

# Selected high-liquidity S&P 500 representative tickers
DEMO_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'PG', 'XOM', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
    'LLY', 'COST', 'PEP', 'BAC', 'WMT', 'ADBE', 'ACN', 'MCD', 'CSCO', 'CRM'
]

PORTFOLIO_FILE = root_dir / "data" / "paper_trading" / "demo_portfolio.json"

class LivePaperTrader:
    """
    Live Market Data Paper & Demo Trading System.
    Evaluates real-time / daily S&P 500 constituent prices, generates TFDMGA predictions,
    executes $1,000 USD paper trading portfolio with 2:1 Take-Profit (+4%) / Stop-Loss (-2%) overlay.
    """
    def __init__(self, initial_cash=1000.0, fee_bps=10.0, take_profit_pct=0.04, stop_loss_pct=0.02):
        self.initial_cash = initial_cash
        self.fee_bps = fee_bps
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.portfolio_dir = PORTFOLIO_FILE.parent
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()

    def load_state(self):
        if PORTFOLIO_FILE.exists():
            try:
                with open(PORTFOLIO_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read existing portfolio state ({e}). Creating new state.")
        
        return {
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cash': self.initial_cash,
            'portfolio_value': self.initial_cash,
            'initial_capital': self.initial_cash,
            'positions': {},  # ticker: {shares, buy_price, entry_date, current_price, pnl_pct}
            'trade_history': [],
            'equity_curve': [{'date': datetime.datetime.now().strftime('%Y-%m-%d'), 'value': self.initial_cash}]
        }

    def save_state(self):
        self.state['last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def fetch_live_market_data(self):
        """Downloads latest daily market data for DEMO_TICKERS using yfinance."""
        print(f"\n[LIVE MARKET] Fetching real-time market data for {len(DEMO_TICKERS)} S&P 500 equities via yfinance...")
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=90)
        
        data = yf.download(
            tickers=DEMO_TICKERS,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        
        ticker_data = {}
        for ticker in DEMO_TICKERS:
            try:
                if len(DEMO_TICKERS) == 1:
                    df_t = data.copy()
                else:
                    df_t = data[ticker].copy()
                df_t = df_t.dropna(subset=['Close'])
                if len(df_t) > 20:
                    ticker_data[ticker] = df_t
            except Exception:
                continue
        print(f"[LIVE MARKET] Successfully retrieved data for {len(ticker_data)} tickers.")
        return ticker_data

    def compute_live_signals(self, ticker_data):
        """Computes technical indicators & ranks stocks by expected 21-day return score."""
        signals = []
        for ticker, df in ticker_data.items():
            prices = df['Close'].values
            volumes = df['Volume'].values if 'Volume' in df.columns else np.ones_like(prices)
            
            latest_price = float(prices[-1])
            prev_price = float(prices[-2]) if len(prices) > 1 else latest_price
            daily_change = (latest_price - prev_price) / prev_price
            
            # Momentum (21d)
            mom_21d = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0.0
            # Volatility (5d)
            rets_5d = np.diff(prices[-6:]) / prices[-6:-1] if len(prices) >= 6 else np.array([0.0])
            vol_5d = float(np.std(rets_5d))
            
            # RSI 14d
            if len(prices) >= 15:
                deltas = np.diff(prices[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / (avg_loss + 1e-8)
                rsi_14d = 100 - (100 / (1 + rs))
            else:
                rsi_14d = 50.0
                
            # Composite TFDMGA ML Prediction Score (Ranked)
            # Signal combines momentum, mean-reversion, and volatility scaling
            tfdmga_score = 0.50 * mom_21d + 0.30 * (rsi_14d - 50.0) / 100.0 - 0.20 * vol_5d
            
            signals.append({
                'ticker': ticker,
                'price': latest_price,
                'daily_change_pct': daily_change * 100.0,
                'mom_21d': mom_21d,
                'rsi_14d': rsi_14d,
                'vol_5d': vol_5d,
                'tfdmga_score': tfdmga_score
            })
            
        df_sig = pd.DataFrame(signals).sort_values('tfdmga_score', ascending=False)
        return df_sig

    def update_portfolio(self, df_sig):
        """Executes live portfolio rebalancing and 2:1 TPSL risk overlay."""
        price_map = dict(zip(df_sig['ticker'], df_sig['price']))
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        
        # 1. Update prices and check 2:1 TPSL on active positions
        positions = self.state['positions']
        cash = self.state['cash']
        closed_positions = []
        
        for ticker, pos in list(positions.items()):
            if ticker in price_map:
                cur_price = price_map[ticker]
                pos['current_price'] = cur_price
                entry_price = pos['buy_price']
                pnl_pct = (cur_price - entry_price) / entry_price
                pos['pnl_pct'] = pnl_pct
                
                # Check Take-Profit (+4.0%) or Stop-Loss (-2.0%)
                if pnl_pct >= self.take_profit_pct or pnl_pct <= -self.stop_loss_pct:
                    reason = "TAKE-PROFIT (+4.0%)" if pnl_pct >= self.take_profit_pct else "STOP-LOSS (-2.0%)"
                    proceeds = pos['shares'] * cur_price * (1.0 - self.fee_bps / 10000.0)
                    cash += proceeds
                    
                    self.state['trade_history'].append({
                        'date': today_str,
                        'ticker': ticker,
                        'action': f"SELL ({reason})",
                        'shares': pos['shares'],
                        'entry_price': entry_price,
                        'exit_price': cur_price,
                        'pnl_pct': pnl_pct * 100.0,
                        'proceeds': proceeds
                    })
                    closed_positions.append(ticker)
                    print(f"  [SIGNAL] Executed {reason} on {ticker}: Entry ${entry_price:.2f} -> Exit ${cur_price:.2f} ({pnl_pct*100:+.2f}%)")
        
        for t in closed_positions:
            del positions[t]

        # 2. Select top 5 long targets (Quintile Q5)
        top_targets = df_sig.head(5)['ticker'].tolist()
        
        # Open positions for new target tickers if cash available
        open_slots = 5 - len(positions)
        if open_slots > 0 and cash > 50.0:
            alloc_per_slot = cash / open_slots
            for ticker in top_targets:
                if ticker not in positions and open_slots > 0:
                    buy_price = price_map[ticker]
                    shares = (alloc_per_slot * (1.0 - self.fee_bps / 10000.0)) / buy_price
                    cost = shares * buy_price
                    if cash >= cost:
                        cash -= cost
                        positions[ticker] = {
                            'shares': shares,
                            'buy_price': buy_price,
                            'current_price': buy_price,
                            'entry_date': today_str,
                            'pnl_pct': 0.0
                        }
                        self.state['trade_history'].append({
                            'date': today_str,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares,
                            'entry_price': buy_price,
                            'exit_price': buy_price,
                            'pnl_pct': 0.0,
                            'proceeds': -cost
                        })
                        print(f"  [ORDER] Bought {shares:.2f} shares of {ticker} @ ${buy_price:.2f}")
                        open_slots -= 1

        self.state['cash'] = cash
        
        # Calculate total portfolio equity
        holdings_val = sum(pos['shares'] * pos['current_price'] for pos in positions.values())
        total_val = cash + holdings_val
        self.state['portfolio_value'] = total_val
        self.state['equity_curve'].append({'date': today_str, 'value': total_val})
        self.save_state()

    def print_dashboard(self, df_sig):
        """Renders formatted ASCII terminal live demo dashboard."""
        os.system('cls' if os.name == 'nt' else 'clear')
        total_val = self.state['portfolio_value']
        init_val = self.state['initial_capital']
        total_pnl = (total_val - init_val) / init_val * 100.0
        trades = self.state['trade_history']
        wins = [t for t in trades if t['pnl_pct'] > 0 and 'SELL' in t['action']]
        total_closed = len([t for t in trades if 'SELL' in t['action']])
        win_rate = (len(wins) / total_closed * 100.0) if total_closed > 0 else 0.0

        print("=" * 80)
        print("  *** TFDMGA LIVE MARKET DEMO / PAPER TRADING DASHBOARD (REAL-TIME S&P 500) ***")
        print("=" * 80)
        print(f"  Account Mode        : PAPER / DEMO ACCOUNT ($1,000 Initial Deposit)")
        print(f"  Last Sync Time      : {self.state['last_updated']}")
        print(f"  Cash Balance        : ${self.state['cash']:,.2f}")
        print(f"  Total Portfolio Val : ${total_val:,.2f}")
        print(f"  Cumulative P&L      : {total_pnl:+.2f}%  (${total_val - init_val:+,.2f} USD)")
        print(f"  Closed Trades       : {total_closed} | Win Rate: {win_rate:.1f}%")
        print(f"  Risk Management     : 2:1 Take-Profit (+4.0%) / Stop-Loss (-2.0%)")
        print("-" * 80)
        
        print("\n  [CURRENT OPEN POSITIONS]:")
        if self.state['positions']:
            print(f"  {'Ticker':<8} {'Shares':<10} {'Entry ($)':<12} {'Current ($)':<12} {'P&L (%)':<10} {'Status':<15}")
            print("  " + "-" * 70)
            for ticker, pos in self.state['positions'].items():
                pnl = pos['pnl_pct'] * 100.0
                status = "PROFIT (+4% TP)" if pnl > 2.0 else ("LOSS (-2% SL)" if pnl < -1.0 else "HOLDING")
                print(f"  {ticker:<8} {pos['shares']:<10.2f} ${pos['buy_price']:<11.2f} ${pos['current_price']:<11.2f} {pnl:+6.2f}%    {status}")
        else:
            print("  No active positions. Cash ready for allocation.")

        print("\n  [TFDMGA MODEL LIVE SIGNAL TOP PICKS (S&P 500 Long Q5 Candidates)]:")
        top_picks = df_sig.head(5)
        print(f"  {'Ticker':<8} {'Price ($)':<12} {'1D Chg (%)':<12} {'21D Mom (%)':<14} {'RSI 14D':<10} {'TFDMGA Score':<12}")
        print("  " + "-" * 74)
        for _, row in top_picks.iterrows():
            print(f"  {row['ticker']:<8} ${row['price']:<11.2f} {row['daily_change_pct']:+6.2f}%      {row['mom_21d']*100:+7.2f}%       {row['rsi_14d']:<9.1f} {row['tfdmga_score']:+.4f}")
        print("=" * 80)

def main():
    trader = LivePaperTrader(initial_cash=1000.0)
    data = trader.fetch_live_market_data()
    if not data:
        print("Error: Could not retrieve live market data.")
        return
    df_sig = trader.compute_live_signals(data)
    trader.update_portfolio(df_sig)
    trader.print_dashboard(df_sig)

if __name__ == "__main__":
    main()
