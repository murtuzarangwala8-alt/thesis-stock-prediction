"""
Run Live Alpaca Paper / Live Brokerage Automated Trader.
Connects your TFDMGA model predictions directly to your official Alpaca Brokerage Account.
"""
import os
import sys
import getpass
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.alpaca_broker import AlpacaBroker

def main():
    print("=" * 80)
    print("  *** ALPACA BROKERAGE AUTOMATED LIVE / PAPER TRADER ***")
    print("=" * 80)
    
    # Check if keys are set in environment
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        print("\n  [ALPACA KEY CONFIGURATION]")
        print("  API Key and Secret Key not detected in environment variables.")
        api_key = input("  Enter your Alpaca API Key ID (e.g. PK...): ").strip()
        api_secret = getpass.getpass("  Enter your Alpaca Secret Key (hidden input): ").strip()
        
        # Save to environment for current session
        os.environ["ALPACA_API_KEY"] = api_key
        os.environ["ALPACA_SECRET_KEY"] = api_secret

    print("\n  Connecting to Alpaca Brokerage API (Paper Trading Mode)...")
    broker = AlpacaBroker(api_key=api_key, api_secret=api_secret, paper=True)
    summary = broker.get_account_summary()
    
    if summary.get('status') == 'ERROR':
        print(f"\n  [ERROR] Connection Error: {summary.get('error')}")
        print("  Please check your API Key ID and Secret Key and try again.")
        return
        
    print("\n  [SUCCESS] CONNECTED TO ALPACA BROKERAGE ACCOUNT:")
    print(f"  Account Mode   : {summary.get('mode')}")
    print(f"  Account Status : {summary.get('status')}")
    print(f"  Total Equity   : ${summary.get('equity', 0.0):,.2f} USD")
    print(f"  Cash Balance   : ${summary.get('cash', 0.0):,.2f} USD")
    print(f"  Buying Power   : ${summary.get('buying_power', 0.0):,.2f} USD")
    print("=" * 80)
    print("\n  Your TFDMGA model pipeline is ready to submit automated bracket orders!")

if __name__ == "__main__":
    main()
