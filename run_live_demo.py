"""
Run Live S&P 500 Market Demo / Paper Trading System.
Fetches real-time market data, generates TFDMGA predictions, and updates $1,000 USD Demo Account.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from scripts.live_paper_trader import main

if __name__ == "__main__":
    main()
