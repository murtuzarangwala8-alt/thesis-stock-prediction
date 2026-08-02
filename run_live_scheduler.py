"""
Continuous Live Automated Trading Scheduler Daemon.
Runs in the background, downloads live data daily, computes 53 TFDMGA features,
and executes Alpaca bracket orders automatically at market close.
"""
import os
import sys
import time
import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from run_alpaca_trader import main as run_trading_cycle

def run_scheduler_loop(interval_hours: float = 24.0):
    print("=" * 80)
    print("  *** TFDMGA CONTINUOUS AUTOMATED TRADING DAEMON STARTED ***")
    print("  Will automatically download live data and execute trades daily.")
    print("=" * 80)
    
    while True:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[DAEMON CYCLE {now}] Starting daily automated trading pipeline...")
        try:
            run_trading_cycle()
        except Exception as e:
            print(f"[DAEMON ERROR] Cycle error: {e}")
            
        print(f"\n[DAEMON SLEEP] Sleeping for {interval_hours} hours until next market cycle...")
        time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    run_scheduler_loop()
