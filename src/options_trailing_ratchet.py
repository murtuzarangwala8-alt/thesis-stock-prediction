"""
Options Dynamic Trailing Stop-Loss & Trailing Take-Profit Ratchet Module.
Tracks real-time high-water mark prices for open Option Contracts.
Dynamically ratchets up Stop-Loss levels as option premiums rise to lock in peak profits.
"""

import os
import json
from pathlib import Path
from datetime import datetime

RATCHET_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "options_trailing_ratchet.json"

class OptionsTrailingRatchet:
    """
    Manages Dynamic Trailing Stop-Loss & Trailing Take-Profit for Options Contracts.
    - Tracks peak contract high-water marks.
    - Ratchets Trailing Stop-Loss upward to lock in profits.
    - Triggers automated market sell when price pulls back from peak by trailing_pct (default 10%).
    """
    def __init__(self, trailing_pct: float = 0.10, activation_gain_pct: float = 0.15):
        self.trailing_pct = trailing_pct          # Trailing distance (10% below peak)
        self.activation_gain_pct = activation_gain_pct  # Gain required to activate trail (+15%)
        self.state_file = RATCHET_STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.active_contracts = self.load_state()

    def load_state(self):
        """Loads active options trailing ratchet state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_state(self):
        """Saves current trailing ratchet state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.active_contracts, f, indent=2)

    def register_contract(self, order_id: str, symbol: str, option_type: str, entry_premium: float, qty: int = 1):
        """Registers a newly purchased Option Contract into the Trailing Ratchet System."""
        self.active_contracts[symbol] = {
            'order_id': order_id,
            'symbol': symbol,
            'option_type': option_type,
            'entry_premium': entry_premium,
            'high_water_mark': entry_premium,
            'current_stop_loss': round(entry_premium * (1.0 - 0.15), 2), # Initial 15% stop
            'trailing_active': False,
            'qty': qty,
            'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.save_state()
        print(f"  [TRAILING RATCHET] Registered {option_type.upper()} Contract {symbol} @ ${entry_premium:.2f}/sh | Initial Stop: ${self.active_contracts[symbol]['current_stop_loss']:.2f}")

    def update_contract_price(self, symbol: str, current_premium: float):
        """
        Updates current option contract price and ratchets trailing stop-loss upward if price hits new high.
        Returns:
            dict with {'action': 'HOLD'|'TRIGGER_SELL', 'reason': str, 'locked_profit_pct': float}
        """
        if symbol not in self.active_contracts:
            return {'action': 'HOLD'}

        data = self.active_contracts[symbol]
        entry = data['entry_premium']
        peak = data['high_water_mark']
        current_stop = data['current_stop_loss']

        # 1. Update High-Water Mark if current price makes a new high
        if current_premium > peak:
            data['high_water_mark'] = current_premium
            peak = current_premium

        gain_from_entry = (current_premium - entry) / entry

        # 2. Check Trailing Ratchet Activation Threshold (+15% gain)
        if gain_from_entry >= self.activation_gain_pct:
            data['trailing_active'] = True

        # 3. Ratchet Trailing Stop-Loss Upward if Trailing Mode is Active
        if data['trailing_active']:
            new_trailing_stop = round(peak * (1.0 - self.trailing_pct), 2)
            if new_trailing_stop > current_stop:
                data['current_stop_loss'] = new_trailing_stop
                current_stop = new_trailing_stop
                locked_gain = (current_stop - entry) / entry * 100.0
                print(f"  [TRAILING RATCHET UPGRADE] {symbol} Peak: ${peak:.2f} -> Trailing Stop Ratcheted Up to ${current_stop:.2f} (Locks in {locked_gain:+.1f}% Profit!)")

        # 4. Check if Current Premium has fallen below Trailing Stop-Loss
        if current_premium <= current_stop:
            locked_pnl = (current_premium - entry) / entry * 100.0
            print(f"  [TRAILING RATCHET TRIGGER] {symbol} Current Price ${current_premium:.2f} <= Trailing Stop ${current_stop:.2f}! Triggering Exit (Locked PnL: {locked_pnl:+.1f}%)")
            del self.active_contracts[symbol]
            self.save_state()
            return {
                'action': 'TRIGGER_SELL',
                'reason': f"Trailing Stop Triggered at ${current_stop:.2f} (Peak ${peak:.2f})",
                'locked_profit_pct': locked_pnl
            }

        self.save_state()
        return {'action': 'HOLD', 'current_stop': current_stop, 'peak': peak}
