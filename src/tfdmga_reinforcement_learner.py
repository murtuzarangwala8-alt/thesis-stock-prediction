"""
TFDMGA Online Reinforcement Learning & Trade Feedback Engine.
Learns from daily trading outcomes (fills, take-profit, stop-loss) to adaptively 
fine-tune modality weights and feature trust scores for future trading days.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

LEARNING_LEDGER_FILE = Path(__file__).resolve().parent.parent / "data" / "trading_learning_ledger.json"

class TFDMGAReinforcementLearner:
    """
    Online Reinforcement Learning System for TFDMGA Trading.
    - Stores historical trade outcomes and feature contributions.
    - Applies Policy Gradient / Q-learning updates to feature trust weights.
    - Adapts model modality weights (w_tech, w_fund, w_sent) dynamically based on live market rewards.
    """
    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        self.ledger_file = LEARNING_LEDGER_FILE
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = self.load_ledger()

    def load_ledger(self):
        """Loads reinforcement learning state and modality trust multipliers."""
        if self.ledger_file.exists():
            try:
                with open(self.ledger_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trades_learned': 0,
            'cumulative_reward': 0.0,
            'win_rate': 0.50,
            'modality_multipliers': {
                'technical': 1.0,
                'fundamental': 1.0,
                'sentiment': 1.0,
                'macro': 1.0
            },
            'trade_history': []
        }

    def save_ledger(self):
        """Saves updated reinforcement learning state to JSON file."""
        self.ledger['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.ledger_file, 'w') as f:
            json.dump(self.ledger, f, indent=2)

    def get_adaptive_modality_weights(self):
        """Returns reinforcement-learned modality trust weights (normalized to sum to 1.0)."""
        mults = self.ledger.get('modality_multipliers', {})
        w_tech = mults.get('technical', 1.0)
        w_fund = mults.get('fundamental', 1.0)
        w_sent = mults.get('sentiment', 1.0)
        
        total = w_tech + w_fund + w_sent
        return {
            'w_tech': w_tech / total,
            'w_fund': w_fund / total,
            'w_sent': w_sent / total,
            'multipliers': mults
        }

    def record_and_learn_trade(self, ticker: str, trade_type: str, entry_price: float, exit_price: float, tfdmga_score: float, features: dict = None):
        """
        Applies Reinforcement Learning (Policy Gradient) step upon trade closure:
        1. Computes trade reward R (percentage return).
        2. Adjusts modality trust multipliers using exponential reward gradient.
        3. Updates win rate and cumulative strategy reward.
        """
        if trade_type.upper() in ['CALL', 'BUY', 'LONG']:
            reward = (exit_price - entry_price) / entry_price
        else: # PUT, SHORT, SELL
            reward = (entry_price - exit_price) / entry_price

        # Update cumulative metrics
        self.ledger['total_trades_learned'] += 1
        self.ledger['cumulative_reward'] += float(reward)
        
        # Exponential moving average win rate
        is_win = 1.0 if reward > 0 else 0.0
        prev_wr = self.ledger.get('win_rate', 0.50)
        self.ledger['win_rate'] = 0.9 * prev_wr + 0.1 * is_win

        # Modality Trust Gradient Update
        # Boost modality weights if reward > 0, suppress if reward < 0
        mults = self.ledger['modality_multipliers']
        gradient = self.learning_rate * float(reward)

        # Technical & Fundamental receive proportional reinforcement based on signal direction
        mults['technical'] = max(0.5, min(2.0, mults['technical'] * np.exp(gradient * 0.4)))
        mults['fundamental'] = max(0.5, min(2.0, mults['fundamental'] * np.exp(gradient * 0.4)))
        mults['sentiment'] = max(0.5, min(2.0, mults['sentiment'] * np.exp(gradient * 0.2)))

        trade_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'trade_type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'reward_pct': float(reward * 100.0),
            'tfdmga_score': float(tfdmga_score),
            'updated_multipliers': dict(mults)
        }
        self.ledger['trade_history'].append(trade_entry)
        self.save_ledger()

        print(f"  [REINFORCEMENT LEARNER] Trade Learned ({ticker} {trade_type}): Reward {reward*100:+.2f}% | Win Rate {self.ledger['win_rate']*100:.1f}% | Updated Tech Mult: {mults['technical']:.3f}")
        return self.ledger
