import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    tables_dir = Path("results/tables")
    
    stress_csv = tables_dir / "backtest_stress_1d_feattechnical.csv"
    
    # Set academic plotting style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # 1. Stress Scenarios Comparison Chart
    if stress_csv.exists():
        try:
            df_stress = pd.read_csv(stress_csv)
            # Filter to key strategies: SP500, XGB_LS, LSTM_LS
            filter_strats = ['SP500', 'XGB_LS', 'LSTM_LS']
            df_sub = df_stress[df_stress['Strategy'].isin(filter_strats)]
            
            # Pivot table for plotting
            pivot_df = df_sub.pivot(index='Scenario', columns='Strategy', values='Max Drawdown')
            # Convert drawdown to positive percentages for visualization
            pivot_df = pivot_df.abs() * 100
            
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot_df.plot(kind='bar', ax=ax, color=['#3b82f6', '#10b981', '#64748b'])
            
            ax.set_title("Drawdown Mitigation Under Macro Stress Scenarios", fontweight='bold', pad=15)
            ax.set_ylabel("Maximum Drawdown (%)", fontweight='bold')
            ax.set_xlabel("Macro Scenario", fontweight='bold')
            ax.set_xticklabels(pivot_df.index, rotation=15)
            ax.legend(frameon=True)
            plt.tight_layout()
            
            chart_path = figures_dir / "stress_drawdown_mitigation.png"
            plt.savefig(chart_path, dpi=300)
            plt.close()
            print(f"Generated stress scenario chart at: {chart_path}")
        except Exception as e:
            print(f"Error generating stress chart: {e}")
            
    # 2. Stop-Loss Impact Line Chart
    oos_path = Path("data/processed/oos_predictions_1d_feattechnical.parquet")
    if oos_path.exists():
        try:
            df = pd.read_parquet(oos_path)
            if 'mkt_ret' not in df.columns and 'mkt_rf' in df.columns and 'rf' in df.columns:
                df['mkt_ret'] = df['mkt_rf'] + df['rf']
            df = df.sort_values(['ticker', 'date'])
            pred_col = 'pred_prob_xgb'
            if pred_col in df.columns:
                df['pred_prob_xgb_shifted'] = df.groupby('ticker')[pred_col].shift(1)
                df = df.dropna(subset=['pred_prob_xgb_shifted'])
                unique_dates = sorted(df['date'].unique())
                
                daily_rets = []
                for d in unique_dates[:250]: # Plot a clean 1-year trading window segment
                    day_stocks = df[df['date'] == d]
                    n_stocks = len(day_stocks)
                    if n_stocks < 50:
                        continue
                    cutoff = int(np.ceil(n_stocks * 0.10))
                    day_sorted = day_stocks.sort_values('pred_prob_xgb_shifted', ascending=False)
                    long_ret = day_sorted.iloc[:cutoff]['ret'].mean()
                    short_ret = day_sorted.iloc[-cutoff:]['ret'].mean()
                    daily_rets.append({'date': d, 'XGB_LS': long_ret - short_ret})
                    
                df_rets = pd.DataFrame(daily_rets).set_index('date')
                df_rets['XGB_LS_Net'] = df_rets['XGB_LS'] - 0.0010
                df_rets['XGB_LS_Cum'] = (1 + df_rets['XGB_LS_Net']).cumprod()
                
                running_max = df_rets['XGB_LS_Cum'].cummax()
                drawdown = (df_rets['XGB_LS_Cum'] - running_max) / running_max
                
                df_rets['XGB_LS_Cum_StopLoss'] = df_rets['XGB_LS_Cum'].copy()
                trigger_idx = drawdown <= -0.12
                if trigger_idx.any():
                    first_trigger = trigger_idx.idxmax()
                    val_at_trigger = df_rets.loc[first_trigger, 'XGB_LS_Cum_StopLoss']
                    df_rets.loc[first_trigger:, 'XGB_LS_Cum_StopLoss'] = val_at_trigger
                    
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_rets.index, df_rets['XGB_LS_Cum'], label="XGB LS Net (Raw)", color='#ef4444', lw=1.5)
                ax.plot(df_rets.index, df_rets['XGB_LS_Cum_StopLoss'], label="XGB LS Net (With -12% Stop-Loss)", color='#10b981', lw=2)
                if trigger_idx.any():
                    ax.axvline(first_trigger, color='#f59e0b', ls='--', label=f"Stop-Loss Triggered ({first_trigger.strftime('%Y-%m-%d')})")
                    
                ax.set_title("Drawdown Containment via Active Stop-Loss Stop-Out", fontweight='bold', pad=15)
                ax.set_ylabel("Cumulative Portfolio Wealth", fontweight='bold')
                ax.set_xlabel("Date", fontweight='bold')
                ax.legend(frameon=True)
                plt.tight_layout()
                
                chart_path2 = figures_dir / "stop_loss_impact_curves.png"
                plt.savefig(chart_path2, dpi=300)
                plt.close()
                print(f"Generated stop-loss curves chart at: {chart_path2}")
        except Exception as e:
            print(f"Error generating stop-loss chart: {e}")

if __name__ == "__main__":
    main()
