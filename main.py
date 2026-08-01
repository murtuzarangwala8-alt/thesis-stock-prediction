import warnings
from pathlib import Path
from src.utils import setup_logger, setup_plotting_theme

# Import pipeline components
from src.data_pipeline import DataProcessor
from src.features import FeatureEngineer
from src.baseline_models import BaselineModels
from src.ml_models import MLEngine
from src.interpretability import InterpretabilityEngine
from src.backtest import BacktestEngine

warnings.filterwarnings('ignore')

logger = setup_logger("MainOrchestrator")

def main():
    setup_plotting_theme()
    
    BASE_DIR = Path(__file__).resolve().parent
    if BASE_DIR.name == 'code':
        BASE_DIR = BASE_DIR.parent
        DATA_DIR = BASE_DIR / "data"
        RESULTS_DIR = BASE_DIR / "results"
        if (BASE_DIR / "thesis" / "figures").exists():
            FIGURES_DIR = BASE_DIR / "thesis" / "figures"
        else:
            FIGURES_DIR = BASE_DIR / "figures"
    else:
        DATA_DIR = BASE_DIR / "data"
        RESULTS_DIR = BASE_DIR / "results"
        FIGURES_DIR = BASE_DIR / "figures"
    
    logger.info("==================================================")
    logger.info("  THESIS FINAL 2.0 - ELITE ACADEMIC PIPELINE")
    logger.info("==================================================")
    
    # Check if we already have the pre-processed master parquet with Bloomberg features
    features_path = DATA_DIR / "processed" / "master_panel_features.parquet"
    skip_pipeline = False
    if features_path.exists():
        try:
            import pandas as pd
            pd.read_parquet(features_path, columns=['book_to_market_rank', 'quality_score_rank', 'pe_ratio_rank'])
            skip_pipeline = True
        except Exception:
            pass
            
    if skip_pipeline:
        logger.info("\nPre-processed database with Bloomberg fundamental features detected.")
        logger.info("Skipping Phase 1 (Data Pipeline) and Phase 2 (Feature Engineering) to preserve Bloomberg data.")
    else:
        # 1. Data Engineering (2015-2025)
        logger.info("\n--- PHASE 1: DATA PIPELINE ---")
        dp = DataProcessor(start_date="2014-12-01", end_date="2025-05-27", data_dir=DATA_DIR)
        df_clean = dp.run_pipeline()
        
        # 2. Feature Engineering
        logger.info("\n--- PHASE 2: FEATURE ENGINEERING ---")
        fe = FeatureEngineer(data_dir=DATA_DIR)
        df_feat = fe.build_features(df_clean)
        
    # 3. Baseline Econometrics (Fama-French Fama-MacBeth)
    logger.info("\n--- PHASE 3: BASELINE econometric MODELS ---")
    bm = BaselineModels(data_dir=DATA_DIR, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
    bm.run_fama_macbeth()
    
    # 4. Machine Learning, Interpretability, and Backtesting Loop
    for horizon in ['21d']:
        for feat_size in ['technical', 'tech_fund', 'tech_fund_macro', 'all_selected']:
            logger.info(f"\n==================================================")
            logger.info(f"  RUNNING PIPELINE FOR HORIZON: {horizon} | FEATURE SPACE: {feat_size}")
            logger.info(f"==================================================")
            
            logger.info(f"\n--- PHASE 4: MACHINE LEARNING & TUNING ({horizon}, Feat: {feat_size}) ---")
            ml = MLEngine(data_dir=DATA_DIR, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, feature_size=feat_size)
            ml.run_walk_forward(horizon=horizon)
            
            logger.info(f"\n--- PHASE 4.5: PYTORCH LSTM SYSTEM ({horizon}, Feat: {feat_size}) ---")
            from src.lstm_model import LSTMEngine
            lstm = LSTMEngine(data_dir=DATA_DIR, results_dir=RESULTS_DIR, feature_size=feat_size)
            lstm.train_lstm_model(horizon=horizon)
            
            logger.info(f"\n--- PHASE 5: INTERPRETABILITY ({horizon}, Feat: {feat_size}) ---")
            ie = InterpretabilityEngine(data_dir=DATA_DIR, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, feature_size=feat_size)
            ie.run_shap_analysis(horizon=horizon)
            
            logger.info(f"\n--- PHASE 6: BACKTESTING ({horizon}, Feat: {feat_size}) ---")
            bt = BacktestEngine(data_dir=DATA_DIR, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, feature_size=feat_size)
            bt.run_backtest(horizon=horizon)
            
            # --- PHASE 7: CUSTOM TFDMGA MODEL (Skipped as requested) ---
            # if feat_size == 'all_selected':
            #     logger.info(f"\n--- PHASE 7: TFDMGA CUSTOM DEEP LEARNING MODEL ({horizon}) ---")
            #     try:
            #         from TFDMGA.train import main as tfdmga_main
            #         tfdmga_argv = [
            #             "--data_path", str(DATA_DIR / "processed" / "master_panel_features.parquet"),
            #             "--checkpoint_dir", str(RESULTS_DIR / "checkpoints" / "TFDMGA"),
            #             "--log_dir", str(RESULTS_DIR / "logs" / "TFDMGA"),
            #             "--results_dir", str(RESULTS_DIR / "results" / "TFDMGA"),
            #             "--max_epochs", "2",
            #             "--batch_size", "256",
            #             "--num_workers", "0",
            #             "--no_compile",
            #             "--run_optuna",
            #             "--n_optuna_trials", "5",
            #             "--n_trial_epochs", "3"
            #         ]
            #         tfdmga_main(tfdmga_argv)
            #     except Exception as e:
            #         logger.error(f"Error running TFDMGA training: {e}")
    
    logger.info("==================================================")
    logger.info("  PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
