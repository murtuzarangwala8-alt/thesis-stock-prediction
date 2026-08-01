import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from scipy.stats import spearmanr
from .utils import setup_logger, export_csv_table

logger = setup_logger("MLModels")

class MLEngine:
    """
    Executes walk-forward machine learning pipeline.
    Trains LASSO, Elastic Net, Random Forest, and XGBoost classifiers.
    Implements a strict Train/Validation/Test chronological split.
    Tunes hyperparameters on the Validation Set using ROC-AUC.
    """
    def __init__(self, data_dir: Path, results_dir: Path, figures_dir: Path, feature_size='11'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.tables_dir = self.results_dir / "tables"
        self.models_dir = self.results_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.data_dir / "processed" / "master_panel_features.parquet"
        self.feature_size = feature_size

        self.features_11 = [
            'mom_21d_rank', 'mom_252d_rank', 'vol_21d_rank', 'rsi_14_rank', 
            'beta_mkt_5f_252d_rank', 'beta_smb_5f_252d_rank', 'beta_hml_5f_252d_rank',
            'beta_rmw_5f_252d_rank', 'beta_cma_5f_252d_rank',
            'book_to_market_rank', 'quality_score_rank'
        ]
        
        self.features_50 = [
            'mom_1d_rank', 'mom_5d_rank', 'mom_21d_rank', 'mom_63d_rank', 'mom_126d_rank', 'mom_252d_rank',
            'rsi_14_rank', 'macd_hist_rank', 'bb_pct_b_rank',
            'vol_5d_rank', 'vol_21d_rank', 'vol_63d_rank', 'vol_126d_rank', 'vol_ratio_rank',
            'beta_252d_rank',
            'earnings_yield_rank', 'book_to_market_rank', 'quality_score_rank',
            'pe_ratio_rank', 'pb_ratio_rank', 'return_on_equity_rank', 'return_on_assets_rank',
            'return_on_inv_capital_rank', 'sales_growth_rank', 'net_profit_margin_rank',
            'oper_margin_rank', 'asset_turnover_rank', 'debt_to_equity_fundamentals_rank',
            'debt_to_assets_rank', 'piotroski_f_score_rank',
            'buy_ratio_rank', 'pt_upside_rank', 'best_analyst_rating_rank', 'price_target_rank', 'analyst_count_rank',
            'oil_wti_backup_rank', 'oil_brent_backup_rank', 'gold_backup_rank', 'silver_backup_rank',
            'dollar_index_backup_rank', 'eurusd_backup_rank', 'usdjpy_backup_rank',
            'vix_rank', 'vxn_backup_rank', 'vvix_backup_rank',
            'yield_3m_backup_rank', 'yield_5y_backup_rank', '10y_yield_rank', 'yield_30y_backup_rank',
            'fed_funds_rank'
        ]
        
        self.features_80 = self.features_50 + [
            'acct_rcv_rank', 'altman_z_score_rank', 'total_assets_rank', 'bs_tot_liab2_rank',
            'capital_expend_rank', 'cf_cash_from_oper_rank', 'cf_dvd_paid_rank', 'cf_free_cash_flow_rank',
            'cur_mkt_cap_rank', 'ebit_rank', 'ebitda_rank', 'eqy_float_rank', 'eqy_sh_out_rank',
            'ev_to_t12m_ebitda_rank', 'free_cash_flow_per_sh_rank', 'is_eps_rank', 'is_oper_inc_rank',
            'long_term_borrow_rank', 'net_income_rank', 'px_volume_rank', 'sales_rev_turn_rank',
            'short_term_borrow_rank', 'tax_rate_reported_rank', 'tot_common_eqy_rank',
            'tot_return_index_gross_dvds_rank', 'eps_trailing_rank', 'ebitda_to_revenue_rank',
            'px_to_sales_ratio_rank', 'px_to_free_cash_flow_rank', 'beta_mkt_5f_252d_rank'
        ]
        
        selected_path = self.data_dir / "processed" / "selected_features.json"
        
        if self.feature_size == '11':
            self.features = self.features_11
        elif self.feature_size == '50':
            self.features = self.features_50
        elif self.feature_size == '80':
            self.features = self.features_80
        elif self.feature_size in ['technical', 'fundamental', 'macro', 'sentiment', 'all_selected', 'selected', 'tech_fund', 'tech_fund_macro']:
            if selected_path.exists():
                logger.info(f"Loading feature size '{self.feature_size}' from {selected_path}")
                import json
                with open(selected_path, "r") as f:
                    sel_data = json.load(f)
                if self.feature_size == 'technical':
                    self.features = sel_data["tech_cols"]
                elif self.feature_size == 'tech_fund':
                    self.features = sel_data["tech_cols"] + sel_data["fund_cols"]
                elif self.feature_size == 'tech_fund_macro':
                    self.features = sel_data["tech_cols"] + sel_data["fund_cols"] + sel_data["macro_cols"]
                elif self.feature_size == 'fundamental':
                    self.features = sel_data["fund_cols"]
                elif self.feature_size == 'macro':
                    self.features = sel_data["macro_cols"]
                elif self.feature_size == 'sentiment':
                    self.features = sel_data["sent_cols"]
                else: # 'all_selected' or 'selected'
                    self.features = sel_data["tech_cols"] + sel_data["fund_cols"] + sel_data["macro_cols"] + sel_data["sent_cols"]
            else:
                raise FileNotFoundError(f"Feature selection file not found at {selected_path}. Run select_features.py first.")
        else:
            raise ValueError(f"Invalid feature_size: {feature_size}. Must be '11', '50', '80', 'technical', 'tech_fund', 'tech_fund_macro', 'fundamental', 'macro', 'sentiment', or 'all_selected'.")
            
        self.target = 'target_excess_1d'

    def run_walk_forward(self, horizon='1d', start_val_date=None, start_test_date=None):
        """Runs validation tuning and saves out-of-sample predictions for optimized models over multiple folds."""
        prediction_path = self.data_dir / "processed" / f"oos_predictions_{horizon}_feat{self.feature_size}.parquet"
        if prediction_path.exists():
            logger.info(f"OOS predictions already exist at {prediction_path}. Skipping training phase to save time.")
            return
            
        logger.info(f"Loading master panel for ML (Horizon: {horizon}, Feature Size: {self.feature_size})...")
        df = pd.read_parquet(self.features_path)
        
        self.target = f"target_excess_{horizon}"
        df['target_binary'] = (df[self.target] > 0).astype(int)
        
        # Drop rows with NaNs in target (specifically for 21d horizon which shifts future data)
        df = df.dropna(subset=[self.target, 'target_binary'])
        
        # Define the 5 chronological folds strictly within the 2014-2024 range
        folds = [
            {
                "test_year": 2020,
                "train_end": "2018-12-31",
                "val_start": "2019-01-01",
                "val_end": "2019-12-31",
                "test_start": "2020-01-01",
                "test_end": "2020-12-31"
            },
            {
                "test_year": 2021,
                "train_end": "2019-12-31",
                "val_start": "2020-01-01",
                "val_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2021-12-31"
            },
            {
                "test_year": 2022,
                "train_end": "2020-12-31",
                "val_start": "2021-01-01",
                "val_end": "2021-12-31",
                "test_start": "2022-01-01",
                "test_end": "2022-12-31"
            },
            {
                "test_year": 2023,
                "train_end": "2021-12-31",
                "val_start": "2022-01-01",
                "val_end": "2022-12-31",
                "test_start": "2023-01-01",
                "test_end": "2023-12-31"
            },
            {
                "test_year": 2024,
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-12-31"
            }
        ]
        
        # If parameters were explicitly passed and don't match the defaults, support a single split
        if start_val_date is not None and start_test_date is not None:
            logger.info("Custom single split requested...")
            folds = [
                {
                    "test_year": int(start_test_date.split('-')[0]),
                    "train_end": (pd.to_datetime(start_val_date) - timedelta(days=1)).strftime('%Y-%m-%d'),
                    "val_start": start_val_date,
                    "val_end": (pd.to_datetime(start_test_date) - timedelta(days=1)).strftime('%Y-%m-%d'),
                    "test_start": start_test_date,
                    "test_end": "2025-12-31"
                }
            ]
            
        fold_test_predictions = []
        
        # Track final trained models for downstream SHAP analysis
        opt_lasso = None
        opt_elasticnet = None
        opt_rf = None
        opt_xgb = None
        
        for fold in folds:
            test_yr = fold['test_year']
            logger.info(f"\n==================================================")
            logger.info(f"  PROCESSING FOLD FOR TEST YEAR: {test_yr} ({horizon})")
            logger.info(f"==================================================")
            
            # Rolling 5-year train, single-year val, single-year test
            # WALK-FORWARD EMBARGO FIX (Audit Fix C4)
            # =========================================
            # Add purging gap between splits to prevent overlapping holding
            # period leakage. For 21d horizon, the target at train_end uses
            # returns spanning into the validation period. The embargo removes
            # these contaminated rows. (de Prado, Advances in Financial ML, 2018)
            embargo_days = 25 if horizon == '21d' else 2  # ~21 trading days + buffer
            
            train_end_dt = pd.to_datetime(fold['train_end'])
            train_start_str = (train_end_dt - pd.DateOffset(years=5) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Embargo: truncate training data to end 'embargo_days' before val_start
            embargo_train_end = (pd.to_datetime(fold['val_start']) - pd.Timedelta(days=embargo_days)).strftime('%Y-%m-%d')
            train_df = df[(df['date'] >= train_start_str) & (df['date'] <= embargo_train_end)].dropna(subset=self.features)
            
            # Embargo: truncate validation data to end 'embargo_days' before test_start
            embargo_val_end = (pd.to_datetime(fold['test_start']) - pd.Timedelta(days=embargo_days)).strftime('%Y-%m-%d')
            val_df = df[(df['date'] >= fold['val_start']) & (df['date'] <= embargo_val_end)].dropna(subset=self.features)
            
            test_df = df[(df['date'] >= fold['test_start']) & (df['date'] <= fold['test_end'])].copy().dropna(subset=self.features)
            
            if test_df.empty:
                logger.warning(f"Test set is empty for fold {test_yr}. Skipping.")
                continue
                
            logger.info(f"Train Set: {train_df.shape[0]:,} rows (pre-{fold['val_start']})")
            logger.info(f"Val Set:   {val_df.shape[0]:,} rows ({fold['val_start']} to {fold['val_end']})")
            logger.info(f"Test Set:  {test_df.shape[0]:,} rows ({fold['test_start']} to {fold['test_end']})")
            
            X_train, y_train = train_df[self.features], train_df['target_binary']
            X_val, y_val = val_df[self.features], val_df['target_binary']
            X_test, y_test = test_df[self.features], test_df['target_binary']
            
            # Subsample train and val sets for 10x training speedup
            np_rand = np.random.RandomState(42)
            if len(X_train) > 80000:
                train_idx = np_rand.choice(X_train.index, size=80000, replace=False)
                X_train = X_train.loc[train_idx]
                y_train = y_train.loc[train_idx]
            if len(X_val) > 20000:
                val_idx = np_rand.choice(X_val.index, size=20000, replace=False)
                X_val = X_val.loc[val_idx]
                y_val = y_val.loc[val_idx]
            
            # --- 1. LASSO Tuning ---
            logger.info("\n  Tuning LASSO (Logistic L1)...")
            best_lasso_auc = -1.0
            best_lasso_c = 1.0
            lasso_grid = [0.01, 0.1, 1.0]
            
            for c in lasso_grid:
                model = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=50, tol=1e-3, random_state=42)
                model.fit(X_train, y_train)
                pred_val = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, pred_val)
                logger.info(f"    C={c:<5} | Val AUC: {val_auc:.5f}")
                if val_auc > best_lasso_auc:
                    best_lasso_auc = val_auc
                    best_lasso_c = c
                    
            logger.info(f"  Optimized LASSO C: {best_lasso_c} (Val AUC: {best_lasso_auc:.5f})")
            opt_lasso = LogisticRegression(penalty='l1', C=best_lasso_c, solver='saga', max_iter=100, tol=1e-3, random_state=42)
            opt_lasso.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
            test_df['pred_prob_lasso'] = opt_lasso.predict_proba(X_test)[:, 1]
            
            # --- 2. Elastic Net Tuning ---
            logger.info("\n  Tuning Elastic Net (Logistic L1+L2)...")
            best_en_auc = -1.0
            best_en_params = {}
            en_grid_c = [0.01, 0.1, 1.0]
            en_grid_l1 = [0.2, 0.5, 0.8]
            
            for c in en_grid_c:
                for l1 in en_grid_l1:
                    model = LogisticRegression(penalty='elasticnet', C=c, l1_ratio=l1, solver='saga', max_iter=50, tol=1e-3, random_state=42)
                    model.fit(X_train, y_train)
                    pred_val = model.predict_proba(X_val)[:, 1]
                    val_auc = roc_auc_score(y_val, pred_val)
                    logger.info(f"    C={c:<5}, l1_ratio={l1:.2f} | Val AUC: {val_auc:.5f}")
                    if val_auc > best_en_auc:
                        best_en_auc = val_auc
                        best_en_params = {'C': c, 'l1_ratio': l1}
                        
            logger.info(f"  Optimized Elastic Net params: {best_en_params} (Val AUC: {best_en_auc:.5f})")
            opt_elasticnet = LogisticRegression(penalty='elasticnet', C=best_en_params['C'], l1_ratio=best_en_params['l1_ratio'], solver='saga', max_iter=100, tol=1e-3, random_state=42)
            opt_elasticnet.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
            test_df['pred_prob_elasticnet'] = opt_elasticnet.predict_proba(X_test)[:, 1]
            
            # --- 3. Random Forest Tuning ---
            logger.info("\n  Tuning Random Forest via Optuna...")
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def rf_objective(trial):
                max_depth = trial.suggest_int('max_depth', 4, 8)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 500, 2000, step=500)
                n_estimators = trial.suggest_int('n_estimators', 50, 100)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                pred_val = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, pred_val)
                
            rf_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            rf_study.optimize(rf_objective, n_trials=5)
            best_rf_params = rf_study.best_params
            best_rf_auc = rf_study.best_value
            logger.info(f"  Optimized RF params via Optuna: {best_rf_params} (Val AUC: {best_rf_auc:.5f})")
            
            opt_rf = RandomForestClassifier(
                n_estimators=best_rf_params.get('n_estimators', 100),
                max_depth=best_rf_params['max_depth'],
                min_samples_leaf=best_rf_params['min_samples_leaf'],
                n_jobs=-1,
                random_state=42
            )
            opt_rf.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
            test_df['pred_prob_rf'] = opt_rf.predict_proba(X_test)[:, 1]
            
            # --- 4. XGBoost Tuning ---
            logger.info("\n  Tuning XGBoost via Optuna (with Early Stopping)...")
            # Check GPU availability for XGBoost acceleration
            import torch
            xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            xgb_tree = 'hist' if torch.cuda.is_available() else 'auto'
            logger.info(f"  [XGBoost Device] Routing model execution to: {xgb_device} ({xgb_tree})")
            
            def xgb_objective(trial):
                max_depth = trial.suggest_int('max_depth', 3, 6)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
                n_estimators = trial.suggest_int('n_estimators', 80, 200)
                
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method=xgb_tree,
                    device=xgb_device,
                    random_state=42,
                    early_stopping_rounds=15,
                    eval_metric="auc"
                )
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                best_ntree_limit = getattr(model, "best_iteration", n_estimators)
                trial.set_user_attr("best_ntree_limit", int(best_ntree_limit))
                
                pred_val = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, pred_val)
                
            xgb_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            xgb_study.optimize(xgb_objective, n_trials=5)
            best_xgb_params = xgb_study.best_params
            best_xgb_auc = xgb_study.best_value
            best_ntree_limit = xgb_study.best_trial.user_attrs.get("best_ntree_limit", 100)
            logger.info(f"  Optimized XGB params via Optuna: {best_xgb_params} (Trees: {best_ntree_limit}, Val AUC: {best_xgb_auc:.5f})")
            
            opt_xgb = xgb.XGBClassifier(
                n_estimators=best_ntree_limit,
                max_depth=best_xgb_params['max_depth'],
                learning_rate=best_xgb_params['learning_rate'],
                reg_alpha=1.0,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method=xgb_tree,
                device=xgb_device,
                random_state=42
            )
            opt_xgb.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
            test_df['pred_prob_xgb'] = opt_xgb.predict_proba(X_test)[:, 1]
            
            fold_test_predictions.append(test_df)
            
        # --- Out-of-Sample (OOS) Concatenation & Evaluation ---
        logger.info("\n==================================================")
        logger.info(f"  CONCATENATING AND EVALUATING OOS FOLDS ({horizon})")
        logger.info("==================================================")
        
        oos_df = pd.concat(fold_test_predictions, ignore_index=True)
        y_oos = oos_df['target_binary']
        
        metrics = []
        for name, col in [
            ("LASSO", "pred_prob_lasso"),
            ("Elastic Net", "pred_prob_elasticnet"),
            ("Random Forest", "pred_prob_rf"),
            ("XGBoost", "pred_prob_xgb")
        ]:
            m = self._evaluate_model(name, oos_df[col], y_oos, oos_df)
            metrics.append(m)
            
        metrics_df = pd.DataFrame(metrics).set_index("Model")
        logger.info("\n" + metrics_df.to_string())
        export_csv_table(metrics_df, self.tables_dir / f"ml_evaluation_metrics_{horizon}_feat{self.feature_size}.csv")
        if self.feature_size == '11':
            export_csv_table(metrics_df, self.tables_dir / f"ml_evaluation_metrics_{horizon}.csv")
            if horizon == '1d':
                export_csv_table(metrics_df, self.tables_dir / "ml_evaluation_metrics.csv")
        
        # Save models and predictions specifically for this horizon and feature size
        suffix = f"_{horizon}_feat{self.feature_size}"
        if opt_lasso is not None:
            with open(self.models_dir / f"lasso_model{suffix}.pkl", "wb") as f:
                pickle.dump(opt_lasso, f)
            if horizon == '1d' and self.feature_size == '11':
                with open(self.models_dir / "lasso_model.pkl", "wb") as f:
                    pickle.dump(opt_lasso, f)
                    
        if opt_elasticnet is not None:
            with open(self.models_dir / f"elasticnet_model{suffix}.pkl", "wb") as f:
                pickle.dump(opt_elasticnet, f)
                
        if opt_rf is not None:
            with open(self.models_dir / f"rf_model{suffix}.pkl", "wb") as f:
                pickle.dump(opt_rf, f)
            if horizon == '1d' and self.feature_size == '11':
                with open(self.models_dir / "rf_model.pkl", "wb") as f:
                    pickle.dump(opt_rf, f)
                    
        if opt_xgb is not None:
            with open(self.models_dir / f"xgb_model{suffix}.pkl", "wb") as f:
                pickle.dump(opt_xgb, f)
            if horizon == '1d' and self.feature_size == '11':
                with open(self.models_dir / "xgb_model.pkl", "wb") as f:
                    pickle.dump(opt_xgb, f)
                
        prediction_path = self.data_dir / "processed" / f"oos_predictions_{horizon}_feat{self.feature_size}.parquet"
        oos_df.to_parquet(prediction_path, index=False)
        if self.feature_size == '11':
            oos_df.to_parquet(self.data_dir / "processed" / f"oos_predictions_{horizon}.parquet", index=False)
            if horizon == '1d':
                oos_df.to_parquet(self.data_dir / "processed" / "oos_predictions.parquet", index=False)
            
        logger.info(f"Saved optimized models and out-of-sample predictions for horizon {horizon}.")

    def _evaluate_model(self, name: str, y_pred_prob, y_true, test_df):
        """Calculates Accuracy, ROC-AUC, and Daily Information Coefficient (IC)."""
        y_pred_class = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred_class)
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate daily IC using a clean temp dataframe to avoid warnings
        temp_df = pd.DataFrame({
            'pred': y_pred_prob,
            'target': test_df[self.target].values,
            'date': test_df['date'].values
        })
        
        daily_ic = []
        for d in temp_df['date'].unique():
            day_data = temp_df[temp_df['date'] == d]
            if len(day_data) > 30:
                ic, _ = spearmanr(day_data['pred'], day_data['target'])
                if not np.isnan(ic):
                    daily_ic.append(ic)
                    
        ic_mean = np.mean(daily_ic) if daily_ic else 0
        ic_std = np.std(daily_ic) if daily_ic else 1
        icir = (ic_mean / ic_std) * np.sqrt(252)
        
        return {
            'Model': name,
            'Accuracy': acc,
            'ROC_AUC': roc_auc,
            'Daily_IC': ic_mean,
            'ICIR': icir
        }

