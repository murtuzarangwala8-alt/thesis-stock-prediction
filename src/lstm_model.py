import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import timedelta
from sklearn.metrics import roc_auc_score, accuracy_score
from .utils import setup_logger

logger = setup_logger("LSTMBaseline")

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, feature_size, seq_len=5, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # LSTM processes sequence inputs (batch, seq_len, feature_size)
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Linear classifier head using the final hidden state
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len * feature_size) -> reshape to (batch_size, seq_len, feature_size)
        x = x.view(-1, self.seq_len, self.feature_size)
        
        # out shape: (batch, seq_len, hidden_dim)
        out, (hn, cn) = self.lstm(x)
        
        # Take the final step's hidden representation
        final_state = out[:, -1, :]
        
        # Classify
        logits = self.classifier(final_state).squeeze(-1)
        return logits

class LSTMEngine:
    def __init__(self, data_dir: Path, results_dir: Path, feature_size='technical', seq_len=5):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = self.results_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.data_dir / "processed" / "master_panel_features.parquet"
        self.feature_size = feature_size
        self.seq_len = seq_len
        
        # Load features list from MLEngine
        from .ml_models import MLEngine
        ml_temp = MLEngine(data_dir, results_dir, results_dir, feature_size=feature_size)
        self.base_features = ml_temp.features
        
    def _create_lagged_data(self, df):
        """Creates lagged columns in pandas to build sequence inputs efficiently without memory fragmentation."""
        logger.info(f"Generating lagged sequence columns (seq_len={self.seq_len})...")
        df = df.sort_values(by=["ticker", "date"]).copy()
        
        new_cols = {}
        lagged_feature_cols = []
        
        # LSTM SEQUENCE ORDER FIX (Audit Fix M8)
        # =========================================
        # Build sequence in CHRONOLOGICAL order: oldest lag first, current day last.
        # The LSTM's final hidden state (out[:, -1, :]) then captures the
        # most RECENT information, which is the correct temporal ordering.
        # Previous code had lag0 (today) at timestep 0 and lag4 (oldest) at
        # timestep 4, reversing the intended information flow.
        
        # Groupby once per lag level and shift
        gp = df.groupby("ticker")[self.base_features]
        
        # Build from oldest to newest: lag(seq_len-1), lag(seq_len-2), ..., lag(1), lag(0)
        for lag in reversed(range(1, self.seq_len)):
            shifted = gp.shift(lag)
            for col in self.base_features:
                col_name = f"{col}_lag{lag}"
                new_cols[col_name] = shifted[col]
                lagged_feature_cols.append(col_name)
                
        # Lag 0 (current day) — goes at the END of the sequence (most recent)
        for col in self.base_features:
            new_cols[f"{col}_lag0"] = df[col]
            lagged_feature_cols.append(f"{col}_lag0")
                
        # Single concatenation step to prevent pandas memory fragmentation copies
        new_cols_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_cols_df], axis=1)
        
        # Drop rows where lagged variables are NaNs (beginning of each stock's series)
        df_clean = df.dropna(subset=lagged_feature_cols).copy()
        return df_clean, lagged_feature_cols

    def train_lstm_model(self, horizon='1d'):
        prediction_path = self.data_dir / "processed" / f"oos_predictions_{horizon}_feat{self.feature_size}.parquet"
        if prediction_path.exists():
            try:
                exist_df = pd.read_parquet(prediction_path)
                if "pred_prob_lstm" in exist_df.columns:
                    logger.info(f"LSTM predictions already exist in {prediction_path}. Skipping training phase.")
                    return
            except Exception:
                pass
        
        logger.info(f"Loading master panel for LSTM Baseline (Horizon: {horizon}, Feature Size: {self.feature_size})...")
        df = pd.read_parquet(self.features_path)
        
        self.target = f"target_excess_{horizon}"
        df['target_binary'] = (df[self.target] > 0).astype(int)
        df = df.dropna(subset=[self.target, 'target_binary'])
        
        # Create lagged inputs
        df_lagged, lagged_cols = self._create_lagged_data(df)
        
        folds = [
            {"test_year": 2020, "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
            {"test_year": 2021, "train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"test_year": 2022, "train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
            {"test_year": 2023, "train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
            {"test_year": 2024, "train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}
        ]
        
        fold_test_predictions = []
        
        for fold in folds:
            test_yr = fold['test_year']
            logger.info(f"\nTraining LSTM for Test Year: {test_yr} ({horizon})")
            
            # Splitting Train/Val/Test
            train_end_dt = pd.to_datetime(fold['train_end'])
            train_start_str = (train_end_dt - pd.DateOffset(years=5) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            train_df = df_lagged[(df_lagged['date'] >= train_start_str) & (df_lagged['date'] <= fold['train_end'])]
            val_df = df_lagged[(df_lagged['date'] >= fold['val_start']) & (df_lagged['date'] <= fold['val_end'])]
            test_df = df_lagged[(df_lagged['date'] >= fold['test_start']) & (df_lagged['date'] <= fold['test_end'])].copy()
            
            if test_df.empty:
                logger.warning(f"Test set empty for fold {test_yr}. Skipping.")
                continue
                
            X_train, y_train = train_df[lagged_cols].values, train_df['target_binary'].values
            X_val, y_val = val_df[lagged_cols].values, val_df['target_binary'].values
            X_test, y_test = test_df[lagged_cols].values, test_df['target_binary'].values
            
            # Subsample to keep speeds optimal
            np_rand = np.random.RandomState(42)
            if len(X_train) > 40000:
                train_idx = np_rand.choice(len(X_train), size=40000, replace=False)
                X_train, y_train = X_train[train_idx], y_train[train_idx]
            if len(X_val) > 10000:
                val_idx = np_rand.choice(len(X_val), size=10000, replace=False)
                X_val, y_val = X_val[val_idx], y_val[val_idx]
                
            train_dataset = TabularDataset(X_train, y_train)
            val_dataset = TabularDataset(X_val, y_val)
            test_dataset = TabularDataset(X_test, y_test)
            
            pin_mem = torch.cuda.is_available()
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=pin_mem)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, pin_memory=pin_mem)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=pin_mem)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"  [LSTM Device] Using accelerator: {device}")
            
            # Model Definition & Optuna Tuning
            logger.info(f"\n  Tuning LSTM via Optuna for Year: {test_yr}...")
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def lstm_objective(trial):
                hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64])
                num_layers = trial.suggest_int('num_layers', 1, 2)
                dropout = trial.suggest_float('dropout', 0.1, 0.3)
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                
                trial_model = LSTMModel(
                    feature_size=len(self.base_features),
                    seq_len=self.seq_len,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
                if torch.cuda.device_count() > 1:
                    trial_model = nn.DataParallel(trial_model)
                trial_model = trial_model.to(device)
                
                trial_criterion = nn.BCEWithLogitsLoss()
                trial_optimizer = optim.Adam(trial_model.parameters(), lr=lr, weight_decay=1e-5)
                
                for epoch in range(15):
                    trial_model.train()
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        trial_optimizer.zero_grad()
                        outputs = trial_model(X_batch)
                        loss = trial_criterion(outputs, y_batch)
                        loss.backward()
                        trial_optimizer.step()
                        
                    # Evaluate epoch on validation set for pruning
                    trial_model.eval()
                    val_preds = []
                    val_trues = []
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            outputs = torch.sigmoid(trial_model(X_batch))
                            val_preds.extend(outputs.cpu().numpy())
                            val_trues.extend(y_batch.cpu().numpy())
                    val_auc = roc_auc_score(val_trues, val_preds) if len(np.unique(val_trues)) > 1 else 0.5
                    
                    trial.report(val_auc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                        
                return val_auc
                
            lstm_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            lstm_study.optimize(lstm_objective, n_trials=5)
            best_lstm_params = lstm_study.best_params
            best_lstm_auc = lstm_study.best_value
            logger.info(f"  Optimized LSTM params via Optuna (with MedianPruner): {best_lstm_params} (Val AUC: {best_lstm_auc:.5f})")
            
            # Re-train final model using best parameters on concat Train+Val dataset
            model = LSTMModel(
                feature_size=len(self.base_features),
                seq_len=self.seq_len,
                hidden_dim=best_lstm_params['hidden_dim'],
                num_layers=best_lstm_params['num_layers'],
                dropout=best_lstm_params['dropout']
            )
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=best_lstm_params['lr'], weight_decay=1e-5)
            
            best_auc = -1.0
            best_model_state = None
            epochs = 20
            patience = 5
            patience_counter = 0
            
            # Combine train and validation datasets for final OOS training
            concat_X = np.concatenate([X_train, X_val], axis=0)
            concat_y = np.concatenate([y_train, y_val], axis=0)
            final_dataset = TabularDataset(concat_X, concat_y)
            final_loader = DataLoader(final_dataset, batch_size=512, shuffle=True, pin_memory=pin_mem)
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in final_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Check performance on val to save best weights
                model.eval()
                val_preds = []
                val_trues = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = torch.sigmoid(model(X_batch))
                        val_preds.extend(outputs.cpu().numpy())
                        val_trues.extend(y_batch.cpu().numpy())
                val_auc = roc_auc_score(val_trues, val_preds) if len(np.unique(val_trues)) > 1 else 0.5
                logger.info(f"  Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss/len(final_loader):.4f} - Val AUC: {val_auc:.4f}")
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = (model.module.state_dict().copy() if isinstance(model, nn.DataParallel) else model.state_dict().copy())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"  Early stopping triggered after {epoch+1} epochs. Restoring best model weights.")
                    break
            
            # Load best weights for OOS inference
            if best_model_state is not None:
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(best_model_state)
                else:
                    model.load_state_dict(best_model_state)
            
            # OOS Inference
            model.eval()
            test_preds = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = torch.sigmoid(model(X_batch))
                    test_preds.extend(outputs.cpu().numpy())
                    
            test_df["pred_prob_lstm"] = test_preds
            fold_test_predictions.append(test_df[["date", "ticker", "pred_prob_lstm"]])
            
            # Save the trained model checkpoint
            torch.save(best_model_state, self.models_dir / f"lstm_model_fold_{test_yr}_{horizon}.pt")
            
        # Merge all predictions
        all_lstm_preds = pd.concat(fold_test_predictions)
        
        # Merge with existing predictions parquet
        if prediction_path.exists():
            existing_df = pd.read_parquet(prediction_path)
            # Ensure date column types match
            existing_df["date"] = pd.to_datetime(existing_df["date"])
            all_lstm_preds["date"] = pd.to_datetime(all_lstm_preds["date"])
            
            # Drop column if already exists in target
            if "pred_prob_lstm" in existing_df.columns:
                existing_df = existing_df.drop(columns=["pred_prob_lstm"])
                
            merged_df = pd.merge(existing_df, all_lstm_preds, on=["date", "ticker"], how="left")
            
            # Fill missing predictions with S&P 500 median baseline (0.50)
            merged_df["pred_prob_lstm"] = merged_df["pred_prob_lstm"].fillna(0.50)
            
            merged_df.to_parquet(prediction_path, index=False)
            logger.info(f"Merged LSTM predictions into {prediction_path}.")
            
            # If default feature config, sync to standard predictions parquet files
            if self.feature_size == '11':
                merged_df.to_parquet(self.data_dir / "processed" / f"oos_predictions_{horizon}.parquet", index=False)
                if horizon == '1d':
                    merged_df.to_parquet(self.data_dir / "processed" / "oos_predictions.parquet", index=False)
        else:
            logger.error(f"Base prediction file missing at {prediction_path}. Run MLEngine first.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default="1d")
    parser.add_argument("--features", type=str, default="technical")
    args = parser.parse_args()
    
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    engine = LSTMEngine(BASE_DIR / "data", BASE_DIR / "results", feature_size=args.features)
    engine.train_lstm_model(horizon=args.horizon)
