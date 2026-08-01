import pandas as pd
import numpy as np
from pathlib import Path
from .utils import setup_logger

logger = setup_logger("FeatureEngineering")

class FeatureEngineer:
    """
    Constructs cross-sectional features for the machine learning pipeline.
    Ensures strict adherence to point-in-time data to prevent lookahead bias.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = self.data_dir / "processed"
        self.features_path = self.processed_dir / "master_panel_features.parquet"

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes time-series features per ticker."""
        logger.info("Computing rolling time-series features...")
        
        df = df.sort_values(['ticker', 'date'])
        
        # 1. Momentum (Short-term 1-Month)
        # Captures short-term trend continuation / pricing momentum
        df['mom_21d'] = df.groupby('ticker')['close'].pct_change(periods=21)
        
        # 2. Momentum (Long-term 12-Month Skip-1 Month)
        # Standard academic factor momentum definition to skip short-term reversal
        df['ret_252d'] = df.groupby('ticker')['close'].pct_change(periods=252)
        df['mom_252d_skip1'] = df.groupby('ticker')['ret_252d'].shift(21)
        
        # 3. Idiosyncratic Volatility proxy (21d rolling standard deviation)
        # Standard risk characteristic representing daily dispersion of returns
        df['vol_21d'] = df.groupby('ticker')['ret'].transform(lambda x: x.rolling(21).std())
        
        # 4. Technical Indicator: Relative Strength Index (14d RSI)
        # Normalizes price velocity to detect overbought/oversold regimes
        def compute_rsi(series, window=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
            
        df['rsi_14d'] = df.groupby('ticker')['close'].transform(compute_rsi)
        
        # 5. Value & Quality Proxies
        # Reversal represents value proxy; Inverse volatility represents quality
        df['value_proxy'] = -df['ret_252d']
        df['quality_proxy'] = 1.0 / (df['vol_21d'] + 1e-6)
        
        return df

    def _compute_rolling_ff_betas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling 3-factor and 5-factor Fama-French betas in a vectorized manner."""
        logger.info("Computing rolling 3-factor and 5-factor Fama-French betas...")
        
        df = df.sort_values(['ticker', 'date'])
        window = 252
        
        # 1. Precompute factor covariance elements globally (same for all stocks)
        factors_df = df.groupby('date')[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']].first().sort_index()
        
        # For 3-Factor Model
        var_mkt = factors_df['mkt_rf'].rolling(window).var()
        var_smb = factors_df['smb'].rolling(window).var()
        var_hml = factors_df['hml'].rolling(window).var()
        cov_m_s = factors_df['mkt_rf'].rolling(window).cov(factors_df['smb'])
        cov_m_h = factors_df['mkt_rf'].rolling(window).cov(factors_df['hml'])
        cov_s_h = factors_df['smb'].rolling(window).cov(factors_df['hml'])
        
        factor_cov_3f = pd.DataFrame({
            'var_mkt': var_mkt,
            'var_smb': var_smb,
            'var_hml': var_hml,
            'cov_m_s': cov_m_s,
            'cov_m_h': cov_m_h,
            'cov_s_h': cov_s_h
        }, index=factors_df.index)
        
        # For 5-Factor Model
        factor_names = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']
        cov_elements_5f = {}
        for i in range(len(factor_names)):
            for j in range(i, len(factor_names)):
                f1, f2 = factor_names[i], factor_names[j]
                key = f"cov_{f1}_{f2}" if f1 != f2 else f"var_{f1}"
                cov_elements_5f[key] = factors_df[f1].rolling(window).cov(factors_df[f2])
                
        factor_cov_5f = pd.DataFrame(cov_elements_5f, index=factors_df.index)
        # Drop duplicate columns to prevent pandas from renaming them on merge
        factor_cov_5f = factor_cov_5f.drop(columns=['var_smb', 'var_hml'], errors='ignore')
        
        # Merge factor covariances into stock panel
        df = pd.merge(df, factor_cov_3f, on='date', how='left')
        df = pd.merge(df, factor_cov_5f, on='date', how='left')
        
        # Compute stock covariances with factors
        df['ret_excess'] = df['ret'] - df['rf']
        df['cov_y_mkt_rf'] = df.groupby('ticker')['ret_excess'].transform(lambda x: x.rolling(window).cov(df.loc[x.index, 'mkt_rf']))
        df['cov_y_smb'] = df.groupby('ticker')['ret_excess'].transform(lambda x: x.rolling(window).cov(df.loc[x.index, 'smb']))
        df['cov_y_hml'] = df.groupby('ticker')['ret_excess'].transform(lambda x: x.rolling(window).cov(df.loc[x.index, 'hml']))
        df['cov_y_rmw'] = df.groupby('ticker')['ret_excess'].transform(lambda x: x.rolling(window).cov(df.loc[x.index, 'rmw']))
        df['cov_y_cma'] = df.groupby('ticker')['ret_excess'].transform(lambda x: x.rolling(window).cov(df.loc[x.index, 'cma']))
        
        n = len(df)
        
        # --- Solve 3-Factor System ---
        sigma_xx_3f = np.zeros((n, 3, 3))
        sigma_xx_3f[:, 0, 0] = df['var_mkt'].values
        sigma_xx_3f[:, 0, 1] = df['cov_m_s'].values
        sigma_xx_3f[:, 0, 2] = df['cov_m_h'].values
        sigma_xx_3f[:, 1, 0] = df['cov_m_s'].values
        sigma_xx_3f[:, 1, 1] = df['var_smb'].values
        sigma_xx_3f[:, 1, 2] = df['cov_s_h'].values
        sigma_xx_3f[:, 2, 0] = df['cov_m_h'].values
        sigma_xx_3f[:, 2, 1] = df['cov_s_h'].values
        sigma_xx_3f[:, 2, 2] = df['var_hml'].values
        
        sigma_xy_3f = np.zeros((n, 3, 1))
        sigma_xy_3f[:, 0, 0] = df['cov_y_mkt_rf'].values
        sigma_xy_3f[:, 1, 0] = df['cov_y_smb'].values
        sigma_xy_3f[:, 2, 0] = df['cov_y_hml'].values
        
        valid_mask_3f = ~df[['var_mkt', 'cov_y_mkt_rf']].isna().any(axis=1).values
        betas_3f = np.full((n, 3), np.nan)
        if np.any(valid_mask_3f):
            solved_3f = np.linalg.solve(sigma_xx_3f[valid_mask_3f], sigma_xy_3f[valid_mask_3f])
            betas_3f[valid_mask_3f] = solved_3f.squeeze(-1)
            
        df['beta_mkt_252d'] = betas_3f[:, 0]
        df['beta_smb_252d'] = betas_3f[:, 1]
        df['beta_hml_252d'] = betas_3f[:, 2]
        
        # --- Solve 5-Factor System ---
        sigma_xx_5f = np.zeros((n, 5, 5))
        for i in range(5):
            for j in range(5):
                f1, f2 = factor_names[i], factor_names[j]
                if i <= j:
                    key = f"cov_{f1}_{f2}" if f1 != f2 else f"var_{f1}"
                else:
                    key = f"cov_{f2}_{f1}" if f2 != f1 else f"var_{f2}"
                sigma_xx_5f[:, i, j] = df[key].values
                
        sigma_xy_5f = np.zeros((n, 5, 1))
        for i in range(5):
            f = factor_names[i]
            sigma_xy_5f[:, i, 0] = df[f'cov_y_{f}'].values
            
        valid_mask_5f = ~df[['var_mkt_rf', 'cov_y_mkt_rf']].isna().any(axis=1).values
        betas_5f = np.full((n, 5), np.nan)
        if np.any(valid_mask_5f):
            solved_5f = np.linalg.solve(sigma_xx_5f[valid_mask_5f], sigma_xy_5f[valid_mask_5f])
            betas_5f[valid_mask_5f] = solved_5f.squeeze(-1)
            
        df['beta_mkt_5f_252d'] = betas_5f[:, 0]
        df['beta_smb_5f_252d'] = betas_5f[:, 1]
        df['beta_hml_5f_252d'] = betas_5f[:, 2]
        df['beta_rmw_5f_252d'] = betas_5f[:, 3]
        df['beta_cma_5f_252d'] = betas_5f[:, 4]
        
        # Drop temporary columns
        cols_to_drop = [
            'var_mkt', 'var_smb', 'var_hml', 'cov_m_s', 'cov_m_h', 'cov_s_h',
            'cov_y_mkt_rf', 'cov_y_smb', 'cov_y_hml', 'cov_y_rmw', 'cov_y_cma', 'ret_excess'
        ]
        cols_to_drop.extend(cov_elements_5f.keys())
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        return df

    def _cross_sectional_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks features cross-sectionally per day to normalize data and remove outliers.
        This resolves lookahead bias by ranking within each daily slice.
        """
        logger.info("Computing daily cross-sectional ranks...")
        
        raw_features = [
            'mom_21d', 'mom_252d_skip1', 'vol_21d', 'rsi_14d', 
            'beta_mkt_252d', 'beta_smb_252d', 'beta_hml_252d',
            'beta_mkt_5f_252d', 'beta_smb_5f_252d', 'beta_hml_5f_252d', 'beta_rmw_5f_252d', 'beta_cma_5f_252d',
            'value_proxy', 'quality_proxy'
        ]
        
        # Drop rows where we can't compute features yet (e.g. first 252 days)
        df = df.dropna(subset=raw_features).copy()
        
        for feat in raw_features:
            rank_col = f"{feat}_rank"
            # Rank strictly within each date group, scaling to [0, 1]
            df[rank_col] = df.groupby('date')[feat].transform(lambda x: x.rank(pct=True))
            
        return df

    def build_features(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Executes the full feature engineering pipeline."""
        df_feat = self._compute_rolling_features(df_clean)
        df_feat_ff = self._compute_rolling_ff_betas(df_feat)
        df_ranked = self._cross_sectional_ranking(df_feat_ff)
        
        # Save master panel
        df_ranked.to_parquet(self.features_path)
        logger.info(f"Saved master feature panel to {self.features_path}")
        logger.info(f"Master panel shape: {df_ranked.shape}")
        
        return df_ranked
