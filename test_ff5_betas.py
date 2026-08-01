import pandas as pd
import numpy as np
import time

def test_ff5_betas():
    print("Testing FF5 rolling beta calculation...")
    
    # Generate mock data: 1000 days, 100 stocks
    np.random.seed(42)
    n_days = 1000
    n_stocks = 100
    n_rows = n_days * n_stocks
    
    dates = pd.date_range("2020-01-01", periods=n_days)
    tickers = [f"STK_{i}" for i in range(n_stocks)]
    
    df_list = []
    for t in tickers:
        temp = pd.DataFrame({
            'date': dates,
            'ticker': t,
            'ret': np.random.normal(0.0005, 0.02, n_days),
            'rf': 0.0001
        })
        df_list.append(temp)
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Mock FF5 factors
    ff5_df = pd.DataFrame({
        'date': dates,
        'mkt_rf': np.random.normal(0.0003, 0.01, n_days),
        'smb': np.random.normal(0.0001, 0.005, n_days),
        'hml': np.random.normal(-0.0001, 0.006, n_days),
        'rmw': np.random.normal(0.0002, 0.004, n_days),
        'cma': np.random.normal(0.0001, 0.003, n_days)
    })
    
    df = pd.merge(df, ff5_df, on='date', how='left')
    df = df.sort_values(['ticker', 'date'])
    
    # Implement rolling FF5 betas
    start_time = time.time()
    
    window = 252
    factors_df = df.groupby('date')[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']].first().sort_index()
    
    # Compute rolling covariance matrix elements of factors
    factor_names = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']
    cov_elements = {}
    for i in range(len(factor_names)):
        for j in range(i, len(factor_names)):
            f1, f2 = factor_names[i], factor_names[j]
            key = f"cov_{f1}_{f2}" if f1 != f2 else f"var_{f1}"
            cov_elements[key] = factors_df[f1].rolling(window).cov(factors_df[f2])
            
    factor_cov_df = pd.DataFrame(cov_elements, index=factors_df.index)
    df = pd.merge(df, factor_cov_df, on='date', how='left')
    
    # Compute stock covariance with factors
    df['ret_excess'] = df['ret'] - df['rf']
    for f in factor_names:
        df[f'cov_y_{f}'] = df.groupby('ticker')['ret_excess'].transform(lambda x: x.rolling(window).cov(df.loc[x.index, f]))
        
    # Build systems Sigma_XX * Beta = Sigma_XY
    n = len(df)
    sigma_xx = np.zeros((n, 5, 5))
    for i in range(5):
        for j in range(5):
            f1, f2 = factor_names[i], factor_names[j]
            # Get the correct key (since matrix is symmetric)
            if i <= j:
                key = f"cov_{f1}_{f2}" if f1 != f2 else f"var_{f1}"
            else:
                key = f"cov_{f2}_{f1}" if f2 != f1 else f"var_{f2}"
            sigma_xx[:, i, j] = df[key].values
            
    sigma_xy = np.zeros((n, 5, 1))
    for i in range(5):
        f = factor_names[i]
        sigma_xy[:, i, 0] = df[f'cov_y_{f}'].values
        
    valid_mask = ~df[['var_mkt_rf', 'cov_y_mkt_rf']].isna().any(axis=1).values
    betas = np.full((n, 5), np.nan)
    
    if np.any(valid_mask):
        solved = np.linalg.solve(sigma_xx[valid_mask], sigma_xy[valid_mask])
        betas[valid_mask] = solved.squeeze(-1)
        
    for i, f in enumerate(factor_names):
        df[f'beta_{f}_5f_252d'] = betas[:, i]
        
    end_time = time.time()
    print(f"Computed FF5 rolling betas for {n:,} rows in {end_time - start_time:.3f} seconds.")
    print("Columns in df:", [c for c in df.columns if 'beta_' in c])
    print("Sample betas:")
    print(df[df['beta_mkt_rf_5f_252d'].notna()][['ticker', 'date', 'beta_mkt_rf_5f_252d', 'beta_smb_5f_252d']].head(5))

if __name__ == "__main__":
    test_ff5_betas()
