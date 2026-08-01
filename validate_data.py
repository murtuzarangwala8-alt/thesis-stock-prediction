import pandas as pd
import numpy as np

print('Loading dataset...')
df = pd.read_parquet('data/processed/master_panel_features.parquet')

print('\n--- Dataset Validation Report ---')
print(f'Total Rows: {len(df):,}')
print(f'Total Columns: {len(df.columns)}')
print(f"Unique Tickers: {df['ticker'].nunique()}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")

# Check for NaNs
nan_counts = df.isna().sum()
cols_with_nans = nan_counts[nan_counts > 0]
print(f'\nColumns with NaNs: {len(cols_with_nans)}')
if len(cols_with_nans) > 0:
    print('Top 5 Columns with NaNs:')
    print(cols_with_nans.sort_values(ascending=False).head(5))

# Check for Infinity
numeric_cols = df.select_dtypes(include=[np.number])
inf_counts = np.isinf(numeric_cols).sum()
cols_with_infs = inf_counts[inf_counts > 0]
print(f'\nColumns with Infinity: {len(cols_with_infs)}')
if len(cols_with_infs) > 0:
    print('Columns containing Infinity:')
    print(cols_with_infs.sort_values(ascending=False).head(5))

# Validate Targets (Using actual column names in the parquet)
print('\nTarget Variables:')
print(f"target_ret_1d Mean: {df['target_ret_1d'].mean():.6f}, Std: {df['target_ret_1d'].std():.6f}")
print(f"target_ret_21d Mean: {df['target_ret_21d'].mean():.6f}, Std: {df['target_ret_21d'].std():.6f}")
print('\nValidation Complete!')
