import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
t0 = time.time()
print("Starting...")
import pandas as pd
print("Imported pandas:", time.time() - t0)
DATA_PATH = "data/processed/master_panel_features.parquet"
df = pd.read_parquet(DATA_PATH)
print("Read parquet:", time.time() - t0)
float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].astype('float32')
print("Cast to float32:", time.time() - t0)
