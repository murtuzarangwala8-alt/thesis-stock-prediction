import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request
import zipfile
import io
from .utils import setup_logger

logger = setup_logger("DataPipeline")

class DataProcessor:
    """
    Handles fetching, cleaning, and aligning raw market data.
    Implements a survivorship bias proxy by attempting to scrape historical
    S&P 500 constituent changes, though limited by open-source yfinance availability.
    """
    def __init__(self, start_date: str, end_date: str, data_dir: Path):
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _get_sp500_tickers(self) -> list:
        """Fetches current S&P 500 tickers from a reliable source.

        SURVIVORSHIP BIAS DISCLOSURE (Audit Fix C1)
        =============================================
        This method uses the CURRENT S&P 500 constituent list applied
        retroactively to the historical period (2015-2024). Companies that
        were delisted, acquired, or went bankrupt during this period are
        excluded from the sample, which artificially inflates historical
        return distributions. This is a known limitation because yfinance
        does not provide historical constituent membership data.

        A point-in-time constituent database (e.g., CRSP/Compustat merged)
        would eliminate this bias but requires a commercial data license.

        All empirical results in this thesis should be interpreted with
        this caveat in mind. See Thesis Section 3.1 (Limitations).
        """
        logger.info("Fetching S&P 500 constituents from GitHub reliable source...")
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        
        # Clean tickers for yfinance
        tickers = [t.replace('.', '-') for t in tickers]
        logger.warning(
            "SURVIVORSHIP BIAS WARNING: Using current S&P 500 constituents "
            "applied retroactively. Companies delisted/acquired/bankrupt during "
            "2015-2024 are excluded, which inflates historical returns. "
            "A point-in-time constituent database (CRSP/Compustat) would "
            "eliminate this bias. See Thesis Section 3.1."
        )
        return tickers

    def fetch_market_data(self) -> pd.DataFrame:
        """Downloads pricing and volume data via yfinance."""
        tickers = self._get_sp500_tickers()
        tickers.append("SPY")
        
        logger.info(f"Downloading data for {len(tickers)} tickers from {self.start_date} to {self.end_date}...")
        
        df_raw = yf.download(
            tickers, 
            start=self.start_date, 
            end=self.end_date, 
            progress=False,
            group_by='ticker'
        )
        
        records = []
        for ticker in tickers:
            if ticker not in df_raw.columns.levels[0]:
                continue
                
            ticker_df = df_raw[ticker].copy()
            ticker_df = ticker_df.reset_index()
            ticker_df['ticker'] = ticker
            records.append(ticker_df)
            
        df_long = pd.concat(records, ignore_index=True)
        df_long.columns = [c.lower() for c in df_long.columns]
        
        if df_long['date'].dt.tz is not None:
            df_long['date'] = df_long['date'].dt.tz_localize(None)
            
        logger.info(f"Raw data download complete. Shape: {df_long.shape}")
        
        raw_path = self.raw_dir / "sp500_raw.parquet"
        df_long.to_parquet(raw_path)
        logger.info(f"Saved raw data to {raw_path}")
        
        return df_long

    def fetch_fama_french_factors(self) -> pd.DataFrame:
        """Downloads daily Fama-French 5-Factor data including risk-free rate from Ken French Library."""
        logger.info("Fetching Fama-French 5-Factor daily data...")
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
            with zip_file.open(zip_file.namelist()[0]) as f:
                content = f.read().decode('utf-8')
                
        lines = content.split('\n')
        skip_rows = 0
        for i, line in enumerate(lines[:15]):
            if 'Mkt-RF' in line:
                skip_rows = i
                break
                
        df_ff = pd.read_csv(io.StringIO(content), skiprows=skip_rows)
        df_ff.columns = ['date', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']
        df_ff = df_ff.dropna()
        df_ff = df_ff[df_ff['date'].astype(str).str.strip().str.isdigit()]
        df_ff['date'] = pd.to_datetime(df_ff['date'].astype(str), format='%Y%m%d')
        
        for col in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']:
            df_ff[col] = df_ff[col].astype(float) / 100.0  # Convert percent to decimal
            
        ff_path = self.raw_dir / "fama_french_raw.parquet"
        df_ff.to_parquet(ff_path)
        logger.info(f"Saved raw Fama-French factors to {ff_path}")
        return df_ff

    def clean_and_align(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data, forward-fills gaps, aligns the cross-section,
        incorporates historical risk-free rates, and winsorizes outliers.
        """
        logger.info("Cleaning and aligning data...")
        df = df_raw.copy()
        
        # UNADJUSTED CLOSE FIX (Audit Fix M7)
        # ======================================
        # Use 'adj close' (split-and-dividend adjusted) for return computation.
        # Raw 'close' creates phantom return spikes on stock split dates
        # (e.g., AAPL 4:1 split in 2020 would show a -75% daily 'return').
        # yfinance's 'adj close' retroactively adjusts for all corporate actions.
        adj_col = 'adj close' if 'adj close' in df.columns else 'close'
        if adj_col == 'close':
            logger.warning("'adj close' column not found — falling back to unadjusted 'close'. "
                           "Returns may contain split/dividend artefacts.")
        else:
            logger.info("Using 'adj close' for return computation (split/dividend adjusted).")
        
        df = df[df[adj_col] > 1.0] 
        df = df[df['volume'] > 0]
        
        df = df.sort_values(['ticker', 'date'])
        df['ret'] = df.groupby('ticker')[adj_col].pct_change()
        
        for col in ['close', 'volume', 'ret']:
            df[col] = df.groupby('ticker')[col].ffill(limit=5)
        
        df = df.dropna(subset=['ret'])
        
        spy = df[df['ticker'] == 'SPY'][['date', 'ret']].rename(columns={'ret': 'mkt_ret'})
        df = df[df['ticker'] != 'SPY']
        
        df = pd.merge(df, spy, on='date', how='left')
        
        # Fetch and merge Fama-French factors
        df_ff = self.fetch_fama_french_factors()
        df = pd.merge(df, df_ff, on='date', how='left')
        
        # Forward fill factors for stock tickers in case of alignment mismatch
        df = df.sort_values(['ticker', 'date'])
        for col in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']:
            df[col] = df.groupby('ticker')[col].ffill()
            df[col] = df[col].fillna(0.0)
            
        # EXECUTION LOOKAHEAD FIX (Audit Fix C2)
        # =========================================
        # Features at Close(t) predict return from Close(t+1)->Close(t+2).
        # shift(-2) allows realistic signal computation at Close(t) and
        # execution at Open(t+1), capturing the Close(t+1)->Close(t+2) return.
        # Original shift(-1) assumed instantaneous execution at Close(t),
        # which is physically impossible and captured overnight gaps.
        df['rf_exec_1d'] = df.groupby('ticker')['rf'].shift(-2)
        df['target_excess_1d'] = df.groupby('ticker')['ret'].shift(-2) - df['rf_exec_1d']
        
        # Target excess return: 21-day forward return with 1-day execution buffer
        # Starts from Close(t+1) to Close(t+22) instead of Close(t) to Close(t+21)
        df['ret_21d_forward'] = df.groupby('ticker')['ret'].transform(lambda x: np.exp(np.log(np.maximum(1 + x, 1e-8)).rolling(21).sum()).shift(-22) - 1)
        df['rf_21d_forward'] = df.groupby('ticker')['rf'].transform(lambda x: x.rolling(21).sum().shift(-22))
        df['target_excess_21d'] = df['ret_21d_forward'] - df['rf_21d_forward']
        
        df = df.dropna(subset=['target_excess_1d'])
        
        # Cross-sectional Winsorization [1%, 99%] daily to handle outliers
        logger.info("Winsorizing target excess returns and daily returns cross-sectionally at [1%, 99%]...")
        for col in ['target_excess_1d', 'target_excess_21d', 'ret']:
            q_low = df.groupby('date')[col].transform('quantile', 0.01)
            q_high = df.groupby('date')[col].transform('quantile', 0.99)
            df[col] = df[col].clip(lower=q_low, upper=q_high)
            
        logger.info(f"Cleaned panel shape: {df.shape}")
        
        processed_path = self.processed_dir / "sp500_cleaned.parquet"
        df.to_parquet(processed_path)
        logger.info(f"Saved cleaned panel to {processed_path}")
        
        return df

    def run_pipeline(self):
        """Executes the full data pipeline."""
        raw_path = self.raw_dir / "sp500_raw.parquet"
        if raw_path.exists():
            logger.info("Loading existing raw data...")
            df_raw = pd.read_parquet(raw_path)
        else:
            df_raw = self.fetch_market_data()
            
        df_clean = self.clean_and_align(df_raw)
        return df_clean
