import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os

# Determine the base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

MIN_ROWS_THRESHOLD = 10  # minimum number of rows required for a column to be kept

def load_data(filepath, parse_dates=['Date'], index_col='Date'):
    """
    Load a CSV file into a Pandas DataFrame.
    """
    df = pd.read_csv(filepath, parse_dates=parse_dates, index_col=index_col)
    return df

def compute_log_returns(series):
    """
    Compute the log returns for a price series.
    Replace 0 with NaN to avoid divide-by-zero warnings.
    """
    series = series.replace(0, np.nan)
    return np.log(series).diff().dropna()

def winsorize_series(series, lower_percentile=0.5, upper_percentile=99.5):
    """
    Winsorize a series by clipping its values to the specified lower and upper percentiles.
    """
    if series.empty:
        return series
    lower_bound = np.percentile(series, lower_percentile)
    upper_bound = np.percentile(series, upper_percentile)
    return series.clip(lower=lower_bound, upper=upper_bound)

def fill_missing(series):
    """
    Fill missing values using forward-fill.
    """
    return series.ffill()

def standardize_series(series):
    """
    Standardize the series to have zero mean and unit variance.
    """
    return (series - series.mean()) / series.std()

def adf_test(series):
    """
    Perform the Augmented Dickey-Fuller (ADF) test on a series.
    """
    result = adfuller(series)
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4]
    }

def preprocess_price_series(series):
    """
    Preprocess a raw price series:
      1. Compute log returns.
      2. Winsorize extreme values.
      3. Forward-fill missing values.
      4. Standardize the series.
    """
    log_ret = compute_log_returns(series)
    if log_ret.empty:
        return log_ret
    winsorized = winsorize_series(log_ret)
    filled = fill_missing(winsorized)
    standardized = standardize_series(filled)
    return standardized

def process_all_datasets():
    """
    Load and process all raw datasets and save processed outputs to the processed folder.
    Expected raw files:
      - ../data/raw/energy_close_V2.csv
      - ../data/raw/energy_market_cap_V2.csv
      - ../data/raw/energy_volume_V2.csv
      - ../data/raw/energy.csv  (metadata)
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # ---------------- Process energy_close_V2.csv (price data) ----------------
    close_filepath = os.path.join(RAW_DIR, 'energy_close_V2.csv')
    df_close = load_data(close_filepath)  # assumes header "Date"
    print("[DEBUG] energy_close_V2.csv initial shape:", df_close.shape)
    
    processed_cols = {}  # dictionary to hold processed series for each column
    for col in df_close.columns:
        if pd.api.types.is_numeric_dtype(df_close[col]):
            print(f"[DEBUG] Processing column: {col}")
            processed_series = preprocess_price_series(df_close[col])
            print(f"[DEBUG] Processed column {col} length: {len(processed_series)}")
            # Only include if processed_series meets our minimum row threshold
            if len(processed_series) >= MIN_ROWS_THRESHOLD:
                processed_cols[col] = processed_series
            else:
                print(f"[DEBUG] Skipping column {col} due to insufficient data.")
            
    # Concatenate processed columns using an inner join (keep only common dates)
    if processed_cols:
        df_close_processed = pd.concat(processed_cols, axis=1, join='inner')
    else:
        df_close_processed = pd.DataFrame()
    print("[DEBUG] energy_close_processed final shape:", df_close_processed.shape)
    df_close_processed.to_csv(os.path.join(PROCESSED_DIR, 'energy_close_processed.csv'))
    print("Processed energy_close_V2.csv saved.")
    print("---------------------------------------------------\n")
    
    # ---------------- Process energy_market_cap_V2.csv (market cap data) ----------------
    market_cap_filepath = os.path.join(RAW_DIR, 'energy_market_cap_V2.csv')
    df_market_cap = load_data(market_cap_filepath, parse_dates=['date'], index_col='date')
    df_market_cap = df_market_cap[~df_market_cap.index.duplicated(keep='first')]
    print("[DEBUG] energy_market_cap_V2.csv initial shape:", df_market_cap.shape)
    
    processed_cols_cap = {}
    for col in df_market_cap.columns:
        if pd.api.types.is_numeric_dtype(df_market_cap[col]):
            print(f"[DEBUG] Processing market cap column: {col}")
            processed_series = preprocess_price_series(df_market_cap[col])
            print(f"[DEBUG] Processed market cap column {col} length: {len(processed_series)}")
            if len(processed_series) >= MIN_ROWS_THRESHOLD:
                processed_cols_cap[col] = processed_series
            else:
                print(f"[DEBUG] Skipping market cap column {col} due to insufficient data.")
    if processed_cols_cap:
        df_market_cap_processed = pd.concat(processed_cols_cap, axis=1, join='inner')
    else:
        df_market_cap_processed = pd.DataFrame()
    print("[DEBUG] energy_market_cap_processed final shape:", df_market_cap_processed.shape)
    df_market_cap_processed.to_csv(os.path.join(PROCESSED_DIR, 'energy_market_cap_processed.csv'))
    print("Processed energy_market_cap_V2.csv saved.")
    print("---------------------------------------------------\n")
    
    # ---------------- Process energy_volume_V2.csv (volume data) ----------------
    volume_filepath = os.path.join(RAW_DIR, 'energy_volume_V2.csv')
    df_volume = load_data(volume_filepath)  # assumes header "Date"
    print("[DEBUG] energy_volume_V2.csv initial shape:", df_volume.shape)
    
    processed_cols_vol = {}
    for col in df_volume.columns:
        if pd.api.types.is_numeric_dtype(df_volume[col]):
            print(f"[DEBUG] Processing volume column: {col}")
            processed_series = preprocess_price_series(df_volume[col])
            print(f"[DEBUG] Processed volume column {col} length: {len(processed_series)}")
            if len(processed_series) >= MIN_ROWS_THRESHOLD:
                processed_cols_vol[col] = processed_series
            else:
                print(f"[DEBUG] Skipping volume column {col} due to insufficient data.")
    if processed_cols_vol:
        df_volume_processed = pd.concat(processed_cols_vol, axis=1, join='inner')
    else:
        df_volume_processed = pd.DataFrame()
    print("[DEBUG] energy_volume_processed final shape:", df_volume_processed.shape)
    df_volume_processed.to_csv(os.path.join(PROCESSED_DIR, 'energy_volume_processed.csv'))
    print("Processed energy_volume_V2.csv saved.")
    print("---------------------------------------------------\n")
    
    # ---------------- Process energy.csv (metadata) ----------------
    metadata_filepath = os.path.join(RAW_DIR, 'energy.csv')
    df_metadata = pd.read_csv(metadata_filepath)  # load without parse_dates/index_col
    df_metadata.to_csv(os.path.join(PROCESSED_DIR, 'energy_metadata.csv'), index=False)
    print("Metadata energy.csv saved to processed folder.")

if __name__ == '__main__':
    process_all_datasets()
