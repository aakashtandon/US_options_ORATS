import numpy as np
from numba import njit
import pandas as pd
import polars as pl

def quick_float32_convert(df):

    """Fastest possible float64→float32 conversion
        makes the dataframe small and process faster
                                                    """  
    float_cols = df.select_dtypes('float64').columns
    return df.astype({col: 'float32' for col in float_cols}) if len(float_cols) > 0 else df


def quick_float32_convert2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fastest possible float64 to float32 conversion for a pandas DataFrame,
    while excluding any columns that end with '_volume'. since  'volume' is very large sometimes
    """
    # 1. Select all columns that are of type float64
    float64_cols = df.select_dtypes(include='float64').columns
    
    # 2. From that list, create a new list excluding columns that end with '_volume'
    cols_to_convert = [col for col in float64_cols if not col.endswith('_volume')]
    
    if not cols_to_convert:
        # If no columns need converting, return the original DataFrame
        return df
        
    # 3. Create the dictionary for only the columns we want to change
    dtype_map = {col: 'float32' for col in cols_to_convert}
    
    # 4. Apply the conversion and return the new DataFrame
    return df.astype(dtype_map)


# Usage - replace your entire optimization loop with:
#com_df2 = quick_float32_convert2(com_df2)

import pandas as pd
import numpy as np
from numba import njit

# The Numba helper function _resample_fast remains unchanged from the previous version.
# It now accepts t0 as an argument - this is start 
@njit
def _resample_fast2(ts, open_, high, low, close, volume, bin_ns, t0):
    # ... (function content is identical to the previous answer)
    n = ts.size
    max_bins = n
    out_open = np.empty(max_bins, dtype=np.float64)
    out_high = np.empty(max_bins, dtype=np.float64)
    out_low = np.empty(max_bins, dtype=np.float64)
    out_close = np.empty(max_bins, dtype=np.float64)
    out_volume = np.empty(max_bins, dtype=np.float64)
    out_ts = np.empty(max_bins, dtype=np.int64)
    if n == 0:
        return 0, out_open, out_high, out_low, out_close, out_volume, out_ts
    current_bin = (ts[0] - t0) // bin_ns
    agg_ts = t0 + current_bin * bin_ns
    agg_open = open_[0]
    agg_high = high[0]
    agg_low = low[0]
    agg_close = close[0]
    agg_volume = volume[0]
    out_idx = 0
    for i in range(1, n):
        bin_id = (ts[i] - t0) // bin_ns
        if bin_id != current_bin:
            out_ts[out_idx] = agg_ts
            out_open[out_idx] = agg_open
            out_high[out_idx] = agg_high
            out_low[out_idx] = agg_low
            out_close[out_idx] = agg_close
            out_volume[out_idx] = agg_volume
            out_idx += 1
            current_bin = bin_id
            agg_ts = t0 + current_bin * bin_ns
            agg_open = open_[i]
            agg_high = high[i]
            agg_low = low[i]
            agg_close = close[i]
            agg_volume = volume[i]
        else:
            if high[i] > agg_high:
                agg_high = high[i]
            if low[i] < agg_low:
                agg_low = low[i]
            agg_close = close[i]
            agg_volume += volume[i]
    out_ts[out_idx] = agg_ts
    out_open[out_idx] = agg_open
    out_high[out_idx] = agg_high
    out_low[out_idx] = agg_low
    out_close[out_idx] = agg_close
    out_volume[out_idx] = agg_volume
    out_idx += 1
    return out_idx, out_open, out_high, out_low, out_close, out_volume, out_ts


def resample_data_crypto_fast_numba_origin(df, timeframe='30min', highcol='High', lowcol='Low', opencol='Open', closecol='Close', volumecol='Volume', origin='start_day'):
    """
    Fast OHLCV resampling using Numba with support for a daily or custom time origin.

    parameter: origin -- 'start_day'-- means resampling start from 00:00 ( midnight)
                      -- '9:30' -- means starts from 9:30  

    """


    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    if df.empty:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    df = df.sort_index()
    
    ts = df.index.to_numpy().astype(np.int64)

    # --- NEW LOGIC TO HANDLE CUSTOM TIME ORIGIN ---
    first_timestamp = df.index[0]
    if origin == 'start_day':
        # Normalize to the beginning of that day (midnight)
        t0 = first_timestamp.normalize().value
    elif isinstance(origin, str) and ':' in origin:
        # Handle a time string like '09:15'
        try:
            # Combine the date of the first timestamp with the custom time origin
            start_of_day = first_timestamp.normalize()
            custom_time = pd.to_datetime(origin).time()
            custom_origin_ts = pd.Timestamp.combine(start_of_day.date(), custom_time)
            
            # If the first data point is before the custom origin on the first day, 
            # set the origin to the previous day's custom time.
            if first_timestamp < custom_origin_ts:
                custom_origin_ts = custom_origin_ts - pd.Timedelta(days=1)
            
            t0 = custom_origin_ts.value
        except ValueError:
            raise ValueError(f"Could not parse the custom time origin string: '{origin}'")
    else:
        # Default behavior: use the very first timestamp as the origin
        t0 = ts[0]
    
    # --- The rest of the function remains the same ---
    tf = timeframe.strip().lower()
    # ... (rest of the function is identical)
    unit = ''.join(filter(str.isalpha, tf))
    num_str = ''.join(filter(str.isdigit, tf))
    if not num_str:
        raise ValueError("Timeframe must include a number (e.g., '15min', '1d').")
    num = int(num_str)
    if unit == 'min':
        bin_ns = num * 60 * 1_000_000_000
    elif unit == 'h':
        bin_ns = num * 60 * 60 * 1_000_000_000
    elif unit == 'd':
        bin_ns = num * 24 * 60 * 60 * 1_000_000_000
    elif unit == 'w':
        bin_ns = num * 7 * 24 * 60 * 60 * 1_000_000_000
    else:
        raise ValueError("Unsupported timeframe.")
    open_ = df[opencol].values.astype(np.float64)
    high = df[highcol].values.astype(np.float64)
    low = df[lowcol].values.astype(np.float64)
    close = df[closecol].values.astype(np.float64)
    volume = df[volumecol].values.astype(np.float64) if volumecol in df.columns else np.zeros_like(open_)
    out = _resample_fast2(ts, open_, high, low, close, volume, bin_ns, t0)
    rdf = pd.DataFrame({
        'Open': out[1][:out[0]],
        'High': out[2][:out[0]],
        'Low': out[3][:out[0]],
        'Close': out[4][:out[0]],
        'Volume': out[5][:out[0]],
    }, index=pd.to_datetime(out[6][:out[0]], unit='ns'))

    return rdf





def resample_data_crypto_fast_numba(df, timeframe='30min' , highcol='High', lowcol='Low', opencol='Open', closecol='Close', volumecol='Volume'):
    """
    Fast OHLCV resampling using Numba.
    
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    df = df.sort_index()
    
    
    #ts = df.index.astype('int64').to_numpy() * 1000  # convert to ns

    ts = pd.to_datetime(df.index).astype('int64').to_numpy()

    tf = timeframe.strip().lower()
    unit = ''.join(filter(str.isalpha, tf))
    num = int(''.join(filter(str.isdigit, tf)))
    if unit == 'min':
        bin_ns = num * 60 * 1_000_000_000
    elif unit == 'h':
        bin_ns = num * 60 * 60 * 1_000_000_000
    elif unit == 'd':
        bin_ns = num * 24 * 60 * 60 * 1_000_000_000
    elif unit == 'w':
        bin_ns = num * 7 * 24 * 60 * 60 * 1_000_000_000
    else:
        raise ValueError("Unsupported timeframe.")

    open_ = df[opencol].values.astype(np.float64)
    high = df[highcol].values.astype(np.float64)
    low = df[lowcol].values.astype(np.float64)
    close = df[closecol].values.astype(np.float64)
    volume = df[volumecol].values.astype(np.float64) if volumecol in df.columns else np.zeros_like(open_)

    out = _resample_fast(ts, open_, high, low, close, volume, bin_ns)

    rdf = pd.DataFrame({
        'Open': out[1][:out[0]],
        'High': out[2][:out[0]],
        'Low': out[3][:out[0]],
        'Close': out[4][:out[0]],
        'Volume': out[5][:out[0]],
    }, index=pd.to_datetime(out[6][:out[0]], unit='ns'))

    return rdf


@njit
def _resample_fast(ts, open_, high, low, close, volume, bin_ns):
    n = ts.size
    max_bins = n  # worst case: every row in a separate bin

    out_open = np.empty(max_bins, dtype=np.float64)
    out_high = np.empty(max_bins, dtype=np.float64)
    out_low = np.empty(max_bins, dtype=np.float64)
    out_close = np.empty(max_bins, dtype=np.float64)
    out_volume = np.empty(max_bins, dtype=np.float64)
    out_ts = np.empty(max_bins, dtype=np.int64)

    if n == 0:
        return 0, out_open, out_high, out_low, out_close, out_volume, out_ts

    t0 = ts[0]
    current_bin = (ts[0] - t0) // bin_ns
    agg_ts = t0 + current_bin * bin_ns
    agg_open = open_[0]
    agg_high = high[0]
    agg_low = low[0]
    agg_close = close[0]
    agg_volume = volume[0]

    out_idx = 0

    for i in range(1, n):
        bin_id = (ts[i] - t0) // bin_ns
        if bin_id != current_bin:
            out_ts[out_idx] = agg_ts
            out_open[out_idx] = agg_open
            out_high[out_idx] = agg_high
            out_low[out_idx] = agg_low
            out_close[out_idx] = agg_close
            out_volume[out_idx] = agg_volume
            out_idx += 1

            current_bin = bin_id
            agg_ts = t0 + current_bin * bin_ns
            agg_open = open_[i]
            agg_high = high[i]
            agg_low = low[i]
            agg_close = close[i]
            agg_volume = volume[i]
        else:
            if high[i] > agg_high:
                agg_high = high[i]
            if low[i] < agg_low:
                agg_low = low[i]
            agg_close = close[i]
            agg_volume += volume[i]

    # Final bin
    out_ts[out_idx] = agg_ts
    out_open[out_idx] = agg_open
    out_high[out_idx] = agg_high
    out_low[out_idx] = agg_low
    out_close[out_idx] = agg_close
    out_volume[out_idx] = agg_volume
    out_idx += 1

    return out_idx, out_open, out_high, out_low, out_close, out_volume, out_ts



#==============================================================================


@njit
def strict_prior_day_low(values: np.ndarray, dates: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the minimum low price over the prior n days, excluding the current day.
    
    Enhanced to handle NaN values properly and to be fully Numba-compatible
    with multi-dimensional input arrays.
    """
    # This initial setup remains the same
    result = np.empty_like(values)
    result.fill(np.nan)
    
    date_changes = np.where(dates[1:] != dates[:-1])[0] + 1
    date_starts = np.concatenate((np.array([0]), date_changes))
    date_ends = np.concatenate((date_changes, np.array([len(dates)])))
    
    daily_mins = np.empty(len(date_starts))
    daily_mins.fill(np.nan)
    
    for i in range(len(date_starts)):
        day_values = values[date_starts[i]:date_ends[i]]
        
        # --- FIX APPLIED HERE ---
        # Before (Problematic Line):
        # valid_values = day_values[~np.isnan(day_values)]
        
        # After (Corrected Logic):
        # 1. Flatten the 2D array slice into a 1D array.
        # 2. Apply the boolean mask to the 1D array.
        flat_day_values = day_values.flatten()
        valid_values = flat_day_values[~np.isnan(flat_day_values)]

        if len(valid_values) > 0:
            daily_mins[i] = np.min(valid_values)
    
    # This second loop for the rolling window remains the same
    for i in range(1, len(date_starts)):
        window_start = max(0, i - n)
        window_mins = daily_mins[window_start:i]
        
        valid_mins = window_mins[~np.isnan(window_mins)]
        if len(valid_mins) > 0:
            current_min = np.min(valid_mins)
            result[date_starts[i]:date_ends[i]] = current_min
            
    return result



def get_x_day_low_numba(df, n, column='Low'):
    """Safe wrapper that guarantees no future data usage"""
    dates = df.index.normalize().values.astype('datetime64[D]').astype(np.int64)
    values = df[column].values
    
    result = strict_prior_day_low(values, dates, n)
    return pd.Series(result, index=df.index)

#======================
import numpy as np
from numba import njit

@njit
def strict_prior_day_high(values: np.ndarray, dates: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the maximum high price over the prior n days, excluding the current day.
    
    Enhanced with proper NaN handling and corrected for Numba's multi-dimensional
    indexing limitations.
    """
    result = np.empty_like(values)
    result.fill(np.nan)
    
    # Find all date boundaries
    date_changes = np.where(dates[1:] != dates[:-1])[0] + 1
    date_starts = np.concatenate((np.array([0]), date_changes))
    date_ends = np.concatenate((date_changes, np.array([len(dates)])))
    
    # Calculate each day's max (ignoring NaNs)
    daily_maxs = np.empty(len(date_starts))
    daily_maxs.fill(np.nan)
    
    for i in range(len(date_starts)):
        day_values = values[date_starts[i]:date_ends[i]]
        
        # --- FIX APPLIED HERE ---
        # Before (Problematic line):
        # valid_values = day_values[~np.isnan(day_values)]

        # After (Corrected logic for Numba):
        # Flatten the 2D array slice into 1D before filtering.
        flat_day_values = day_values.flatten()
        valid_values = flat_day_values[~np.isnan(flat_day_values)]
        
        if len(valid_values) > 0:
            daily_maxs[i] = np.max(valid_values)
            
    # Apply rolling window EXCLUDING current day
    for i in range(1, len(date_starts)):
        window_start = max(0, i - n)
        window_maxs = daily_maxs[window_start:i]
        
        # Only calculate max if we have at least one valid value
        valid_maxs = window_maxs[~np.isnan(window_maxs)]
        if len(valid_maxs) > 0:
            current_max = np.max(valid_maxs)
            # Apply to current day's values
            result[date_starts[i]:date_ends[i]] = current_max
            
    return result

def get_x_day_high_numba(df, n, column='High'):
    """
    Safe wrapper that guarantees no future data usage
    
    Parameters:
        df: DataFrame with datetime index
        n: Lookback window in days
        column: Name of column containing high prices
        
    Returns:
        pd.Series with prior n-day highs (NaN where insufficient data)
    """
    dates = df.index.normalize().values.astype('datetime64[D]').astype(np.int64)
    values = df[column].values
    
    result = strict_prior_day_high(values, dates, n)
    return pd.Series(result, index=df.index, name=f'prior_{n}day_high')


#=== Time based window high and low


@njit
def _time_window_min(values: np.ndarray, timestamps: np.ndarray, start_sec: int, end_sec: int) -> np.ndarray:
    """Ultra-fast time window minimum calculator"""
    result = np.empty_like(values)
    result.fill(np.nan)
    current_min = np.inf
    current_date = 0
    
    for i in range(len(timestamps)):
        ts = timestamps[i]
        date = ts // (24 * 3600 * 1e9)
        time_of_day = (ts % (24 * 3600 * 1e9)) / 1e9
        
        if date != current_date:
            current_date = date
            current_min = np.inf
            in_window = False
        
        if start_sec <= time_of_day <= end_sec:
            in_window = True
            if values[i] < current_min:
                current_min = values[i]
            result[i] = current_min
        elif in_window:
            result[i] = current_min
    
    return result

@njit
def _time_window_max(values: np.ndarray, timestamps: np.ndarray, start_sec: int, end_sec: int) -> np.ndarray:
    """Ultra-fast time window maximum calculator"""
    result = np.empty_like(values)
    result.fill(np.nan)
    current_max = -np.inf
    current_date = 0
    
    for i in range(len(timestamps)):
        ts = timestamps[i]
        date = ts // (24 * 3600 * 1e9)
        time_of_day = (ts % (24 * 3600 * 1e9)) / 1e9
        
        if date != current_date:
            current_date = date
            current_max = -np.inf
            in_window = False
        
        if start_sec <= time_of_day <= end_sec:
            in_window = True
            if values[i] > current_max:
                current_max = values[i]
            result[i] = current_max
        elif in_window:
            result[i] = current_max
    
    return result

def get_x_min_low_numba(df, column, start='09:15', end='10:15'):
    """Calculate time-windowed minimum values (optimized)"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be datetime")
    
    start_sec = (int(start[:2]) * 3600 + int(start[3:]) * 60)
    end_sec = (int(end[:2]) * 3600 + int(end[3:]) * 60)
    
    return pd.Series(
        _time_window_min(
            df[column].values,
            df.index.values.astype(np.int64),
            start_sec,
            end_sec
        ),
        index=df.index
    )

def get_x_min_high_numba(df, column, start='09:15', end='10:15'):
    """Calculate time-windowed maximum values (optimized)"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be datetime")
    
    start_sec = (int(start[:2]) * 3600 + int(start[3:]) * 60)
    end_sec = (int(end[:2]) * 3600 + int(end[3:]) * 60)
    
    return pd.Series(
        _time_window_max(
            df[column].values,
            df.index.values.astype(np.int64),
            start_sec,
            end_sec
        ),
        index=df.index
    )
    
    

from tqdm import tqdm  # For progress bars

# ======= Core Numba Functions =======
@njit
def kendall_tau(x, y):
    """Numba-optimized Kendall Tau (no ties handling)."""
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            x_dir = x[i] - x[j]
            y_dir = y[i] - y[j]
            if x_dir * y_dir > 0:
                concordant += 1
            elif x_dir * y_dir < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else np.nan

@njit
def rolling_kendall(ret1, ret2, window):
    """Rolling Kendall Tau between two return series."""
    n = len(ret1)
    out = np.empty(n)
    out[:] = np.nan  # Initialize with NaNs
    for i in range(window-1, n):
        out[i] = kendall_tau(ret1[i-window+1:i+1], ret2[i-window+1:i+1])
    return out

# ======= Parameterized Wrapper =======
def calculate_all_kendall(df, symbols, window=24, benchmark='BTCUSDT'):
    """
    Calculate rolling Kendall correlations for all symbols against benchmark.
    
    Parameters:
        df (pd.DataFrame): DataFrame with '_close' columns for each symbol
        symbols (list): List of symbol prefixes (e.g., ['ETHUSDT', 'SOLUSDT'])
        window (int): Rolling window length
        benchmark (str): Benchmark symbol (default: 'BTCUSDT')
    
    Returns:
        pd.DataFrame: Original DataFrame with new '*_kendall' columns
    """
    # 1. Calculate log returns for all symbols
    
    benchmark_ret = df[f'{benchmark}_log_ret'].values
    
    # 2. Compute rolling Kendall for each symbol
    for sym in tqdm(symbols, desc='Calculating Kendall correlations'):
        sym_ret = df[f'{sym}_log_ret'].values
        df[f'{benchmark}_{sym}_kendall'] = rolling_kendall(benchmark_ret, sym_ret, window)
    
    return df

# # ======= Usage Example =======
# # Define your symbols (excluding BTC)
# final_symbols2 = ['ETHUSDT', 'SOLUSDT', '1000BONKUSDT']  

# # Run the calculation
# com_df2 = calculate_all_kendall(
#     df=com_df2,
#     symbols=final_symbols,
#     window=24,           # Customizable window
#     benchmark='BTCUSDT'  # Change to 'ETHUSDT' if needed
# )

# Result columns will be:
# 'BTCUSDT_ETHUSDT_kendall', 'BTCUSDT_SOLUSDT_kendall', etc.



@njit
def _time_window_close(values: np.ndarray, timestamps: np.ndarray, start_sec: int, end_sec: int) -> np.ndarray:
    """Ultra-fast time window close price calculator"""
    result = np.empty_like(values)
    result.fill(np.nan)
    last_close = np.nan
    current_date = 0
    
    for i in range(len(timestamps)):
        ts = timestamps[i]
        date = ts // (24 * 3600 * 1e9)
        time_of_day = (ts % (24 * 3600 * 1e9)) / 1e9
        
        if date != current_date:
            current_date = date
            last_close = np.nan
        
        if start_sec <= time_of_day <= end_sec:
            last_close = values[i]
            result[i] = last_close
        elif last_close is not np.nan:
            result[i] = last_close
    
    return result

def get_x_min_close_numba(df, column='Close', start='09:15', end='10:15'):
    """
    Calculate time-windowed close prices (optimized with Numba).
    
    Parameters:
        df (pd.DataFrame): Input dataframe with datetime index
        column (str): Column name containing price data (default: 'Close')
        start (str): Window start time in 'HH:MM' format (default: '09:15')
        end (str): Window end time in 'HH:MM' format (default: '10:15')
        
    Returns:
        pd.Series: Series with last close price within the time window
        
    Example:
        >>> df['window_close'] = get_x_min_close_numba(df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be datetime")
    
    start_sec = (int(start[:2]) * 3600 + int(start[3:]) * 60)
    end_sec = (int(end[:2]) * 3600 + int(end[3:]) * 60)
    
    return pd.Series(
        _time_window_close(
            df[column].values,
            df.index.values.astype(np.int64),
            start_sec,
            end_sec
        ),
        index=df.index,
        name=f'close_{start.replace(":","")}_{end.replace(":","")}'
    )

    

def calculate_prev_day_close(df, close_col='close', n_days=1):
    
    """
    Correctly calculates previous n trading days' value from intraday data
    
    """

    # Get the last close of each trading day
    daily_closes = df.groupby(df.index.normalize())[close_col].last()
    
    # Create a mapping of date to previous n-day close
    prev_closes = daily_closes.shift(n_days)
    
    # Map back to intraday data
    result = df.index.normalize().map(prev_closes).values
    
    return result


def calculate_prev_day_open(df, close_col='close', n_days=1):
    
    """
    Correctly calculates previous n trading days' value from intraday data
    
    """

    # Get the last close of each trading day
    daily_closes = df.groupby(df.index.normalize())[close_col].first()
    
    # Create a mapping of date to previous n-day close
    prev_closes = daily_closes.shift(n_days)
    
    # Map back to intraday data
    result = df.index.normalize().map(prev_closes).values
    
    return result


import pandas_ta as ta
import numpy as np

def calculate_atr_mtf_vectorized(df, symbols, resample_tf='1D', atr_period=5 ,  origin_time='09:30:00'):
    
    """
    Calculates a multi-timeframe ATR for all symbols at once, correctly
    shifting the data to prevent lookahead bias.

    #== faster than calling a single atr function
    """
    print(f"Vectorized MTF ATR for {len(symbols)} symbols...")
    
    # --- Step 1: Resample all symbols' OHLC data in one operation ---
    #print("  - Resampling to daily timeframe...")
    agg_rules = {}
    for symbol in symbols:
        agg_rules[f'{symbol}_open'] = 'first'
        agg_rules[f'{symbol}_high'] = 'max'
        agg_rules[f'{symbol}_low'] = 'min'
        agg_rules[f'{symbol}_close'] = 'last'
    
    daily_df = df.resample(resample_tf).agg(agg_rules , origin=origin_time).dropna(how='all').shift(1)

    # --- Step 2: Calculate daily ATR for each symbol ---
    #print(f"  - Calculating {atr_period}-period ATR on daily data...")
    atr_results = {}

    timeframe_suffix = resample_tf.upper()

    for s in symbols:

        column_name = f"{s}_{timeframe_suffix}_ATR"

        atr_results[column_name] = ta.atr(
            high=daily_df[f"{s}_high"],
            low=daily_df[f"{s}_low"],
            close=daily_df[f"{s}_close"],
            length=atr_period
        )
    daily_atr_df = pd.DataFrame(atr_results, index=daily_df.index)

   
    
    # --- Step 4: Reindex and join all new features back ---
    #print("  - Broadcasting daily ATR values back to the original index...")
    atr_reindexed = daily_atr_df.reindex(df.index, method='ffill')
    
    #print("✅ Vectorized ATR calculation complete.")
    return atr_reindexed

# 

# # This single function call replaces your entire for loop for the ATR
# atr_features_df = calculate_atr_mtf_vectorized(
#     df=com_df2, 
#     symbols=final_symbols,
#     resample_tf='1D',
#     atr_period=3 , origin_time='start'
# )

# Join the results back
# com_df2 = com_df2.join(atr_features_df)

# print("\n--- Final DataFrame Tail ---")
# # Note how the ATR value is constant throughout a single day
# # and is the value from the *previous* day.
# print(com_df2.filter(like='dailyATR').tail(10))

#=========================================================================

def calculate_atr_pandas_ta_vectorized(df: pd.DataFrame, symbols: list, atr_period: int = 14) -> pd.DataFrame:
    
    import pandas_ta as ta

    """

    Calculates ATR for all symbols in a fully vectorized way using pandas-ta,
    eliminating the symbol loop entirely.

    """
    # 1. Prepare column names and filter the DataFrame
    ohlc_cols = [f"{s}_{t}" for s in symbols for t in ('high', 'low', 'close')]
    data = df[ohlc_cols].copy()
    
    # 2. Reshape columns to a MultiIndex: (Symbol, OHLC_Type)
    data.columns = pd.MultiIndex.from_tuples(
        [c.split('_', 1) for c in data.columns],
        names=['symbol', 'type']
    )

    # 3. Stack the DataFrame to convert from wide to long format
    # The result has a MultiIndex of (timestamp, symbol)
    long_df = data.stack(level='symbol')

    # 4. Calculate ATR on the grouped data in a single pass
    # Use .apply() to run the ATR function on each group (symbol) separately
    atr_series = long_df.groupby(level='symbol', group_keys=False).apply(
        lambda group: group.ta.atr(length=atr_period, append=False) )
        
    # 5. Reshape the result back to the original wide format
    atr_df = atr_series.unstack(level='symbol')
    
    # 6. Rename columns to the final desired format (e.g., 'AAPL_ATR')
    atr_df.columns = [f"{col}_ATR" for col in atr_df.columns]
    
    return atr_df




import polars as pl
from functools import reduce

def fast_polars_resample_multi(
    df_pd: pd.DataFrame, 
    timeframe: str = '15m', 
    market_open_time: str = '09:31'
) -> pd.DataFrame:
    # Clean the input DataFrame
    if df_pd.columns.has_duplicates:
        df_pd = df_pd.loc[:, ~df_pd.columns.duplicated(keep='first')]

    # Convert to Polars
    df_pl = pl.from_pandas(df_pd.reset_index())

    # Calculate offset differently for daily vs intraday
    if 'd' in timeframe.lower():
        # For daily, offset is the market open time (e.g., "09:31")
         # For daily, convert market open time to duration format (e.g., "11:50" -> "11h50m")
        hours, minutes = map(int, market_open_time.split(':'))
        offset_str = f"{hours}h{minutes}m"
    else:
        # Intraday (minutes/hours) - existing logic
        hours, minutes = map(int, market_open_time.split(':'))
        if 'm' in timeframe.lower():
            timeframe_minutes = int(timeframe.rstrip('m'))
            offset_minutes = minutes % timeframe_minutes
            offset_str = f"{offset_minutes}m"
        elif 'h' in timeframe.lower():
            timeframe_hours = int(timeframe.rstrip('h'))
            offset_hours = hours % timeframe_hours
            offset_str = f"{offset_hours}h{minutes}m"


    # Create a list of resampled DataFrames for each aggregation type
    resampled_parts = [
        df_pl.select(["timestamp", pl.col("^.*_open$")]).group_by_dynamic(
            index_column="timestamp", 
            every=timeframe, 
            offset=offset_str, 
            label="left"
        ).agg(pl.all().first()),
        
        df_pl.select(["timestamp", pl.col("^.*_high$")]).group_by_dynamic(
            index_column="timestamp", 
            every=timeframe, 
            offset=offset_str, 
            label="left"
        ).agg(pl.all().max()),
        
        df_pl.select(["timestamp", pl.col("^.*_low$")]).group_by_dynamic(
            index_column="timestamp", 
            every=timeframe, 
            offset=offset_str, 
            label="left"
        ).agg(pl.all().min()),
        
        df_pl.select(["timestamp", pl.col("^.*_close$")]).group_by_dynamic(
            index_column="timestamp", 
            every=timeframe, 
            offset=offset_str, 
            label="left"
        ).agg(pl.all().last()),
        
        df_pl.select(["timestamp", pl.col("^.*_volume$")]).group_by_dynamic(
            index_column="timestamp", 
            every=timeframe, 
            offset=offset_str, 
            label="left"
        ).agg(pl.all().sum())
    ]
    




    
    result_df = reduce(
        lambda left, right: left.join(right, on="timestamp", how="outer_coalesce"),
        resampled_parts
    )

    final_pd = result_df.to_pandas().set_index('timestamp')
    original_cols_in_order = [col for col in df_pd.columns if col in final_pd.columns]
    return final_pd[original_cols_in_order]




def calculate_pivots_vectorized(df, symbols, freq='W' , origin='start'):
    """
    Calculates multi-timeframe pivot points for all symbols at once in a 
    fast, vectorized way.

    Args:
        df (pd.DataFrame): The main DataFrame with a DatetimeIndex.
        symbols (list): A list of all ticker symbols to process.
        freq (str): The timeframe to resample to (e.g., 'W' for weekly, 'D' for daily).

    Returns:
        pd.DataFrame: A DataFrame containing all the new pivot point columns.
    """
    print(f"Vectorized Pivot Point Calculation for {len(symbols)} symbols...")

    # --- Step 1: Resample all symbols' OHLC data in one operation ---
    print(f"  - Resampling all symbols to {freq} timeframe...")
    agg_rules = {}
    for s in symbols:
        agg_rules[f'{s}_open'] = 'first'
        agg_rules[f'{s}_high'] = 'max'
        agg_rules[f'{s}_low'] = 'min'
        agg_rules[f'{s}_close'] = 'last'
    
    # Use standard pandas resample, which is highly optimized
    resampled_df = df.resample(freq ,origin=origin).agg(agg_rules).dropna(how='all')

    # --- Step 2: Calculate all pivot points using vectorized DataFrame math ---
    print("  - Calculating all pivot points...")
    
    # Get the previous period's data for all symbols at once
    prev_close = resampled_df.filter(like='_close').shift(1)
    prev_high = resampled_df.filter(like='_high').shift(1)
    prev_low = resampled_df.filter(like='_low').shift(1)
    
    # Rename columns to remove suffix for easy calculation
    prev_close.columns = symbols
    prev_high.columns = symbols
    prev_low.columns = symbols

    # Calculate all pivot levels for all symbols simultaneously
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    #r3 = pivot + 2 * (prev_high - prev_low)
    s3 = pivot - 2 * (prev_high - prev_low)
    
    # Combine all pivot levels into a single DataFrame
    # And add suffixes to the column names (e.g., 'AAPL_Pivot', 'MSFT_R1')
    all_pivots_df = pd.concat([
        pivot.add_suffix('_Pivot'), r1.add_suffix('_R1'), s1.add_suffix('_S1'),
        r2.add_suffix('_R2'), s2.add_suffix('_S2'),
        s3.add_suffix('_S3')
    ], axis=1)

    # --- Step 3: Reindex all pivot columns back to the original index ---
    print("  - Broadcasting pivot values back to the original index...")
    pivots_reindexed = all_pivots_df.reindex(df.index, method='ffill')
    
    print("✅ Vectorized calculation complete.")
    return pivots_reindexed


import numba

def calculate_vwap_with_numba(df: pd.DataFrame, symbols: list):
    """
    Calculates a daily-resetting VWAP for multiple symbols using a Numba-JIT compiled kernel.
    This is the fastest and most memory-efficient method.
    """
    
    @numba.njit
    def _vwap_kernel(timestamps_as_int, high, low, close, volume):
        """
        Numba-compiled function to calculate session VWAP. Works on NumPy arrays.
        """
        n = len(high)
        out = np.full(n, np.nan, dtype=np.float64)
        
        if n == 0:
            return out

        cumulative_pv = 0.0
        cumulative_vol = 0.0
        
        for i in range(n):
            # Check if the integer representing the day has changed
            if i > 0 and timestamps_as_int[i] != timestamps_as_int[i-1]:
                # Reset for the new day
                cumulative_pv = 0.0
                cumulative_vol = 0.0

            typical_price = (high[i] + low[i] + close[i]) / 3.0
            pv = typical_price * volume[i]
            
            cumulative_pv += pv
            cumulative_vol += volume[i]
            
            if cumulative_vol != 0:
                out[i] = cumulative_pv / cumulative_vol
        
        return out

    vwap_results = pd.DataFrame(index=df.index)
    
    # --- THE FIX: Convert date objects to integer codes ---
    # pd.factorize() returns an array of integers and a unique index of the original values
    timestamps_date_int = pd.factorize(df.index.date)[0]
    
    for symbol in symbols:
        high_col, low_col, close_col, vol_col = (
            f'{symbol}_high', f'{symbol}_low', f'{symbol}_close', f'{symbol}_volume'
        )

        if not all(c in df.columns for c in [high_col, low_col, close_col, vol_col]):
            continue

        # Pass the integer array of dates to the Numba function
        vwap_values = _vwap_kernel(
            timestamps_date_int,
            df[high_col].values,
            df[low_col].values,
            df[close_col].values,
            df[vol_col].values
        )
        vwap_results[f'{symbol}_vwap'] = vwap_values
        
    return vwap_results

# # Calculate VWAP for all symbols
# vwap_df2 = calculate_vwap_with_numba(com_df2, symbols=final_symbols)
