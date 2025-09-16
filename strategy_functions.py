#=================================
import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import pandas_ta as ta
from faster_numba_strategy_functions import *

import os
from datetime import datetime, timedelta

import zipfile
from io import BytesIO
import time


def setup_logger(logger_name, base_directory, log_file="app.log", to_console=False):
    
    import logging
    from logging.handlers import RotatingFileHandler
    
    """
    
    Set up a logger with a specified name and directory.

    Args:
        logger_name (str): Name of the logger.
        base_directory (str): Base directory where the 'logs' folder will be created.
        log_file (str): Name of the log file. Defaults to 'app.log'.
        to_console (bool): Whether to log messages to the console. Defaults to False.
        
    """
    
    # Create the logs directory inside the specified base directory
    
    logs_directory = os.path.join(base_directory, "logs")
    os.makedirs(logs_directory, exist_ok=True)

    # Full path to the log file
    log_file_path = os.path.join(logs_directory, log_file)

    # Configure the logger
    logger = logging.getLogger(logger_name)

    # Ensure handlers are not duplicated
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Define log format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Rotate logs when they reach 25MB, keep last 3 backups
        file_handler = RotatingFileHandler(log_file_path, maxBytes=25*1024*1024, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)  # Always log to file

        # Conditionally add console handler
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)  # Log to console only if requested

    return logger



def close_logger(logger):
    """
    Properly closes all handlers of a logger to release resources.

    Parameters:
    - logger (logging.Logger): The logger instance to close.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)






def get_public_ip():
    
    """
    
    Fetches the public IP address of the machine using the ipify service.

    Returns:
    
    - str: The public IP address or an error message if the fetch fails.
    
    """
    
    response = requests.get("https://api.ipify.org?format=json")
    if response.status_code == 200:
        return response.json()["ip"]
    else:
        return "Could not fetch public IP"



def setup_custom_logger(name, log_file, is_production=False, stg_folder_name=None):
    from pathlib import Path

    from logging.handlers import RotatingFileHandler
    """
    Set up a custom logger with a specific name and log file, optionally creating a subfolder in logs.

    Parameters:
    - name (str): The name of the logger.
    - log_file (str): The file to write the logs to (just the file name, not the full path).
    - is_production (bool): Flag to indicate if the environment is production.
      -- If true, won't generate console logs.
    - stg_folder_name (str, optional): Subfolder name to dynamically create under the logs directory.
      -- If None, the log file will be created directly in the "logs" directory.

    Returns:
    - logging.Logger: The configured logger.
    """
    # Determine log file path
    base_log_dir = Path("logs")
    if stg_folder_name:
        log_path = base_log_dir / stg_folder_name  # Append the subfolder
    else:
        log_path = base_log_dir

    # Ensure log file directory exists
    log_path.mkdir(parents=True, exist_ok=True)

    # Full log file path
    log_file_path = log_path / log_file  # Combine directory and file name

    # Create or get an existing logger
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # Clear handlers to avoid duplication
        logger.handlers.clear()

    # Set logger level
    logger.setLevel(logging.DEBUG)

    # File handler with rotating log files
    file_handler = RotatingFileHandler(log_file_path, maxBytes=100**6, backupCount=5)  # 1 MB per file, keep 5 backups
    file_handler.setLevel(logging.DEBUG)

    # Console handler (optional, for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    if not is_production:  # Only add console handler if not in production
        logger.addHandler(console_handler)

    return logger



def sync_windows_clock():
    os.system("w32tm /resync")
    print("✅ Windows clock synced using NTP!")


def analyze_trading_performance(trade_log, position_cap=None, max_positions=None, total_cost_bps=30):
    """
    Analyzes trading performance metrics from a trade log DataFrame.
    
    Parameters:
    -----------
    trade_log : pd.DataFrame
        Must contain these columns:
        - 'entry_date', 'exit_date' (datetime)
        - 'Side' (string: 'buy' or 'sell')
        - 'entry_price', 'exit_price', 'profit' (numeric)
        - Optional: 'symbol' for multi-asset analysis
        
    position_cap : float, optional
        Capital allocated per position for equity curve calculation
        
    max_positions : int, optional
        Maximum concurrent positions for equity curve calculation
        
    total_cost_bps : float
        Total round-trip trading costs in basis points (default: 30bps)
    
    Returns:
    --------
    dict
        Dictionary containing all performance metrics
    pd.DataFrame
        Enhanced trade log with additional calculated columns
    """
    
    # Input validation
    required_columns = { 'exit_date', 'Side', 'entry_price', 'exit_price', 'profit'}
    if not required_columns.issubset(trade_log.columns):
        missing = required_columns - set(trade_log.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Create a copy to avoid modifying original
    local_trade_log2 = trade_log.copy()
    
    # Sort by profit and set index
   
    
    if 'entry_date' in local_trade_log2.columns:
        local_trade_log2.set_index('entry_date', inplace=True)
        local_trade_log2.sort_index(ascending=True, inplace=True)
    
    # Calculate profit in basis points (adjusted for costs)
    local_trade_log2['profit_bps'] = np.where(
        local_trade_log2['Side'].str.lower() == 'buy',
        ((local_trade_log2['exit_price'] - local_trade_log2['entry_price']) / local_trade_log2['entry_price'] * 10000) ,
        ((local_trade_log2['entry_price'] - local_trade_log2['exit_price']) / local_trade_log2['entry_price'] * 10000)  )
    
    # Basic trade statistics
    total_trade_count = len(local_trade_log2)
    num_long_trades = len(local_trade_log2[local_trade_log2['Side'].str.lower() == 'buy'])
    num_short_trades = len(local_trade_log2[local_trade_log2['Side'].str.lower() == 'sell'])
    
    # Profit metrics
    average_trade_profit = local_trade_log2['profit'].mean()
    avg_profit_bps = local_trade_log2['profit_bps'].mean()
    
    # Winning/losing trades
    winners = local_trade_log2[local_trade_log2['profit'] > 0]
    losers = local_trade_log2[local_trade_log2['profit'] <= 0]
    num_losing_trades = len(losers)
    losers_percentage = (num_losing_trades / total_trade_count) * 100 if total_trade_count > 0 else 0
    
    # Long/short performance
    avg_long_profit_bps = local_trade_log2[local_trade_log2['Side'].str.lower() == 'buy']['profit_bps'].mean()
    avg_short_profit_bps = local_trade_log2[local_trade_log2['Side'].str.lower() == 'sell']['profit_bps'].mean()
    profit_total_long = local_trade_log2[local_trade_log2['Side'].str.lower() == 'buy']['profit'].sum()
    profit_total_short = local_trade_log2[local_trade_log2['Side'].str.lower() == 'sell']['profit'].sum()
    
    # Holding period analysis
    local_trade_log2['holding_days'] = (local_trade_log2['exit_date'] - local_trade_log2.index).dt.days
    avg_holding_days = local_trade_log2['holding_days'].mean()
    
    # Extreme losses
    loss_threshold_bps = -1000  # -10%
    deep_losers = local_trade_log2[local_trade_log2['profit_bps'] < loss_threshold_bps]
    num_deep_losers = len(deep_losers)
    
    # Equity curve calculation
    if position_cap and max_positions:
        local_trade_log2['EC'] = (position_cap * max_positions) + local_trade_log2['profit'].cumsum()
    
    # Trade frequency
    max_entries_per_day = local_trade_log2.groupby(local_trade_log2.index.date).size().max()
    
    # Prepare results dictionary
    results = {
        'total_trades': total_trade_count,
        'long_trades': num_long_trades,
        'short_trades': num_short_trades,
        'avg_trade_profit': average_trade_profit,
        'avg_profit_bps': avg_profit_bps,
        'avg_long_profit_bps': avg_long_profit_bps,
        'avg_short_profit_bps': avg_short_profit_bps,
        'total_long_profit': profit_total_long,
        'total_short_profit': profit_total_short,
        'winning_trades': len(winners),
        'losing_trades': num_losing_trades,
        'losing_percentage': losers_percentage,
        'avg_winning_trade': winners['profit'].mean(),
        'avg_losing_trade': losers['profit'].mean(),
        'deep_losers': num_deep_losers,
        'avg_holding_days': avg_holding_days,
        'max_daily_trades': max_entries_per_day,
        'cost_adjusted_bps': total_cost_bps
    }
    
    # Add position statistics if available
    if hasattr(local_trade_log2, 'position_count'):
        open_positions = list(local_trade_log2.position_count.values())
        results.update({
            'max_long_positions': np.max(open_positions),
            'max_short_positions': np.max(open_short_positions)
        })
    
    return results, local_trade_log2


def print_performance_report(results):
    """Prints a formatted performance report"""
    print("\n=== TRADING PERFORMANCE REPORT ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Long/Short Trades Ratio: {results['long_trades']}/{results['short_trades']}")
    print(f"\nProfitability:")
    print(f"Average Trade Profit: ${results['avg_trade_profit']:.2f}")
    print(f"Average Profit (bps): {results['avg_profit_bps']:.1f}")
    print(f"Long/Short Avg Profit (bps): {results['avg_long_profit_bps']:.1f}/{results['avg_short_profit_bps']:.1f}")
    print(f"Total Long/Short Profit: ${results['total_long_profit']:.2f}/${results['total_short_profit']:.2f}")
    
    print(f"\nWin/Loss Analysis:")
    print(f"\n ========================================================\n:")
    print(f"Winning Trades: {results['winning_trades']} ({100-results['losing_percentage']:.1f}%)")
    print(f"Losing Trades: {results['losing_trades']} ({results['losing_percentage']:.1f}%)")
    print(f"Average Win/Loss: ${results['avg_winning_trade']:.2f}/${results['avg_losing_trade']:.2f}")
    print(f"Deep Losses (>10%): {results['deep_losers']}")
    
    print(f"\nTrade Characteristics:")
    print(f"Average Holding Days: {results['avg_holding_days']:.1f}")
    print(f"Max Daily Trades: {results['max_daily_trades']}")
    print(f"Cost Adjusted (bps): {results['cost_adjusted_bps']}")
    
    if 'max_long_positions' in results:
        print(f"\nPosition Management:")
        print(f"Max Long/Short Positions: {results['max_long_positions']}/{results['max_short_positions']}")


#Example usage:
# results, enhanced_log = analyze_trading_performance(local_trade_log2, position_cap=10000, max_positions=20)
# print_performance_report(results)
# enhanced_log['EC'].plot(title="Equity Curve")





def find_common_symbols(exchanges, exchange_names, allowed_quotes=['USDC', 'USDT', 'USD']):
    """
    Find common symbols across multiple exchanges and return symbol_id columns for each.

    Note: Only works on the find_liquid_symbols_of_exchange function we use to find liquidity of each exchange from COINAPI

    Parameters:
    - exchanges: List of DataFrames with columns ['symbol_id', 'asset_id_base', 'asset_id_quote']
    - exchange_names: List of exchange names corresponding to the DataFrames
    - allowed_quotes: List of allowed quote currencies

    Returns:
    - pd.DataFrame with one column per exchange: symbol_id_{exchange}
    """

    if not exchanges or len(exchanges) != len(exchange_names):
        raise ValueError("Mismatch in exchanges and exchange_names input.")

    processed = []

    for df, name in zip(exchanges, exchange_names):
        if 'symbol_id' not in df.columns:
            print(f"[ERROR] 'symbol_id' column not found in DataFrame for exchange: {name}")
            return None
        if 'asset_id_base' not in df.columns or 'asset_id_quote' not in df.columns:
            print(f"[ERROR] 'asset_id_base' or 'asset_id_quote' column missing in exchange: {name}")
            return None

        filtered = df[df['asset_id_quote'].isin(allowed_quotes)]
        if filtered.empty:
            print(f"[WARNING] No symbols found with allowed quote currencies in exchange: {name}")
        
        renamed = filtered[['symbol_id', 'asset_id_base', 'asset_id_quote']].rename(columns={
            'symbol_id': f'symbol_id_{name}',
            'asset_id_base': 'asset_id_base',
            'asset_id_quote': 'asset_id_quote'
        })
        processed.append(renamed)

    if not processed:
        print("[ERROR] No valid exchange DataFrames to process.")
        return None

    merged = processed[0]
    for other in processed[1:]:
        merged = pd.merge(merged, other, on=['asset_id_base'], how='inner')

    if merged.empty:
        print("[WARNING] No common symbols found across the given exchanges.")
        return None

    symbol_columns = [f'symbol_id_{name}' for name in exchange_names]
    return merged[symbol_columns].reset_index(drop=True)




def fetch_futures_daily_metrics(symbol, days, output_folder):
    """
    Fetches Binance futures daily metrics data for a given symbol.
    
    Parameters:
        symbol (str): The trading pair symbol (e.g., BTCUSDT).
        days (int): Number of days to fetch data for (default 180).
        output_folder (str): The directory where the extracted CSV files will be saved.
    
    Returns:
        list: Paths to the downloaded files.
    """
    try:
        # Create output folder based on symbol
        symbol_folder = os.path.join(output_folder, f"{symbol}_metrics")
        os.makedirs(symbol_folder, exist_ok=True)
        
        current_date = datetime.now().date()
        #current_date = datetime(2022 , 7 , 18).date()
        end_date = current_date - timedelta(days=1)  # Data is typically available with 1-day delay
        start_date = end_date - timedelta(days=days - 1)
        
        downloaded_files = []
        missed_files = []
        
        #print(f"Fetching daily metrics for {symbol} from {start_date} to {end_date}")
        
        while start_date <= end_date:
            current_date_str = start_date.strftime('%Y-%m-%d')
            url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{current_date_str}.zip"
            
            # Check if the file exists
            response = requests.get(url)
            if response.status_code == 200:
                # Extract the ZIP file content
                with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
                    for file_name in zip_ref.namelist():
                        if file_name.endswith('.csv'):
                            # Extract CSV file and save it
                            csv_file_path = os.path.join(symbol_folder, f"{symbol}_metrics_{current_date_str}.csv")
                            with zip_ref.open(file_name) as file:
                                with open(csv_file_path, 'wb') as output_file:
                                    output_file.write(file.read())
                            print(f"Saved {csv_file_path}")
                            downloaded_files.append(csv_file_path)
            else:
                #print(f"Metrics data for {current_date_str} not available, skipping...")
                missed_files.append(current_date_str)
            
            # Move to the next day
            start_date += timedelta(days=1)
            time.sleep(0.5)  # Be polite with their servers
            
        print(f"Downloaded and extracted {len(downloaded_files)} metrics files.")
        if missed_files:
            print(f"Missed {len(missed_files)} dates: {missed_files}")
        
        return downloaded_files
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# #Example usage:
# symbol = "LTCUSDT"
# days = 1200
# output_folder = r"C:\Users\aakas\Desktop\Crypto_Data\Perpetuals_Metrics_Daily_5_min\Binance"
# downloaded_files = fetch_futures_daily_metrics(symbol, days, output_folder)




  
def US_stock_monthly_expiry(df , expiry=0):
    
    from datetime import date, timedelta
    import pandas_market_calendars as mcal
    
    """
    Finds the nearest monthly expiry for all dates between start date and end_date 
    
    
    parameters : df with datetime index
    expiry = 0 is the current month and 1 , next etc etc
    
    
    """
    start_date = df.index.date.min()
    end_date = df.index.date.max()+ timedelta(days=360)
    
    us_cal = mcal.get_calendar('CBOE_Equity_Options')
    
    valid_US_days = us_cal.valid_days(start_date=start_date , end_date=end_date).date
    
    
     # Find all Thursdays
    all_days = pd.date_range(start_date, end_date, freq='B')
    fridays = all_days[all_days.to_series().dt.weekday == 4] 
    #print("all fridays are: \n " ,fridays )
    
    
    # Get the last Thursday of each month
    third_friday = fridays.to_series().groupby([fridays.year, fridays.month]).nth(2)  # nth is 0-based, so 2 means the third item
    last_fridays = fridays.to_series().groupby([fridays.year, fridays.month]).last()
    
    
    
    # If the last Thursday is not a valid day, find the previous valid day
    #last_thursdays = last_thursdays.apply(lambda d: valid_nse_days[valid_nse_days <= pd.to_datetime(d).date()].max() if pd.to_datetime(d).date() not in valid_nse_days else d)
    
    third_fridays = third_friday.apply(lambda d: valid_US_days[valid_US_days <= pd.to_datetime(d).date()].max() if pd.to_datetime(d).date() not in valid_US_days else d)
    

        # Remove the time part from 'last_thursdays'
    # Convert back to Timestamp and remove the time part from 'last_thursdays'
    #last_thursdays_date = pd.to_datetime(last_thursdays).dt.date
    
    third_friday_date = pd.to_datetime(third_fridays).dt.date
    

    # Remove the time part from the DataFrame's index
    df_date = df.index.date

    # Find the nearest expiry date for each date in the DataFrame
    #df['Expiry_Date'] = [last_thursdays_date[last_thursdays_date >= date].min() for date in df_date]
    
    
    
    # Find the nearest expiry date for each date in the DataFrame
    if expiry>=0: 
        df['Month_Expiry_Date'] = [sorted(third_friday_date[third_friday_date >= date])[expiry] if len(third_friday_date[third_friday_date >= date]) > expiry else np.nan for date in df_date]
    elif expiry<0:
        
        df['Month_Expiry_Date'] = [sorted(third_friday_date[third_friday_date < date])[expiry]  if len(third_friday_date[third_friday_date < date]) > abs(expiry) - 1 else np.nan for date in df_date]
    
        
    return df


def get_data_timeframe(df):
    
    
    if isinstance(df.index, pd.DatetimeIndex):
        
        df.sort_index(inplace=True)
        
        time_diffs = np.diff(df.index)
        min_time_diff = min(diff for diff in time_diffs if diff != np.timedelta64(0, 'ns'))
        time_frame = min_time_diff.astype('timedelta64[m]').astype(int)
        
        
        return time_frame
    
    else:
        print("\n Index is not DatetimeIndex ... convert index to datetime for further processing ")
    







def fetch_VIX(vix_file_path):
    
    df = pd.read_csv(vix_file_path)
    
    if 'Date' in df.columns:
        df.Date = pd.to_datetime(df['Date'] , format="%Y-%m-%d")
        df.set_index('Date' , inplace=True)
    return df


def z_score_price(df, column, window_size=60):
    
    """
    This function calculates the rolling Z-Score for the specified column in the given DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column for which to calculate the rolling Z-Score.
    window_size (int): Size of the rolling window.

    Returns:
    pandas.Series: A Series representing the rolling Z-Score of the specified column.
    """
    
     # Calculate the rolling mean and standard deviation
    rolling_mean = df[column].rolling(window=window_size , min_periods=3).mean()
    rolling_std = df[column].rolling(window=window_size, min_periods=3).std()

    # Calculate the rolling Z-Score and handle division by zero
    rolling_z_score = (df[column] - rolling_mean) / rolling_std

    # Replace infinities with NaNs if any
    rolling_z_score.replace([np.inf, -np.inf], np.nan, inplace=True)

    return rolling_z_score
    


def supertrend( highcol='High' , lowcol= 'Low' , closecol = 'Close' , length=2 , multiplier=2.5):
    import pandas_ta as pta

    super_df  = pta.supertrend(highcol , lowcol , closecol , length=length , multiplier=multiplier)
    # Check if the DataFrame is empty or all values are NaN
    if super_df.empty or super_df.isnull().all().all():
        print("The DataFrame is either empty or contains only NaN values.")
        return None  # handle this as appropriate for your use case
    
    # If DataFrame is non-empty and not all NaN, return the first column
    return super_df.iloc[:, 0]
    
    

# pta.supertrend(rdf['High'] , rdf['Low'] , rdf['Close'] , length=14 , multiplier=2)

def ATR(highcol='High' , lowcol = 'Low' , closecol='Close', length=21 , mamode='RMA'  , talib=True  , percent=False):
    
    import pandas_ta as pta
    
    atrv = pta.atr(highcol , lowcol , closecol , length=length , mamode=mamode  , talib=talib  , percent=percent )
    
    return atrv


def daily_ATR_zcore(df , length=14  , highcol='high' , lowcol = 'low' , closecol='close'  ,  mamode='RMA'  , talib=True  , percent=False):

    import pandas_ta as pta

    # Assuming 'data' is your DataFrame with daily data
    # Replace 'YourHighColumn', 'YourLowColumn', and 'YourCloseColumn' with actual column names from your data
    daily_data = df.resample('D').agg({
        highcol: 'max',
        lowcol: 'min',
        closecol: 'last'
    })

    daily_data.dropna(how='all' , inplace=True)
    #print( "\n resapleed data is \n " , daily_data)
    # Calculate daily ATR for the last 3 days
    last_3_days_atr = pta.atr(daily_data[highcol], daily_data[lowcol], daily_data[closecol], length=length ,  mamode=mamode  , talib=talib  , percent=percent)
    #print(last_3_days_atr)
    
    rolling_mean = last_3_days_atr.rolling(window=40 , min_periods=3).mean()
    rolling_std = last_3_days_atr.rolling(window=40, min_periods=3).std()

    # Calculate the rolling Z-Score and handle division by zero
    rolling_z_score = (last_3_days_atr - rolling_mean) / rolling_std

    # Replace infinities with NaNs if any
    rolling_z_score.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    
    last_3_days_atr = last_3_days_atr.reindex(df.index, method='ffill')
    rolling_z_score = rolling_z_score.reindex(df.index, method='ffill')
        
    
    print( "\n \n Z_score", rolling_z_score)
    
    # Print or use the ATR values for the last 3 days
    #last_3_days_atr.plot()
    return rolling_z_score
        




def add_time_to_column_or_index(df, column=None, specific_time="09:15"):
    
    """
    Adds a specific time to the date in a DataFrame column or index if the time component is missing.

    :param - df: DataFrame to which the operation will be applied.
    :param - column: The column name to which the operation will be applied. If None, applies to the index.
    :param - specific_time: The specific time to add if the time component is missing. Default is "09:15".
    :return: DataFrame with updated times.
    """

    def add_specific_time(timestamp):
        if timestamp.time() == pd.Timestamp("00:00").time():
            return pd.Timestamp(f"{timestamp.date()} {specific_time}")
        else:
            return timestamp

    if column:
        # Apply to a specific column
        corrected_ts = pd.to_datetime(df[column]).map(add_specific_time)
    else:
        # Apply to the index
        corrected_ts = pd.to_datetime(df.index).map(add_specific_time)

    return corrected_ts




def intraday_high(df, high_col, agg_func):
    if agg_func not in ['max']:
        raise ValueError("agg_func must be - 'max'")

    #intraday_high_col = f'intraday_{high_col}_{agg_func}'
    
    if agg_func == 'max':
       int_high =  df.groupby(df.index.date)[high_col].transform(lambda x: x.expanding().max())
    else:
       int_high = None
    return int_high



def intraday_low(df, low_col, agg_func):
    if agg_func not in ['min']:
        raise ValueError("agg_func must be - 'min'")

    intraday_low_col = f'intraday_{low_col}_{agg_func}'
    
    if agg_func == 'min':
        int_low = df.groupby(df.index.date)[low_col].transform(lambda x: x.expanding().min())
    else:
        int_low = None
    return int_low


#== calculates ROC from daily prices
def daily_ROC(df , column='Close' , ROC_period=3  , highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open'  , timeframe='1D' , volcol='Volume'):
    
    resamp = resample_data_crypto_fast_numba(df=df , timeframe=timeframe, highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volumecol=volcol ).shift(1)
    
    print(resamp)
        
    resamp["ROC"] = (resamp[column] - resamp[column].shift(ROC_period)) / (resamp[column].shift(ROC_period)) 
    #resamp["ROC_z_score"] = z_score_price(df=resamp , column='ROC' , window_size=z_score_period)
    
    resamp = resamp.reindex(df.index , method='ffill')
        
    return resamp['ROC']
    

    
def x_min_close(df , column , start='09:15' , end='10:15' ):
    """
    Get the previous x min value of column value of your choice for each row in the DataFrame.
    example you want first hour close:
        x_min_close(df , 'Close' , start='09:15' , end='10:15'):
    
    """
    
    if column in df.columns:
    
        xm_close = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end)).ffill()
        return xm_close


def x_min_low(df , column , start='9:15' , end='10:15'):
    
    """
    Get the previous x min lowest low for value of your choice for each row in the DataFrame.
    example you want first hour low:
        x_min_low(df , 'Low' , start='9:15' , end='10:15'):
    
    """
    
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        print("Error: DataFrame index is not a datetime index.")
        return None
    
    
    
    if column in df.columns:
    
        xm_low = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end).expanding().min()).ffill()
        return xm_low

def x_min_high(df , column , start='09:15' , end='10:15' ):
    """
    Get the previous x min high high for value of your choice for each row in the DataFrame.
    example you want first hour highest high:
        x_min_low(df , 'High' , start='09:15' , end='10:15'):
    
    """
    
    if column in df.columns:
    
        xm_high = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end).expanding().max()).ffill()
        return xm_high



def get_x_day_low(df, n , column='Low'):
    """
    Get the previous low of the last n known dates for each row in the DataFrame.

    n>=1
    """
    df = df.copy()
    df['Date'] = df.index.normalize()
    unique_dates = df['Date'].unique()

    # Create a dictionary to store the x_day_low for each date
    x_day_low_dict = {}

    for i, current_date in enumerate(unique_dates):
        # Find the start date index such that the difference in known dates is at least n days
        start_date_index = i - n
        if start_date_index < 0:
            start_date_index = 0

        start_date = unique_dates[start_date_index]
        mask = (df['Date'] < current_date) & (df['Date'] >= start_date)
        x_day_low_dict[current_date] = df.loc[mask, column].min()

    # Apply the x_day_low values from the dictionary to the DataFrame
    x_day_low = df['Date'].map(x_day_low_dict)

    # Remove the temporary Date column
    #df.drop(columns='Date', inplace=True)

    return x_day_low




def convert_datetime(df, column_name , is_index=0):
    
    # convert index to datetime
    formats = [ '%Y%m%d %H%M%S', '%Y-%d-%m %H:%M:%S', '%d-%m-%Y %H:%M:%S' ,  '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S' , '%m-%d-%Y %H:%M:%S' , '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%m%d%Y %H:%M:%S', '%Y/%m/%d' , "%Y-%d-%m" ,"%Y-%m-%d",  '%Y-%m-%dT%H:%M:%S.%f']
    for fmt in formats:
        try:
            if is_index ==0:
                df[column_name] = pd.to_datetime(df[column_name], format=fmt, errors='raise')
                return df
            if is_index==1:
                
                df.index = pd.to_datetime(df.index ,format=fmt, errors='raise' )
                return df
                       
             
            break
                    
        except ValueError:
            pass
        

 
        
def x_min_cum_vol(df ,column , start='09:15' , end='10:15' ):
    
    
    
    if column in df.columns:
    
        xm_cvol = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end).expanding().sum()).ffill()
        return xm_cvol
 



def get_x_day_high(df, n , column='high'):
    """
    Get the previous low of the last n known dates for each row in the DataFrame.

    n>=1
    """
    df = df.copy()
    df['Date'] = df.index.normalize()
    unique_dates = df['Date'].unique()

    # Create a dictionary to store the x_day_low for each date
    x_day_low_dict = {}

    for i, current_date in enumerate(unique_dates):
        # Find the start date index such that the difference in known dates is at least n days
        start_date_index = i - n
        if start_date_index < 0:
            start_date_index = 0

        start_date = unique_dates[start_date_index]
        mask = (df['Date'] < current_date) & (df['Date'] >= start_date)
        x_day_low_dict[current_date] = df.loc[mask, column].max()

    # Apply the x_day_low values from the dictionary to the DataFrame
    x_day_high = df['Date'].map(x_day_low_dict)

    # Remove the temporary Date column
    #df.drop(columns='Date', inplace=True)

    return x_day_high




def get_x_day_close_vectorized(df, n, column='Close'):
    """
    Efficiently get the closing price of the last n calendar days for each row in the DataFrame
    using vectorized operations for improved performance.

    Parameters:
    - df: DataFrame with a DateTimeIndex.
    - n: Integer, the number of calendar days to look back for the last known close price.
    - column: String, the name of the column from which to get the close price.
    
    Returns:
    - A Series containing the last known close price from the last n calendar days for each row in the DataFrame.
    """
    # Normalize the DateTimeIndex to remove time and keep only dates
    
    normalized_dates = df.index.normalize()
    
    # Find the unique dates to establish the range of trading days
    unique_dates = pd.Series(normalized_dates.unique()).sort_values()
    
    # Map each date in the DataFrame to its 'n days ago' equivalent
    n_days_ago_mapping = {date: unique_dates[unique_dates.searchsorted(date, side='left') - n] 
                          for date in unique_dates[n:]}
    
    # Create a mapping from each date to the last close of its 'n days ago' date
    #last_close_mapping = df[column].groupby(normalized_dates).last().reindex(unique_dates).fillna(method='ffill')
    
     # Create a mapping from each date to the last close of its 'n days ago' date
    last_close_mapping = (
        df[column]
        .groupby(normalized_dates)
        .last()
        .reindex(unique_dates)
        .ffill()  # Replaced fillna(method='ffill') with ffill()
    )
    
    last_close_mapping = {date: last_close_mapping[n_days_ago] for date, n_days_ago in n_days_ago_mapping.items()}
    
    # Apply the mapping to get the 'n days ago' close price for each date in the DataFrame
    x_day_close = normalized_dates.map(last_close_mapping)
    
    return x_day_close




def get_x_day_range(df  , n , high_col = 'High' , low_col='Low' ):
    
    
    xdh = get_x_day_high(df , n=n , column=high_col)
    xdl = get_x_day_low(df , n=n , column=low_col)
    
    x_day_range = xdh/xdl-1
    
    return x_day_range
    


def get_data_timeframe(df):
    
    
    if isinstance(df.index, pd.DatetimeIndex):
        
        df.sort_index(inplace=True)
        
        time_diffs = np.diff(df.index)
        min_time_diff = min(diff for diff in time_diffs if diff != np.timedelta64(0, 'ns'))
        #time_frame = min_time_diff.astype('timedelta64[m]').astype(int)
        
        time_frame = min_time_diff / np.timedelta64(1, 'm')
        
        return time_frame
    
    else:
        print("\n Index is not DatetimeIndex ... convert index to datetime for further processing ")


        

def resample_data_crypto(df  , origin_for_sample = 'start_day' , highcol = 'High' , lowcol = 'Low' , closecol = 'Close' ,opencol = 'Open'  , volcol = 'Volume' , timeframe = '30min'):
    
    """
    Gets the resampled upsample time-series for a OHLCV dataframe
    
    
    """
    orig_tf = get_data_timeframe(df)
    
    # Parse timeframe string to check if it's in minutes
    if 'T' in timeframe or 'min' in timeframe:
        # Extract numeric value from timeframe string
        timeframe_numeric = int(''.join(filter(str.isdigit, timeframe)))
        
        # Perform comparison if orig_tf is not None
        if orig_tf is not None and timeframe_numeric <= orig_tf:
            print(f"Invalid timeframe resample request {timeframe} for original timeframe {orig_tf} minutes.")
            return
    elif orig_tf is None:
        print("\n Original timeframe could not be determined.")
        return
        
    
    rdf = pd.DataFrame()

    rdf['High'] = df[highcol].resample(timeframe , origin=origin_for_sample).max().dropna()
    rdf['Low'] = df[lowcol].resample(timeframe , origin=origin_for_sample ).min().dropna()
    rdf['Close'] = df[closecol].resample(timeframe, origin=origin_for_sample).last().dropna()
    rdf['Open'] = df[opencol].resample(timeframe , origin=origin_for_sample).first().dropna()
    if volcol in df.columns:
        rdf['Volume'] = df[volcol].resample(timeframe , origin=origin_for_sample).sum().dropna()
    
    
    # Automatically adjust timestamps based on timeframe
    if timeframe.startswith('W'):
        rdf.index = rdf.index - pd.offsets.Week(weekday=0)  # Moves to Monday
    elif timeframe.startswith('M'):
        rdf.index = rdf.index - pd.offsets.MonthBegin()  # Moves to first day of the month

    #rdf = rdf.dropna()  # Remove NaN rows that result from resampling
    
    
    return rdf   




def atr_mtf(com_df2, highcol, lowcol, closecol, opencol, volcol, resample_tf='1D', atr_period=3):
    """
    Computes ATR (3) on a resampled timeframe and maps it back to the original intraday data.

    Parameters:
    - com_df2 (DataFrame): The main OHLCV dataset.
    - symbol (str): The crypto symbol (e.g., 'BTCUSDT').
    - highcol (str): Column name for High prices.
    - lowcol (str): Column name for Low prices.
    - closecol (str): Column name for Close prices.
    - opencol (str): Column name for Open prices.
    - volcol (str): Column name for Volume.
    - resample_tf (str): The timeframe to resample to (e.g., '1D', '4H', '12H').
    - atr_period (int): The ATR lookback period (default=3).

    Returns:
    - Updated com_df2 with a new column '{symbol}_ATR_resampled'
    """
    
    # Resample the data dynamically
    resampled_df = resample_data_crypto_fast_numba_origin(
        df=com_df2, 
        timeframe=resample_tf,
        highcol=highcol, 
        lowcol=lowcol,
        closecol=closecol,
        opencol=opencol,
        volumecol=volcol,
        origin='9:30'
         )
    
    
    
    #print(resampled_df.tail(5))
    
    
    resampled_df = resampled_df.shift(1)
    
    
    # Compute ATR on the resampled data
    resampled_df['ATR'] = ta.atr(
        resampled_df["High"], resampled_df["Low"], resampled_df["Close"], length=atr_period)

    
     # Reindex to align with the original timestamps, using forward fill to carry forward last known ATR
    atr_series = resampled_df["ATR"].reindex(com_df2.index, method="ffill")

    
    return atr_series




    
def rsi_timeframe(df , column='Close' , period=14,  highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open' , volcol='Volume' , timeframe='1D'):
    
    """
    Compute RSI on a higher timeframe OHLCV resampled version of the input dataframe.

    Parameters:
    - df (DataFrame): The original intraday OHLCV DataFrame with a DateTimeIndex.
    - column (str): The column name to compute RSI on (usually 'Close').
    - period (int): RSI lookback period.
    - highcol (str): Name of the high column for resampling.
    - lowcol (str): Name of the low column for resampling.
    - closecol (str): Name of the close column for resampling.
    - opencol (str): Name of the open column for resampling.
    - volcol (str): Name of the volume column for resampling.
    - timeframe (str): Target timeframe for resampling (e.g., '1D', '4H').

    Returns:
    - Series: A Series of RSI values, forward-filled and aligned to the original df's index.
    """
    
    
    
    import pandas_ta as pta
    
    resamp = resample_data_crypto_fast_numba(df=df , timeframe = timeframe , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volumecol=volcol )
    
    rsi_htf = pta.rsi(resamp[column].shift(1) , length=period) 
    
    rsi_htf = rsi_htf.reindex(df.index, method='ffill')
    
    
    return rsi_htf


def rsi_custom_variable(df, value_col, period=14, timeframe='1D'):
    
    """
    Computes RSI on a single variable after converting it to a specific timeframe.
    
    Parameters:
    - df (DataFrame): The input DataFrame with a DateTimeIndex.
    - value_col (str): Column name of the custom variable to compute RSI on.
    - period (int): RSI lookback period.
    - timeframe (str): Target timeframe to resample the variable to (e.g., '1D', '4H').

    Returns:
    - Series: RSI values indexed to match the original DataFrame.
    """
    import pandas_ta as pta

    # Resample the custom variable to the specified timeframe
    resampled_series = df[value_col].resample(timeframe , origin='start_day').mean().dropna()

    # Compute RSI on the resampled series
    rsi_htf = pta.rsi(resampled_series.shift(1), length=period)

    # Reindex back to the original DataFrame
    rsi_htf = rsi_htf.reindex(df.index, method='ffill')

    return rsi_htf





def daily_percentile(df, column='close' , window=10 , output_col_name='daily-percentile'):
    
    """
    Get the rolling Daily percentile of a series from intraday data 

    n>=1
    """
    
    
    if column not in df.columns:
        raise ValueError("Column not found in df")
    
    df_daily = df.resample('D' , origin = 'start_day')[column].last().dropna()

    # Define the window size
    window_size = window
    df_daily = df_daily.to_frame()
    # Calculate the rolling percentile rank
    
    df_daily[output_col_name] = df_daily[column].shift().rolling(window_size).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    df_daily = df_daily.reindex(df.index, method='ffill')
    df = pd.concat([df, df_daily[output_col_name]], axis=1)
    return df


def rsi_custom_variable(df, value_col, period=14, timeframe='1D'):
    """
    Computes RSI on a single variable after converting it to a specific timeframe.
    
    Parameters:
    - df (DataFrame): The input DataFrame with a DateTimeIndex.
    - value_col (str): Column name of the custom variable to compute RSI on.
    - period (int): RSI lookback period.
    - timeframe (str): Target timeframe to resample the variable to (e.g., '1D', '4H').

    Returns:
    - Series: RSI values indexed to match the original DataFrame.
    """
    import pandas_ta as pta

    # Resample the custom variable to the specified timeframe
    resampled_series = df[value_col].resample(timeframe , origin='start_day').mean().dropna()

    # Compute RSI on the resampled series
    rsi_htf = pta.rsi(resampled_series.shift(1), length=period)

    # Reindex back to the original DataFrame
    rsi_htf = rsi_htf.reindex(df.index, method='ffill')

    return rsi_htf





def robust_mtf_rsi(
    df_15min: pd.DataFrame,
    symbols: list,
    period: int = 14,
    timeframe: str = '45min' , origin = '09:30'
    ) -> pd.DataFrame:
    """
    A robust and fast function to calculate multi-timeframe RSI that
    avoids index type errors by using a more stable processing pattern.

    Args:
        df_15min (pd.DataFrame): DataFrame with 15-min OHLC data.
                                 Index must be a timezone-aware DatetimeIndex.
        symbols (list): A list of the stock symbols.
        period (int): The RSI calculation period.
        timeframe (str): The timeframe to resample to.

    Returns:
        pd.DataFrame: A DataFrame with the aligned, higher-timeframe RSI.
    """
    # 1. Prepare session-based close prices. This is fast and correct.
    resampled_closes = df_15min.filter(regex='_close$') \
                               .resample(timeframe, origin=origin, closed='right', label='right') \
                               .last() \
                               .dropna(how='all')

    # 2. Calculate RSI for each column and store results in a list
    rsi_results_list = []
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in resampled_closes.columns:
            # Calculate RSI for the single symbol's resampled series
            rsi_series = ta.rsi(
                resampled_closes[close_col],
                length=period,
                mamode=None  # Use Wilder's smoothing to match TradingView
            )
            # Rename the series to the desired final column name
            rsi_series.name = f'{symbol}_RSI'
            rsi_results_list.append(rsi_series)

    # 3. Combine all calculated RSI series into a single DataFrame.
    # This correctly preserves the DatetimeIndex.
    if not rsi_results_list:
        return pd.DataFrame(index=df_15min.index) # Return empty if no symbols processed
    
    rsi_df = pd.concat(rsi_results_list, axis=1)

    # 4. Align back to the original 15-minute index using forward-fill.
    # This step will now work correctly.
    final_aligned_rsi = rsi_df.reindex(df_15min.index, method='ffill')
    
    return final_aligned_rsi





def daily_patterns(df, highcol , lowcol , closecol , opencol , volcol):


    if isinstance(df.index, pd.DatetimeIndex):
        ddf = resample_data_crypto(df=df , origin_for_sample = 'start_day' , highcol='high' , lowcol='low' , closecol='close' , opencol='open' , volcol='volume' , timeframe='1D')
    else:
        print("\n Error in index of df.... Not datetime.")
    
        
    ddf = ddf.shift()


    ddf['body'] = ddf['Close'] - ddf['Open']
    ddf['range'] = ddf['High'] - ddf['Low']
    # Calculate the rolling average of the body size for the last 1 hour
    ddf['body_avg'] = abs(ddf['body']).rolling(window=9).mean()

        
    # Apply the condition to create a new column
    ddf['candle'] = np.where(ddf['body'] >= 0, 1, -1)

    
    
    # Check if the current negative candle's body is more than twice the average of the last 1 hour
    ddf['big_rise'] = (ddf['candle']>=1) & (ddf['body'].abs() > 2 * ddf['body_avg'])
    ddf['big_fall'] = (ddf['candle']<=-1) & (ddf['body'].abs() > 2 * ddf['body_avg'])

# 2. Continuation Patterns (Bigger Follow-Through)
    ddf['bullish_continuation'] = (
        (ddf['candle_dir'].shift() == 1) &  # Prev day green
        (ddf['candle_dir'] == 1) &          # Current green
        (ddf['body_pct'] > 1.5 * ddf['body_pct'].rolling(5).mean()) &  # Bigger body
        (ddf[closecol] > ddf[highcol].shift())  # Closes above prev high
    ).astype(int)
    
    

    ddf['gap_up'] = ddf['Low'] > ddf['High'].shift(1)
    ddf['gap_dn'] = ddf['High'] < ddf['Low'].shift(1)

    # Calculate the 'Reversal' pattern directly
    ddf['Reversal'] = ((ddf['Close'].shift(1) < ddf['Open'].shift(1)) &  # Previous day bearish
                    (ddf['Close'] > ddf['Open']) &  # Today bullish
                    (ddf['Close'] > ddf['Open'].shift(1)))
    
    
    ddf = ddf.reindex(df.index , method='ffill')

    return ddf



    
def pivot_points_classic(df , close_col = 'Close' , low_col = 'Low' , high_col='High'):
    Prev_close = df[close_col].groupby(df.index.date).last().shift().reindex(df.index.date).values
    Prev_High = df[high_col].groupby(df.index.date).max().shift().reindex(df.index.date).values
    Prev_low = df[low_col].groupby(df.index.date).min().shift().reindex(df.index.date).values
    
    Pivot = (Prev_High + Prev_low + Prev_close) / 3
    R1 = 2 * Pivot - Prev_low
    S1 = 2 * Pivot - Prev_High
    R2 = Pivot + (Prev_High - Prev_low)
    S2 = Pivot - (Prev_High - Prev_low)
    R3 = Pivot + 2 * (Prev_High - Prev_low)
    S3 = Pivot - 2 * (Prev_High - Prev_low)

    # Add columns to the DataFrame
    df['Pivot'] = Pivot
    df['R1'] = R1
    df['S1'] = S1
    df['R2'] = R2
    df['S2'] = S2
    df['R3'] = R3
    df['S3'] = S3

    return df


def pivot_points_each(df , close_col = 'Close' , low_col = 'Low' , high_col='High'):
    
    Prev_close = df[close_col].groupby(df.index.date).last().shift().reindex(df.index.date).values
    Prev_High = df[high_col].groupby(df.index.date).max().shift().reindex(df.index.date).values
    Prev_low = df[low_col].groupby(df.index.date).min().shift().reindex(df.index.date).values
    
    Pivot = (Prev_High + Prev_low + Prev_close) / 3
    R1 = 2 * Pivot - Prev_low
    S1 = 2 * Pivot - Prev_High
    R2 = Pivot + (Prev_High - Prev_low)
    S2 = Pivot - (Prev_High - Prev_low)
    R3 = Pivot + 2 * (Prev_High - Prev_low)
    S3 = Pivot - 2 * (Prev_High - Prev_low)

    return pd.Series(R1, index=df.index), pd.Series(S1, index=df.index), pd.Series(R2, index=df.index), pd.Series(S2, index=df.index)

    

def pivot_points_mtf(df, close_col='Close', low_col='Low', high_col='High', open_col = 'Open' , freq='W'):
    
    """
    
        Calculate multi-timeframe pivot points for financial data. for weekly use freq='W'
        
        Parameters:
        - df (DataFrame): The input DataFrame with a DateTimeIndex.
        - close_col (str): Column name for the close prices.
        - low_col (str): Column name for the low prices.
        - high_col (str): Column name for the high prices.
        - open_col (str): Column name for the open prices.
        - freq (str): Frequency for resampling the data (e.g., 'W' for weekly).
        
        Returns:
        - DataFrame: A DataFrame containing pivot points and corresponding support and resistance levels.
    
            # Example usage
        # df is your DataFrame with a DateTimeIndex
        weekly_pivots = pivot_points_mtf(df, freq='W-SUN')  # For weekly pivot points

    
        
    """
    
    
    resampled_df = resample_data_crypto(df=df , origin_for_sample='start_day' , timeframe=freq , highcol=high_col , lowcol=low_col , closecol=close_col , opencol=open_col , volcol='volume' )
    #print(resampled_df)
    
    #Calculate the pivot points using shifted values to avoid look-ahead bias
    Prev_close = resampled_df['Close'].shift(1)
    Prev_High = resampled_df['High'].shift(1)
    Prev_low = resampled_df["Low"].shift(1)
    
    
    # Calculate pivot points
    Pivot = (Prev_High + Prev_low + Prev_close) / 3
    R1 = 2 * Pivot - Prev_low
    S1 = 2 * Pivot - Prev_High
    R2 = Pivot + (Prev_High - Prev_low)
    S2 = Pivot - (Prev_High - Prev_low)
    R3 = Pivot + 2 * (Prev_High - Prev_low)
    S3 = Pivot - 2 * (Prev_High - Prev_low)
    
    # Create a DataFrame for pivot points
    pivot_df = pd.DataFrame({
        'Pivot': Pivot,
        'R1': R1,
        'S1': S1,
        'R2': R2,
        'S2': S2,
        'R3': R3,
        'S3': S3
    })

    # Reindex to match the original DataFrame index and backfill missing values
    pivot_df = pivot_df.reindex(df.index, method='bfill')

    return pivot_df


def pivot_points_mtf_individual(df, close_col='Close', low_col='Low', high_col='High', open_col='Open', freq='W'):
    """
    Calculate multi-timeframe pivot points and return individual Series (Pivot, R1, S1, etc.).
    """

    resampled_df = resample_data_crypto(df=df , origin_for_sample='start_day' , timeframe=freq , highcol=high_col , lowcol=low_col , closecol=close_col , opencol=open_col )

    # Debugging: Print resampled columns
    #print("Resampled Columns:", resampled_df.columns)

    # Calculate pivot points using shifted values to avoid look-ahead bias
    Prev_close = resampled_df['Close'].shift(1)
    Prev_High = resampled_df['High'].shift(1)
    Prev_low = resampled_df['Low'].shift(1)

    # Calculate pivot points
    Pivot = (Prev_High + Prev_low + Prev_close) / 3
    R1 = 2 * Pivot - Prev_low
    S1 = 2 * Pivot - Prev_High
    R2 = Pivot + (Prev_High - Prev_low)
    S2 = Pivot - (Prev_High - Prev_low)
    R3 = Pivot + 2 * (Prev_High - Prev_low)
    S3 = Pivot - 2 * (Prev_High - Prev_low)

    # Reindex to match the original DataFrame index and backfill missing values
    Pivot = Pivot.reindex(df.index, method='ffill')
    R1 = R1.reindex(df.index, method='ffill')
    S1 = S1.reindex(df.index, method='ffill')
    R2 = R2.reindex(df.index, method='ffill')
    S2 = S2.reindex(df.index, method='ffill')
    #R3 = R3.reindex(df.index, method='bfill')
    S3 = S3.reindex(df.index, method='ffill')

    return Pivot, R1, S1, R2, S2, S3
    

import pandas as pd
import numpy as np

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






def bars_since_condition(df, column):
    # Calculate the cumulative sum which resets at each '1' to create group identifiers
    group_ids = (df[column] == 1).cumsum()
    
    # Use 'cumcount' to count the number of occurrences since the last reset, adding 1 to start from 1
    bars_since = df.groupby(group_ids).cumcount() 
    
    # Wherever the condition is true, reset 'bars_since' to 1
    return bars_since.where(df[column] == 0, 1)


def get_nearest_future_exp(date_series , symbol='NIFTY'):
    """
    Finds the nearest future expiry (>= current date) for all rows in a dataframe 
    
    Parameters:
    date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
    
    Returns:
    nearest_future_expiry (pandas.Series): A pandas series with nearest expiry (>= current date) for each datetime in date_series
    
    """   
    #== find all valid expiries between start and end of df
    expirylist = fetchExpiryDays(symbol=symbol)
    exp_days = expirylist['dates'].tolist()
        
    df_dates = np.array(date_series ,dtype='datetime64[ns]')
    exp_dates = np.array(exp_days, dtype='datetime64[ns]')
    
    future_diffs = exp_dates - df_dates[:, np.newaxis]
    future_diffs[future_diffs < np.timedelta64(0)] = np.timedelta64(99999999, 'D') # Set past values to large number
    nearest_future_indices = np.argmin(future_diffs, axis=1)
   
    # Use the nearest indices to lookup the corresponding dates in exp_dates
    nearest_future_dates = exp_dates[nearest_future_indices]
    
    return nearest_future_dates


def get_prev_exp(date_series ,  symbol='NIFTY'):
    
    """
    Finds the previous expiry for all rows in a dataframe 
    
    Parameters:
    date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
    expiry_dt (pandas.Series): A pandas series of expiry datetime 
    
    Returns:
    days_to_expiry (pandas.Series): A pandas series with business days difference to nearest expiry for each datetime in date_series
    
    """
    
    # Generate a list of all business days between start and end dates
    start_date = date_series.min().date()- timedelta(days=10)
    end_date = date_series.max().date()+ timedelta(days=10)
    
    expirylist = fetchExpiryDays(symbol=symbol)
    exp_days = expirylist['dates'].tolist()
    #exp_days = [date.date() for date in exp_days]
    df_dates = np.array(date_series ,dtype='datetime64[ns]')
    exp_dates = np.array(exp_days, dtype='datetime64[ns]')
    
    prev_diffs = df_dates[:, np.newaxis] - exp_dates
    prev_diffs[prev_diffs < np.timedelta64(0)] = np.timedelta64(99999999, 'D') # Set negative values to large number
    nearest_previous_indices = np.argmin(prev_diffs, axis=1)
   
    # Use the nearest indices to lookup the corresponding dates in exp_dates
    nearest_previous_dates = exp_dates[nearest_previous_indices]
    #prev_exp = [date+datetime.time(15 , 15) for date in nearest_previous_dates]
    
    #nearest_previous_dates = pd.Series(nearest_previous_dates)+pd.Timedelta(hours=15, minutes=15)
    
    return nearest_previous_dates


def get_expiry(dt,expiry_offset , symbol , expiry_list):
    """
        get_expiry- calulates the expiry to trade for a datetime(row of df) on based on the expiry_offset
            
        Parameters
        ----------
        dt : datetime.date
            current date of the trade. 
            
        expiry_offset: int
            nth expiry away from current(latest) expiry 
            
        expiry_list : pd.DataFrame
            a dataframe with dates of all expiries    
                        
    """
    
    #expiriesList = fetchExpiryDays(symbol=symbol)
    possible_dates = expiry_list[expiry_list['dates'] >= dt]
    if not possible_dates.empty:
        if 0 <= expiry_offset < len(possible_dates):
            return possible_dates.iloc[expiry_offset]['dates']
        else:
            return "Not found"
    return "Not found"




def find_slope_and_prediction(df, column='Close', window=21):
    
    def calc_slope_intercept(y):
        if len(y) < window:
            return np.nan, np.nan
        x = np.arange(len(y))
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x**2)
        sum_xy = np.sum(x*y)
        N = len(x)

        # Calculate slope and intercept
        slope = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / N
        return slope, intercept

    # Apply the function over a rolling window and extract results
    results = df[column].rolling(window=window).apply(
        lambda y: calc_slope_intercept(y)[0], raw=True
    )

    slopes = results
    intercepts = df[column].rolling(window=window).apply(
        lambda y: calc_slope_intercept(y)[1], raw=True
    )

    # Calculate predicted value for the last point in each window
    predictions = slopes * (window - 1) + intercepts

    return slopes, predictions



def days_to_expiry_weekly(date_series , symbol='NIFTY'):
    
    """
    Finds the business days to nearest expiry for weekly Indian options
    
    Parameters:
    date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
    expiry_dt (pandas.Series): A pandas series of expiry datetime 
    
    Returns:
    days_to_expiry (pandas.Series): A pandas series with business days difference to nearest expiry for each datetime in date_series
    
    """
        
    # Generate a list of all business days between start and end dates
    start_date = date_series.min().date()
    end_date = date_series.max().date()+ timedelta(days=10)
    
    #== find all valid expiries between start and end of df
    expirylist = fetchExpiryDays(symbol=symbol)
    exp_days = expirylist['dates'].tolist()
    
    
    df_dates = np.array(date_series.date )
    exp_dates = np.array(exp_days)

    trading_days = fetchTradingDays()
    nse_cal = trading_days['dates']

    nearest_indices = np.argmax(exp_dates[:, np.newaxis] >= df_dates, axis=0)
    
    # Use the nearest indices to lookup the corresponding dates in exp_dates
    nearest_dates = exp_dates[nearest_indices]


    # Calculate the number of business days between each date in df and the nearest date in expiry
    
    # , holidays=nse_cal.holidays().holidays
    business_days = np.busday_count(df_dates.astype('datetime64[D]'), nearest_dates.astype('datetime64[D]') , weekmask='1111100' )
    
    # Check if the nearest date is the same as the input date, and set business days to 0 if so
    same_date_indices = np.where(nearest_dates == df_dates)[0]
    business_days[same_date_indices] = 0
    
    return business_days

#==========================================================

def days_since_expiry_weekly(date_series , symbol='NIFTY'):
    
    """
    Finds the business days from nearest  previous expiry for weekly Indian options
    
    Parameters:
    date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
    
    
    Returns:
    days_to_expiry (pandas.Series): A pandas series with business days difference to nearest expiry for each datetime in date_series
    
    """
    
    
    
    start_date = date_series.min().date()
    end_date = date_series.max().date()+ timedelta(days=10)
        
    #== find all valid expiries between start and end of df
    expirylist = fetchExpiryDays(symbol=symbol)
    exp_days = expirylist['dates'].tolist()
        
        
    df_dates = np.array(date_series.date )
    exp_dates = np.array(exp_days)

    # Convert to numpy arrays of type 'datetime64[D]'
    df_dates_np = np.array([np.datetime64(d) for d in df_dates])
    exp_dates_np = np.array([np.datetime64(d) for d in exp_dates])

    # Find the nearest date in exp_dates that is less than or equal to each date in df_dates
    nearest_dates = np.array([exp_dates_np[exp_dates_np <= d][-1] if np.any(exp_dates_np <= d) else np.datetime64('NaT') 
                            for d in df_dates_np])


    days_since_expiry = np.busday_count(nearest_dates ,df_dates_np  , weekmask='1111100' )

    return days_since_expiry



def calculate_anchored_vwap(df, open_col, high_col, low_col, close_col, volume_col, start_col):
    """
    Calculate Anchored VWAP for each row in the DataFrame, restarting the calculation each time 'a condition' is 1.
    
    Can be used to find vwap from say an expiry or an event
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the price and volume data.
    open_col (str): Name of the column containing open prices.
    high_col (str): Name of the column containing high prices.
    low_col (str): Name of the column containing low prices.
    close_col (str): Name of the column containing close prices.
    volume_col (str): Name of the column containing volume data.
    start_col (str): Name of the binary column indicating the start of the VWAP period.
    
    Returns:
    pd.Series: A Series containing the Anchored VWAP values.
    """
    # Create a rolling identifier for each VWAP period
    vwap_period_id = df[start_col].cumsum()

    # Calculate typical price
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3

    # Calculate price-volume product
    pv_product = typical_price * df[volume_col]

    # Calculate cumulative price-volume and cumulative volume for each period
    cumulative_pv = pv_product.groupby(vwap_period_id).cumsum()
    cumulative_vol = df[volume_col].groupby(vwap_period_id).cumsum()

    # Calculate Anchored VWAP
    anchored_vwap = cumulative_pv / cumulative_vol

    return anchored_vwap



def calc_intraday_low(series):
    # intraday_low_col = f'intraday_low_{agg_func}'
    spread_intraday_low = series.groupby(series.index.date).transform(lambda x: x.expanding().min())

    
def calc_spread(action,ratio,close_series):
    """
        calc_spread- calculates the spread of the options legs
            
        Parameters
        ----------
        action : string
            "buy"/"sell" based on this the spread is calculated. 
            
        ratio: int
            multiplier of the leg position
            
        close_series    : pd.series    
            close price of the option contract
            
        low_series    : pd.series    
            low price of the option contract
    """
    if(action =='buy'):
        spread_price = spread_price.add(ratio*close_series, fill_value=0)
    elif(action == 'sell'):
        spread_price = spread_price.sub(ratio*close_series, fill_value=0)
    

def get_nth_day(dt,offset):
        """
            get_nth_day- calulates the day  based on the offset
                
            Parameters
            ----------
            dt : datetime.date
                current date of the trade. 
                
            offset: int
                nth day from current(latest) day        
        """
        possible_dates = [fetchTradingDays['dates'] >= dt]
        if not possible_dates.empty:
            if 0 <= offset < len(possible_dates):
                return possible_dates.iloc[offset]['dates']
            else:
                return "Not found"
        return "Not found"



def intraday_vwap(df , high_col = 'High' , low_col='Low' , close_col = 'Close' , vol_col = 'Volume'):
    
    """
    
    Get the intraday vwap for a df.
    
    requirements: df with datetime as index and HLCV
    
    
    """
       
    df2 = df.copy(deep=True)

    df2['typical_p'] = ((df2[high_col] + df2[low_col] + df2[close_col])/3).astype('float64') 
    cumulative_volume = df2.groupby(df2.index.date)[vol_col].apply(lambda x: x.expanding().sum())
    
    df2['cumulative_volume'] = cumulative_volume.values
    
    df2['cumm'] = df2['typical_p']*df2[vol_col]
    
    expanding_sum = df2.groupby(df.index.date)['cumm'].apply(lambda x: x.expanding().sum())
    
    df2['expanding_sum'] = expanding_sum.values    
    
    xx = df2['expanding_sum']/df2['cumulative_volume']
    #df.drop(columns=['typical_p' , 'cumm' , 'expanding_sum'] , inplace=True)
    
    return xx.values
 
 
#=== This is faster    
def vwap(df, label='vwap', window=3, fillna=True , highcol='High' , lowcol='Low' , closecol='Close' , opencol = 'Open' , volcol='Volume'):
        
        import pandas_ta as ta
        from ta.volume import VolumeWeightedAveragePrice
        
        df[label] = VolumeWeightedAveragePrice(high=df[highcol], low=df[lowcol], close=df[closecol], volume=df[volcol], window=window, fillna=fillna).volume_weighted_average_price()
        
        return df    

def daily_moving_average(df, timeframe, column, periods=3, agg_func='last'):
    # Calculate the moving average excluding the latest value
    moving_average = df.resample(timeframe , origin='start_day')[column].agg(agg_func).dropna(how='all').rolling(window=periods).mean().shift(1)

    # Reindex the moving average back to the original DataFrame
    moving_average_reindexed = moving_average.reindex(df.index, method='ffill')

    # Add the moving average to the original DataFrame
     

    return moving_average_reindexed 

import pandas as pd
import numpy as np

#= very fast for a large dataframe


def calculate_daily_ma_vectorized(df, symbols, periods=3, agg_func='last', col_suffix='close'):
    """
    Calculates a daily moving average for multiple symbols at once in a vectorized way.

    Args:
        df (pd.DataFrame): The main DataFrame with a DatetimeIndex.
        symbols (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        periods (int): The number of days for the rolling window.
        agg_func (str): The aggregation function for resampling (e.g., 'last', 'first').
        col_suffix (str): The suffix of the columns to use (e.g., 'close').

    Returns:
        pd.DataFrame: A DataFrame containing the new daily moving average columns.
    """
    print(f"Calculating {periods}-day MA for {len(symbols)} symbols...")
    
    # 1. Select all the columns we need to process in one go
    target_cols = [f"{symbol}_{col_suffix}" for symbol in symbols]
    
    # 2. Resample all columns simultaneously to the daily timeframe
    daily_data = df[target_cols].resample('1D', origin='start_day').agg(agg_func)
    
     # This replaces the flawed .dropna() logic.
    if agg_func == 'sum':
        daily_data = daily_data[daily_data.sum(axis=1) > 0]
    # If using 'last' or 'first', the .dropna() logic is correct.
    else:
        daily_data = daily_data.dropna(how='all')

    # 3. Calculate the rolling average for all columns at once
    daily_ma = daily_data.rolling(window=periods, min_periods=1).mean()
    
    # 4. Shift the results to avoid lookahead bias
    shifted_daily_ma = daily_ma.shift(1)
    
    # 5. Reindex back to the original DataFrame's index and forward-fill
    # This efficiently broadcasts the daily values to the intraday index
    ma_reindexed = shifted_daily_ma.reindex(df.index, method='ffill')
    
    # 6. Rename the columns to their final desired names
    new_col_name = f'_{periods}d_MA'
    rename_map = {col: col + new_col_name for col in target_cols}
    ma_reindexed.rename(columns=rename_map, inplace=True)
    
    return ma_reindexed




import pandas as pd
import pandas_ta as pta
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

        atr_results[column_name] = pta.atr(
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


# # This single function call replaces your entire for loop for the ATR
# atr_features_df = calculate_atr_mtf_vectorized(
#     df=com_df2, 
#     symbols=final_symbols,
#     resample_tf='1D',
#     atr_period=3 , origin_time='start'
# )


def analyze_market_condition(price_change, oi_change):
    
    """
    Get the OI intrepetation for a list with price_change and oi_change
            
    """
   
    if price_change > 0 and oi_change > 0:
        return 'Long Buildup'
    elif price_change < 0 and oi_change < 0:
        return 'Long unwinding'
    elif price_change < 0 and oi_change > 0:
        return 'Short Buildup'
    elif price_change > 0 and oi_change < 0:
        return 'Short Covering'
    else:
        return 'Indeterminate/Other'    
    
    


def fetch_prev_day_OI_analysis(und_df , symbol , callput ):

    
    """
    Finds the previous day OI analysis for a symbol using its curretn options.
    Finds the top two strikes with highest OI and its initrepretation using price and OI change
    
    
    requirements: underlying data(OHLC and future expiry dates) with datetime as index and symbol name
    
    """

    # Initialize variable to keep track of the previous date
    previous_date = None

    unique_dates = np.unique(und_df.index.date)
    
    comb_df = pd.DataFrame()
    
    #=== Process option chain for each day 
    
    for current_date in unique_dates:
        
        print("\n Processing date" , current_date)
               
        #-- skip the first date as else we can see the future
        if previous_date is not None:
            
            #print( "\n current_date", current_date )
            #print( "\n \n ============== previous_date", previous_date )
        
            data_day = und_df[und_df.index.date==current_date]
            
            #print("\n Days data is " , data_day)
            
            
            #=== OI is only relevant for days with atleast 1 day since expiry
                      
                
            if data_day['days_since_expiry'].iloc[0]>=1:

                # Determine the day's low and high, rounded to the nearest 100
                
                day_low = np.ceil(data_day['Low'].min() / 100) * 100 
                day_high = np.ceil(data_day['High'].max() / 100) * 100
                
                call_limit = np.ceil(data_day['High'].max()*1.02 / 100) * 100
                put_low_limit = np.floor(data_day['Low'].min()*0.98 / 100) * 100
                
                strike_range_call = np.arange(day_low, call_limit , 100)
                strike_range_put = np.arange(put_low_limit, day_high , 100)
                                
                
                # Get start , end and expiry dates
                exp_Date = data_day['next_exp'].dt.date.iloc[0]
                #print("\n"  , "put_range: " ,strike_range_put  , exp_Date)
                
                #======================
                
                start_date = min(data_day['near_Exp'].dt.date.iloc[0] ,previous_date)-timedelta(days=5) 
                end_date = previous_date
                
                chain = fetchAllOptions(symbol=symbol , startDate=start_date , endDate=end_date , upper_strike=strike_range_call[-1] , lower_strike=strike_range_call[0] , expiryDate=exp_Date , callPut=callput , timeframe='1D')
                
                #print("\n Unique days found in chain" , chain.index.unique())               
                
                chain = chain.assign(
                                OI_pct_change=lambda x: x.groupby('Strike')['OI'].pct_change(),
                                Close_pct_change=lambda x: x.groupby('Strike')['Close'].pct_change()
                            )
                
                #print( "\n the chain is : ", chain)
                
                
                #== Fetch the Option chain and create a pivot table for Close and OI and changes in them
                pivot = chain.pivot_table(index=chain.index.date, columns='Strike', values=['OI' , 'Close' , 'OI_pct_change' ,'Close_pct_change' ], aggfunc='max')
                                
                #==== Filter to last row which is previous date
                if previous_date in pivot.index:
                    pivot = pivot.loc[previous_date]
                    
                    pivot = pivot.dropna(how='all')
                    
                    if pivot is None or pivot.empty:
                        print("\n ========Empty pivot: " )
                        
                    top_2_strikes = pivot['OI'].nlargest(2).index.get_level_values('Strike')

                    # Create a list of tuples for the column headers
                    top_2_strikes_columns = [(level, strike) for strike in top_2_strikes for level in [ 'OI', 'OI_pct_change', 'Close_pct_change']]
                    #print("\n top_2_strikes_columns are  " , top_2_strikes_columns)
                    existing_columns = [col for col in top_2_strikes_columns if col in pivot.index]
                    #print( "\n Exising columns are: ", existing_columns)
                    
                    if top_2_strikes_columns==existing_columns:
                        # Retrieve all columns for these top 2 strikes
                        #print("\n" , "Columns are equal")
                        top_2_strikes_data = pivot[top_2_strikes_columns]
                        #print(top_2_strikes_data)
                    
                    # To view the result
                    #print( "\n======= Data of top 2 strikes", top_2_strikes_data)
                    
                    if (top_2_strikes_data is None or  top_2_strikes_data.empty):
                        print("\n ========Empty dataframe: " )
                                                        
                                    # Extract unique strikes
                    unique_strikes = top_2_strikes_data.index.get_level_values('Strike').unique()

                    
                    
                    rows = []

                    for strike in unique_strikes:
                        price_change = top_2_strikes_data.loc[('Close_pct_change', strike)]
                        oi_change = top_2_strikes_data.loc[('OI_pct_change', strike)]
                        condition = analyze_market_condition(price_change, oi_change)

                        # Create a dictionary for each row
                        row_data = {
                            'Date': previous_date,
                            'Strike': strike,
                            'Buildup': condition
                        }
                        rows.append(row_data)

                    # Create DataFrame from the list of dictionaries
                    df1 = pd.DataFrame(rows)
        
                                        
                    comb_df = pd.concat([comb_df , df1] , axis=0)
                    
                    
                    # ... rest of your code to process pivot_day_data
                else:
                    print(f"No data available for {previous_date}")
                                #pivot = pivot.loc[previous_date]
               
                
        previous_date = current_date
   
    
    comb_df = comb_df.groupby('Date').agg({'Strike': list, 'Buildup': list}).reset_index()    
    # Add time of 9:15 to the 'Date' column
    #comb_df['Date'] = pd.to_datetime(comb_df['Date']) + pd.Timedelta(hours=9, minutes=15)
    #print(comb_df['Date'])
    comb_df.set_index('Date' , inplace=True)
       
    return comb_df        
        


def fetch_single_option(symbol , expiry_list , und_df_row,und_df , strike_ref_col='Close' , moneyness=0 , min_strike_chang=100 , expiry_offset=0 , option_type='CE'):
    
    
    option_data = []  #
    
        
    curr_date = und_df_row.name.date()
    #print("\n\n", curr_date)
    
    expiry_to_trade = get_expiry( curr_date,expiry_offset , symbol ,  expiry_list )
    
    strike_of_focus = round(und_df_row[strike_ref_col],-2)
    
    strike_to_fetch = strike_of_focus if moneyness == 0 else ((moneyness*min_strike_chang) + strike_of_focus  if option_type == 'CE' else strike_of_focus - (moneyness*min_strike_chang))

    print("\n Strike to fetch - " , strike_to_fetch)

    end_data = expiry_to_trade
    
    curr_date = curr_date - datetime.timedelta(days=3)
    
    #print("\n Starting and ending days: " , curr_date  , end_data)
    option_df_temp = fetchDataOptions(symbol=symbol,startDate = curr_date, endDate= end_data ,strike = strike_to_fetch ,expiryDate = expiry_to_trade,callPut = option_type,timeframe = '15min')
    
    new_columns = option_df_temp.columns.map(lambda x: 'opt_' + str(x))
    option_df_temp.columns = new_columns
    # option_data.append(option_df_temp)
    # del(option_df_temp)

    #== Now we combine the signal(underlying) data with option data 
    filtered_df = und_df[(und_df.index >= option_df_temp.index[0]) & (und_df.index <= option_df_temp.index[-1])]
    #print(filtered_df)
    result = pd.concat([option_df_temp, filtered_df], axis=1,join='outer')
    
    return result
    
    
def get_leg_info(leg_id):
    for leg_info in config_object.legs_info:
        if leg_info['leg_id'] == leg_id:
            return leg_info
    return None  # return None if no matching leg_id is found    



def fetch_all_options_combine_und(config , symbol, expiry_list,und_df_row, und_df , strike_ref_col , min_strike_chang , time_frame , trading_days ):
    
    option_data = []  #
    
    for leg in config['legs_info']:
        
        #print("\n Processing leg info" , leg)
        
        curr_date = und_df_row.name.date()
                    
        expiry_to_trade = get_expiry(curr_date,leg['expiry_offset'] , symbol , expiry_list )
        
        strike_of_focus = round(und_df_row[strike_ref_col],-2)
        
        if leg["call_put"] == 'CE':
            strike_to_fetch = (leg["moneyness"] * min_strike_chang) + strike_of_focus
        else:
            strike_to_fetch = strike_of_focus - (leg["moneyness"] * min_strike_chang)
        
        
        end_data = expiry_to_trade
        
        #print( "\n Expiry : " , expiry_to_trade , "\n strike: " , strike_to_fetch)
        
        curr_date = curr_date - datetime.timedelta(days=3)
                
        option_df_temp = fetchDataOptions(symbol=symbol,startDate = curr_date, endDate= end_data ,strike = strike_to_fetch ,expiryDate = expiry_to_trade,callPut = leg['call_put'],timeframe = time_frame , trading_days=trading_days)
        #leg_id = add_leg(name=f"{leg['action']}_{leg['ratio']}_{leg['call_put']}_{strike_of_focus}_{str(expiry_to_trade)}")
        
        if option_df_temp is not None and not option_df_temp.empty and len(option_df_temp) > 30:
            
            new_columns = option_df_temp.columns.map(lambda x: leg["leg_id"]+'_' + str(x))
            
            option_df_temp.columns = new_columns
            
            option_data.append(option_df_temp)
            
            del(option_df_temp)

    
    if option_data:  # This ensures that option_data is not empty
        combined_df = pd.concat(option_data, axis=1)
        filtered_df = und_df[(und_df.index >= combined_df.index[0]) & (und_df.index <= combined_df.index[-1])]
        result = pd.concat([combined_df, filtered_df], axis=1, join='outer')
    else:
        print("Warning: No options data fetched. Returning an empty DataFrame.")
        result = None  # Return an empty DataFrame or handle as needed
   
   
    #print("\n\n\n" , "Result or net dataframe is : \n" ,result )
    return result



def expanding_high_since_expiry(df, expiry_series , column='indHigh'):
    
    #==== Function to find the expanding high since a date column..eg expiry
    
    
    # Ensure the expiry_series is aligned with df's index
    #df['Temp_Expiry'] = expiry_series.reindex(df.index)

    # Use groupby on Temp_Expiry and then compute expanding max on the 'High' column
    expanding_high = df.groupby(expiry_series)[column].expanding().max().reset_index(level=0, drop=True)
    
    # Drop the temporary expiry column
    #df.drop(columns=['Temp_Expiry'], inplace=True)
    
    return expanding_high


def expanding_low_since_expiry(df, expiry_series , column='indLow'):
    
    #==== Function to find the expanding high since a date column..eg expiry
    
    
    # Ensure the expiry_series is aligned with df's index
    #df['Temp_Expiry'] = expiry_series.reindex(df.index)

    # Use groupby on Temp_Expiry and then compute expanding max on the 'High' column
    expanding_low = df.groupby(expiry_series)[column].expanding().min().reset_index(level=0, drop=True)
    
    # Drop the temporary expiry column
    #df.drop(columns=['Temp_Expiry'], inplace=True)
    
    return expanding_low

def expanding_high_since_condition(df, high_col='High', condition_col='is_exp'):
    # Create a segment identifier that increments each time condition_col is 1
    df['segment'] = df[condition_col].cumsum()

    # Calculate expanding high within each segment
    expanding_highs = df.groupby('segment')[high_col].expanding().max().reset_index(level=0, drop=True)

    return expanding_highs




def create_tradelog2(df, symbol):
    
    print("\n Creating trade log for", symbol)
    
    if df.empty:
        print("\n DataFrame is empty. No data to process.\n")
        return pd.DataFrame()
    
    
    df['position_change'] = df[f'{symbol}_position'].diff()
    
    print(df.columns)
    
        # Construct column names dynamically
    columns = [
        f'{symbol}bar_count', f'{symbol}_position',  'position_change' , f'{symbol}_ret', f'{symbol}_Close' , f'{symbol}_order' , f'{symbol}_exit_price']

    # Now filter the DataFrame to include only the columns for this symbol
    df = df[columns]
    
    
    # Assuming 'bar_count_col' and other columns are defined as per your data
    bar_count_col = f'{symbol}bar_count'

    # Check if the essential column exists in the DataFrame
    if bar_count_col not in df.columns:
        print(f"Missing essential column: {bar_count_col}")
        return pd.DataFrame()
    
    # Identifying the end of a trade
    df['trade_end'] = (df[bar_count_col].shift(-1) < df[bar_count_col]) | (df[bar_count_col].shift(-1) == 0) | df[bar_count_col].eq(0)

    # Creating 'trade_start' to identify the start of a trade
    df['trade_start'] = df['trade_end'].shift(1).fillna(False) | df[bar_count_col].eq(1)
    
    df['trade_start'] = df['trade_start'].astype(int)

    # Generating 'trade_id' for each trade
    df['trade_id'] = df['trade_start'].cumsum()

    # Filtering out rows not part of any trade (i.e., before any trade has started)
    df = df[df['trade_id'] > 0]
    
    # Processing each trade to gather trade details
    trades_info = []
    for trade_id, trade_group in df.groupby('trade_id'):
        if trade_group.empty or trade_group[bar_count_col].max() == 0:
            continue  # Skip empty trades or trades without bar counts
        
        print("\n Individual trade found as " , trade_group)
        
        #print("\n Exit price: " , trade_group.iloc[-1][f'{symbol}_Close'])
        positionn = trade_group[f'{symbol}_position'].iloc[0]
        side = 'long' if positionn == 1 else 'Short' if positionn == -1 else 'flat'
        
        trade_details = {
            'Symbol': symbol,
            'Side': side,
            'Trade ID': trade_id,
            'Start Time': trade_group.index[0],
            'End Time': trade_group.index[-1],
            'Entry Price': trade_group.iloc[0][f'{symbol}_Close'],  # Assuming 'Close' is the close price column
            'Exit Price': trade_group.iloc[-1][f'{symbol}_exit_price'],  # Modify as per your DataFrame's column names
            # Include other details as needed
        }
        trades_info.append(trade_details)
    
    # Converting the trades information into a DataFrame
    trade_summary_df = pd.DataFrame(trades_info)
    return trade_summary_df

# Replace 'your_symbol' and 'Close' with actual column names as per your DataFrame
# Ensure your DataFrame 'df' is indexed appropriately (e.g., datetime index) and contains the necessary columns





def plot_signal2_options(config ,  tradelog ,unddf, signal_date):
        """
            plot_signal- Iterates over the tradelist and plot the signal u want
            
            Parameters
            ----------
            
            signal_date: string or datetime
                the date for which you want to see the signals
                      
        
         """
         
        import mplfinance as mpf 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        
        
        opt_timeframe = str(config['timeframe']) + "min" 
        print("\n Plotting on timeframe : " , opt_timeframe)
         
        if isinstance(signal_date, str):
            signal_date = datetime.datetime.strptime(signal_date, '%Y-%m-%d %H:%M:%S')
        
        df_date =tradelog[tradelog['Time of Entry'].dt.date == signal_date]
        df_date.reset_index(inplace=True)
                
        fig = plt.figure(figsize=(12, 8))

        # create GridSpec with 2 rows and 2 columns
        gs = gridspec.GridSpec(2, 2, figure=fig)
        # Create my own `marketcolors` style:
        mc = mpf.make_marketcolors(up='b', down='r')
        s = mpf.make_mpf_style(base_mpl_style='bmh', marketcolors=mc)
        
        for i, trade in df_date.iterrows():
            # Fetch historical data
            entry_time = trade['Time of Entry']
            exit_time = trade['Time of Exit']

            string_leg = trade['name']
                        
            # Split the string if '_' is present, else assign 'NA'
            parts = string_leg.split('_') if '_' in string_leg else ['NA', 'NA', 'NA']

            # Unpack the parts to callput, strike, and expiry_to_trade
            callput, strike, expiry_to_trade = parts[:3]
            
                                
            option_df_temp = fetchDataOptions(symbol=config['symbol'],startDate = entry_time.date(), endDate= exit_time.date() ,strike = strike ,expiryDate = expiry_to_trade,callPut = callput,timeframe = opt_timeframe , trading_days=fetchTradingDays())
            #print(option_df_temp)
            t = fig.add_subplot(gs[0, i])
            t.grid(True, color='darkgrey')
            mpf.plot(option_df_temp,style=s,type='candle',vlines=dict(vlines=[f'{str(entry_time)}',f'{str(exit_time)}'],linewidths=(1,1)),ax=t,axtitle=f"Strikes = {strike}")

        #For underlyiung
        # Modify the date filtering for und_df
        unddf = unddf[(unddf.index.date >= signal_date - timedelta(days=1)) & (unddf.index.date <= signal_date + timedelta(days=1))]

        t = fig.add_subplot(gs[1, :])
        t.grid(True, color='darkgrey')
        mpf.plot(unddf,style=s,type='candle',ax=t,axtitle=f"Underlying")
        plt.tight_layout()
        plt.show()



def process_VIX(loc = r"C:\Users\aakas\Desktop\Data\Indices_Futures\Indices_Futures\India_VIX_2020_Jan_2024_clean.csv"):
    
    vixfile = loc

    vixdf = pd.read_csv(vixfile)
    if 'TIMESTAMP' in vixdf.columns:
        vixdf['TIMESTAMP'] = pd.to_datetime(vixdf['TIMESTAMP'] , format="%Y-%m-%d")
        vixdf['TIMESTAMP'] = vixdf['TIMESTAMP'] + timedelta(hours=15, minutes=15)
        vixdf.set_index('TIMESTAMP' , inplace=True)
        
        vixdf = vixdf['VIX_Close']
    else:
        print("\n Timeframe not present in VIX dataframe")
        return None
    
        
    return vixdf    


#==== Functions related to peak and trough



def find_peak(df , column='High' , threshold=4 , distance=3):
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(df[column] , threshold=threshold , distance=distance )
    
        # Extracting peak values
    peak_values = df[column].iloc[peaks]
    #print(peak_values)

    return peak_values


def find_trough(df , column='Low' , threshold=4 , distance=3):
    from scipy.signal import find_peaks
    

    # Invert the data to turn troughs into peaks
    inverted_data = -df[column]

    # Find peaks in the inverted data, which correspond to troughs in the original data
    troughs, _ = find_peaks(inverted_data  , threshold=threshold , distance=distance )

    # Extracting trough values from the original data
    trough_values = df[column].iloc[troughs]
    
    
    return trough_values


def find_peak_multi_tf(df , column='High', highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open' , volcol='Volume' , threshold=4 , distance=3 , timeframe='1D'):
    
    #=== resample the data to the timeframe 
    resamp = resample_data_crypto_fast_numba(df=df , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volumecol=volcol , timeframe=timeframe)
    
    #=== find the peak series and reindex to the resampled dataframe
    resamp = resamp.shift(1)
    
    peak_values = find_peak(resamp , column=column , threshold=threshold , distance=distance)
    
    peakss = peak_values.reindex(df.index, method='ffill')
    
    return peakss
    
def find_trough_multi_tf(df , column='High', highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open' , volcol='Volume' , threshold=4 , distance=3 , timeframe='1D'):     
    
    
     #=== resample the data to the timeframe 
    resamp = resample_data_crypto_fast_numba(df=df , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volumecol=volcol , timeframe=timeframe)
    
    
    resamp = resamp.shift(1)
    
    trough_values = find_trough(resamp , column=column , threshold=threshold , distance=distance)
    
    troughs = trough_values.reindex(df.index , method='ffill')
    
    return troughs

#==================================================================



        

def resistance_all_levels(und_df , price_increase_percentage):
    
    import pandas as pd
    import numpy as np

    # Assuming und_df is your DataFrame and it's already defined with necessary columns
    
    # Your existing code
    
    target_price = und_df['today_open'] + (und_df['prevday_rng'] * price_increase_percentage)
     
    resistance_cols = ['prevday_close', 'prevdayh', 'expiry_high', 'R1', 'S1', 'Pivot', '20DMA', 'R2', 'today_open', 'all_time_high', 'nearest_100_fhr', 'round_level' , 'ind_3highc' , '100DEMA']
     
    # Convert DataFrame to NumPy array for efficient computation
    
    resistance_values = und_df[resistance_cols].values
    target_prices = target_price.values[:, np.newaxis]  # Convert to column vector for broadcasting
    today_opens = und_df['today_open'].values[:, np.newaxis]

    # Filter out values below target price and today's open
    valid_resistances = np.where((resistance_values > target_prices) & (resistance_values > today_opens), resistance_values, np.nan)

    # Sort along axis=1 and take the first three values
    sorted_resistances = np.sort(valid_resistances, axis=1)
    nearest_three_resistances = sorted_resistances[:, :3]  # Take first three columns which are the nearest resistances

    # Convert back to DataFrame
    resistance_three = list(nearest_three_resistances)

    return resistance_three



#--- Make global the tradelog dictionary

def make_tradelog( action , position_book, id ,name  ,dtime ,Side ,price , qty , trade_book):
        """
        make_tradelog - adds trade values to 'temp_dict_entry'
        
        Parameters
        ----------
            action : string
                entry/exit
                
            dtime: datetime
                timestamp of the trade
                
            price  : float    
                price point of the trade
        """        

        if(action == 'entry'):
            position_book[id]['name'] = name
            position_book[id]['inposition'] = True
            position_book[id]['Time of Entry'] = dtime
            position_book[id]['Side'] = Side
            position_book[id]['Entry Price'] = price
            position_book[id]['qty'] = qty
            #tradelog = pd.concat([tradelog,pd.DataFrame([position_book[id]])]) 
                        
        elif(action == 'exit'):
            position_book[id]['inposition'] = False
            position_book[id]['Time of Exit'] = dtime
            #position_book[id]['Side'] = Side           
            position_book[id]['Exit Price'] = price
            #position_book[id]['Exit Reason'] =  'reason'
            position_book[id]['qty'] = qty
            trade = pd.DataFrame([position_book[id]])
            print("\n \n Entire_trade in position_book is " ,trade )
            
            return trade
            
            
    
    





def support_all_levels(und_df , price_increase_percentage):
    
    """
    Finds the support levels below today open and % of previous range from open.
    Levels considered are 5 ,20 , 100 DMA and Pivot points and prevday low and closes etc
    
    
    requirements: underlying data(OHLC) with datetime as index and price_increase_percentage() e.g) 0.2
    
    """
    
    # Assuming und_df is your DataFrame and it's already defined with necessary columns
    
    # Your existing code
    
    target_price = und_df['today_open'] - (und_df['prevday_rng'] * price_increase_percentage)
     
    resistance_cols = ['prevday_close', 'prevdayl', 'expiry_low', 'R1', 'S1', 'S2', 'Pivot', '20DMA', 'R2', 'fh_low_round', 'round_support' , 'ind_3dlowc' , '100DEMA']
     
    # Convert DataFrame to NumPy array for efficient computation
    
    resistance_values = und_df[resistance_cols].values
    target_prices = target_price.values[:, np.newaxis]  # Convert to column vector for broadcasting
    today_opens = und_df['today_open'].values[:, np.newaxis]

    # Filter out values above target price and above today's open
    valid_supports = np.where((resistance_values < target_prices) & (resistance_values < today_opens), resistance_values, today_opens*0.99)


    # Sort along axis=1 and take the first three values
    sorted_resistances = -np.sort(-valid_supports, axis=1)

    nearest_three_resistances = sorted_resistances[:, :3]  # Take first three columns which are the nearest resistances

    # Convert back to DataFrame
    resistance_three = list(nearest_three_resistances)

    return resistance_three

    
    

def yang_zhang_vol(df , periods=20 , trading_days=252 , clean=True , highcol = 'High' , lowcol='Low' , closecol='Close' , opencol='Open'):
    
    import math
    """
    Calculate the Yang-Zhang estimator of volatility for an array of open, high, low, and closing prices( daily is better).
    K is a factor to correct for bias. Default is 0.34 for daily returns.

    :param open_prices: array-like of opening prices
    :param high_prices: array-like of high prices
    :param low_prices: array-like of low prices
    :param close_prices: array-like of closing prices
    :return: the Yang-Zhang estimate of volatility
    """
    
    
    logho = (df[highcol]/df[opencol]).apply(np.log)
    
    loglo = (df[lowcol]/df[opencol]).apply(np.log)
    
    logco = (df[closecol]/df[opencol]).apply(np.log)
    
    logoc = (df[opencol]/df[closecol].shift(1)).apply(np.log)
    
    logoc_sq = logoc ** 2
    
    logcc = (df[closecol]/df[closecol].shift(1)).apply(np.log)
    
    logcc_sq = logcc **2
    
    
    rs = logho *( logho - logco) + loglo * (loglo - logco)
    
    close_vol = logcc_sq.rolling(window=periods , center=False ).sum() * ( 1.0 /(periods - 1.0))
    
    open_vol = logoc_sq.rolling(window=periods , center=False).sum() * ( 1.0 /(periods - 1.0))
    
    window_rs = rs.rolling(window=periods , center=False).sum() * (1/(periods-1))
    
    
    k = 0.34/ (1 + (periods +1) / (periods -1)) 
    
    result = (open_vol + k*close_vol + (1-k)*window_rs).apply(np.sqrt)* math.sqrt(trading_days)
    
    if clean:
        return result.dropna()
    else:
        return result
        

def plot_signal2_equity_from_file( tradelog_file ,unddf_file, signal_date , stk):
        
        
        """
            plot_signal- Iterates over the tradelist and plot the signal u want
            
            Parameters
            ----------
            
            signal_date: string or datetime
                the date for which you want to see the signals
                   
        
        """
         
            
        import mplfinance as mpf 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from datetime import timedelta
         
        #=== read the trade log 
        
        tradelog = pd.read_csv(tradelog_file)
        print( "\n Trade log columns are: ", tradelog.columns)
        
        tradelog['Start Time'] = pd.to_datetime(tradelog['Start Time'])
        tradelog['End Time'] = pd.to_datetime(tradelog['End Time'])
        
        print("\n Found trade log as : " , tradelog)
        und_df = pd.read_csv(unddf_file)
        
        und_df.columns = und_df.columns.str.strip().str.capitalize()
      
        if 'Date' in und_df.columns:
            
            und_df['Date'] = pd.to_datetime(und_df['Date'])
            und_df.set_index('Date' , inplace=True)
            
        
        print("\n Underlyng datatframe is " , und_df)
        
        
        
        #=== if stk is not provided than an index 
        columns = [f"{stk}_High", f"{stk}_Low", f"{stk}_Close", f"{stk}_Open"] if stk else ["High", "Low", "Close", "Open"]
        missing_columns = [col for col in columns if col not in und_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns {missing_columns} in the DataFrame. Available columns: {und_df.columns.tolist()}")

        
        und_df = und_df[columns]
        
        
        if 'Unnamed: 0' in und_df.columns:

            und_df['Unnamed: 0'] = pd.to_datetime(und_df['Unnamed: 0'] , format ="%Y-%m-%d %H:%M:%S")
            und_df.rename(columns={'Unnamed: 0':'DateTime'} , inplace=True)
            
            und_df.set_index('DateTime' , inplace=True)    
                

       
        
        
             # Renaming columns for mplfinance
        plot_columns = {
            f'{stk}_Open': 'Open',
            f'{stk}_High': 'High',
            f'{stk}_Low': 'Low',
            f'{stk}_Close': 'Close'
        }
        
                
                # Check if all specified old column names are present in the DataFrame
        if all(column in und_df.columns for column in plot_columns.keys()):
            # If all columns are present, rename them
            und_df.rename(columns=plot_columns, inplace=True)
        else:
            print("Not all columns are present in the DataFrame.")
        
        
        
        if isinstance(signal_date, str):
            signal_date = datetime.datetime.strptime(signal_date, '%Y-%m-%d %H:%M:%S')
        
        df_date =tradelog[tradelog['Start Time'].dt.date == signal_date]
        df_date.reset_index(inplace=True)
        
        print("Found trade as : " , df_date)
        
        fig = plt.figure(figsize=(12, 8))

        # create GridSpec with 2 rows and 2 columns
        gs = gridspec.GridSpec(2, 2, figure=fig)
        # Create my own `marketcolors` style:
        mc = mpf.make_marketcolors(up='b', down='r')
        s = mpf.make_mpf_style(base_mpl_style='bmh', marketcolors=mc)
        
        for i, trade in df_date.iterrows():
            # Fetch historical data
            entry_time = trade['Start Time']
            exit_time = trade['End Time']

            string_leg = trade['Side']
            
            und_req = und_df[(und_df.index.date>=entry_time.date()-timedelta(days=3)) & (und_df.index.date<=exit_time.date()+timedelta(days=2)) ]
           
            #print(option_df_temp)
            t = fig.add_subplot(gs[0, i])
            t.grid(True, color='darkgrey')
            mpf.plot(und_req,style=s,type='candle',vlines=dict(vlines=[f'{str(entry_time)}',f'{str(exit_time)}'],linewidths=(1,1)),ax=t,axtitle="Trade")
            plt.tight_layout()
            plt.show()
        #For underlyiung
#         unddf = unddf[unddf.index.date == signal_date]
#         t = fig.add_subplot(gs[1, :])
#         t.grid(True, color='darkgrey')
#         mpf.plot(unddf,style=s,type='candle',ax=t,axtitle=f"Underlying")
        

#----- function to read trade logs and find total profit from all of symbols for them

def read_trade_log(logloc):
    
    import os
    
    
    basename = os.path.basename(logloc)
    if '_' in basename:
        symbol = os.path.splitext(basename)[0].lstrip('_')
    else:
        
        symbol = os.path.splitext(basename)[0]
        # Handle the case where no underscore is present
        print("\n Symbol is widout underscore--  " , symbol)
    df = pd.read_csv(logloc)
    
    if df.empty:
        print("\n No trades found")
        return None
    else:
        
       
        
        if 'Entry Price' in df.columns:
            df['Entry Price'] = df['Entry Price'].astype('float')
        
        else:
            
            print("\n Entry price column not found on trade file")
            
        df['Exit Price'] = df['Exit Price'].astype('float')
        
        # Drop rows where entry_price or exit_price is NaN
        df.dropna(subset=['Entry Price', 'Exit Price'], inplace=True)
                
        
        df['returns'] = np.where(df['Side']=='long' , (df['Exit Price']/df['Entry Price'])-1 , (df['Entry Price']-df['Exit Price'])/df['Entry Price'])
        
        
        net_return = (100000*df["returns"]).cumsum()
        net_return.dropna(inplace=True)
        print( "\n Symbol return from trade log is : ", net_return.iloc[-1]) 
        return symbol , net_return.iloc[-1]


    
def concatenate_trades(logloc , comb_df , add_slippages=0 , slippage_in_ticks=1 ,min_movement = 0.05):
    
    import os
    
    
    
        # Get the base name of the file
    basnam = os.path.basename(logloc)
    #print("\n parsing file: \n" , basnam)
        
        
    

    # Check if the basename contains an underscore
    if '_' in basnam:
        symbol = os.path.splitext(basnam)[0].lstrip('_')
        print("\n SYmbol with underscore  is: \n " , symbol)
    else:
        
        symbol = os.path.splitext(basnam)[0]
        print("\n SYmbol is: \n " , symbol)
        
    
    
    df = pd.read_csv(logloc)
    if df.empty:
        print("\n No trades found")
        return None
    else:
        
        # Convert entry_price and exit_price to numeric, set errors='coerce' to handle non-numeric
        df['Entry Price'] = pd.to_numeric(df['Entry Price'], errors='coerce')
        df['Exit Price'] = pd.to_numeric(df['Exit Price'], errors='coerce')
  
        
        
        if add_slippages>0:


            df['Entry Price'] = np.where(df['Side']=='long' , (df['Entry Price'] + (slippage_in_ticks*min_movement))  ,  (df['Entry Price'] - (slippage_in_ticks*min_movement)))

            df['Exit Price'] = np.where(df['Side']=='long' , (df['Exit Price'] - (slippage_in_ticks*min_movement))  ,  (df['Exit Price'] + (slippage_in_ticks*min_movement)))

            #print("\n After adding slippages" , df)

               
        # Drop rows where entry_price or exit_price is NaN
        df.dropna(subset=['Entry Price', 'Exit Price'], inplace=True)
        
        comb_df = pd.concat([comb_df , df] , axis=0 , join='outer')
        
        
        return comb_df
    
    
    
    
    

def plot_OHLC_cuff(df , out_pth ):
    
    
        
    
    df = df[['Open' , 'High' , 'Low' , 'Close']]
    
    import os
    import cufflinks as cf
    from plotly.offline import iplot
    import plotly.offline as pyo
    # Setup to run Plotly in offline mode
    cf.set_config_file(offline=True)

    
    cf.set_config_file(theme='pearl')
        
    # Create the QuantFig object with your data
    qf = cf.QuantFig(df, title='First Quant Figure', legend='top', name='Bank Nifty')


    # Generate the figure
    fig = qf.figure()

    # Update layout to treat the x-axis as categorical
    fig.update_layout(
        xaxis=dict(
            type='category' , rangeslider=dict(visible=True)   # Treats the x-axis values as categorical, skipping gaps automatically
        )
    )

    # Plot the figure with a range slider
    #iplot(fig)    
    
        # Path for the HTML file
    output_file = os.path.join(out_pth , "OHLC.html" )

    # Save the plot
    
    pyo.plot(fig, filename=output_file, auto_open=True)
        
    
      
def calc_realised_vol(df , col , window=10):
    
    
    tm = get_data_timeframe(df)
       
    print("\n Current timeframe is: \n " , tm)
    
    if tm<1000:
        
        print("\n Please check timeframe as it is not daily \n")
    
    
    ret = df[col].pct_change()
    
    
    realized_vol = ret.rolling(window).std()
       
    
    
    return realized_vol
    
    

def get_sec_list():
    
    
    
    sec_list = {'NIFTY 50': ['BPCL', 'HDFCLIFE', 'HCLTECH', 'TCS', 'MARUTI', 'ONGC', 'WIPRO', 'SBILIFE', 'INFY', 'LT',
                 'TATASTEEL', 'UPL', 'BHARTIARTL', 'SBIN', 'TECHM', 'COALINDIA', 'ULTRACEMCO', 'EICHERMOT', 'M&M',
                 'TATAMOTORS', 'ADANIPORTS', 'DRREDDY', 'HINDALCO', 'ASIANPAINT', 'BAJAJ', 'HEROMOTOCO', 'ADANIENT',
                 'TATACONSUM', 'DIVISLAB', 'LTIM', 'SUNPHARMA', 'JSWSTEEL', 'ICICIBANK', 'HINDUNILVR', 'TITAN',
                 'NTPC', 'HDFCBANK', 'BAJFINANCE', 'NESTLEIND', 'APOLLOHOSP', 'CIPLA', 'AXISBANK', 'RELIANCE',
                 'BAJAJFINSV', 'KOTAKBANK', 'GRASIM', 'ITC', 'INDUSINDBK', 'BRITANNIA', 'POWERGRID'],
    'NIFTY AUTO': ['HEROMOTOCO', 'BAJAJ-AUTO', 'TVSMOTOR', 'MARUTI', 'BOSCHLTD', 'MOTHERSON', 'M&M', 'TATAMOTORS',
                   'SONACOMS', 'EICHERMOT', 'ASHOKLEY', 'BHARATFORG', 'MRF', 'TIINDIA', 'BALKRISIND'],
    'NIFTY BANK': ['ICICIBANK', 'KOTAKBANK', 'BANDHANBNK', 'HDFCBANK', 'AXISBANK', 'FEDERALBNK', 'BANKBARODA', 'SBIN',
                   'AUBANK', 'PNB', 'INDUSINDBK', 'IDFCFIRSTB'],
    'NIFTY COMMODITIES': ['TATAPOWER', 'ADANIGREEN', 'SHREECEM', 'GRASIM', 'SRF', 'ULTRACEMCO', 'NTPC', 'RELIANCE',
                          'APLAPOLLO', 'NAVINFLUOR', 'JSWSTEEL', 'DEEPAKNTR', 'TATASTEEL', 'AMBUJACEM', 'ACC', 'PIIND',
                          'UPL', 'ADANIPOWER', 'HINDALCO', 'BPCL', 'DALBHARAT', 'TATACHEM', 'VEDL', 'ONGC', 'JINDALSTEL',
                          'SAIL', 'PIDILITIND', 'COALINDIA', 'HINDPETRO', 'IOC'],
    'NIFTY CPSE': ['POWERGRID', 'NTPC', 'COCHINSHIP', 'NBCC', 'BEL', 'ONGC', 'OIL', 'COALINDIA', 'NHPC', 'SJVN',
                   'NLCINDIA'],
    'NIFTY ENERGY': ['TATAPOWER', 'ADANIGREEN', 'POWERGRID', 'NTPC', 'RELIANCE', 'BPCL', 'ONGC', 'COALINDIA', 'IOC',
                     'ADANIENSOL'],
    'NIFTY FIN SERVICE': ['ICICIBANK', 'ICICIGI', 'BAJAJFINSV', 'SHRIRAMFIN', 'KOTAKBANK', 'SBICARD', 'HDFCAMC',
                          'ICICIPRULI', 'BAJFINANCE', 'LICHSGFIN', 'HDFCBANK', 'IEX', 'AXISBANK', 'MUTHOOTFIN',
                          'HDFCLIFE', 'CHOLAFIN', 'RECLTD', 'SBIN', 'SBILIFE', 'PFC']}
    # Add other sectors following the same pattern }
    
    
    return sec_list 
    
    
def x_min_cum_vol(df ,column , start='09:15' , end='10:15' ):
    
    
    
    if column in df.columns:
    
        xm_cvol = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end).expanding().sum()).ffill()
        return xm_cvol    
    
    
    
def calculate_beta(alt_ret, btc_ret, window=12):
    """Calculate rolling beta for a single altcoin"""
    covar = alt_ret.rolling(window).cov(btc_ret)
    btc_var = btc_ret.rolling(window).var()
    return covar / btc_var

# # Apply to all symbols in one go
# for sym in final_symbols:
#     com_df2[f'{sym}_beta'] = calculate_beta(com_df2[f'{sym}_log_ret'])
# 


def heikin_ashi_supertrend_fast(df, length=24, multiplier=3):
    o = df['Open'].values
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values

    ha_close = (o + h + l + c) / 4
    ha_open = (np.roll(o, 1) + np.roll(c, 1)) / 2
    ha_open[0] = o[0]

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    ha_df = pd.DataFrame({
        'high': ha_high,
        'low': ha_low,
        'close': ha_close
    })

    st = ta.supertrend(ha_df['high'], ha_df['low'], ha_df['close'], length=length, multiplier=multiplier)

    # Format multiplier exactly as pandas-ta does
    mult_fmt = f"{float(multiplier):.1f}"
    st_col_name = f"SUPERT_{length}_{mult_fmt}"

    if st_col_name not in st.columns:
        raise KeyError(f"Supertrend column '{st_col_name}' not found. Available: {st.columns.tolist()}")

    return st[st_col_name]




def apply_supertrend_to_com_df2_fast(com_df2, symbols):
    for sym in symbols:
        ohlc = pd.DataFrame({
            'Open': com_df2[f'{sym}_open'],
            'High': com_df2[f'{sym}_high'],
            'Low':  com_df2[f'{sym}_low'],
            'Close':com_df2[f'{sym}_close']
        })
        ha_st = heikin_ashi_supertrend_fast(ohlc , length=24 , multiplier=3)
        com_df2[f'{sym}_HA_ST'] = ha_st.values
    return com_df2