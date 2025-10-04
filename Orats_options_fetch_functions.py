

import pandas as pd

from typing import List


import pandas as pd
import pandas_market_calendars as mcal
import duckdb

def find_missing_data_days(con, ticker, path_template, start_date_str, end_date_str):
    """
    Checks a date range against an S3 source using DuckDB and returns a list 
    of trading days with missing data files.

    Args:
        con: An initialized and configured DuckDB connection object.
        ticker (str): The stock ticker (e.g., 'SPY').
        path_template (str): A formatted string for the S3 path, e.g., 
                             "s3://bucket/folder/ticker={ticker}/day={date}/*.parquet".
        start_date_str (str): The first day of the range to check ('YYYY-MM-DD').
        end_date_str (str): The last day of the range to check ('YYYY-MM-DD').

    Returns:
        list: A list of date strings ('YYYY-MM-DD') for which data is missing.
    """
    # 1. Generate a definitive list of all expected trading days
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date_str, end_date=end_date_str)
    expected_days = [d.strftime('%Y-%m-%d') for d in schedule.index]
    
    print(f"Checking {len(expected_days)} expected trading days between {start_date_str} and {end_date_str}...")

    missing_days = []
    # 2. Loop through each expected day and try to query it
    for day_str in expected_days:
        # Fill in the path template with the specific ticker and date
        path_to_check = path_template.format(ticker=ticker, date=day_str)
        query = f"SELECT COUNT(*) FROM read_parquet('{path_to_check}');"
        
        try:
            # Try to run the lightweight count query
            result = con.execute(query).fetchone()
            if result[0] == 0:
                # File exists but is empty
                print(f"  ⚠️ INFO: File for {day_str} exists but is empty.")
                missing_days.append(day_str)
                
        except duckdb.IOException as e:
            # This error means the file was not found
            print(f"  ❌ MISSING: No file found for {day_str}.")
            missing_days.append(day_str)
        except Exception as e:
            # Handle other potential errors
            print(f"  An error occurred checking {day_str}: {e}")
            missing_days.append(day_str)

    return missing_days

# ==============================================================================
# --- Example Usage ---
# ==============================================================================


# # 2. Define the path template with placeholders for the ticker and date
# path_template = "s3://duckdata/ORATS/Options/ticker={ticker}/day={date}/*.parquet"

# # # 3. Define the date range you want to audit
# # start_date = com_df.index[0].strftime('%Y-%m-%d')
# # end_date = com_df.index[-1].strftime('%Y-%m-%d')

# start_date = datetime(2024 ,1 , 1)
# end_date = datetime(2024 , 12 , 31)


# # 4. Call the function
# # This will connect to your S3 and check each day in the range.
# missing_dates = find_missing_data_days(con, 'SPY', path_template, start_date, end_date)

import numpy as np

def round_to_nearest(number, interval=1):
    """
    Rounds a number to the nearest specified interval.

    Args:
        number (float or np.ndarray): The number or array of numbers to round.
        interval (float): The interval to round to (e.g., 1, 0.5, 5).

    Returns:
        float or np.ndarray: The rounded number or array.
    """
    return interval * np.round(number / interval)



def get_option_bid_price(con, timestamp, strike, expiry_str):
    """Gets the current BID price for a specific option (for selling)."""
    utc_timestamp_str = timestamp.tz_localize('America/New_York').tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
    date_str = timestamp.strftime('%Y-%m-%d')
    
    query = f"""
        SELECT bidPrice
        FROM read_parquet('s3://duckdata/ORATS/Options/ticker=SPY/day={date_str}/*.parquet')
        WHERE 
            ts = '{utc_timestamp_str}' AND
            optionType = 1 AND
            strike = {strike} AND
            expiry = '{expiry_str}';
    """
    price_data = con.execute(query).df()
    return price_data['bidPrice'].iloc[0] if not price_data.empty else None


def get_option_ask_price(con, timestamp, strike, expiry_str):
    """Gets the current ASK price for a specific option (for buying)."""
    utc_timestamp_str = timestamp.tz_localize('America/New_York').tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
    date_str = timestamp.strftime('%Y-%m-%d')
    
    query = f"""
        SELECT askPrice
        FROM read_parquet('s3://duckdata/ORATS/Options/ticker=SPY/day={date_str}/*.parquet')
        WHERE 
            ts = '{utc_timestamp_str}' AND
            optionType = 1 AND
            strike = {strike} AND
            expiry = '{expiry_str}';
    """
    price_data = con.execute(query).df()
    return price_data['askPrice'].iloc[0] if not price_data.empty else None


def fetch_single_option_all_day(con, ticker, date_str, strike, expiry_str, option_type):
    """
    Fetches a full day of minute-level data for a single, specific option contract.

    Args:
        con: The DuckDB connection object.
        ticker (str): The stock ticker (e.g., 'SPY').
        date_str (str): The day to query, in 'YYYY-MM-DD' format.
        strike (float): The specific strike price.
        expiry_str (str): The expiry date to filter for, in 'YYYY-MM-DD' format.
        option_type (int): The type of option to fetch (1 for Calls, 0 for Puts).

    Returns:
        pd.DataFrame: A DataFrame containing the minute-by-minute data for the
                      specified contract, or an empty DataFrame if not found.
    """
    option_name = "Call" if option_type == 1 else "Put"
    print(f"Fetching {date_str} data for {ticker} {strike}-strike {option_name} expiring {expiry_str}...")
    
    try:
        # This optimized query selects only the required columns
        query = f"""
            SELECT 
                ts, strike, expiry, dte, optionType, volume, oi,
                bidPrice, askPrice, bidIv, askIv, stockPrice, ticker
            FROM read_parquet('s3://duckdata/ORATS/Options/ticker={ticker}/day={date_str}/*.parquet')
            WHERE 
                strike = {strike} AND
                expiry = '{expiry_str}' AND
                optionType = {option_type}
            ORDER BY
                ts;
        """
        
        strike_df = con.execute(query).df()
        
        if strike_df.empty:
            print("INFO: No data found for this specific contract.")
        
        return strike_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

# --- Example Usage ---

# Assume 'con' is your active DuckDB connection

# 1. Define the parameters for the contract you want
# target_ticker = 'SPY'
# target_day = '2024-09-06'
# target_strike = 530.0
# target_expiry = '2024-09-06'
# target_option_type = 1 # 1 for Call

# 2. Call the function to get the data
# single_option_df = fetch_single_option_all_day(
#     con=con,
#     ticker=target_ticker,
#     date_str=target_day,
#     strike=target_strike,
#     expiry_str=target_expiry,
#     option_type=target_option_type
# )

# if not single_option_df.empty:
#     print("\n--- Found Data for Single Contract ---")
#     print(single_option_df.head())


def fetch_single_option_at_timestamp(con, ticker, timestamp, strike, expiry_str, option_type , DTE=1):
    
    """
    Fetches data for a single, specific option contract at a single timestamp.

    Args:
        con: The DuckDB connection object.
        ticker (str): The stock ticker (e.g., 'SPY').
        timestamp (pd.Timestamp): The specific timestamp to query (timezone-naive, e.g., EST).
        strike (float): The specific strike price.
        expiry_str (str): The expiry date, in 'YYYY-MM-DD' format.
        option_type (int): The option type (1 for Calls, 0 for Puts).

    Returns:
        pd.DataFrame: A DataFrame with a single row of data, or an empty DataFrame.

    """

    try:
        # Convert the local timestamp to UTC for the query
        utc_timestamp_str = timestamp.tz_localize('America/New_York').tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
        date_str = timestamp.strftime('%Y-%m-%d')

        query = f"""
            SELECT 
                ts, strike, expiry, dte, optionType, volume, oi,close,
                bidPrice, askPrice, bidIv, askIv, iv, stockPrice, ticker
            FROM read_parquet('s3://duckdata/ORATS/Options/ticker={ticker}/day={date_str}/*.parquet')
            WHERE 
                ts = '{utc_timestamp_str}' AND
                strike = {strike} AND
                dte={DTE} AND
                expiry = '{expiry_str}' AND
                optionType = {option_type};
        """
        
        option_data = con.execute(query).df()
        
        return option_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()



def find_straddle_at_timestamp(con, ticker, timestamp, expiry_str, underlying_price, percentage_away):
    """
    Finds the closest matching call and put for a straddle at a single timestamp.

    Args:
        con: The DuckDB connection object.
        ticker (str): The stock ticker (e.g., 'SPY').
        timestamp (pd.Timestamp): The specific timestamp to query (e.g., in EST).
        expiry_str (str): The expiry date for the options, in 'YYYY-MM-DD' format.
        underlying_price (float): The reference stock price to calculate strikes from.
        percentage_away (float): The percentage to calculate strike distance.

    Returns:
        pd.DataFrame: A DataFrame with the data for the two selected options,
                      or an empty DataFrame if not found.
    """
    # date_str = timestamp.strftime('%Y-%m-%d')
    # utc_timestamp_str = timestamp.tz_localize('America/New_York').tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Check if the input timestamp is naive (has no timezone info)
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
        # If it's naive, localize it to New York time
        timestamp = timestamp.tz_localize('America/New_York')
    
    # 2. Now that we're sure it's timezone-aware, we can safely convert to UTC for the query
    date_str = timestamp.strftime('%Y-%m-%d')
    utc_timestamp_str = timestamp.tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
    

    print(f"Finding straddle for {ticker} at {timestamp} with {percentage_away:.2%} offset...")

    try:
        # Calculate target strikes directly in Python
        target_call_strike = underlying_price * (1 + percentage_away)
        target_put_strike = underlying_price * (1 - percentage_away)

        # The query is now faster with the added 'ts' filter
        query = f"""
            SELECT 
                ts, strike, expiry, dte, optionType, volume, oi,
                bidPrice, askPrice, bidIv, askIv, iv, stockPrice, ticker
            FROM read_parquet('s3://duckdata/ORATS/Options/ticker={ticker}/day={date_str}/*.parquet')
            WHERE
                ts = '{utc_timestamp_str}' AND -- <-- KEY CHANGE: Filter for the exact timestamp
                expiry = '{expiry_str}'
            QUALIFY
                ROW_NUMBER() OVER (
                    PARTITION BY optionType 
                    ORDER BY 
                        ABS(strike - CASE 
                                        WHEN optionType = 1 THEN {target_call_strike}
                                        ELSE {target_put_strike}
                                     END)
                ) = 1;
        """
        
        straddle_df = con.execute(query).df()
        
        if straddle_df.empty:
            print("INFO: Could not find matching straddle options at this timestamp.")
        
        straddle_df['ts'] = pd.to_datetime(straddle_df['ts'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None))

        return straddle_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()



def fetch_bulk_option_data(con, ticker, end_date_str, n_days, strikes: List[float], expiries: List[str], option_type):
    
    """
    Fetches a range of minute-level data for MULTIPLE option contracts
    in a single, efficient query.
    """
    # 1. Get the list of business days to query (same as before)
    nyse = mcal.get_calendar('NYSE')
    start_buffer = pd.to_datetime(end_date_str) - pd.Timedelta(days=n_days)
    schedule = nyse.schedule(start_date=start_buffer, end_date=end_date_str)
    business_day_list = [d.strftime('%Y-%m-%d') for d in schedule.index[-n_days:]]
    
    # 2. Build the list of S3 paths (same as before)
    path_list = [
        f"'s3://duckdata/ORATS/Options/ticker={ticker}/day={d}/*.parquet'"
        for d in business_day_list ]
    
    # ▼▼▼ NEW QUERY LOGIC ▼▼▼
    # 3. Format the lists for the SQL 'IN' clause
    strikes_str = ",".join(map(str, strikes)) # For numbers: 470.0,471.0,472.0
    expiries_str = ",".join([f"'{e}'" for e in expiries]) # For strings: '2024-01-05','2024-01-08'
    
    query = f"""
        SELECT ts, strike, expiry, close, bidPrice, askPrice, volume, oi, dte, optionType , iv
        FROM read_parquet([{",".join(path_list)}])
        WHERE 
            CAST(strike AS FLOAT) IN ({strikes_str}) AND
            expiry IN ({expiries_str}) AND
            optionType = {option_type}
        ORDER BY ts;
    """
    
    try:
        bulk_df = con.execute(query).df()
        
        if bulk_df.empty:
            return pd.DataFrame()
            
        # Perform timezone conversion and set index (same as before)
        bulk_df['ts'] = pd.to_datetime(bulk_df['ts'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None))
        bulk_df.set_index('ts' , inplace=True)
        return bulk_df
        
    except Exception as e:
        print(f"An error occurred during bulk fetch: {e}")
        return pd.DataFrame()



# # --- Example Usage ---

# # Assume 'con' is your active DuckDB connection
# # 1. Define the parameters
# target_ticker = 'SPY'
# target_timestamp = pd.to_datetime('2024-09-09 09:30:00') # Specific time
# target_expiry = '2024-09-09'
# reference_price = 540.15 
# percentage_offset = 0.005 

# # 2. Call the function
# straddle_data_df = find_straddle_at_timestamp(
#     con=con,
#     ticker=target_ticker,
#     timestamp=target_timestamp,
#     expiry_str=target_expiry,
#     underlying_price=reference_price,
#     percentage_away=percentage_offset
# )

# if not straddle_data_df.empty:
#     print("\n--- Found Straddle Data at Timestamp ---")
#     print(straddle_data_df)

def check_available_strikes(con, ticker, date_str, expiry_str):
    """Check what strikes exist for a specific expiry date"""
    query = f"""
    SELECT DISTINCT strike 
    FROM read_parquet('s3://duckdata/ORATS/Options/ticker={ticker}/day={date_str}/*.parquet')
    WHERE expiry = '{expiry_str}'
    ORDER BY strike
    """
    
    strikes = con.execute(query).df()
    print(f"Available strikes for {ticker} on {date_str} expiring {expiry_str}:")
    print(strikes)
    return strikes




import pandas as pd
import pandas_market_calendars as mcal

def fetch_option_data_for_n_days(con, ticker, end_date_str, n_days, strike, expiry_str, option_type):
    """
    Fetches a range of minute-level data for a single option contract
    for a specified number of business days ending on a given date.

    Args:
        con: The DuckDB connection object.
        ticker (str): The stock ticker (e.g., 'SPY').
        end_date_str (str): The last day to query, in 'YYYY-MM-DD' format.
        n_days (int): The number of business days of data to fetch.
        strike (float): The specific strike price.
        expiry_str (str): The expiry date to filter for, in 'YYYY-MM-DD' format.
        option_type (int): The type of option to fetch (1 for Calls, 0 for Puts).

    Returns:
        pd.DataFrame: A DataFrame containing the minute-by-minute data.
    """
    option_name = "Call" if option_type == 1 else "Put"
    print(f"Fetching last {n_days} b-days of data ending on {end_date_str} for {ticker} {strike} {option_name} expiring {expiry_str}...")
    
    try:
        # --- Business Day Calculation Logic ---
        # 1. Get the market calendar
        nyse = mcal.get_calendar('NYSE')
        
        # 2. To be safe, find a start date far enough in the past
        #    (n * 2 is a safe buffer for weekends/holidays)
        start_buffer = pd.to_datetime(end_date_str) - pd.Timedelta(days=n_days)
        
        # 3. Generate a schedule of valid trading days
        schedule = nyse.schedule(start_date=start_buffer, end_date=end_date_str)
        
        # 4. Get the last 'n_days' from the schedule and format them
        business_day_list = [d.strftime('%Y-%m-%d') for d in schedule.index[-n_days:]]
        
        # --- Query Logic (remains the same) ---
        path_list = [
            f"'s3://duckdata/ORATS/Options/ticker={ticker}/day={d}/*.parquet'"
            for d in business_day_list
        ]
        
        query = f"""
            SELECT ts, strike, expiry, dte, optionType, bidPrice, askPrice, stockPrice
            FROM read_parquet([{",".join(path_list)}])
            WHERE 
                strike = {strike} AND
                expiry = '{expiry_str}' AND
                optionType = {option_type}
            ORDER BY ts;
        """
        
        multi_day_df = con.execute(query).df()
        
        if multi_day_df.empty:
            print("INFO: No data found for this contract in the specified date range.")
        
        multi_day_df['ts'] = pd.to_datetime(multi_day_df['ts'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None))
        multi_day_df.set_index('ts' , inplace=True)
        return multi_day_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

# ==============================================================================
# --- Example Usage ---

# # Get data for the 5 business days ending on September 12, 2025
# multi_day_df = fetch_option_data_for_n_days(
#     con=con,
#     ticker='SPY',
#     end_date_str='2024-04-12', # The last day of the range
#     n_days=5,                  # How many business days to go back
#     strike=550.0,
#     expiry_str='2024-04-15',
#     option_type=1              # 1 for Call
# )

# if not multi_day_df.empty:
#     print("\n--- Found Data for Date Range ---")
#     print(multi_day_df)





def fetch_market_data_for_range(con, ticker, start_date, end_date, start_time, end_time, option_type=1):
    """
    Generates minute-by-minute timestamps for a date range and fetches market data for each one.

    Args:
        con: The database connection object.
        ticker (str): The ticker symbol to fetch.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        start_time (datetime.time): The start time for the daily filter.
        end_time (datetime.time): The end time for the daily filter.
        option_type (int): The option type to fetch (e.g., 1 for Call).

    Returns:
        Tuple[pd.DataFrame, List]: A tuple containing:
            - A DataFrame of successfully fetched stock prices.
            - A list of timestamps for which data fetching failed.
    """
    # --- Step 1: Generate All Trading Timestamps ---
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    list_of_daily_timestamps = [
        pd.date_range(start=day.market_open, end=day.market_close, freq='1min') 
        for _, day in schedule.iterrows()
    ]
    all_trading_minutes = pd.DatetimeIndex([]).append(list_of_daily_timestamps)

    # --- Step 2: Filter Timestamps to the Desired Window ---
    mask = (all_trading_minutes.tz_convert('America/New_York').time >= start_time) & \
           (all_trading_minutes.tz_convert('America/New_York').time <= end_time)
    filtered_utc_minutes = all_trading_minutes[mask]

    # --- Step 3: Loop and Fetch Data ---
    successful_results = []
    failed_timestamps = set()
    
    for ts in filtered_utc_minutes:
        atm_call_data = fetch_atm_option_at_timestamp(
            con=con,
            ticker=ticker,
            timestamp=ts,
            option_type=option_type
        )
        
        if not atm_call_data.empty:
            successful_results.append({
                'timestamp': atm_call_data.iloc[0]['ts'],
                'stockPrice': atm_call_data.iloc[0]['stockPrice']
            })
        else:
            failed_timestamps.add(ts)

    # --- Step 4: Process and Return Results ---
    if successful_results:
        results_df = pd.DataFrame(successful_results)
        # Convert timestamp column, handle timezone, and set as index
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
        results_df.set_index('timestamp', inplace=True)
    else:
        results_df = pd.DataFrame()
        
    return results_df, sorted(list(failed_timestamps))