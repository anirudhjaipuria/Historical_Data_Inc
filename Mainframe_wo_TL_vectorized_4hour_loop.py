import pandas as pd
import warnings
import os
import numpy as np
import multiprocessing as mp
import uuid
import tempfile
from pathlib import Path

# ────────────────────────────────────────────────
#   Your imports (assuming they exist)
# ────────────────────────────────────────────────
from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from CBearDivg_analysis_vectorized import CBearDivg_analysis
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from Goldenratio_vectorized import calculate_golden_ratios
from Support_Resistance_vectorized import calculate_support_levels
from ABC_Correction import calculate_abc_corrections
from ABC_Extension import calculate_abc_extensions

# Pandas / warnings setup
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented*')

# ────────────────────────────────────────────────
#   SYMBOLS to process
# ────────────────────────────────────────────────
SYMBOLS = ['ADA', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'XRP', 'SOL', 'TRX ', 'LINK']

# SYMBOLS = ['ADANIENT', 'AXISBANK', 'BAJFINANC', 'BANKBAROD', 'BANKINDIA', 'DABUR',
#            'HDFCBANK', 'HINDUNILV', 'ICICIBANK', 'INDUSINDB', 'INFY', 'ITC',
#            'KOTAKBANK', 'LT', 'NOCIL', 'RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN']

# SYMBOLS = ['AAPL', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AMZN', 'AVGO', 'BAC', 'COST', 'CRM',
#            'CSCO', 'CVX', 'DIS', 'GOOGL', 'GS', 'HD', 'JNJ', 'JPM', 'KO', 'LLY',
#            'MA', 'MCD', 'META', 'MRK', 'MSFT', 'NFLX', 'NVDA', 'ORCL', 'PEP', 'PG',
#            'QCOM', 'TSLA', 'UNH', 'V', 'WFC', 'WMT', 'XOM']

def process_chunk(args):
    """
    Process one chunk (up to index i) in its own isolated temporary directory.
    """
    symbol, csv_path, output_dir, i = args

    # Create a unique temporary directory for THIS process
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_parquet = Path(tmpdirname) / f"chunk_{i}.parquet"

        # Read only first i rows and save to unique temp file
        df_chunk = pd.read_csv(csv_path, header=0, parse_dates=['date']).head(i)
        df_chunk.to_parquet(temp_parquet, index=False, engine='pyarrow')

        # Load it back (optional – could also work directly on df_chunk)
        df = pd.read_parquet(temp_parquet)

        # ────────────────────────────────────────────────
        #   All your indicator calculations
        # ────────────────────────────────────────────────
        Initialize_RSI_EMA_MACD(df)
        Level_1_Max_Min(df)
        Candlestick_Type(df)
        CBullDivg_analysis(df, 0.05, 3.25)
        CBullDivg_x2_analysis(df, 0.05, 3.25)
        HBullDivg_analysis(df, 0.05, 3.25)
        CBearDivg_analysis(df, 0.05, 3.25)
        HBearDivg_analysis(df, 0.05, 3.25)

        df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
        df = calculate_golden_ratios(df)
        df = calculate_abc_corrections(df)
        # df = calculate_abc_extensions(df)

        # Save final result with last date in filename
        last_date = df['date'].iloc[-1]
        last_date_str = str(last_date).replace('/', '-').replace(':', '-').replace(' ', '_')
        output_file = Path(output_dir) / f"output_{last_date_str}.parquet"

        # df.to_parquet(output_file, index=False, engine='pyarrow')
        df.tail(400).to_parquet(output_file, index=False, engine='pyarrow')

        # Temporary directory + file will be automatically deleted when leaving the with-block


if __name__ == '__main__':
    for SYMBOL in SYMBOLS:
        print(f"\nProcessing {SYMBOL} ...")

        # ────────────────────────────────────────────────
        #   Paths
        # ────────────────────────────────────────────────
        csv_file_path = rf'C:\PYTHON\historical_data\CRYPTO\{SYMBOL}\{SYMBOL}_4hour_candlesticks_all.csv'
        output_dir    = rf'C:\PYTHON\historical_data\CRYPTO\{SYMBOL}\output_4hour_parquet'

        os.makedirs(output_dir, exist_ok=True)

        # Read full file once to know length
        df_full = pd.read_csv(csv_file_path, header=0, parse_dates=['date'])
        len_df = len(df_full)

        # Prepare arguments for each chunk
        tasks = [(SYMBOL, csv_file_path, output_dir, i) for i in range(200, len_df - 1)]

        # Run in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
        # with mp.Pool(processes=24) as pool:
            pool.map(process_chunk, tasks)

        print(f"Finished processing {SYMBOL}\n")


    print("All symbols processed.")



