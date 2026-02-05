import pandas as pd
import warnings
import os
import numpy as np
import multiprocessing as mp
from pathlib import Path

# ────────────────────────────────────────────────
#   Your custom module imports
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

# SYMBOLS = ['ADA', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'XRP', 'SOL', 'TRX ', 'LINK', 'XLM']

SYMBOLS = ['ADANIENT', 'AXISBANK', 'BAJFINANC', 'BANKBAROD', 'BANKINDIA', 'DABUR',
           'HDFCBANK', 'HINDUNILV', 'ICICIBANK', 'INDUSINDB', 'INFY', 'ITC',
           'KOTAKBANK', 'LT', 'NOCIL', 'RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN']

# SYMBOLS = ['AAPL', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AMZN', 'AVGO', 'BAC', 'COST', 'CRM',
#            'CSCO', 'CVX', 'DIS', 'GOOGL', 'GS', 'HD', 'JNJ', 'JPM', 'KO', 'LLY',
#            'MA', 'MCD', 'META', 'MRK', 'MSFT', 'NFLX', 'NVDA', 'ORCL', 'PEP', 'PG',
#            'QCOM', 'TSLA', 'UNH', 'V', 'WFC', 'WMT', 'XOM']


def process_chunk(args):
    """
    Process prefix of DataFrame up to index i in a separate process.
    """
    symbol, df_full, output_dir, i = args

    # Create a working copy of the prefix (prevents SettingWithCopyWarning)
    df = df_full.iloc[:i].copy()

    # ────────────────────────────────────────────────
    #   All your indicator / pattern calculations
    # ────────────────────────────────────────────────
    Initialize_RSI_EMA_MACD(df)
    Level_1_Max_Min(df)
    Candlestick_Type(df)

    CBullDivg_analysis(df, 0.01, 3.25)
    CBullDivg_x2_analysis(df, 0.01, 3.25)
    HBullDivg_analysis(df, 0.01, 3.25)
    CBearDivg_analysis(df, 0.01, 3.25)
    HBearDivg_analysis(df, 0.01, 3.25)

    df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
    df = calculate_golden_ratios(df)
    df = calculate_abc_corrections(df)
    # df = calculate_abc_extensions(df)   # ← uncomment when ready

    # Determine output filename using the last date in this chunk
    last_date = df['date'].iloc[-1]
    # Safe string for filename
    if pd.api.types.is_datetime64_any_dtype(last_date):
        last_date_str = last_date.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        last_date_str = str(last_date).replace('/', '-').replace(':', '-').replace(' ', '_')

    output_file = Path(output_dir) / f"output_{last_date_str}.parquet"

    # Save only last 400 rows (as in your original code)
    df.tail(400).to_parquet(output_file, index=False, engine='pyarrow')

    # Optional: print when each chunk finishes (helps debugging)
    # print(f"Completed chunk up to row {i:6d} for {symbol} → {output_file.name}")


if __name__ == '__main__':
    for SYMBOL in SYMBOLS:
        print(f"\n{'=' * 60}")
        print(f"Processing {SYMBOL} ...")
        print(f"{'=' * 60}")

        # ────────────────────────────────────────────────
        #   Paths
        # ────────────────────────────────────────────────
        csv_file_path = rf'C:\PYTHON\historical_data\INDIA\{SYMBOL}\{SYMBOL}_2hour_candlesticks_all.csv'
        output_dir = rf'C:\PYTHON\historical_data\INDIA\{SYMBOL}\output_2hour_parquet'

        os.makedirs(output_dir, exist_ok=True)

        # Read full file once — outside the pool
        print("Reading full CSV ...")
        try:
            df_full = pd.read_csv(csv_file_path, header=0, parse_dates=['date'])
        except Exception as e:
            print(f"ERROR reading {csv_file_path}: {e}")
            continue

        len_df = len(df_full)
        print(f"Total rows: {len_df:,}")

        if len_df < 201:
            print(f" → Not enough rows to process (need ≥ 201), skipping.")
            continue

        # Prepare tasks: process prefixes from row 200 to almost the end
        tasks = [(SYMBOL, df_full, output_dir, i) for i in range(200, len_df - 1)]

        # print(f"Starting {len(tasks):,} parallel chunks...")

        # Adjust number of processes according to your machine
        # num_processes = 20  # ← safe default
        num_processes = mp.cpu_count() # ← max CPUs

        with mp.Pool(processes=num_processes) as pool:
            pool.map(process_chunk, tasks)

        print(f"Finished processing {SYMBOL}\n")

    print("\nAll symbols processed.")
    print("Done.")

