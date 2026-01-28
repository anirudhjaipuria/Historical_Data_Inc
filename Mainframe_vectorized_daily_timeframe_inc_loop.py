# ============================================================
# Parallel Multi-Symbol Technical Analysis (CPU-Multiprocessing)
# ============================================================

import os
import warnings
import pandas as pd
from multiprocessing import Pool, cpu_count

from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from CBearDivg_analysis_vectorized import CBearDivg_analysis
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from Trendline_Up_Support_vectorized_inc import calc_TL_Up_Support, update_TL_Up_Support
from Trendline_Up_Resistance_vectorized_inc import calc_TL_Up_Resistance, update_TL_Up_Resistance
from Trendline_Down_Resistance_vectorized_inc import calc_TL_Down_Resistance, update_TL_Down_Resistance
from Goldenratio_vectorized import calculate_golden_ratios
from Support_Resistance_vectorized import calculate_support_levels
from ABC_Correction import calculate_abc_corrections
from ABC_Extension import calculate_abc_extensions


# ------------------------------------------------------------
# Pandas + Warning Settings
# ------------------------------------------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented*")


# ------------------------------------------------------------
# SYMBOL LIST
# ------------------------------------------------------------
SYMBOLS = [
    'BAJFINANC', 'BANKBAROD',  'BANKINDIA', 'DABUR',
    'HDFCBANK', 'HINDUNILV', 'ICICIBANK', 'INDUSINDB', 'INFY', 'ITC',
    'KOTAKBANK', 'LT', 'NOCIL', 'RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN'
]
#
# SYMBOLS = [
#     'ADANIENT', 'AXISBANK', 'BAJFINANC', 'BANKBAROD',  'BANKINDIA', 'DABUR',
#     'HDFCBANK', 'HINDUNILV', 'ICICIBANK', 'INDUSINDB', 'INFY', 'ITC',
#     'KOTAKBANK', 'LT', 'NOCIL', 'RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN'
# ]

# ------------------------------------------------------------
# CORE PROCESSING FUNCTION (ONE SYMBOL = ONE CPU CORE)
# ------------------------------------------------------------
def process_symbol(symbol: str):

    print(f"▶ Starting {symbol}")

    csv_file_path = rf"C:\PYTHON\historical_data\INDIA\{symbol}\{symbol}_1day_candlesticks_all.csv"
    output_dir = rf"C:\PYTHON\historical_data\INDIA\{symbol}\output_1day_parquet"
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    full_df = pd.read_csv(csv_file_path)
    full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce', utc=True)
    full_df = full_df.dropna(subset=['date']).reset_index(drop=True)

    len_df = len(full_df)

    current_df = None

    # Trendline states
    up_support_state = None
    up_resistance_state = None
    down_resistance_state = None

    # --------------------------------------------------------
    # Incremental Walk-Forward Loop
    # --------------------------------------------------------
    for i in range(200, len_df + 1):

        new_row = full_df.iloc[i - 1:i]

        if current_df is None:
            current_df = full_df.head(200).copy()
            current_df['date'] = pd.to_datetime(current_df['date'], utc=True)
        else:
            current_df = pd.concat([current_df, new_row], ignore_index=True)
            current_df['date'] = pd.to_datetime(current_df['date'], utc=True)

        # Indicators & Patterns
        Initialize_RSI_EMA_MACD(current_df)
        Candlestick_Type(current_df)
        Level_1_Max_Min(current_df)

        CBullDivg_analysis(current_df, 0.1, 3.25)
        CBullDivg_x2_analysis(current_df, 0.1, 3.25)
        HBullDivg_analysis(current_df, 0.1, 3.25)
        CBearDivg_analysis(current_df, 0.1, 3.25)
        HBearDivg_analysis(current_df, 0.1, 3.25)

        # Trendlines
        if i == 200:
            current_df, up_support_state = calc_TL_Up_Support(current_df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
            current_df, up_resistance_state = calc_TL_Up_Resistance(current_df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
            current_df, down_resistance_state = calc_TL_Down_Resistance(current_df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
        else:
            current_df, up_support_state = update_TL_Up_Support(current_df, up_support_state, min_gap=20, adjacent_candles=10, exclude_end_points=7)
            current_df, up_resistance_state = update_TL_Up_Resistance(current_df, up_resistance_state, min_gap=20, adjacent_candles=10, exclude_end_points=7)
            current_df, down_resistance_state = update_TL_Down_Resistance(current_df, down_resistance_state, min_gap=20, adjacent_candles=10, exclude_end_points=7)

        # Levels & Projections
        current_df = calculate_support_levels(current_df, lookback_years=25, pivot_threshold=0.25)
        current_df = calculate_golden_ratios(current_df)
        current_df = calculate_abc_corrections(current_df)
        current_df = calculate_abc_extensions(current_df)

        # Save parquet
        last_date = current_df['date'].iloc[-1].strftime('%Y-%m-%d_%H-%M-%S')
        output_file = os.path.join(output_dir, f"output_{last_date}.parquet")
        current_df.to_parquet(output_file, index=False, engine="pyarrow")

    print(f"✔ Finished {symbol}")


# ------------------------------------------------------------
# MULTIPROCESS ENTRY POINT (MANDATORY ON WINDOWS)
# ------------------------------------------------------------
if __name__ == "__main__":

    workers = min(cpu_count(), len(SYMBOLS))
    print(f"\n Running {len(SYMBOLS)} symbols on {workers} CPU cores\n")

    with Pool(processes=workers) as pool:
        pool.map(process_symbol, SYMBOLS)

    print("\n ALL SYMBOLS COMPLETED SUCCESSFULLY\n")
