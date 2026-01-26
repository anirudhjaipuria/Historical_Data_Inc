import pandas as pd
import warnings
import os
import numpy as np
from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from Trendline_Up_Support_vectorized_inc import calc_TL_Up_Support, update_TL_Up_Support
from Trendline_Up_Resistance_vectorized_inc import calc_TL_Up_Resistance, update_TL_Up_Resistance
from Trendline_Down_Resistance_vectorized_inc import calc_TL_Down_Resistance, update_TL_Down_Resistance
from Goldenratio_vectorized import calculate_golden_ratios
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from CBearDivg_analysis_vectorized import CBearDivg_analysis
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented*')

# Paths
csv_file_path = r'C:\Anirudh\Python\IBKR\Final_Version\CRYPTO\BTC\btc_1day_candlesticks_all.csv'
output_dir = r'C:\Anirudh\Python\New_Layout\output_daily_parquet_inc2'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Read the full CSV once and force timezone-aware UTC
full_df = pd.read_csv(csv_file_path, header=0)
full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce', utc=True)
full_df = full_df.dropna(subset=['date']).reset_index(drop=True)
len_df = len(full_df)
# ────────────────────────────────────────────────────────────────

# Sequential processing
current_df = None

# Separate states for each trendline type
up_support_state = None
up_resistance_state = None
down_resistance_state = None
long_state = None
short_state = None

for i in range(200, len_df + 1):
    new_row = full_df.iloc[i - 1:i]

    if current_df is None:
        current_df = full_df.head(200).copy()
        # Force consistent timezone from the very beginning
        current_df['date'] = pd.to_datetime(current_df['date'], utc=True)
    else:
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        # ─── CRITICAL: Force timezone consistency after EVERY append ───
        current_df['date'] = pd.to_datetime(current_df['date'], utc=True)

    Initialize_RSI_EMA_MACD(current_df)
    Candlestick_Type(current_df)
    Level_1_Max_Min(current_df)
    CBullDivg_analysis(current_df, 0.1, 3.25)
    CBullDivg_x2_analysis(current_df, 0.1, 3.25)
    HBullDivg_analysis(current_df, 0.1, 3.25)
    CBearDivg_analysis(current_df, 0.1, 3.25)
    HBearDivg_analysis(current_df, 0.1, 3.25)
    current_df = calculate_golden_ratios(current_df)

    if i == 200:
        # Initialize all three trendline types
        current_df, up_support_state = calc_TL_Up_Support(current_df, min_gap=20, adjacent_candles=10,
                                                          exclude_end_points=7)
        current_df, up_resistance_state = calc_TL_Up_Resistance(current_df, min_gap=20, adjacent_candles=10,
                                                                exclude_end_points=7)
        current_df, down_resistance_state = calc_TL_Down_Resistance(current_df, min_gap=20, adjacent_candles=10,
                                                                    exclude_end_points=7)
    else:
        # Incremental updates
        current_df, up_support_state = update_TL_Up_Support(current_df, up_support_state, min_gap=20,
                                                            adjacent_candles=10, exclude_end_points=7)
        current_df, up_resistance_state = update_TL_Up_Resistance(current_df, up_resistance_state, min_gap=20,
                                                                  adjacent_candles=10, exclude_end_points=7)
        current_df, down_resistance_state = update_TL_Down_Resistance(current_df, down_resistance_state, min_gap=20,
                                                                      adjacent_candles=10, exclude_end_points=7)

    # Save output with clean, filename-safe date
    last_date = current_df['date'].iloc[-1]
    last_date_clean = last_date.strftime('%Y-%m-%d_%H-%M-%S')  # Example: 2025-01-15_14-30-00
    output_file = os.path.join(output_dir, f'output_{last_date_clean}.parquet')

    current_df.to_parquet(output_file, index=False, engine='pyarrow')
    # current_df.tail(400).to_parquet(output_file, index=False, engine='pyarrow')

print("Processing complete!")