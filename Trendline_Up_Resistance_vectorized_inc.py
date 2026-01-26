import pandas as pd
import numpy as np
import warnings
from numba import jit, prange
from itertools import combinations

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@jit(nopython=True, parallel=True)
def is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n, start_idx, end_idx):
    """
    Vectorized check for invalid upward resistance trendlines.
    Excludes start and end points from intersection checks.
    Returns True if the trendline is invalid.
    """
    x = np.arange(n)
    trendline_y = slope * x + y_intercept

    # Exclude start and end points from checks
    mask = np.ones(n, dtype=np.bool_)
    mask[start_idx] = False
    mask[end_idx] = False

    # Check for intersections with candle bodies (body above trendline)
    body_intersections = (
        ((opens < trendline_y) & (closes >= trendline_y)) |
        ((opens >= trendline_y) & (closes < trendline_y))
    )[mask]

    # Check for upper wick deviations (high above, close below)
    upper_wick_deviations = ((highs > trendline_y) & (closes <= trendline_y))[mask]

    # Check for candles entirely above the trendline
    above_trendline = (
        (highs > trendline_y) & (lows > trendline_y) &
        (opens > trendline_y) & (closes > trendline_y)
    )[mask]

    return (
        body_intersections.sum() > 0 or
        upper_wick_deviations.sum() > 1 or
        above_trendline.any()
    )

def calc_TL_Up_Resistance(df, min_gap, adjacent_candles, exclude_end_points):
    """
    Full calculation of upward resistance trendlines.
    Returns DataFrame + all valid trendlines (sorted) as state for incremental updates.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)

    # Initialize dummy columns
    first_date = df['date'].iloc[0]
    last_date = df['date'].iloc[-1]
    for i in range(1, 8):
        df[f'TL_U_Resistance_{i}_Start_Date'] = pd.NaT
        df[f'TL_U_Resistance_{i}_End_Date'] = pd.NaT
        df[f'TL_U_Resistance_{i}_Start_Price'] = 0
        df[f'TL_U_Resistance_{i}_End_Price'] = 0
        df[f'TL_U_Resistance_{i}_next_file_Price_projection'] = 0
        df[f'TL_U_Resistance_{i}_Date_Gap'] = 0

    last_row_idx = len(df) - 1
    for i in range(1, 8):
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_Start_Date'] = first_date
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_End_Date'] = last_date
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_Date_Gap'] = 0

    # Cache arrays
    dates = df['date'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    lm_highs = df['LM_High_window_2_CS'].values
    n = len(df)

    # Identify local highs
    local_high_indices = np.where(lm_highs != 0)[0]

    # Generate valid pairs
    pairs = np.array(list(combinations(local_high_indices, 2)))
    valid_pairs = pairs[pairs[:, 1] - pairs[:, 0] >= min_gap]
    valid_pairs = valid_pairs[highs[valid_pairs[:, 1]] >= highs[valid_pairs[:, 0]]]
    valid_pairs = valid_pairs[valid_pairs[:, 1] < n - exclude_end_points]

    # Calculate trendlines
    trendlines = []
    for i in prange(len(valid_pairs)):
        start_idx, end_idx = valid_pairs[i]
        slope = (highs[end_idx] - highs[start_idx]) / (end_idx - start_idx)
        y_intercept = highs[start_idx] - slope * start_idx
        if not is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n, start_idx, end_idx):
            date_gap = end_idx - start_idx
            trendlines.append((start_idx, highs[start_idx], end_idx, highs[end_idx],
                               dates[start_idx], dates[end_idx], date_gap, slope, y_intercept))

    # Sort by start_idx, end_idx for consistent filtering
    trendlines.sort(key=lambda x: (x[0], x[2]))

    # Filtering: keep the lowest (closest to price action) when clustered
    filtered_trendlines = []
    for tl in trendlines:
        start_idx, start_high, end_idx, end_high, start_date, end_date, date_gap, slope, y_intercept = tl
        is_adjacent = False
        adjacent_trendlines = []

        for existing in filtered_trendlines:
            ex_start_idx, ex_start_high, ex_end_idx, ex_end_high, _, _, _, _, _ = existing
            if (abs(start_idx - ex_start_idx) <= adjacent_candles or
                abs(end_idx - ex_end_idx) <= adjacent_candles):
                is_adjacent = True
                adjacent_trendlines.append(existing)

        if is_adjacent:
            all_lower = all(start_high <= atl[1] and end_high <= atl[3] for atl in adjacent_trendlines)
            if all_lower:
                filtered_trendlines = [etl for etl in filtered_trendlines if etl not in adjacent_trendlines]
                filtered_trendlines.append(tl)
        else:
            filtered_trendlines.append(tl)

    # Sort by start date descending and take top 7
    sorted_trendlines = sorted(filtered_trendlines, key=lambda x: x[4], reverse=True)[:7]

    # Assign to last row
    last_row_idx = n - 1
    if sorted_trendlines:
        for idx, tl in enumerate(sorted_trendlines, 1):
            start_idx, start_high, end_idx, end_high, start_date, end_date, date_gap, slope, y_intercept = tl
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_Start_Date'] = pd.Timestamp(start_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_End_Date'] = pd.Timestamp(end_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_Start_Price'] = start_high
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_End_Price'] = end_high
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_Date_Gap'] = date_gap

            if date_gap != 0:
                next_x = n
                projection = slope * next_x + y_intercept
                df.loc[last_row_idx, f'TL_U_Resistance_{idx}_next_file_Price_projection'] = projection

    return df, trendlines  # all valid trendlines (sorted) as state


def update_TL_Up_Resistance(df, previous_all_valid, min_gap, adjacent_candles, exclude_end_points):
    """
    Incremental update for upward resistance trendlines.
    """
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)

    dates = df['date'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    lm_highs = df['LM_High_window_2_CS'].values
    n = len(df)
    old_n = n - 1

    # Revalidate previous valid trendlines
    revalidated = []
    for tl in previous_all_valid:
        start_idx, start_high, end_idx, end_high, start_date, end_date, date_gap, slope, y_intercept = tl
        invalid = is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n, start_idx, end_idx)
        if not invalid:
            revalidated.append((start_idx, start_high, end_idx, end_high, start_date, end_date, date_gap, slope, y_intercept))

    # Generate new trendlines (with buffer)
    buffer = 20
    local_high_indices = np.where(lm_highs != 0)[0]
    potential_new_ends = [idx for idx in local_high_indices if idx >= old_n - buffer]

    new_trendlines = []
    for end_idx in potential_new_ends:
        for start_idx in local_high_indices:
            if start_idx >= end_idx:
                continue
            if end_idx - start_idx < min_gap:
                continue
            if highs[end_idx] < highs[start_idx]:
                continue
            if end_idx >= n - exclude_end_points:
                continue

            slope = (highs[end_idx] - highs[start_idx]) / (end_idx - start_idx)
            y_intercept = highs[start_idx] - slope * start_idx
            invalid = is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n, start_idx, end_idx)
            if not invalid:
                date_gap = end_idx - start_idx
                new_trendlines.append((start_idx, highs[start_idx], end_idx, highs[end_idx],
                                       dates[start_idx], dates[end_idx], date_gap, slope, y_intercept))

    # Combine and sort
    candidates = revalidated + new_trendlines
    candidates.sort(key=lambda x: (x[0], x[2]))

    # Filtering
    filtered_trendlines = []
    for tl in candidates:
        start_idx, start_high, end_idx, end_high, start_date, end_date, date_gap, slope, y_intercept = tl
        is_adjacent = False
        adjacent_trendlines = []

        for existing in filtered_trendlines:
            ex_start_idx, ex_start_high, ex_end_idx, ex_end_high, _, _, _, _, _ = existing
            if (abs(start_idx - ex_start_idx) <= adjacent_candles or
                abs(end_idx - ex_end_idx) <= adjacent_candles):
                is_adjacent = True
                adjacent_trendlines.append(existing)

        if is_adjacent:
            all_lower = all(start_high <= atl[1] and end_high <= atl[3] for atl in adjacent_trendlines)
            if all_lower:
                filtered_trendlines = [etl for etl in filtered_trendlines if etl not in adjacent_trendlines]
                filtered_trendlines.append(tl)
        else:
            filtered_trendlines.append(tl)

    # Sort and take top 7
    sorted_trendlines = sorted(filtered_trendlines, key=lambda x: x[4], reverse=True)[:7]

    # Assign to last row
    last_row_idx = n - 1
    first_date = df['date'].iloc[0]
    last_date = df['date'].iloc[-1]
    for i in range(1, 8):
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_Start_Date'] = first_date
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_End_Date'] = last_date
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_Start_Price'] = 0
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_End_Price'] = 0
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_next_file_Price_projection'] = 0
        df.loc[last_row_idx, f'TL_U_Resistance_{i}_Date_Gap'] = 0

    if sorted_trendlines:
        for idx, tl in enumerate(sorted_trendlines, 1):
            start_idx, start_high, end_idx, end_high, start_date, end_date, date_gap, slope, y_intercept = tl
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_Start_Date'] = pd.Timestamp(start_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_End_Date'] = pd.Timestamp(end_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_Start_Price'] = start_high
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_End_Price'] = end_high
            df.loc[last_row_idx, f'TL_U_Resistance_{idx}_Date_Gap'] = date_gap

            if date_gap != 0:
                next_x = n
                projection = slope * next_x + y_intercept
                df.loc[last_row_idx, f'TL_U_Resistance_{idx}_next_file_Price_projection'] = projection

    return df, candidates