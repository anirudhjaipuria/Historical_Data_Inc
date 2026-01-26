import pandas as pd
import numpy as np
import warnings
from numba import jit
from itertools import combinations

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@jit(nopython=True, parallel=True)
def is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n):
    """
    Vectorized check for invalid trendlines using NumPy operations.
    Returns (invalid: bool, wick_count: int)
    """
    x = np.arange(n)
    trendline_y = slope * x + y_intercept

    # Check for intersections with candle bodies
    body_intersections = (
        ((opens < trendline_y) & (closes > trendline_y)) |
        ((opens > trendline_y) & (closes < trendline_y))
    )

    # Check for intersections with lower wicks
    lower_wick_intersections = (closes > trendline_y) & (lows < trendline_y)

    # Check for candles entirely below the trendline
    below_trendline = (
        (highs < trendline_y) & (lows < trendline_y) &
        (opens < trendline_y) & (closes < trendline_y)
    )

    body_sum = body_intersections.sum()
    wick_sum = lower_wick_intersections.sum()
    below_any = below_trendline.any()

    invalid = (wick_sum > 1) or (body_sum > 0) or below_any
    return invalid, wick_sum

def calc_TL_Up_Support(df, min_gap, adjacent_candles, exclude_end_points):
    """
    Calculate upward trendlines for the entire DataFrame up to the last row and add their coordinates and dates to the last row.
    If no trendlines are found, populate with dummy values (0 for prices, first/last dates for dates).
    Uses LM_Low_window_2_CS for local lows.

    Parameters:
    - df: pandas DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'LM_Low_window_2_CS']
    - min_gap: Minimum gap between candles for trendline points
    - adjacent_candles: Number of adjacent candles to check for clustering
    - exclude_end_points: Number of data points to exclude from the end for the second low

    Returns:
    - df: Updated DataFrame
    - all_valid: List of all valid trendline tuples (sorted), for incremental updates
    """
    # Make a copy and ensure date is datetime, sort by date, and reset index
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)

    # Initialize trendline columns with dummy values
    first_date = df['date'].iloc[0]
    last_date = df['date'].iloc[-1]
    for i in range(1, 8):
        df[f'TL_U_Support_{i}_Start_Date'] = pd.NaT
        df[f'TL_U_Support_{i}_End_Date'] = pd.NaT
        df[f'TL_U_Support_{i}_Start_Price'] = 0
        df[f'TL_U_Support_{i}_End_Price'] = 0
        df[f'TL_U_Support_{i}_next_file_Price_projection'] = 0
        df[f'TL_U_Support_{i}_Date_Gap'] = 0

    # Set dummy values in the last row
    last_row_idx = len(df) - 1
    for i in range(1, 8):
        df.loc[last_row_idx, f'TL_U_Support_{i}_Start_Date'] = first_date
        df.loc[last_row_idx, f'TL_U_Support_{i}_End_Date'] = last_date

    # Cache DataFrame columns as NumPy arrays for faster access
    dates = df['date'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    lm_lows = df['LM_Low_window_2_CS'].values
    n = len(df)

    # Identify indices of local lows
    local_low_indices = np.where(lm_lows != 0)[0]

    # Generate all possible pairs of local lows with minimum gap
    pairs = list(combinations(local_low_indices, 2))
    valid_pairs = [p for p in pairs if p[1] - p[0] >= min_gap]

    # Further filter pairs where second low is not lower (upward trendlines)
    valid_pairs = [p for p in valid_pairs if lows[p[1]] >= lows[p[0]]]

    # Exclude pairs where the second point is within exclude_end_points from the end
    valid_pairs = [p for p in valid_pairs if p[1] < n - exclude_end_points]

    # Calculate slopes and y-intercepts for all valid pairs
    trendlines = []
    for start_idx, end_idx in valid_pairs:
        slope = (lows[end_idx] - lows[start_idx]) / (end_idx - start_idx)
        y_intercept = lows[start_idx] - slope * start_idx
        invalid, wick_count = is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n)
        if not invalid:
            date_gap = end_idx - start_idx
            trendlines.append((start_idx, lows[start_idx], end_idx, lows[end_idx], dates[start_idx],
                               dates[end_idx], date_gap, slope, y_intercept, wick_count))

    # Sort trendlines by start_idx asc, end_idx asc for consistent processing order
    trendlines.sort(key=lambda x: (x[0], x[2]))

    # Filter trendlines with adjacent lows
    filtered_trendlines = []
    for tl in trendlines:
        start_idx, start_low, end_idx, end_low, start_date, end_date, date_gap, slope, y_intercept, wick_count = tl
        is_adjacent = False
        adjacent_trendlines = []

        # Check for adjacency with existing filtered trendlines
        for existing in filtered_trendlines:
            ex_start_idx, ex_start_low, ex_end_idx, ex_end_low, _, _, _, _, _, _ = existing
            if (abs(start_idx - ex_start_idx) <= adjacent_candles or
                    abs(end_idx - ex_end_idx) <= adjacent_candles):
                is_adjacent = True
                adjacent_trendlines.append(existing)

        if is_adjacent:
            # Replace if current trendline has lower lows
            all_lower = all(start_low <= atl[1] and end_low <= atl[3] for atl in adjacent_trendlines)
            if all_lower:
                filtered_trendlines = [
                    etl for etl in filtered_trendlines if etl not in adjacent_trendlines
                ]
                filtered_trendlines.append(tl)
        else:
            filtered_trendlines.append(tl)

    # Sort trendlines by start date (most recent first) and take up to 7
    sorted_trendlines = sorted(filtered_trendlines, key=lambda x: x[4], reverse=True)[:7]

    # Assign trendline data to the last row
    last_row_idx = n - 1
    if sorted_trendlines:
        for idx, tl in enumerate(sorted_trendlines, 1):
            start_idx, start_low, end_idx, end_low, start_date, end_date, date_gap, slope, y_intercept, wick_count = tl
            # Store as timezone-naive datetime objects
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Start_Date'] = pd.Timestamp(start_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Support_{idx}_End_Date'] = pd.Timestamp(end_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Start_Price'] = start_low
            df.loc[last_row_idx, f'TL_U_Support_{idx}_End_Price'] = end_low
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Date_Gap'] = date_gap

            # Calculate projection for next file (assuming next index n)
            if date_gap != 0:  # Avoid division by zero
                next_x = n
                projection = slope * next_x + y_intercept
                df.loc[last_row_idx, f'TL_U_Support_{idx}_next_file_Price_projection'] = projection

    return df, trendlines  # Return sorted all valid as state

def update_TL_Up_Support(df, previous_all_valid, min_gap, adjacent_candles, exclude_end_points):
    """
    Incremental update for upward trendlines when a new row is added.
    """
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)

    # Cache arrays
    dates = df['date'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    lm_lows = df['LM_Low_window_2_CS'].values
    n = len(df)
    old_n = n - 1  # Assuming one row added

    # Revalidate previous all valid trendlines with full check on current data
    revalidated = []
    for tl in previous_all_valid:
        start_idx, start_low, end_idx, end_low, start_date, end_date, date_gap, slope, y_intercept, _ = tl
        invalid, wick_count = is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n)
        if not invalid:
            new_tl = (start_idx, start_low, end_idx, end_low, start_date, end_date, date_gap, slope, y_intercept, wick_count)
            revalidated.append(new_tl)

    # Identify all current local lows
    local_low_indices = np.where(lm_lows != 0)[0]

    # Find potential new ends with buffer
    buffer = 20
    potential_new_ends = [idx for idx in local_low_indices if idx >= old_n - buffer]

    # Generate new trendlines
    new_trendlines = []
    for end_idx in potential_new_ends:
        for start_idx in local_low_indices:
            if start_idx >= end_idx:
                continue
            if end_idx - start_idx < min_gap:
                continue
            if lows[end_idx] < lows[start_idx]:
                continue
            if end_idx >= n - exclude_end_points:
                continue

            slope = (lows[end_idx] - lows[start_idx]) / (end_idx - start_idx)
            y_intercept = lows[start_idx] - slope * start_idx
            invalid, wick_count = is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n)
            if not invalid:
                date_gap = end_idx - start_idx
                new_trendlines.append((start_idx, lows[start_idx], end_idx, lows[end_idx], dates[start_idx],
                                       dates[end_idx], date_gap, slope, y_intercept, wick_count))

    # Combine revalidated and new as candidates
    candidates = revalidated + new_trendlines

    # Sort candidates by start_idx asc, end_idx asc for consistent processing order
    candidates.sort(key=lambda x: (x[0], x[2]))

    # Re-apply filtering to the sorted candidates
    filtered_trendlines = []
    for tl in candidates:
        start_idx, start_low, end_idx, end_low, start_date, end_date, date_gap, slope, y_intercept, wick_count = tl
        is_adjacent = False
        adjacent_trendlines = []

        # Check for adjacency with existing filtered trendlines
        for existing in filtered_trendlines:
            ex_start_idx, ex_start_low, ex_end_idx, ex_end_low, _, _, _, _, _, _ = existing
            if (abs(start_idx - ex_start_idx) <= adjacent_candles or
                    abs(end_idx - ex_end_idx) <= adjacent_candles):
                is_adjacent = True
                adjacent_trendlines.append(existing)

        if is_adjacent:
            # Replace if current trendline has lower lows
            all_lower = all(start_low <= atl[1] and end_low <= atl[3] for atl in adjacent_trendlines)
            if all_lower:
                filtered_trendlines = [
                    etl for etl in filtered_trendlines if etl not in adjacent_trendlines
                ]
                filtered_trendlines.append(tl)
        else:
            filtered_trendlines.append(tl)

    # Sort for assignment (most recent first) and take up to 7
    sorted_trendlines = sorted(filtered_trendlines, key=lambda x: x[4], reverse=True)[:7]

    # Assign to last row (reset dummies first)
    last_row_idx = n - 1
    first_date = df['date'].iloc[0]
    last_date = df['date'].iloc[-1]
    for i in range(1, 8):
        df.loc[last_row_idx, f'TL_U_Support_{i}_Start_Date'] = first_date
        df.loc[last_row_idx, f'TL_U_Support_{i}_End_Date'] = last_date
        df.loc[last_row_idx, f'TL_U_Support_{i}_Start_Price'] = 0
        df.loc[last_row_idx, f'TL_U_Support_{i}_End_Price'] = 0
        df.loc[last_row_idx, f'TL_U_Support_{i}_next_file_Price_projection'] = 0
        df.loc[last_row_idx, f'TL_U_Support_{i}_Date_Gap'] = 0

    if sorted_trendlines:
        for idx, tl in enumerate(sorted_trendlines, 1):
            start_idx, start_low, end_idx, end_low, start_date, end_date, date_gap, slope, y_intercept, wick_count = tl
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Start_Date'] = pd.Timestamp(start_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Support_{idx}_End_Date'] = pd.Timestamp(end_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Start_Price'] = start_low
            df.loc[last_row_idx, f'TL_U_Support_{idx}_End_Price'] = end_low
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Date_Gap'] = date_gap

            if date_gap != 0:
                next_x = n
                projection = slope * next_x + y_intercept
                df.loc[last_row_idx, f'TL_U_Support_{idx}_next_file_Price_projection'] = projection

    return df, candidates