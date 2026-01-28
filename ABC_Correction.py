import pandas as pd
import numpy as np


def detect_abc_v1(ohlc: pd.DataFrame, ath_idx) -> dict:
    """
    Detects ABC points up to the provided DataFrame, starting after ath_idx.
    Returns dict with 'ZERO', 'A', 'B' tuples (index, value) if valid, else None.
    Enforces that B is not greater than ZERO.
    """
    if ohlc.empty or 'high' not in ohlc.columns:
        return None

    mask_after_ATH = ohlc.index >= ath_idx
    if not mask_after_ATH.any():
        return None

    # Find non-zero indices for Level_2_High_window_1_CS after ath_idx
    high_mask = (ohlc['Level_2_High_window_1_CS'] != 0) & mask_after_ATH
    high_indices = ohlc[high_mask].index.tolist()

    if len(high_indices) < 1:
        return None

    # Get last index for ZERO after ath_idx
    idx_ZERO = high_indices[-1]
    ZERO = ohlc.loc[idx_ZERO, 'Level_2_High_window_1_CS']

    # Find first non-zero low after idx_ZERO up to now
    after_ZERO_mask = (ohlc.index > idx_ZERO)
    if not after_ZERO_mask.any():
        return None
    lows_after_ZERO = ohlc.loc[after_ZERO_mask, 'Level_1_Low_window_1_CS']
    non_zero_lows_after = lows_after_ZERO[lows_after_ZERO != 0]

    if non_zero_lows_after.empty:
        return None

    idx_first_low = non_zero_lows_after.index[0]

    # Find non-zero highs after idx_first_low up to now
    after_first_low_mask = (ohlc.index > idx_first_low)
    if not after_first_low_mask.any():
        return None
    highs_after_first_low = ohlc.loc[after_first_low_mask, 'Level_1_High_window_1_CS']
    non_zero_highs_after = highs_after_first_low[highs_after_first_low != 0]

    if non_zero_highs_after.empty:
        return None

    B = non_zero_highs_after.max()
    # Get the index of the first occurrence of B
    idx_B = non_zero_highs_after[non_zero_highs_after == B].index[0]

    # Check if B > ZERO
    if B > ZERO:
        return None

    # Check for higher highs after idx_B up to now
    after_B_mask = (ohlc.index > idx_B)
    if after_B_mask.any():
        highs_after_B = ohlc.loc[after_B_mask, 'high']
        if highs_after_B.max() > B:
            return None

    # Find non-zero lows between idx_ZERO and idx_B (exclusive)
    between_mask = (ohlc.index > idx_ZERO) & (ohlc.index < idx_B)
    if not between_mask.any():
        return None
    lows_between = ohlc.loc[between_mask, 'Level_1_Low_window_1_CS']
    non_zero_lows_between = lows_between[lows_between != 0]

    if non_zero_lows_between.empty:
        return None

    A = non_zero_lows_between.min()
    # Get the index of the first occurrence of A
    idx_A = non_zero_lows_between[non_zero_lows_between == A].index[0]

    # Return points
    return {
        'ZERO': (idx_ZERO, ZERO),
        'A': (idx_A, A),
        'B': (idx_B, B)
    }


def detect_abc_v2(ohlc: pd.DataFrame, ath_idx) -> dict:
    """
    Detects ABC points up to the provided DataFrame, starting after ath_idx.
    Returns dict with 'ZERO', 'A', 'B' tuples (index, value) if valid, else None.
    Enforces that B is not greater than ZERO.
    """
    if ohlc.empty or 'high' not in ohlc.columns:
        return None

    mask_after_ATH = ohlc.index >= ath_idx
    if not mask_after_ATH.any():
        return None

    # Find non-zero indices for Level_2_High_window_1_CS after ath_idx
    high_mask = (ohlc['Level_2_High_window_1_CS'] != 0) & mask_after_ATH
    high_indices = ohlc[high_mask].index.tolist()

    if len(high_indices) < 1:
        return None

    # Get initial index for B
    idx_B = high_indices[-1]
    B = ohlc.loc[idx_B, 'Level_2_High_window_1_CS']

    # Find the max Level_2_High_window_1_CS after ath_idx for ZERO
    idx_ZERO = ohlc.loc[mask_after_ATH, 'Level_2_High_window_1_CS'].idxmax()
    ZERO = ohlc.loc[idx_ZERO, 'Level_2_High_window_1_CS']

    # Check if B > ZERO (initial check, before loop)
    if B > ZERO:
        return None

    # Loop to adjust B if necessary
    while True:
        # Find non-zero lows between idx_ZERO and idx_B (exclusive)
        between_mask = (ohlc.index > idx_ZERO) & (ohlc.index < idx_B)
        if not between_mask.any():
            return None
        lows_between = ohlc.loc[between_mask, 'Level_1_Low_window_1_CS']
        non_zero_lows_between = lows_between[lows_between != 0]

        if non_zero_lows_between.empty:
            return None

        A = non_zero_lows_between.min()
        # Get the index of the first occurrence of A
        idx_A = non_zero_lows_between[non_zero_lows_between == A].index[0]

        # Check if any non-zero LM_High_window_1_CS between idx_A and idx_B is higher than B
        between_A_B_mask = (ohlc.index > idx_A) & (ohlc.index < idx_B)
        if not between_A_B_mask.any():
            break
        lm_highs_between = ohlc.loc[between_A_B_mask, 'LM_High_window_1_CS']
        non_zero_lm_between = lm_highs_between[lm_highs_between != 0]

        if non_zero_lm_between.empty or non_zero_lm_between.max() <= B:
            break

        # Update B to the max LM_High
        max_lm = non_zero_lm_between.max()
        # Get the last index of the max
        idx_new_B = non_zero_lm_between[non_zero_lm_between == max_lm].index[-1]
        idx_B = idx_new_B
        B = max_lm

        # Check if updated B > ZERO
        if B > ZERO:
            return None

    # Check if any high after final idx_B is higher than final B
    after_B_mask = ohlc.index > idx_B
    if after_B_mask.any():
        highs_after_B = ohlc.loc[after_B_mask, 'high']
        if highs_after_B.max() > B:
            return None

    # Return points
    return {
        'ZERO': (idx_ZERO, ZERO),
        'A': (idx_A, A),
        'B': (idx_B, B)
    }


def find_intermediate_low(ohlc: pd.DataFrame) -> dict:
    """
    Starting from the all-time high (ATH), ignores any data before it.
    Then, starting from the last non-zero value of Level_2_High_window_1_CS after ATH (ZERO),
    finds the first non-zero Level_1_Low_window_1_CS after it.
    Then, from there, finds all non-zero Level_1_High_window_1_CS and takes the highest value (B).
    Then, finds the lowest non-zero Level_1_Low_window_1_CS between the indices of ZERO and B (A).
    Calculates the Fibonacci retracement levels (C_1_long, C_1618_long) if conditions are met.
    After B, checks the minimum low value after B's index; if it is <= a C level, that C is not included.
    After B, if any high > B, returns empty.
    Returns a dict with indices and values for ZERO, A, B and valid C levels if found, else None.
    """
    # Find the index of the all-time high
    if ohlc.empty or 'high' not in ohlc.columns:
        return None
    idx_ATH = ohlc['high'].idxmax()

    # Find non-zero indices for Level_1_High_window_1_CS after ATH
    high_mask = (ohlc['Level_1_High_window_1_CS'] != 0) & (ohlc.index >= idx_ATH)
    high_indices = ohlc[high_mask].index.tolist()

    if len(high_indices) < 1:
        return None

    # Get last index for ZERO after ATH
    idx_ZERO = high_indices[-1]
    ZERO = ohlc.loc[idx_ZERO, 'Level_1_High_window_1_CS']

    # Find first non-zero low after idx_ZERO
    after_ZERO_mask = (ohlc.index > idx_ZERO)
    lows_after_ZERO = ohlc.loc[after_ZERO_mask, 'LM_Low_window_1_CS']
    non_zero_lows_after = lows_after_ZERO[lows_after_ZERO != 0]

    if non_zero_lows_after.empty:
        return None

    idx_first_low = non_zero_lows_after.index[0]

    # Find non-zero highs after idx_first_low
    after_first_low_mask = (ohlc.index > idx_first_low)
    highs_after_first_low = ohlc.loc[after_first_low_mask, 'LM_High_window_2_CS']
    non_zero_highs_after = highs_after_first_low[highs_after_first_low != 0]

    if non_zero_highs_after.empty:
        return None

    B = non_zero_highs_after.max()
    # Get the index of the first occurrence of B
    idx_B = non_zero_highs_after[non_zero_highs_after == B].index[0]

    # ADD THIS CHECK
    if B > ZERO:
        return None  # Critical: B must be a lower high

    # Check for higher highs after B
    after_B_mask = (ohlc.index > idx_B)
    if after_B_mask.any():
        highs_after_B = ohlc.loc[after_B_mask, 'high']
        if highs_after_B.max() > B:
            return None

    # Find non-zero lows between idx_ZERO and idx_B (exclusive)
    between_mask = (ohlc.index > idx_ZERO) & (ohlc.index < idx_B)
    lows_between = ohlc.loc[between_mask, 'LM_Low_window_1_CS']
    non_zero_lows_between = lows_between[lows_between != 0]

    if non_zero_lows_between.empty:
        return None

    A = non_zero_lows_between.min()
    # Get the index of the first occurrence of A
    idx_A = non_zero_lows_between[non_zero_lows_between == A].index[0]

    # Calculate Fibonacci levels (renamed)
    C_1_long = B - 1.000 * (ZERO - A)
    C_1618_long = B - 1.618 * (ZERO - A)

    # Find min low after idx_B
    if after_B_mask.any():
        lows_after_B = ohlc.loc[after_B_mask, 'low']
        min_low_after_B = lows_after_B.min()
    else:
        min_low_after_B = np.inf  # No data after, so no breach

    # Prepare levels dictionary
    levels = {}
    if min_low_after_B > C_1_long:
        levels['1_long'] = C_1_long
    if min_low_after_B > C_1618_long:
        levels['1618_long'] = C_1618_long

    # Return points and levels
    return {
        'ZERO': (idx_ZERO, ZERO),
        'A': (idx_A, A),
        'B': (idx_B, B),
        'levels': levels
    }


def calculate_abc_corrections(df):
    """
    Incrementally calculates ABC correction C levels for each datapoint using combined detection logics.
    Maintains up to 10 active configurations, storing valid (not invalidated) C levels
    as C_1_long_1, C_1618_long_1, ..., C_1_long_10, C_1618_long_10.
    For the last row, includes C_1_long, C_1618_long from find_intermediate_low if valid.
    Invalidated if high > B or low <= C level after B.

    FIXED: Only stores positive non-zero values, in DESCENDING order (highest first).
    """
    n = len(df)
    if n == 0:
        return df

    # Initialize columns to 0.0
    for i in range(1, 11):
        df[f'C_1_long_{i}'] = 0.0
        df[f'C_1618_long_{i}'] = 0.0

    # State variables
    current_ath_idx = None
    current_ath_value = -np.inf
    active_configs = []  # List of {'idx_B': idx, 'B': val, 'min_low_after': float, 'levels': {'1_long': val, '1618_long': val}}

    detect_v1 = detect_abc_v1
    detect_v2 = detect_abc_v2
    detect_test = find_intermediate_low

    for end_idx in range(n):
        curr_idx = df.index[end_idx]
        current_high = df['high'].iloc[end_idx]
        current_low = df['low'].iloc[end_idx]

        # Update ATH if necessary
        if current_high > current_ath_value:
            current_ath_value = current_high
            current_ath_idx = curr_idx
            active_configs = []  # Reset all previous configs

        sub_df = df.iloc[:end_idx + 1]

        # Trigger v1 detection on Level_1_High_window_1_CS
        if df['Level_1_High_window_1_CS'].iloc[end_idx] != 0:
            result = detect_v1(sub_df, current_ath_idx)
            if result is not None:
                idx_B, B = result['B']
                # Compute min_low_after up to now
                after_mask = sub_df.index > idx_B
                min_low_after = sub_df.loc[after_mask, 'low'].min() if after_mask.any() else np.inf

                ZERO = result['ZERO'][1]
                A = result['A'][1]
                C_1_long = B - 1.000 * (ZERO - A)
                C_1618_long = B - 1.618 * (ZERO - A)

                levels = {}
                if C_1_long > 0 and min_low_after > C_1_long:
                    levels['1_long'] = C_1_long
                if C_1618_long > 0 and min_low_after > C_1618_long:
                    levels['1618_long'] = C_1618_long

                if levels:
                    new_config = {
                        'idx_B': idx_B,
                        'B': B,
                        'min_low_after': min_low_after,
                        'levels': levels
                    }
                    # Avoid adding duplicates based on idx_B and B value
                    if not any(c['idx_B'] == idx_B and abs(c['B'] - B) < 1e-6 for c in active_configs):
                        active_configs.append(new_config)

        # Trigger v2 detection on Level_2_High_window_1_CS
        if df['Level_2_High_window_1_CS'].iloc[end_idx] != 0:
            result = detect_v2(sub_df, current_ath_idx)
            if result is not None:
                idx_B, B = result['B']
                # Compute min_low_after up to now
                after_mask = sub_df.index > idx_B
                min_low_after = sub_df.loc[after_mask, 'low'].min() if after_mask.any() else np.inf

                ZERO = result['ZERO'][1]
                A = result['A'][1]
                C_1_long = B - 1.000 * (ZERO - A)
                C_1618_long = B - 1.618 * (ZERO - A)

                levels = {}
                if C_1_long > 0 and min_low_after > C_1_long:
                    levels['1_long'] = C_1_long
                if C_1618_long > 0 and min_low_after > C_1618_long:
                    levels['1618_long'] = C_1618_long

                if levels:
                    new_config = {
                        'idx_B': idx_B,
                        'B': B,
                        'min_low_after': min_low_after,
                        'levels': levels
                    }
                    # Avoid adding duplicates based on idx_B and B value
                    if not any(c['idx_B'] == idx_B and abs(c['B'] - B) < 1e-6 for c in active_configs):
                        active_configs.append(new_config)

        # For the last row, apply find_intermediate_low
        if end_idx == n - 1:
            result = detect_test(df)  # Use full DataFrame for final calculation
            if result is not None:
                idx_B, B = result['B']
                after_mask = df.index > idx_B
                min_low_after = df.loc[after_mask, 'low'].min() if after_mask.any() else np.inf
                levels = result.get('levels', {})
                # Filter only positive levels
                levels = {k: v for k, v in levels.items() if v > 0}
                if levels:
                    new_config = {
                        'idx_B': idx_B,
                        'B': B,
                        'min_low_after': min_low_after,
                        'levels': levels
                    }
                    # Avoid adding duplicates based on idx_B and B value
                    if not any(c['idx_B'] == idx_B and abs(c['B'] - B) < 1e-6 for c in active_configs):
                        active_configs.append(new_config)

        # Validate existing configs with current candle
        to_remove = []
        for config in active_configs:
            if curr_idx > config['idx_B']:
                # Check high breach
                if current_high > config['B']:
                    to_remove.append(config)
                    continue
                # Update min low
                config['min_low_after'] = min(config['min_low_after'], current_low)
                # Remove breached levels
                for key in list(config['levels']):
                    if config['min_low_after'] <= config['levels'][key]:
                        del config['levels'][key]
            # Remove if no levels left
            if not config['levels']:
                to_remove.append(config)

        for conf in to_remove:
            if conf in active_configs:
                active_configs.remove(conf)

        # === FILTER & SORT: only positive, descending by level value ===
        for cfg in active_configs:
            cfg['levels'] = {k: v for k, v in cfg['levels'].items() if v > 0}

        # Sort by the HIGHEST remaining level value
        def sort_key(cfg):
            vals = [v for v in cfg['levels'].values() if v > 0]
            return -max(vals) if vals else -np.inf  # descending

        active_configs.sort(key=sort_key)

        # === ASSIGN TOP 10 TO COLUMNS ===
        for j, config in enumerate(active_configs[:10], 1):
            if '1_long' in config['levels']:
                df.at[curr_idx, f'C_1_long_{j}'] = config['levels']['1_long']
            if '1618_long' in config['levels']:
                df.at[curr_idx, f'C_1618_long_{j}'] = config['levels']['1618_long']

    return df
