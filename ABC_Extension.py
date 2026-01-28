# --------------------------------------------------------------
# ABC_Upward_Extension_Strict_No_Higher_A.py
# Upward ABC Extension from ATL
# ALL FILTERS:
#   1. B > ZERO and B < A
#   2. For each ZERO: after first valid A, skip A <= first_A
#   3. NEW: If any Level_1_High_window_1_CS between A and B > A → REJECT ENTIRE PAIR
# --------------------------------------------------------------

import pandas as pd
import numpy as np


def _fmt(idx):
    """Safe index → string (int or Timestamp)"""
    if isinstance(idx, pd.Timestamp):
        return idx.strftime('%Y-%m-%d %H:%M')
    return str(idx)


# def print_point(name, idx, val):
#     print(f"  {name:5}: {_fmt(idx)} | Price: {val:.5f}")


def calculate_abc_extensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function: computes valid upward ABC extensions with strict filters.
    Now outputs C_1_short and C_1618_short levels.
    FIXED: Only stores non-zero positive values, sorted ascending.
    """
    n = len(df)
    if n == 0:
        return df

    # Initialize output columns with _short suffix
    for i in range(1, 11):
        df[f'C_1_short_{i}'] = 0.0
        df[f'C_1618_short_{i}'] = 0.0

    # Find All-Time Low
    atl_idx = df['low'].idxmin()
    mask_after_atl = df.index >= atl_idx

    # Extract signal points
    lows = df.loc[mask_after_atl & (df['Level_1_Low_window_1_CS'] != 0), 'Level_1_Low_window_1_CS']
    highs = df.loc[mask_after_atl & (df['Level_1_High_window_1_CS'] != 0), 'Level_1_High_window_1_CS']

    if lows.empty or highs.empty:
        return df

    low_points = [(idx, val) for idx, val in lows.items()]
    high_points = [(idx, val) for idx, val in highs.items()]

    # Sort by index
    low_points.sort(key=lambda x: x[0])
    high_points.sort(key=lambda x: x[0])

    # Track first valid A per ZERO
    first_a_per_zero = {}

    # All valid configurations
    all_configs = []

    # For every ZERO
    for idx_ZERO, ZERO in low_points:
        a_candidates = [p for p in high_points if p[0] > idx_ZERO]
        if not a_candidates:
            continue

        a_candidates.sort(key=lambda x: x[0])

        for idx_A, A in a_candidates:
            # FILTER 1: Skip if A <= first valid A for this ZERO
            if idx_ZERO in first_a_per_zero and A <= first_a_per_zero[idx_ZERO]:
                continue

            b_candidates = [p for p in low_points if p[0] > idx_A]
            if not b_candidates:
                continue

            found_valid_b = False
            for idx_B, B in b_candidates:
                # FILTER 2: B > ZERO and B < A
                if not (B > ZERO and B < A):
                    continue

                # === FILTER 3: NO Level_1_High_window_1_CS > A between A and B ===
                between_mask = (df.index > idx_A) & (df.index < idx_B)
                if between_mask.any():
                    highs_between = df.loc[between_mask, 'Level_1_High_window_1_CS']
                    nz_highs_between = highs_between[highs_between != 0]
                    if not nz_highs_between.empty and nz_highs_between.max() > A:
                        continue  # REJECT ENTIRE PAIR

                # Compute swing and C levels
                swing = A - ZERO
                C_1_short = B + swing
                C_1618_short = B + 1.618 * swing

                # Final high after B (for breach check)
                after_B_mask = df.index > idx_B
                max_high_after = df.loc[after_B_mask, 'high'].max() if after_B_mask.any() else -np.inf

                levels = {}
                if max_high_after < C_1_short:
                    levels['1'] = C_1_short
                if max_high_after < C_1618_short:
                    levels['1618'] = C_1618_short

                if levels:
                    config = {
                        'idx_ZERO': idx_ZERO,
                        'ZERO': ZERO,
                        'idx_A': idx_A,
                        'A': A,
                        'idx_B': idx_B,
                        'B': B,
                        'max_high_after': max_high_after,
                        'levels': levels
                    }
                    all_configs.append(config)

                    # Record first valid A
                    if idx_ZERO not in first_a_per_zero:
                        first_a_per_zero[idx_ZERO] = A

                    # # Print
                    # print(f"\nVALID ABC PAIR (Strict: No higher A between A-B)")
                    # print_point("ZERO", idx_ZERO, ZERO)
                    # print_point("A",    idx_A,    A)
                    # print_point("B",    idx_B,    B)
                    # if '1' in levels:
                    #     print(f"  C_1  : {levels['1']:.5f}")
                    # if '1618' in levels:
                    #     print(f"  C_1618: {levels['1618']:.5f}")

                    found_valid_b = True

            if found_valid_b:
                continue

    # === ROW-BY-ROW ASSIGNMENT (FIXED: only positive non-zero, sorted ascending) ===
    active = []  # list of live configs

    for end in range(n):
        idx = df.index[end]
        high = df.at[idx, 'high']

        # 1. Activate configs when we pass their B
        for cfg in all_configs:
            if cfg['idx_B'] <= idx and cfg not in active:
                active.append(cfg.copy())

        # 2. Update max_high_after and invalidate breached levels
        to_remove = []
        for cfg in active:
            if idx > cfg['idx_B']:
                cfg['max_high_after'] = max(cfg['max_high_after'], high)
                for k in list(cfg['levels']):
                    if cfg['max_high_after'] >= cfg['levels'][k]:
                        del cfg['levels'][k]
                if not cfg['levels']:
                    to_remove.append(cfg)

        for cfg in to_remove:
            if cfg in active:
                active.remove(cfg)

        # 3. Keep only positive non-zero levels
        for cfg in active:
            cfg['levels'] = {k: v for k, v in cfg['levels'].items() if v > 0}

        # 4. Sort active configs by the smallest remaining level (ascending)
        def sort_key(cfg):
            vals = [v for v in cfg['levels'].values() if v > 0]
            return min(vals) if vals else np.inf

        active.sort(key=sort_key)

        # 5. Assign top 10 active levels to output columns
        for pos, cfg in enumerate(active[:10], start=1):
            if '1' in cfg['levels']:
                df.at[idx, f'C_1_short_{pos}'] = cfg['levels']['1']
            if '1618' in cfg['levels']:
                df.at[idx, f'C_1618_short_{pos}'] = cfg['levels']['1618']

    return df
