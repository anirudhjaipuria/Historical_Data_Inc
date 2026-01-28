import numpy as np
import pandas as pd

def calculate_golden_ratios(df):
    # Convert columns to NumPy arrays for efficiency
    highs = df['high'].values
    lows = df['low'].values
    dates = df.get('timestamp', df.get('date')).values  # Use 'timestamp' or 'date'
    n = len(df)
    last_idx = n - 1

    # Longside masks
    maxima_mask_long = df['LM_High_window_2_CS'].values > 0
    minima_mask_long = df['LM_Low_window_1_CS'].values > 0
    # Shortside masks
    maxima_mask_short = df['LM_High_window_1_CS'].values > 0
    minima_mask_short = df['LM_Low_window_2_CS'].values > 0

    # ---------- HELPER FUNCTION: FILTER MINIMA (LONGSIDE) ----------
    def filter_minima_long(start_idx, end_idx):
        minima_indices = np.where(minima_mask_long[start_idx:end_idx + 1])[0] + start_idx
        if len(minima_indices) == 0:
            return np.array([])

        sorted_indices = minima_indices[np.argsort(dates[minima_indices])]
        filtered_indices = []
        previous_low = None

        for idx in sorted_indices[::-1]:  # Process from latest to earliest
            current_low = lows[idx]
            if previous_low is None or current_low < previous_low:
                filtered_indices.append(idx)
                previous_low = current_low

        return np.array(filtered_indices)[np.argsort(filtered_indices)]

    # ---------- HELPER FUNCTION: FILTER MAXIMA (SHORTSIDE) ----------
    def filter_maxima_short(start_idx, end_idx, all_time_high_idx, all_time_high_date):
        maxima_indices = np.where(maxima_mask_short[start_idx:end_idx + 1])[0] + start_idx
        if len(maxima_indices) == 0:
            return np.array([])

        # Include all-time high if within range
        if start_idx <= all_time_high_idx <= end_idx:
            maxima_indices = np.unique(np.append(maxima_indices, all_time_high_idx))

        # Create array of (index, date, price) tuples
        maxima_data = np.array([(idx, dates[idx], highs[idx]) for idx in maxima_indices],
                               dtype=[('idx', int), ('date', dates.dtype), ('price', highs.dtype)])

        # Sort by date (descending)
        maxima_data = maxima_data[np.argsort(maxima_data['date'])[::-1]]

        filtered_indices = []
        previous_price = None
        previous_date = None

        for max_entry in maxima_data:
            idx, current_date, current_price = max_entry['idx'], max_entry['date'], max_entry['price']
            if all_time_high_date is None or current_date >= all_time_high_date:
                if (previous_price is None or
                        (current_price > previous_price and current_date < previous_date)):
                    filtered_indices.append(idx)
                    previous_price = current_price
                    previous_date = current_date

        return np.array(filtered_indices)

    # ---------- INITIALIZE STATE VARIABLES ----------
    # Longside variables
    reference_high = None
    current_retracements_50_long = []
    current_retracements_618_long = []
    previous_maxima_idx_long = None
    previous_highest_high = None
    processed_maxima_long = set()

    # Shortside variables
    reference_low = None
    reference_low_idx = None
    current_retracements_50_short = []
    current_retracements_618_short = []
    previous_minima_idx_short = None
    previous_lowest_low = None
    processed_minima_short = set()

    # ---------- MAIN LOOP ----------
    for end_idx in range(2, last_idx + 1):
        # --- LONGSIDE CALCULATIONS ---
        # Identify new maxima up to end_idx
        sub_indices_long = np.arange(end_idx + 1)
        new_maxima_mask_long = maxima_mask_long[:end_idx + 1] & ~np.isin(sub_indices_long, list(processed_maxima_long))
        new_maxima_indices_long = np.where(new_maxima_mask_long)[0]

        for max_idx in new_maxima_indices_long:
            processed_maxima_long.add(max_idx)
            current_highest_high = highs[max_idx]

            # Determine minima range and retracement initialization
            if reference_high is None or current_highest_high >= reference_high:
                minima_start_idx_long = 0
                new_retracements_50_long = []
                new_retracements_618_long = []
                reference_high = current_highest_high
            else:
                minima_start_idx_long = previous_maxima_idx_long if previous_maxima_idx_long is not None else 0
                new_retracements_50_long = current_retracements_50_long.copy()
                new_retracements_618_long = current_retracements_618_long.copy()

            # Filter minima up to max_idx
            filtered_minima_long = filter_minima_long(minima_start_idx_long, max_idx)

            # Calculate retracements for each minimum
            for min_idx in filtered_minima_long:
                if min_idx >= max_idx:
                    continue
                previous_min = lows[min_idx]
                r50_long = previous_min + (1 - 0.5) * (current_highest_high - previous_min)
                r618_long = previous_min + (1 - 0.618) * (current_highest_high - previous_min)
                if r50_long not in new_retracements_50_long:
                    new_retracements_50_long.append(r50_long)
                if r618_long not in new_retracements_618_long:
                    new_retracements_618_long.append(r618_long)

            # Sort retracements in descending order
            new_retracements_50_long.sort(reverse=True)
            new_retracements_618_long.sort(reverse=True)

            # Update state
            current_retracements_50_long = new_retracements_50_long
            current_retracements_618_long = new_retracements_618_long
            previous_maxima_idx_long = max_idx
            previous_highest_high = current_highest_high

        # Check for higher highs after the last maxima
        if previous_maxima_idx_long is not None and end_idx > previous_maxima_idx_long:
            current_high = highs[end_idx]
            if (previous_highest_high is not None and
                    current_high > previous_highest_high and
                    (reference_high is None or current_high > reference_high)):
                current_highest_idx = end_idx
                current_highest_high = current_high
                reference_high = current_highest_high

                # Use all minima up to current_highest_idx
                filtered_minima_long = filter_minima_long(0, current_highest_idx)

                # Reset retracements
                current_retracements_50_long = []
                current_retracements_618_long = []

                # Calculate new retracements
                for min_idx in filtered_minima_long:
                    if min_idx >= current_highest_idx:
                        continue
                    previous_min = lows[min_idx]
                    r50_long = previous_min + (1 - 0.5) * (current_highest_high - previous_min)
                    r618_long = previous_min + (1 - 0.618) * (current_highest_high - previous_min)
                    if r50_long not in current_retracements_50_long:
                        current_retracements_50_long.append(r50_long)
                    if r618_long not in current_retracements_618_long:
                        current_retracements_618_long.append(r618_long)

                # Sort retracements
                current_retracements_50_long.sort(reverse=True)
                current_retracements_618_long.sort(reverse=True)

                previous_highest_high = current_highest_high

        # Validate retracements for the current candle (longside)
        if current_retracements_50_long:
            current_low = lows[end_idx]
            valid_retracements_50_long = []
            valid_retracements_618_long = []
            for r50, r618 in zip(current_retracements_50_long, current_retracements_618_long):
                if r50 <= current_low:
                    valid_retracements_50_long.append(r50)
                    valid_retracements_618_long.append(r618)

            # Update current retracements
            current_retracements_50_long = valid_retracements_50_long
            current_retracements_618_long = valid_retracements_618_long

            # Reset reference_high if all retracements are violated
            if not current_retracements_50_long and reference_high is not None:
                reference_high = None

        # --- SHORTSIDE CALCULATIONS ---
        # Find all-time high up to end_idx
        current_highs = highs[:end_idx + 1]
        all_time_high_idx = np.argmax(current_highs)
        all_time_high = current_highs[all_time_high_idx]
        all_time_high_date = dates[all_time_high_idx]

        # Identify new minima up to end_idx
        sub_indices_short = np.arange(end_idx + 1)
        new_minima_mask_short = minima_mask_short[:end_idx + 1] & ~np.isin(sub_indices_short, list(processed_minima_short))
        new_minima_indices_short = np.where(new_minima_mask_short)[0]

        for min_idx in new_minima_indices_short:
            processed_minima_short.add(min_idx)
            current_lowest_low = lows[min_idx]

            # Skip if not after all-time high
            is_after_all_time_high = all_time_high_date is None or dates[min_idx] > all_time_high_date
            if not is_after_all_time_high:
                continue

            # Determine maxima range and retracement initialization
            if reference_low is None or current_lowest_low <= reference_low:
                maxima_start_idx_short = 0
                new_retracements_50_short = []
                new_retracements_618_short = []
                reference_low = current_lowest_low
                reference_low_idx = min_idx
            else:
                maxima_start_idx_short = previous_minima_idx_short if previous_minima_idx_short is not None else 0
                new_retracements_50_short = current_retracements_50_short.copy()
                new_retracements_618_short = current_retracements_618_short.copy()

            # Filter maxima up to min_idx - 1
            filtered_maxima_short = filter_maxima_short(maxima_start_idx_short, min_idx - 1, all_time_high_idx, all_time_high_date)

            # Calculate retracements for each maximum
            for max_idx in filtered_maxima_short:
                if max_idx >= min_idx:
                    continue
                previous_max = highs[max_idx]
                r50_short = previous_max - (1 - 0.5) * (previous_max - current_lowest_low)
                r618_short = previous_max - (1 - 0.618) * (previous_max - current_lowest_low)
                if r50_short not in new_retracements_50_short:
                    new_retracements_50_short.append(r50_short)
                if r618_short not in new_retracements_618_short:
                    new_retracements_618_short.append(r618_short)

            # Sort retracements in ascending order
            new_retracements_50_short.sort()
            new_retracements_618_short.sort()

            # Update state
            current_retracements_50_short = new_retracements_50_short
            current_retracements_618_short = new_retracements_618_short
            previous_minima_idx_short = min_idx
            previous_lowest_low = current_lowest_low

        # Check for lower lows after the last minima
        if previous_minima_idx_short is not None and end_idx > previous_minima_idx_short:
            current_low = lows[end_idx]
            is_after_all_time_high = all_time_high_date is None or dates[end_idx] > all_time_high_date
            if (is_after_all_time_high and
                    previous_lowest_low is not None and
                    current_low < previous_lowest_low and
                    (reference_low is None or current_low < reference_low)):
                current_lowest_idx = end_idx
                current_lowest_low = current_low
                reference_low = current_lowest_low
                reference_low_idx = current_lowest_idx

                # Reset retracements
                current_retracements_50_short = []
                current_retracements_618_short = []

                # Filter maxima up to current_lowest_idx - 1
                filtered_maxima_short = filter_maxima_short(0, current_lowest_idx - 1, all_time_high_idx, all_time_high_date)

                # Calculate new retracements
                for max_idx in filtered_maxima_short:
                    if max_idx >= current_lowest_idx:
                        continue
                    previous_max = highs[max_idx]
                    r50_short = previous_max - (1 - 0.5) * (previous_max - current_lowest_low)
                    r618_short = previous_max - (1 - 0.618) * (previous_max - current_lowest_low)
                    if r50_short not in current_retracements_50_short:
                        current_retracements_50_short.append(r50_short)
                    if r618_short not in current_retracements_618_short:
                        current_retracements_618_short.append(r618_short)

                # Sort retracements
                current_retracements_50_short.sort()
                current_retracements_618_short.sort()

                previous_lowest_low = current_lowest_low

        # Validate retracements for the current candle (shortside)
        if current_retracements_50_short:
            current_high = highs[end_idx]
            valid_retracements_50_short = []
            valid_retracements_618_short = []
            for r50, r618 in zip(current_retracements_50_short, current_retracements_618_short):
                if r50 >= current_high:
                    valid_retracements_50_short.append(r50)
                    valid_retracements_618_short.append(r618)

            # Update current retracements
            current_retracements_50_short = valid_retracements_50_short
            current_retracements_618_short = valid_retracements_618_short

            # Reset reference_low if all retracements are violated
            if not current_retracements_50_short and reference_low is not None:
                reference_low = None
                reference_low_idx = None

    # ---------- ADD RETRACEMENTS TO DATAFRAME ----------
    for i in range(1, 26):
        df[f'Fib_50_long_{i}'] = 0.0
        df[f'Fib_618_long_{i}'] = 0.0
        df[f'Fib_50_short_{i}'] = 0.0
        df[f'Fib_618_short_{i}'] = 0.0

    # Assign longside retracements to the last row
    for i, (r50, r618) in enumerate(zip(current_retracements_50_long[:25], current_retracements_618_long[:25]), 1):
        df.at[last_idx, f'Fib_50_long_{i}'] = r50
        df.at[last_idx, f'Fib_618_long_{i}'] = r618

    # Assign shortside retracements to the last row
    for i, (r50, r618) in enumerate(zip(current_retracements_50_short[:25], current_retracements_618_short[:25]), 1):
        df.at[last_idx, f'Fib_50_short_{i}'] = r50
        df.at[last_idx, f'Fib_618_short_{i}'] = r618

    return df
