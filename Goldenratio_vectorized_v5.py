import numpy as np
import pandas as pd


def calculate_golden_ratios(df):
    highs  = df['high'].values
    lows   = df['low'].values
    dates  = df.get('timestamp', df.get('date')).values
    n      = len(df)
    last_idx = n - 1

    # Precompute masks once
    maxima_mask_long  = df['LM_High_window_2_CS'].values > 0
    minima_mask_long  = df['LM_Low_window_1_CS'].values  > 0
    maxima_mask_short = df['LM_High_window_1_CS'].values > 0
    minima_mask_short = df['LM_Low_window_2_CS'].values  > 0

    # ── OPT 1: Boolean processed-masks instead of set + np.isin ────────────
    processed_long  = np.zeros(n, dtype=bool)
    processed_short = np.zeros(n, dtype=bool)

    # ── OPT 2: All-time-high tracked incrementally ──────────────────────────
    # Loop startet bei end_idx=2, daher highs[0] und highs[1] vorab prüfen
    ath_idx  = int(np.argmax(highs[:2]))   # erster echter ATH aus Index 0 und 1
    ath_val  = highs[ath_idx]
    ath_date = dates[ath_idx]

    # ── HELPER: filter minima (longside) ────────────────────────────────────
    def filter_minima_long(start_idx, end_idx):
        seg = minima_mask_long[start_idx:end_idx + 1]
        if not seg.any():
            return np.array([], dtype=int)
        indices = np.where(seg)[0] + start_idx
        # Sort by date descending, keep only strictly decreasing lows
        order   = np.argsort(dates[indices], kind='stable')[::-1]
        sorted_i = indices[order]
        filtered = []
        prev_low = None
        for idx in sorted_i:
            v = lows[idx]
            if prev_low is None or v < prev_low:
                filtered.append(idx)
                prev_low = v
        return np.sort(filtered)

    # ── HELPER: filter maxima (shortside) ───────────────────────────────────
    # ath_idx_at / ath_date_at: ATH-Stand zum Zeitpunkt des jeweiligen end_idx –
    # exakt wie im Original per Parameter übergeben, nicht aus dem Closure
    def filter_maxima_short(start_idx, end_idx, ath_idx_at, ath_date_at):
        seg = maxima_mask_short[start_idx:end_idx + 1]
        indices = np.where(seg)[0] + start_idx
        # WICHTIG: Längenprüfung VOR ATH-Insert, exakt wie im Original –
        # wenn keine Masken-Treffer, wird ATH ebenfalls NICHT eingefügt
        if len(indices) == 0:
            return np.array([], dtype=int)
        if start_idx <= ath_idx_at <= end_idx:
            indices = np.unique(np.append(indices, ath_idx_at))

        d_arr = dates[indices]
        h_arr = highs[indices]
        order = np.argsort(d_arr, kind='stable')[::-1]
        sorted_i = indices[order]
        sorted_d = d_arr[order]
        sorted_h = h_arr[order]

        filtered = []
        prev_price = None
        prev_date  = None
        for idx, cur_d, cur_h in zip(sorted_i, sorted_d, sorted_h):
            if ath_date_at is None or cur_d >= ath_date_at:
                if prev_price is None or (cur_h > prev_price and cur_d < prev_date):
                    filtered.append(idx)
                    prev_price = cur_h
                    prev_date  = cur_d
        return np.array(filtered)

    # ── OPT 3: Retracement sets instead of list membership check ───────────
    def add_retracements(r50_list, r618_list, r50_set, r618_set,
                         prev_min, highest_high):
        r50  = prev_min + 0.5   * (highest_high - prev_min)
        r618 = prev_min + 0.382 * (highest_high - prev_min)
        if r50 not in r50_set:
            r50_list.append(r50);  r50_set.add(r50)
        if r618 not in r618_set:
            r618_list.append(r618); r618_set.add(r618)

    # ── STATE ────────────────────────────────────────────────────────────────
    # Longside
    reference_high          = None
    cur_ret50_long          = [];  cur_ret50_long_set  = set()
    cur_ret618_long         = [];  cur_ret618_long_set = set()
    prev_maxima_idx_long    = None
    prev_highest_high       = None

    # Shortside
    reference_low           = None
    reference_low_idx       = None
    cur_ret50_short         = [];  cur_ret50_short_set  = set()
    cur_ret618_short        = [];  cur_ret618_short_set = set()
    prev_minima_idx_short   = None
    prev_lowest_low         = None

    # ── MAIN LOOP ────────────────────────────────────────────────────────────
    for end_idx in range(2, last_idx + 1):

        # ── OPT 2: update ATH incrementally ─────────────────────────────────
        if highs[end_idx] > ath_val:
            ath_val  = highs[end_idx]
            ath_idx  = end_idx
            ath_date = dates[end_idx]

        # ════════════════════════════════════════════════════════════════════
        # LONGSIDE
        # ════════════════════════════════════════════════════════════════════
        # OPT 1: find new (unprocessed) maxima without np.isin / set→list
        new_max_long = np.where(
            maxima_mask_long[:end_idx + 1] & ~processed_long[:end_idx + 1]
        )[0]

        for max_idx in new_max_long:
            processed_long[max_idx] = True
            cur_high = highs[max_idx]

            if reference_high is None or cur_high >= reference_high:
                min_start = 0
                new_r50  = [];  new_r50_set  = set()
                new_r618 = [];  new_r618_set = set()
                reference_high = cur_high
            else:
                min_start = prev_maxima_idx_long if prev_maxima_idx_long is not None else 0
                new_r50  = cur_ret50_long.copy();  new_r50_set  = cur_ret50_long_set.copy()
                new_r618 = cur_ret618_long.copy(); new_r618_set = cur_ret618_long_set.copy()

            for min_idx in filter_minima_long(min_start, max_idx):
                if min_idx >= max_idx:
                    continue
                add_retracements(new_r50, new_r618, new_r50_set, new_r618_set,
                                 lows[min_idx], cur_high)

            new_r50.sort(reverse=True);  new_r618.sort(reverse=True)
            cur_ret50_long  = new_r50;   cur_ret50_long_set  = new_r50_set
            cur_ret618_long = new_r618;  cur_ret618_long_set = new_r618_set
            prev_maxima_idx_long = max_idx
            prev_highest_high    = cur_high

        # Check for higher-high candle (not a labeled maxima)
        if (prev_maxima_idx_long is not None and end_idx > prev_maxima_idx_long
                and highs[end_idx] > (prev_highest_high or -np.inf)
                and (reference_high is None or highs[end_idx] > reference_high)):
            cur_high = highs[end_idx]
            reference_high = cur_high
            cur_ret50_long  = [];  cur_ret50_long_set  = set()
            cur_ret618_long = [];  cur_ret618_long_set = set()
            for min_idx in filter_minima_long(0, end_idx):
                if min_idx >= end_idx:
                    continue
                add_retracements(cur_ret50_long, cur_ret618_long,
                                 cur_ret50_long_set, cur_ret618_long_set,
                                 lows[min_idx], cur_high)
            cur_ret50_long.sort(reverse=True);  cur_ret618_long.sort(reverse=True)
            prev_highest_high = cur_high

        # Validate longside retracements
        if cur_ret50_long:
            cur_low = lows[end_idx]
            valid50 = []; valid618 = []; v50_set = set(); v618_set = set()
            for r50, r618 in zip(cur_ret50_long, cur_ret618_long):
                if r50 <= cur_low:
                    valid50.append(r50);  v50_set.add(r50)
                    valid618.append(r618); v618_set.add(r618)
            cur_ret50_long  = valid50;  cur_ret50_long_set  = v50_set
            cur_ret618_long = valid618; cur_ret618_long_set = v618_set
            if not cur_ret50_long:
                reference_high = None

        # ════════════════════════════════════════════════════════════════════
        # SHORTSIDE
        # ════════════════════════════════════════════════════════════════════
        new_min_short = np.where(
            minima_mask_short[:end_idx + 1] & ~processed_short[:end_idx + 1]
        )[0]

        for min_idx in new_min_short:
            processed_short[min_idx] = True
            cur_low = lows[min_idx]

            is_after_ath = ath_date is None or dates[min_idx] > ath_date
            if not is_after_ath:
                continue

            if reference_low is None or cur_low <= reference_low:
                max_start = 0
                new_r50  = [];  new_r50_set  = set()
                new_r618 = [];  new_r618_set = set()
                reference_low     = cur_low
                reference_low_idx = min_idx
            else:
                max_start = prev_minima_idx_short if prev_minima_idx_short is not None else 0
                new_r50  = cur_ret50_short.copy();  new_r50_set  = cur_ret50_short_set.copy()
                new_r618 = cur_ret618_short.copy(); new_r618_set = cur_ret618_short_set.copy()

            for max_idx in filter_maxima_short(max_start, min_idx - 1, ath_idx, ath_date):
                if max_idx >= min_idx:
                    continue
                prev_max = highs[max_idx]
                r50  = prev_max - 0.5   * (prev_max - cur_low)
                r618 = prev_max - 0.382 * (prev_max - cur_low)
                if r50 not in new_r50_set:
                    new_r50.append(r50);   new_r50_set.add(r50)
                if r618 not in new_r618_set:
                    new_r618.append(r618); new_r618_set.add(r618)

            new_r50.sort();   new_r618.sort()
            cur_ret50_short  = new_r50;  cur_ret50_short_set  = new_r50_set
            cur_ret618_short = new_r618; cur_ret618_short_set = new_r618_set
            prev_minima_idx_short = min_idx
            prev_lowest_low       = cur_low

        # Check for lower-low candle (not a labeled minima)
        if (prev_minima_idx_short is not None and end_idx > prev_minima_idx_short
                and (ath_date is None or dates[end_idx] > ath_date)
                and lows[end_idx] < (prev_lowest_low or np.inf)
                and (reference_low is None or lows[end_idx] < reference_low)):
            cur_low = lows[end_idx]
            reference_low     = cur_low
            reference_low_idx = end_idx
            cur_ret50_short  = [];  cur_ret50_short_set  = set()
            cur_ret618_short = [];  cur_ret618_short_set = set()
            for max_idx in filter_maxima_short(0, end_idx - 1, ath_idx, ath_date):
                if max_idx >= end_idx:
                    continue
                prev_max = highs[max_idx]
                r50  = prev_max - 0.5   * (prev_max - cur_low)
                r618 = prev_max - 0.382 * (prev_max - cur_low)
                if r50 not in cur_ret50_short_set:
                    cur_ret50_short.append(r50);   cur_ret50_short_set.add(r50)
                if r618 not in cur_ret618_short_set:
                    cur_ret618_short.append(r618); cur_ret618_short_set.add(r618)
            cur_ret50_short.sort();  cur_ret618_short.sort()
            prev_lowest_low = cur_low

        # Validate shortside retracements
        if cur_ret50_short:
            cur_high = highs[end_idx]
            valid50 = []; valid618 = []; v50_set = set(); v618_set = set()
            for r50, r618 in zip(cur_ret50_short, cur_ret618_short):
                if r50 >= cur_high:
                    valid50.append(r50);  v50_set.add(r50)
                    valid618.append(r618); v618_set.add(r618)
            cur_ret50_short  = valid50;  cur_ret50_short_set  = v50_set
            cur_ret618_short = valid618; cur_ret618_short_set = v618_set
            if not cur_ret50_short:
                reference_low     = None
                reference_low_idx = None

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    for i in range(1, 26):
        df[f'Fib_50_long_{i}']   = 0.0
        df[f'Fib_618_long_{i}']  = 0.0
        df[f'Fib_50_short_{i}']  = 0.0
        df[f'Fib_618_short_{i}'] = 0.0

    for i, (r50, r618) in enumerate(
            zip(cur_ret50_long[:25], cur_ret618_long[:25]), 1):
        df.at[last_idx, f'Fib_50_long_{i}']  = r50
        df.at[last_idx, f'Fib_618_long_{i}'] = r618

    for i, (r50, r618) in enumerate(
            zip(cur_ret50_short[:25], cur_ret618_short[:25]), 1):
        df.at[last_idx, f'Fib_50_short_{i}']  = r50
        df.at[last_idx, f'Fib_618_short_{i}'] = r618

    return df
