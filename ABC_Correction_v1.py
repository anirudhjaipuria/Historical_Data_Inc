import pandas as pd
import numpy as np


# ── Die drei Detektionsfunktionen bleiben UNVERÄNDERT ──────────────────────
# Sie werden nur noch selten aufgerufen (nur bei Signalen) und arbeiten
# dann auf einem Array-Slice statt auf einem wachsenden DataFrame.
# find_intermediate_low wird nur einmal am Ende aufgerufen – kein Änderungsbedarf.

def detect_abc_v1(ohlc: pd.DataFrame, ath_idx) -> dict:
    if ohlc.empty or 'high' not in ohlc.columns:
        return None
    mask_after_ATH = ohlc.index >= ath_idx
    if not mask_after_ATH.any():
        return None
    high_mask = (ohlc['Level_2_High_window_1_CS'] != 0) & mask_after_ATH
    high_indices = ohlc[high_mask].index.tolist()
    if len(high_indices) < 1:
        return None
    idx_ZERO = high_indices[-1]
    ZERO = ohlc.loc[idx_ZERO, 'Level_2_High_window_1_CS']
    after_ZERO_mask = (ohlc.index > idx_ZERO)
    if not after_ZERO_mask.any():
        return None
    lows_after_ZERO = ohlc.loc[after_ZERO_mask, 'Level_1_Low_window_1_CS']
    non_zero_lows_after = lows_after_ZERO[lows_after_ZERO != 0]
    if non_zero_lows_after.empty:
        return None
    idx_first_low = non_zero_lows_after.index[0]
    after_first_low_mask = (ohlc.index > idx_first_low)
    if not after_first_low_mask.any():
        return None
    highs_after_first_low = ohlc.loc[after_first_low_mask, 'Level_1_High_window_1_CS']
    non_zero_highs_after = highs_after_first_low[highs_after_first_low != 0]
    if non_zero_highs_after.empty:
        return None
    B = non_zero_highs_after.max()
    idx_B = non_zero_highs_after[non_zero_highs_after == B].index[0]
    if B > ZERO:
        return None
    after_B_mask = (ohlc.index > idx_B)
    if after_B_mask.any():
        if ohlc.loc[after_B_mask, 'high'].max() > B:
            return None
    between_mask = (ohlc.index > idx_ZERO) & (ohlc.index < idx_B)
    if not between_mask.any():
        return None
    lows_between = ohlc.loc[between_mask, 'Level_1_Low_window_1_CS']
    non_zero_lows_between = lows_between[lows_between != 0]
    if non_zero_lows_between.empty:
        return None
    A = non_zero_lows_between.min()
    idx_A = non_zero_lows_between[non_zero_lows_between == A].index[0]
    return {'ZERO': (idx_ZERO, ZERO), 'A': (idx_A, A), 'B': (idx_B, B)}


def detect_abc_v2(ohlc: pd.DataFrame, ath_idx) -> dict:
    if ohlc.empty or 'high' not in ohlc.columns:
        return None
    mask_after_ATH = ohlc.index >= ath_idx
    if not mask_after_ATH.any():
        return None
    high_mask = (ohlc['Level_2_High_window_1_CS'] != 0) & mask_after_ATH
    high_indices = ohlc[high_mask].index.tolist()
    if len(high_indices) < 1:
        return None
    idx_B = high_indices[-1]
    B = ohlc.loc[idx_B, 'Level_2_High_window_1_CS']
    idx_ZERO = ohlc.loc[mask_after_ATH, 'Level_2_High_window_1_CS'].idxmax()
    ZERO = ohlc.loc[idx_ZERO, 'Level_2_High_window_1_CS']
    if B > ZERO:
        return None
    while True:
        between_mask = (ohlc.index > idx_ZERO) & (ohlc.index < idx_B)
        if not between_mask.any():
            return None
        lows_between = ohlc.loc[between_mask, 'Level_1_Low_window_1_CS']
        non_zero_lows_between = lows_between[lows_between != 0]
        if non_zero_lows_between.empty:
            return None
        A = non_zero_lows_between.min()
        idx_A = non_zero_lows_between[non_zero_lows_between == A].index[0]
        between_A_B_mask = (ohlc.index > idx_A) & (ohlc.index < idx_B)
        if not between_A_B_mask.any():
            break
        lm_highs_between = ohlc.loc[between_A_B_mask, 'LM_High_window_1_CS']
        non_zero_lm_between = lm_highs_between[lm_highs_between != 0]
        if non_zero_lm_between.empty or non_zero_lm_between.max() <= B:
            break
        max_lm = non_zero_lm_between.max()
        idx_new_B = non_zero_lm_between[non_zero_lm_between == max_lm].index[-1]
        idx_B = idx_new_B
        B = max_lm
        if B > ZERO:
            return None
    after_B_mask = ohlc.index > idx_B
    if after_B_mask.any():
        if ohlc.loc[after_B_mask, 'high'].max() > B:
            return None
    return {'ZERO': (idx_ZERO, ZERO), 'A': (idx_A, A), 'B': (idx_B, B)}


def find_intermediate_low(ohlc: pd.DataFrame) -> dict:
    if ohlc.empty or 'high' not in ohlc.columns:
        return None
    idx_ATH = ohlc['high'].idxmax()
    high_mask = (ohlc['Level_1_High_window_1_CS'] != 0) & (ohlc.index >= idx_ATH)
    high_indices = ohlc[high_mask].index.tolist()
    if len(high_indices) < 1:
        return None
    idx_ZERO = high_indices[-1]
    ZERO = ohlc.loc[idx_ZERO, 'Level_1_High_window_1_CS']
    after_ZERO_mask = (ohlc.index > idx_ZERO)
    lows_after_ZERO = ohlc.loc[after_ZERO_mask, 'LM_Low_window_1_CS']
    non_zero_lows_after = lows_after_ZERO[lows_after_ZERO != 0]
    if non_zero_lows_after.empty:
        return None
    idx_first_low = non_zero_lows_after.index[0]
    after_first_low_mask = (ohlc.index > idx_first_low)
    highs_after_first_low = ohlc.loc[after_first_low_mask, 'LM_High_window_2_CS']
    non_zero_highs_after = highs_after_first_low[highs_after_first_low != 0]
    if non_zero_highs_after.empty:
        return None
    B = non_zero_highs_after.max()
    idx_B = non_zero_highs_after[non_zero_highs_after == B].index[0]
    if B > ZERO:
        return None
    after_B_mask = (ohlc.index > idx_B)
    if after_B_mask.any():
        if ohlc.loc[after_B_mask, 'high'].max() > B:
            return None
    between_mask = (ohlc.index > idx_ZERO) & (ohlc.index < idx_B)
    lows_between = ohlc.loc[between_mask, 'LM_Low_window_1_CS']
    non_zero_lows_between = lows_between[lows_between != 0]
    if non_zero_lows_between.empty:
        return None
    A = non_zero_lows_between.min()
    idx_A = non_zero_lows_between[non_zero_lows_between == A].index[0]
    C_1_long    = B - 1.000 * (ZERO - A)
    C_1618_long = B - 1.618 * (ZERO - A)
    if after_B_mask.any():
        min_low_after_B = ohlc.loc[after_B_mask, 'low'].min()
    else:
        min_low_after_B = np.inf
    levels = {}
    if min_low_after_B > C_1_long:
        levels['1_long'] = C_1_long
    if min_low_after_B > C_1618_long:
        levels['1618_long'] = C_1618_long
    return {'ZERO': (idx_ZERO, ZERO), 'A': (idx_A, A), 'B': (idx_B, B), 'levels': levels}


# ── Optimierter Hauptloop ──────────────────────────────────────────────────

def calculate_abc_corrections(df):
    """
    Optimierungen gegenüber Original:
    OPT-1: Signalspalten einmalig als NumPy-Arrays extrahieren –
           kein df.iloc[end_idx] mehr im Loop.
    OPT-2: sub_df = df.iloc[:end_idx+1] nur noch erzeugt wenn
           tatsächlich ein Signal vorliegt (war vorher immer).
    OPT-3: Output in numpy-Arrays, einmalige DataFrame-Zuweisung am Ende –
           keine n×20 df.at-Calls mehr.
    OPT-4: to_remove-Liste ersetzt durch direkte list-comprehension-Filterung.
    OPT-5: sort_key mit max() statt list-comprehension + max().
    Logik und Reihenfolge aller Operationen: identisch zum Original.
    """
    n = len(df)
    if n == 0:
        return df

    # ── OPT-1: Signalspalten einmalig extrahieren ─────────────────────────
    highs_arr   = df['high'].values
    lows_arr    = df['low'].values
    sig_l1_high = df['Level_1_High_window_1_CS'].values  # Trigger für v1
    sig_l2_high = df['Level_2_High_window_1_CS'].values  # Trigger für v2

    # ── OPT-3: Output-Arrays statt n×20 df.at-Calls ──────────────────────
    out_c1    = np.zeros((n, 10), dtype=np.float64)
    out_c1618 = np.zeros((n, 10), dtype=np.float64)

    # State – identisch zum Original
    current_ath_idx   = None
    current_ath_value = -np.inf
    active_configs    = []

    for end_idx in range(n):
        current_high = highs_arr[end_idx]
        current_low  = lows_arr[end_idx]

        # ATH-Update – Reihenfolge: VOR allem anderen, exakt wie Original
        if current_high > current_ath_value:
            current_ath_value = current_high
            current_ath_idx   = df.index[end_idx]
            active_configs    = []

        # ── OPT-2: sub_df nur bei tatsächlichem Signal ────────────────────
        # Reihenfolge: v1-Trigger prüfen, DANN sub_df erzeugen wenn nötig
        trigger_v1 = sig_l1_high[end_idx] != 0
        trigger_v2 = sig_l2_high[end_idx] != 0

        if trigger_v1 or trigger_v2:
            # sub_df einmalig erzeugen falls mindestens ein Trigger aktiv
            sub_df = df.iloc[:end_idx + 1]

            if trigger_v1:
                result = detect_abc_v1(sub_df, current_ath_idx)
                if result is not None:
                    idx_B, B = result['B']
                    after_mask   = sub_df.index > idx_B
                    min_low_after = sub_df.loc[after_mask, 'low'].min() if after_mask.any() else np.inf
                    ZERO = result['ZERO'][1]
                    A    = result['A'][1]
                    C_1_long    = B - 1.000 * (ZERO - A)
                    C_1618_long = B - 1.618 * (ZERO - A)
                    levels = {}
                    if C_1_long > 0 and min_low_after > C_1_long:
                        levels['1_long'] = C_1_long
                    if C_1618_long > 0 and min_low_after > C_1618_long:
                        levels['1618_long'] = C_1618_long
                    if levels:
                        if not any(c['idx_B'] == idx_B and abs(c['B'] - B) < 1e-6
                                   for c in active_configs):
                            active_configs.append({
                                'idx_B': idx_B, 'B': B,
                                'min_low_after': min_low_after, 'levels': levels
                            })

            if trigger_v2:
                result = detect_abc_v2(sub_df, current_ath_idx)
                if result is not None:
                    idx_B, B = result['B']
                    after_mask   = sub_df.index > idx_B
                    min_low_after = sub_df.loc[after_mask, 'low'].min() if after_mask.any() else np.inf
                    ZERO = result['ZERO'][1]
                    A    = result['A'][1]
                    C_1_long    = B - 1.000 * (ZERO - A)
                    C_1618_long = B - 1.618 * (ZERO - A)
                    levels = {}
                    if C_1_long > 0 and min_low_after > C_1_long:
                        levels['1_long'] = C_1_long
                    if C_1618_long > 0 and min_low_after > C_1618_long:
                        levels['1618_long'] = C_1618_long
                    if levels:
                        if not any(c['idx_B'] == idx_B and abs(c['B'] - B) < 1e-6
                                   for c in active_configs):
                            active_configs.append({
                                'idx_B': idx_B, 'B': B,
                                'min_low_after': min_low_after, 'levels': levels
                            })

        # Letzter Schritt: find_intermediate_low – unverändert zum Original
        if end_idx == n - 1:
            result = find_intermediate_low(df)
            if result is not None:
                idx_B, B = result['B']
                after_mask   = df.index > idx_B
                min_low_after = df.loc[after_mask, 'low'].min() if after_mask.any() else np.inf
                levels = {k: v for k, v in result.get('levels', {}).items() if v > 0}
                if levels:
                    if not any(c['idx_B'] == idx_B and abs(c['B'] - B) < 1e-6
                               for c in active_configs):
                        active_configs.append({
                            'idx_B': idx_B, 'B': B,
                            'min_low_after': min_low_after, 'levels': levels
                        })

        # Validierung bestehender Configs – Reihenfolge exakt wie Original:
        # 1. high-Breach prüfen, 2. min_low aktualisieren, 3. Level-Breach prüfen
        # 4. leere Configs markieren
        curr_idx = df.index[end_idx]

        # ── OPT-4: direkte Filterung statt to_remove-Liste ────────────────
        new_active = []
        for config in active_configs:
            if curr_idx > config['idx_B']:
                if current_high > config['B']:
                    continue  # high-Breach → verwerfen
                config['min_low_after'] = min(config['min_low_after'], current_low)
                for key in list(config['levels']):
                    if config['min_low_after'] <= config['levels'][key]:
                        del config['levels'][key]
            if config['levels']:
                new_active.append(config)
        active_configs = new_active

        # Filter positive Werte – exakt wie Original
        for cfg in active_configs:
            cfg['levels'] = {k: v for k, v in cfg['levels'].items() if v > 0}

        # ── OPT-5: sort_key ohne list-comprehension ────────────────────────
        # Reihenfolge: NACH Filterung, exakt wie Original
        def sort_key(cfg):
            vals = cfg['levels'].values()
            pos  = [v for v in vals if v > 0]
            return -max(pos) if pos else -np.inf

        active_configs.sort(key=sort_key)

        # ── OPT-3: In Output-Arrays schreiben statt df.at ─────────────────
        for j, config in enumerate(active_configs[:10]):
            if '1_long'    in config['levels']:
                out_c1[end_idx, j]    = config['levels']['1_long']
            if '1618_long' in config['levels']:
                out_c1618[end_idx, j] = config['levels']['1618_long']

    # ── OPT-3: Einmalige Zuweisung aller Output-Spalten ───────────────────
    for i in range(10):
        df[f'C_1_long_{i+1}']    = out_c1[:,    i]
        df[f'C_1618_long_{i+1}'] = out_c1618[:, i]

    return df
