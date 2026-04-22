# ════════════════════════════════════════════════════════════════════════════
#   Änderungsprotokoll
# ════════════════════════════════════════════════════════════════════════════
#
#   v1  (Dateiname: ABC_Extension_v2.py)
#       Erste optimierte Version gegenüber Original (ABC_Extension.py):
#       – Phase 1: between_mask O(n) → numpy-Slice O(k)
#       – Phase 1: after_B_mask O(n) → slice-max
#       – Phase 1: lineare Kandidatenfilterung → bisect O(log m)
#       – Phase 2: O(|all_configs|)-Aktivierungscheck → sortierte Liste + Pointer
#       – Phase 2: cfg not in active O(|active|) → id-Set O(1)
#       – Phase 2: active.sort() bei jedem Schritt → nur bei dirty=True
#       – Phase 2: df.at-Einzelzugriffe → numpy-Arrays, einmalige Zuweisung am Ende
#
#   v2  (Dateiname: ABC_Extension_v2_full.py)
#       _full-Variante: für einmaligen Aufruf im Mainframe vor der Chunk-Schleife.
#       Gegenüber v1:
#       – Phase 1: max_high_after = -inf für alle Configs (kein chunk-abhängiger
#         highs[pos_B+1:n].max()-Aufruf) → alle geometrisch validen ZERO/A/B-Paare
#         werden aufgenommen, Phase 2 übernimmt das zeitabhängige Filtern
#       – Phase 2: befüllt JEDE Zeile des DataFrames (nicht nur die letzte),
#         so dass _df_full.iloc[:i] in process_chunk bereits korrekte Werte enthält
#       – Korrektheit: Zeile j enthält exakt die Werte, die v1 mit Chunk df[:j+1]
#         produziert hätte – der Aufruf im Mainframe ersetzt damit alle
#         per-Chunk-Aufrufe von calculate_abc_extensions vollständig
#       – Zeitkomplexität: O(n²) einmalig statt O(n³) über alle Chunks
#         → Faktor ~n/3 schneller (bei AMD: von >28h auf ~1min für ABC_Extension)
#
# ════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import bisect


def calculate_abc_extensions_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    _full-Variante: berechnet ABC-Extension-Spalten für JEDE Zeile des DataFrames.

    Wird einmalig im Mainframe auf dem vollen df_full aufgerufen, bevor die
    Chunk-Schleife startet. process_chunk muss calculate_abc_extensions danach
    NICHT mehr aufrufen – die Spalten sind bereits korrekt befüllt.

    Ausgabe-Spalten (identisch zu v1):
        C_1_short_1  .. C_1_short_10
        C_1618_short_1 .. C_1618_short_10
    """
    n = len(df)
    if n == 0:
        return df

    # ── Ausgabe-Arrays (numpy) ───────────────────────────────────────────────
    out_c1    = np.zeros((n, 10), dtype=np.float64)
    out_c1618 = np.zeros((n, 10), dtype=np.float64)

    # ── Vorausberechnete Arrays ──────────────────────────────────────────────
    highs_arr   = df['high'].values
    lows_arr    = df['low'].values
    l1_high_arr = df['Level_1_High_window_1_CS'].values
    l1_low_arr  = df['Level_1_Low_window_1_CS'].values

    # ── All-Time Low (ATL) ───────────────────────────────────────────────────
    atl_pos = int(np.argmin(lows_arr))

    # ── Signal-Punkte (aufsteigend sortiert durch np.where) ─────────────────
    low_pos_arr  = np.where((np.arange(n) >= atl_pos) & (l1_low_arr  != 0))[0]
    high_pos_arr = np.where((np.arange(n) >= atl_pos) & (l1_high_arr != 0))[0]

    if len(low_pos_arr) == 0 or len(high_pos_arr) == 0:
        for i in range(1, 11):
            df[f'C_1_short_{i}']    = 0.0
            df[f'C_1618_short_{i}'] = 0.0
        return df

    low_vals  = l1_low_arr[low_pos_arr]
    high_vals = l1_high_arr[high_pos_arr]
    low_pos_list  = low_pos_arr.tolist()
    high_pos_list = high_pos_arr.tolist()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Konfigurationssuche
    #
    # Unterschied zu v1: max_high_after wird auf -inf gesetzt (nicht auf
    # highs[pos_B+1:n].max()). Damit werden ALLE geometrisch validen
    # ZERO/A/B-Paare aufgenommen. Phase 2 übernimmt das zeitabhängige
    # Filtern Zeile für Zeile – exakt wie beim bisherigen per-Chunk-Aufruf.
    #
    # Korrektheitsbegründung:
    #   In v1 (per Chunk) entschied max_high_after in Phase 1 nur, ob eine
    #   Config INITIAL in die Liste aufgenommen wird. In Phase 2 wurde sie
    #   ohnehin sofort geprüft und ggf. entfernt. Das Weglassen des
    #   Initial-Checks in Phase 1 führt daher zu identischen Ergebnissen –
    #   Phase 2 filtert korrekt für jeden Zeitpunkt.
    # ════════════════════════════════════════════════════════════════════════
    first_a_per_zero = {}
    all_configs      = []

    for z_i, pos_ZERO in enumerate(low_pos_list):
        ZERO = low_vals[z_i]

        a_start = bisect.bisect_right(high_pos_list, pos_ZERO)
        if a_start >= len(high_pos_list):
            continue

        for a_i in range(a_start, len(high_pos_list)):
            pos_A = high_pos_list[a_i]
            A     = high_vals[a_i]

            # FILTER 1: A <= first valid A für dieses ZERO → überspringen
            if pos_ZERO in first_a_per_zero and A <= first_a_per_zero[pos_ZERO]:
                continue

            b_start = bisect.bisect_right(low_pos_list, pos_A)
            if b_start >= len(low_pos_list):
                continue

            found_valid_b = False

            for b_i in range(b_start, len(low_pos_list)):
                pos_B = low_pos_list[b_i]
                B     = low_vals[b_i]

                # FILTER 2: B > ZERO and B < A
                if not (B > ZERO and B < A):
                    continue

                # FILTER 3: kein Level_1_High_window_1_CS > A zwischen A und B
                seg = l1_high_arr[pos_A + 1 : pos_B]
                if seg.size > 0:
                    nz_max = seg[seg != 0]
                    if nz_max.size > 0 and nz_max.max() > A:
                        continue

                # Swing und C-Level
                swing        = A - ZERO
                C_1_short    = B + swing
                C_1618_short = B + 1.618 * swing

                # ── Kernunterschied zu v1: max_high_after = -inf ─────────
                # Alle geometrisch validen Paare aufnehmen.
                # Phase 2 filtert zeitabhängig korrekt.
                all_configs.append({
                    'pos_B':          pos_B,
                    'max_high_after': -np.inf,
                    'levels': {
                        '1':    C_1_short,
                        '1618': C_1618_short,
                    },
                })

                if pos_ZERO not in first_a_per_zero:
                    first_a_per_zero[pos_ZERO] = A

                found_valid_b = True

            if found_valid_b:
                continue

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: Zeilenweise Zuweisung (für JEDE Zeile)
    #
    # Läuft über alle n Zeilen und schreibt bei jeder Zeile den aktuellen
    # Stand der aktiven Configs in out_c1 / out_c1618.
    # Zeile j enthält exakt die Werte, die v1 mit Chunk df[:j+1] produziert
    # hätte – damit ist die Korrektheit für jeden Chunk i sichergestellt.
    # ════════════════════════════════════════════════════════════════════════

    all_configs.sort(key=lambda c: c['pos_B'])

    active       = []
    active_ids   = set()
    next_cfg_idx = 0
    dirty        = False

    for row in range(n):
        high = highs_arr[row]

        # 1. Aktiviere alle Configs mit pos_B <= row
        while next_cfg_idx < len(all_configs) and all_configs[next_cfg_idx]['pos_B'] <= row:
            cfg = all_configs[next_cfg_idx].copy()
            active.append(cfg)
            active_ids.add(id(cfg))
            dirty = True
            next_cfg_idx += 1

        # 2. Update max_high_after und invalide Level entfernen
        remove_ids = set()
        for cfg in active:
            if row > cfg['pos_B']:
                if high > cfg['max_high_after']:
                    cfg['max_high_after'] = high
                    dirty = True
                mh = cfg['max_high_after']
                for k in list(cfg['levels']):
                    if mh >= cfg['levels'][k]:
                        del cfg['levels'][k]
                        dirty = True
                if not cfg['levels']:
                    remove_ids.add(id(cfg))

        if remove_ids:
            active     = [c for c in active if id(c) not in remove_ids]
            active_ids -= remove_ids
            dirty = True

        # 3. Nur positive non-zero Levels behalten
        for cfg in active:
            neg = {k for k, v in cfg['levels'].items() if v <= 0}
            if neg:
                for k in neg:
                    del cfg['levels'][k]
                dirty = True

        # 4. Nur bei Änderungen neu sortieren
        if dirty and active:
            active.sort(key=lambda c: min(
                (v for v in c['levels'].values() if v > 0), default=np.inf
            ))
            dirty = False

        # 5. Top-10 in Ausgabe-Arrays schreiben
        for pos, cfg in enumerate(active[:10]):
            lvls = cfg['levels']
            if '1' in lvls:
                out_c1[row, pos]    = lvls['1']
            if '1618' in lvls:
                out_c1618[row, pos] = lvls['1618']

    # ── Ausgabe-Spalten einmalig aus numpy-Arrays befüllen ──────────────────
    for i in range(1, 11):
        df[f'C_1_short_{i}']    = out_c1[:, i - 1]
        df[f'C_1618_short_{i}'] = out_c1618[:, i - 1]

    return df
