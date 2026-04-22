# ════════════════════════════════════════════════════════════════════════════
#   Änderungsprotokoll
# ════════════════════════════════════════════════════════════════════════════
#
#   v0/v1  
#       Ursprüngliche Version mit Multiprocessing-Optimierung
#
#   v2/v3
#	    Resume-Funktion (Unterbrechung & Wiederaufnahme):
#       – Atomares Schreiben via .tmp → os.replace() verhindert halbfertige Dateien
#       – Ausgabeordner-Scan beim Start: vollständige Dateien (≥ TAIL_ROWS Zeilen)
#         werden per pyarrow-Metadaten-Check erkannt
#       – Plausibilitätscheck der letzten (cpu_count × 3) Dateien mit
#         rückwärts verschiebbarem Prüffenster
#       – Zu wenige vollständige Dateien → alles löschen, Neuberechnung von vorne
#       – MIN_ROWS_INPUT und TAIL_ROWS als parametrisierte Variablen (früher 200/400)
#       – MIN_FULL_FILES als kombinierte Untergrenze für Resume-Check
#
#	v4
#       – Profiling-Modus (PROFILING_MODE) zum Messen der Funktionslaufzeiten
#
#   v5  
#       Pfad-Variablen und Divergenz-Parameter:
#       – Pfade aufgeteilt in OUTPUT_PATH_BASE, OUTPUT_PATH_REGION,
#         OUTPUT_PATH_SUFFIX_ASSET, OUTPUT_PATH_FOLDER_TIMEFRAME
#       – Divergenz-Toleranzen als einzelne Variablen:
#         CBullDivg_Candle_Tol, CBullDivg_MACD_Tol usw.
#       – RAM_FACTOR_PER_WORKER parametrisiert (früher hardcoded 3)
#       – Klarere Log-Meldung beim Resume-Filter
#       – Profiling-Code entfernt (separates Skript bei Bedarf)
#
#	v6
#		- Anpassung in Dateicheck, die ersten Dateien < TAIL_ROWS werden im
#		  Plausibilitätscheck nicht mehr gelösch
#
#	v7
#		- ABC_Extension: Aufruf von calculate_abc_extensions aus process_chunk
#		  herausgelöst und als einmaliger Vorab-Aufruf auf df_full realisiert
#		  (calculate_abc_extensions_full aus ABC_Extension_v2_full.py).
#		  Begründung: calculate_abc_extensions lief bisher für jeden der n Chunks
#		  komplett neu (O(n³) Gesamtkomplexität). Die _full-Variante berechnet
#		  alle Zeilen einmalig in O(n²) → Faktor ~n/3 schneller.
#		  Bei AMD (18478 Zeilen): Reduktion von >28h auf ~1min für ABC_Extension.
#		  Die Ausgabe (C_1_short_i, C_1618_short_i) ist identisch zu v6.
#		  process_chunk übernimmt die Spalten direkt aus _df_full, kein
#		  erneuter Aufruf von calculate_abc_extensions mehr nötig.
#		- Import angepasst: ABC_Extension_v2_full statt ABC_Extension
#
#   Optimierte Berechnungsmodule (deutliche Laufzeitreduktion):
#       – Goldenratio_vectorized.py: ~45–55% schneller durch:
#           • Boolean-Masken statt Set + np.isin() für processed-Tracking
#           • ATH inkrementell statt np.argmax() bei jedem Schritt
#           • Set-basierter Duplikat-Check für Retracements
#           • Bugfix: ATH-Initialisierung auf highs[0..1] (Loop startet bei 2)
#           • Bugfix: filter_maxima_short Längenprüfung VOR ATH-Insert
#           • Bugfix: stable sort in beiden Hilfsfunktionen
#           • Bugfix: None-Check für ath_date konsistent mit Original
#       – ABC_Correction.py: ~60% schneller durch:
#           • sub_df nur noch bei tatsächlichem Signal erzeugt
#           • Output in numpy-Arrays, einmalige DataFrame-Zuweisung am Ende
#           • Direkte list-comprehension statt to_remove-Liste
#       – ABC_Extension_v2_full.py: drastisch schneller durch:
#           • Einmaliger Aufruf vor Chunk-Schleife statt pro Chunk
#           • Phase 1: max_high_after = -inf, Phase 2 filtert zeitabhängig
#           • Alle Optimierungen aus ABC_Extension_v2 (v1) übernommen
#       → Import dieser Module muss auf die optimierten Versionen zeigen
#
# ════════════════════════════════════════════════════════════════════════════

import pandas as pd
import warnings
import os
import psutil
import time
import numpy as np
import multiprocessing as mp
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

# ────────────────────────────────────────────────
#   Your custom module imports
# ────────────────────────────────────────────────
from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from CBearDivg_analysis_vectorized import CBearDivg_analysis
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from Goldenratio_vectorized_v5 import calculate_golden_ratios          # v4: optimierte Version
#from Goldenratio_vectorized import calculate_golden_ratios          # v4: optimierte Version
from Support_Resistance_vectorized import calculate_support_levels
from ABC_Correction_v1 import calculate_abc_corrections               # v4: optimierte Version
#from ABC_Correction import calculate_abc_corrections               # v4: optimierte Version
from ABC_Extension_v2_full import calculate_abc_extensions_full  # v7: einmaliger Vorab-Aufruf

# ────────────────────────────────────────────────
#   Pandas / warnings setup
# ────────────────────────────────────────────────
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented*')

# ────────────────────────────────────────────────
#   SYMBOLS to process
# ────────────────────────────────────────────────
# SYMBOLS = ['ADA', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'XRP', 'SOL', 'TRX ', 'LINK', 'XLM']

# SYMBOLS = ['ADANIENT', 'AXISBANK', 'BAJFINANC', 'BANKBAROD', 'BANKINDIA', 'DABUR',
#            'HDFCBANK', 'HINDUNILV', 'ICICIBANK', 'INDUSINDB', 'INFY', 'ITC',
#            'KOTAKBANK', 'LT', 'NOCIL', 'RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN']

# SYMBOLS = ['AAPL', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AMZN', 'AVGO', 'BAC', 'COST', 'CRM',
#            'CSCO', 'CVX', 'DIS', 'GOOGL', 'GS', 'HD', 'JNJ', 'JPM', 'KO', 'LLY',
#            'MA', 'MCD', 'META', 'MRK', 'MSFT', 'NFLX', 'NVDA', 'ORCL', 'PEP', 'PG',
#            'QCOM', 'TSLA', 'UNH', 'V', 'WFC', 'WMT', 'XOM']

#SYMBOLS = ['HDFCBANK'] # 18.3.26
#SYMBOLS = ['BANKBAROD'] # 18.3.26
#SYMBOLS = ['ADANIENT', 'AXISBANK', 'BAJFINANC'] # 22.3.26
#SYMBOLS = ['BANKINDIA', 'DABUR','HINDUNILV', 'ICICIBANK'] # 23.3.26
#SYMBOLS = ['INDUSINDB', 'INFY'] # 24.3.26
#SYMBOLS = ['ITC', 'KOTAKBANK', 'LT', 'NOCIL'] # 25.3.26
#SYMBOLS = ['RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN'] # 25.3.26
# SYMBOLS = ['AAPL'] # 12.4. Goldenratio_v5__ABC_v1
# SYMBOLS = ['AAPL', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AMZN', 'AVGO', 'BAC', 'COST', 'CRM',
#            'CSCO', 'CVX', 'DIS', 'GOOGL', 'GS', 'HD', 'JNJ', 'JPM', 'KO', 'LLY',
#            'MA', 'MCD', 'META', 'MRK'] # 12.4. Goldenratio_v5__ABC_v1, AAPL Test, ob Dateicheck funktioniert
#SYMBOLS = ['AAPL',  'MSFT', 'NFLX', 'NVDA', 'ORCL', 'PEP', 'PG',
#           'QCOM', 'TSLA', 'UNH', 'V', 'WFC', 'WMT', 'XOM'] # 12.4. Goldenratio_v5__ABC_v1, AAPL Test, ob Dateicheck funktioniert
#SYMBOLS = ['MSFT', 'NFLX']     # 15.4. Goldenratio_v5__ABC_v1, AAPL Test, ob Dateicheck funktioniert, NFLX nach Ausstieg
#SYMBOLS = ['NVDA', 'ORCL', 'PEP', 'PG',
#           'QCOM', 'TSLA', 'UNH', 'V', 'WFC', 'WMT', 'XOM'] # 16.4. Goldenratio_v5__ABC_v1
#SYMBOLS = ['PEP','AVGO', 'AMD', 'ABBV', 'META', 'TSLA'] # 19.4./20.4 Goldenratio_v5__ABC_v1 ABC_extension, Mainframe_v6
SYMBOLS = ['PEP','AVGO', 'AMD', 'ABBV', 'META', 'TSLA', 'NVDA', 'BAC', 'V'] # 21.4. Goldenratio_v5__ABC_v1, ABC_extension_v2_full

# ────────────────────────────────────────────────
#   Einstellungen
# ────────────────────────────────────────────────
reduceNrCores       = 0    # 0 = alle Cores nutzen, 2 = zwei Cores weniger, usw.
RAM_WARN_THRESHOLD  = 95   # % – kurz warten, dann weiterrechnen
RAM_ABORT_THRESHOLD = 98   # % – Chunk abbrechen, später retry
RAM_FACTOR_PER_WORKER = 3  # Schätzfaktor: RAM pro Worker ≈ N × DataFrame-Größe

MIN_ROWS_INPUT = 200   # Mindest-Eingabezeilen im CSV für die Verarbeitung
TAIL_ROWS      = 400   # Anzahl der letzten Zeilen, die pro Parquet-Datei gespeichert werden

# Mindestanzahl vollständiger Dateien (≥ TAIL_ROWS Zeilen) im Ausgabeordner,
# bevor der Plausibilitätscheck greift. Enthält Puffer für (cpu_count × 3) Check-Dateien.
# Falls weniger vorhanden → alles löschen und Neuberechnung von vorne.
MIN_FULL_FILES = TAIL_ROWS + mp.cpu_count() * 3

# ── Divergenz-Toleranzen ────────────────────────
CBullDivg_Candle_Tol   = 0.01
CBullDivg_MACD_Tol     = 3.25
CBullDivg_x2_Candle_Tol = 0.01
CBullDivg_x2_MACD_Tol  = 3.25
HBullDivg_Candle_Tol   = 0.01
HBullDivg_MACD_Tol     = 3.25
CBearDivg_Candle_Tol   = 0.01
CBearDivg_MACD_Tol     = 3.25
HBearDivg_Candle_Tol   = 0.01
HBearDivg_MACD_Tol     = 3.25

# ── Pfad-Konfiguration ──────────────────────────
OUTPUT_PATH_BASE            = r'C:\PYTHON\historical_data'   # Basis-Pfad
#OUTPUT_PATH_REGION          = 'INDIA'                         # Unterordner Region/Markt
OUTPUT_PATH_REGION          = 'US'                         # Unterordner Region/Markt
#OUTPUT_PATH_SUFFIX_ASSET    = ''          # Suffix für Asset-Ordner, z.B. '_Goldenratio_v5__ABC_v1'
                                          # Wird automatisch mit '_' präfixiert wenn nötig
OUTPUT_PATH_SUFFIX_ASSET    = '_Goldenratio_v5__ABC_v1__ABCext_v2_full'          # Suffix für Asset-Ordner
OUTPUT_PATH_FOLDER_TIMEFRAME = 'output_1hour_parquet'         # Unterordner für Ausgabe-Parquets

# ────────────────────────────────────────────────
#   Logging Setup
# ────────────────────────────────────────────────
output_log_folder = r'C:\PYTHON\__logs'  # Pfad anpassen

folder   = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
datum    = datetime.now().strftime("%Y-%m-%d")
log_file = os.path.join(output_log_folder,
                        f"Mainframe_wo_TL_vectorized_1hour_loop__{folder}__{datum}.txt")

os.makedirs(output_log_folder, exist_ok=True)


def log(msg=""):
    """Gibt eine Nachricht auf der Konsole aus und schreibt sie in die Log-Datei."""
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ────────────────────────────────────────────────
#   Global df_full for worker processes
# ────────────────────────────────────────────────
_df_full = None


def init_worker(df):
    """Initialisiert den globalen DataFrame in jedem Worker-Prozess (einmalig pro Worker)."""
    global _df_full
    _df_full = df


def get_expected_filename(df_full, i):
    """Berechnet den erwarteten Dateinamen für Index i deterministisch."""
    last_date = df_full.iloc[:i]['date'].iloc[-1]
    if pd.api.types.is_datetime64_any_dtype(last_date):
        last_date_str = last_date.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        last_date_str = str(last_date).replace('/', '-').replace(':', '-').replace(' ', '_')
    return f"output_{last_date_str}.parquet"


def process_chunk(args):
    """
    Verarbeitet den DataFrame-Prefix bis Index i in einem separaten Prozess.
    """
    symbol, output_dir, i = args

    try:
        # RAM-Check vor der Berechnung
        ram = psutil.virtual_memory()

        if ram.percent > RAM_ABORT_THRESHOLD:
            # Echter Notfall – Chunk abbrechen, später retry
            raise MemoryError(
                f"RAM-Auslastung kritisch ({ram.percent:.1f}%) "
                f"bei Chunk {i} – Chunk übersprungen"
            )
        elif ram.percent > RAM_WARN_THRESHOLD:
            # Erhöhte Auslastung – kurz warten damit andere Prozesse RAM freigeben
            time.sleep(2)

        # Arbeitskopie des Prefix erstellen
        df = _df_full.iloc[:i].copy()

        # ────────────────────────────────────────────────
        #   Indicator / pattern calculations
        # ────────────────────────────────────────────────
        Initialize_RSI_EMA_MACD(df)
        Level_1_Max_Min(df)
        Candlestick_Type(df)

        CBullDivg_analysis(df,    CBullDivg_Candle_Tol,    CBullDivg_MACD_Tol)
        CBullDivg_x2_analysis(df, CBullDivg_x2_Candle_Tol, CBullDivg_x2_MACD_Tol)
        HBullDivg_analysis(df,    HBullDivg_Candle_Tol,    HBullDivg_MACD_Tol)
        CBearDivg_analysis(df,    CBearDivg_Candle_Tol,    CBearDivg_MACD_Tol)
        HBearDivg_analysis(df,    HBearDivg_Candle_Tol,    HBearDivg_MACD_Tol)

        df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
        df = calculate_golden_ratios(df)
        df = calculate_abc_corrections(df)
        # calculate_abc_extensions entfällt hier – Spalten sind bereits in _df_full
        # durch den einmaligen Vorab-Aufruf von calculate_abc_extensions_full() befüllt

        # Output-Dateiname aus letztem Datum des Chunks
        last_date = df['date'].iloc[-1]
        if pd.api.types.is_datetime64_any_dtype(last_date):
            last_date_str = last_date.strftime('%Y-%m-%d_%H-%M-%S')
        else:
            last_date_str = str(last_date).replace('/', '-').replace(':', '-').replace(' ', '_')

        output_file = Path(output_dir) / f"output_{last_date_str}.parquet"
        tmp_file    = Path(output_dir) / f"output_{last_date_str}.parquet.tmp"

        # Erst in .tmp schreiben, dann atomar umbenennen → nie halbfertige .parquet-Dateien
        df.tail(TAIL_ROWS).to_parquet(tmp_file, index=False, engine='pyarrow')
        os.replace(tmp_file, output_file)

    except MemoryError as e:
        print(f"MEMORY WARNING – Chunk {i} für {symbol}: {e}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"MEMORY WARNING – Chunk {i} für {symbol}: {e}\n")

    except Exception as e:
        print(f"ERROR – Chunk {i} für {symbol}: {e}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ERROR – Chunk {i} für {symbol}: {e}\n")
        raise


def run_sequential(failed_tasks, symbol, df_full):
    """Wiederholt fehlgeschlagene Tasks sequenziell (ohne Parallelisierung)."""
    global _df_full
    _df_full = df_full

    log(f"  Starte sequenzielle Wiederholung für {len(failed_tasks):,} fehlgeschlagene Tasks ...")
    for args in failed_tasks:
        _, output_dir, i = args
        log(f"  Wiederhole Chunk {i} ...")
        try:
            process_chunk(args)
            log(f"  ✔ Chunk {i} erfolgreich wiederholt")
        except Exception as e:
            log(f"  ✗ Chunk {i} auch nach Wiederholung fehlgeschlagen: {e}")


if __name__ == '__main__':

    mp.set_start_method('spawn')  # Explizit für Windows setzen

    # Suffix normalisieren: muss mit '_' beginnen oder leer sein
    _suffix = OUTPUT_PATH_SUFFIX_ASSET.strip()
    if _suffix and not _suffix.startswith('_'):
        _suffix = '_' + _suffix

    start_time = datetime.now()
    time_bkp   = start_time

    log(f"{'=' * 60}")
    log(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'=' * 60}")

    for SYMBOL in SYMBOLS:
        log(f"\n{'=' * 60}")
        log(f"Processing {SYMBOL} ...")
        log(f"{'=' * 60}")

        # ────────────────────────────────────────────────
        #   Paths
        # ────────────────────────────────────────────────
        asset_folder  = f"{SYMBOL}{_suffix}"
        csv_file_path = os.path.join(OUTPUT_PATH_BASE, OUTPUT_PATH_REGION,
                                     SYMBOL, f"{SYMBOL}_1hour_candlesticks_all.csv")
        output_dir    = os.path.join(OUTPUT_PATH_BASE, OUTPUT_PATH_REGION,
                                     asset_folder, OUTPUT_PATH_FOLDER_TIMEFRAME)

        os.makedirs(output_dir, exist_ok=True)

        # CSV einlesen
        log("Reading full CSV ...")
        try:
            df_full = pd.read_csv(csv_file_path, header=0, parse_dates=['date'])
        except Exception as e:
            log(f"ERROR reading {csv_file_path}: {e}")
            continue

        len_df = len(df_full)
        log(f"Total rows: {len_df:,}")

        # RAM-Status nach CSV-Einlesen
        ram = psutil.virtual_memory()
        log(f"RAM gesamt:     {ram.total     / 1024**3:.1f} GB")
        log(f"RAM verfügbar:  {ram.available / 1024**3:.1f} GB")
        log(f"RAM verbraucht: {ram.percent:.1f} %")
        log(f"df_full Größe:  {df_full.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        if len_df <= MIN_ROWS_INPUT:
            log(f" → Not enough rows to process (need > {MIN_ROWS_INPUT}), skipping.")
            continue

        # ────────────────────────────────────────────────
        #   v7: ABC_Extension einmalig vorab auf dem vollen Datensatz berechnen
        #   Alle anderen Indikatoren werden weiterhin pro Chunk in process_chunk
        #   aufgerufen. calculate_abc_extensions_full befüllt jede Zeile von
        #   df_full mit den korrekten C_1_short / C_1618_short Werten.
        #   process_chunk übernimmt diese Spalten via _df_full.iloc[:i].copy()
        #   ohne erneuten Aufruf von calculate_abc_extensions.
        # ────────────────────────────────────────────────
        log("Berechne ABC_Extension vorab auf vollem Datensatz ...")
        # Alle Indikatoren die ABC_Extension als Input benötigt müssen vorab
        # auf df_full berechnet werden (Level_1_Max_Min, Initialize_RSI_EMA_MACD)
        _df_abc = df_full.copy()
        Initialize_RSI_EMA_MACD(_df_abc)
        Level_1_Max_Min(_df_abc)
        _df_abc = calculate_abc_extensions_full(_df_abc)
        # Nur die ABC-Extension-Spalten zurück in df_full übernehmen
        abc_cols = ([f'C_1_short_{i}'    for i in range(1, 11)] +
                    [f'C_1618_short_{i}' for i in range(1, 11)])
        for col in abc_cols:
            if col in _df_abc.columns:
                df_full[col] = _df_abc[col].values
        del _df_abc
        log("ABC_Extension vorab fertig.")

        # Tasks und erwartete Dateinamen vorbereiten
        log("Berechne erwartete Dateinamen ...")
        expected_files = {
            get_expected_filename(df_full, i): (SYMBOL, output_dir, i)
            for i in range(MIN_ROWS_INPUT, len_df - 1)
        }

        # ────────────────────────────────────────────────
        #   Resume-Logik: Ausgabeordner prüfen
        # ────────────────────────────────────────────────

        # 1) Halbfertige .tmp-Dateien löschen (entstehen bei hartem Abbruch)
        tmp_files = list(Path(output_dir).glob("*.parquet.tmp"))
        if tmp_files:
            log(f"  Lösche {len(tmp_files)} halbfertige .tmp-Datei(en) ...")
            for tmp_f in tmp_files:
                try:
                    tmp_f.unlink()
                    log(f"    gelöscht: {tmp_f.name}")
                except Exception as e:
                    log(f"    FEHLER beim Löschen von {tmp_f.name}: {e}")

        # 2) Vorhandene .parquet-Dateien ermitteln und nach Timestamp sortieren
        all_existing = sorted(
            f for f in os.listdir(output_dir)
            if f.startswith("output_") and f.endswith(".parquet")
        )
        log(f"  Im Ausgabeordner gefunden: {len(all_existing):,} .parquet-Datei(en)")

        # 3) Vollständige Dateien zählen (≥ TAIL_ROWS Zeilen)
        #    Wir lesen nur die Metadaten (row-count), nicht den gesamten Inhalt.
        full_files = []
        for fname in all_existing:
            try:
                nrows = pq.read_metadata(Path(output_dir) / fname).num_rows
                if nrows >= TAIL_ROWS:
                    full_files.append((fname, nrows))
            except Exception:
                pass  # nicht lesbare Dateien zählen nicht als vollständig

        log(f"  Davon vollständige Dateien (≥ {TAIL_ROWS} Zeilen): {len(full_files):,}")

        # 4) Zu wenige vollständige Dateien → alles löschen, Neuberechnung
        if len(full_files) < MIN_FULL_FILES:
            log(f"  Weniger als MIN_FULL_FILES={MIN_FULL_FILES} vollständige Dateien gefunden "
                f"→ lösche alle {len(all_existing):,} Dateien und starte neu.")
            for fname in all_existing:
                try:
                    (Path(output_dir) / fname).unlink()
                except Exception as e:
                    log(f"    FEHLER beim Löschen von {fname}: {e}")
            existing_files = set()

        else:
            # 5) Plausibilitätscheck der letzten (cpu_count × 3) vollständigen Dateien
            #    Fenster rückwärts verschiebbar, Untergrenze = Index TAIL_ROWS in full_files
            check_count   = mp.cpu_count() * 3
            lower_bound   = TAIL_ROWS          # Index in full_files, nicht unterschreiten
            window_end    = len(full_files)    # exklusiv
            deleted_on_resume = []

            while True:
                window_start = max(lower_bound, window_end - check_count)
                window       = full_files[window_start:window_end]

                if not window:
                    break

                # Referenz-Zeilenzahl: Datei direkt vor dem Fenster, mindestens TAIL_ROWS
                if window_start > 0:
                    ref_rows = max(TAIL_ROWS, full_files[window_start - 1][1])
                else:
                    ref_rows = TAIL_ROWS

                log(f"  Plausibilitätscheck: Dateien [{window_start}–{window_end - 1}] "
                    f"(Referenz-Zeilenzahl: {ref_rows})")

                first_bad = None
                for idx, (fname, nrows) in enumerate(window):
                    if nrows < ref_rows:
                        first_bad = window_start + idx
                        log(f"    ✗ Erste fehlerhafte Datei bei Index {first_bad}: "
                            f"{fname} (Zeilenzahl {nrows} < Referenz {ref_rows})")
                        break

                if first_bad is None:
                    # Alle Dateien im Fenster OK
                    log(f"  ✔ Plausibilitätscheck bestanden – keine Dateien gelöscht")
                    break

                # Alle Dateien ab first_bad löschen
                to_delete = full_files[first_bad:]
                log(f"  Lösche {len(to_delete)} Datei(en) ab Index {first_bad} ...")
                for fname, _ in to_delete:
                    try:
                        (Path(output_dir) / fname).unlink()
                        deleted_on_resume.append(fname)
                        log(f"    gelöscht: {fname}")
                    except Exception as e:
                        log(f"    FEHLER beim Löschen von {fname}: {e}")

                full_files = full_files[:first_bad]
                window_end = first_bad

                # Untergrenze erreicht → Abbruch (MIN_FULL_FILES-Garantie verhindert
                # dass wir ins "kleine Dateien"-Gebiet rutschen)
                if window_end <= lower_bound:
                    log(f"  Unteres Limit ({lower_bound}) erreicht – Check beendet.")
                    break

            log(f"  Check abgeschlossen – {len(deleted_on_resume)} Datei(en) gelöscht")
            # existing_files = alle noch vorhandenen Dateien (vollständige + kleine Anfangsdateien).
            # Kleine Anfangsdateien (< TAIL_ROWS Zeilen) sind legitim, da die ersten Chunks
            # naturgemäß weniger Zeilen haben. Sie werden NICHT neu berechnet.
            # Nur durch den Plausibilitätscheck gelöschte Dateien fehlen jetzt im Ordner.
            existing_files = set(f for f in os.listdir(output_dir)
                                 if f.startswith("output_") and f.endswith(".parquet"))

        # 6) Resume-Filter: nur wirklich fehlende Tasks berechnen
        n_existing     = len(existing_files)
        n_total        = len(expected_files)
        n_small        = len(all_existing) - len(full_files)   # legitime kleine Anfangsdateien
        n_deleted      = len(all_existing) - n_existing        # durch Check gelöscht
        tasks = [
            (SYMBOL, output_dir, i)
            for i in range(MIN_ROWS_INPUT, len_df - 1)
            if get_expected_filename(df_full, i) not in existing_files
        ]
        n_todo = len(tasks)
        log(f"  Erwartet gesamt:          {n_total:,} Dateien")
        log(f"  Vorhanden (inkl. kleine): {n_existing:,} Dateien  → werden übersprungen")
        if n_small > 0:
            log(f"  Davon kleine Anfangsdateien (< {TAIL_ROWS} Zeilen): {n_small:,}  → legitim, kein Neurechnen")
        if n_deleted > 0:
            log(f"  Durch Plausibilitätscheck gelöscht: {n_deleted:,} Dateien")
        log(f"  Zu berechnen:             {n_todo:,} Dateien")

        if not tasks:
            log(f"  ✔ Alle Dateien für {SYMBOL} vollständig vorhanden – überspringe.")
            continue

        # Anzahl Prozesse dynamisch bestimmen
        ram_available_gb  = ram.available / 1024**3
        df_size_gb        = df_full.memory_usage(deep=True).sum() / 1024**3
        ram_per_worker_gb = df_size_gb * RAM_FACTOR_PER_WORKER
        max_by_ram        = max(1, int(ram_available_gb / ram_per_worker_gb))
        max_by_cpu        = mp.cpu_count() - reduceNrCores
        num_processes     = min(max_by_ram, max_by_cpu)

        log(f"RAM verfügbar:          {ram_available_gb:.1f} GB")
        log(f"df_full Größe:          {df_size_gb * 1024:.1f} MB")
        log(f"Geschätzter RAM/Worker: {ram_per_worker_gb:.2f} GB")
        log(f"Max Worker (RAM):       {max_by_ram}")
        log(f"Max Worker (CPU):       {max_by_cpu}")
        log(f"Gewählte Prozesse:      {num_processes}")
        log(f"Anzahl Tasks:           {len(tasks):,}")
        log(f"RAM Warnschwelle:       {RAM_WARN_THRESHOLD} %")
        log(f"RAM Abbruchschwelle:    {RAM_ABORT_THRESHOLD} %")
        log(f"Starte parallele Verarbeitung ...")

        with mp.Pool(processes=num_processes,
                     initializer=init_worker,
                     initargs=(df_full,)) as pool:
            for _ in pool.imap_unordered(process_chunk, tasks, chunksize=50):
                pass

        # ────────────────────────────────────────────────
        #   Fehlende Dateien ermitteln
        # ────────────────────────────────────────────────
        written_files = {
            f for f in os.listdir(output_dir)
            if f.startswith("output_") and f.endswith(".parquet")
        }

        missing_files = {
            fname: args
            for fname, args in expected_files.items()
            if fname not in written_files
        }

        log(f"Erwartet:    {len(expected_files):,} Dateien")
        log(f"Geschrieben: {len(written_files):,} Dateien")
        log(f"Fehlend:     {len(missing_files):,} Dateien")

        # ────────────────────────────────────────────────
        #   Retry fehlgeschlagener Tasks
        # ────────────────────────────────────────────────
        if missing_files:
            log(f"\nStarte Retry für {len(missing_files):,} fehlende Dateien ...")
            failed_tasks = list(missing_files.values())
            run_sequential(failed_tasks, SYMBOL, df_full)

            # Nochmals prüfen nach Retry
            written_files_after_retry = {
                f for f in os.listdir(output_dir)
                if f.startswith("output_") and f.endswith(".parquet")
            }
            still_missing = {
                fname for fname in missing_files
                if fname not in written_files_after_retry
            }

            if still_missing:
                log(f"  ✗ Nach Retry noch immer fehlend: {len(still_missing):,} Dateien:")
                for fname in sorted(still_missing):
                    log(f"    - {fname}")
            else:
                log(f"  ✔ Alle fehlenden Dateien erfolgreich nachgeholt")
        else:
            log(f"✔ Alle Dateien vollständig geschrieben, kein Retry nötig")

        end_time_symbol = datetime.now()
        duration        = end_time_symbol - time_bkp
        total_seconds   = int(duration.total_seconds())
        hours   = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        log(f"Finished processing {SYMBOL}\n")
        log(f"End {SYMBOL}:   {end_time_symbol.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Time {SYMBOL}:  {hours:02d}:{minutes:02d}:{seconds:02d}")
        time_bkp = end_time_symbol

    # ────────────────────────────────────────────────
    #   Gesamtlaufzeit
    # ────────────────────────────────────────────────
    end_time      = datetime.now()
    duration      = end_time - start_time
    total_seconds = int(duration.total_seconds())
    hours   = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    log(f"{'=' * 60}")
    log(f"End:        {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    log(f"All symbols processed.")
    log(f"{'=' * 60}")
