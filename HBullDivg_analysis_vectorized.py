import pandas as pd
import numpy as np
import warnings
import math

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def HBullDivg_analysis(ohlc: pd.DataFrame, Candle_Tol: float, MACD_tol: float) -> pd.DataFrame:
    # Convert necessary columns to numeric, coercing errors
    columns_to_convert = [
        'LM_Low_window_2_CS', 'LM_Low_window_1_CS',
        'LM_Low_window_2_MACD', 'LM_Low_window_1_MACD',
        'RSI', 'macd_histogram'
    ]
    for col in columns_to_convert:
        ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')

    # Parameters
    window_1 = 5
    window_2 = 1
    RSI_tol = 2     # Absolute Difference in RSI Values

    # Extract data to NumPy arrays
    CL = ohlc["low"].to_numpy()
    n = len(CL)
    dates = pd.to_datetime(ohlc['date'], utc=True).to_numpy()
    LM_Low_window_2_CS = ohlc['LM_Low_window_2_CS'].to_numpy()
    LM_Low_window_1_CS = ohlc['LM_Low_window_1_CS'].to_numpy()
    LM_Low_window_2_MACD = ohlc['LM_Low_window_2_MACD'].to_numpy()
    LM_Low_window_1_MACD = ohlc['LM_Low_window_1_MACD'].to_numpy()
    RSI = ohlc['RSI'].to_numpy()
    macd_histogram = ohlc['macd_histogram'].to_numpy()

    # Initialize output arrays for HBullD_1 (Hidden Bullish Divergence - Candlestick Lows)
    HBullD_1 = np.zeros(n)
    HBullD_Lower_Low_1 = np.zeros(n)
    HBullD_Higher_Low_1 = np.zeros(n)
    HBullD_Lower_Low_RSI_1 = np.zeros(n)
    HBullD_Higher_Low_RSI_1 = np.zeros(n)
    HBullD_Lower_Low_MACD_1 = np.zeros(n)
    HBullD_Higher_Low_MACD_1 = np.zeros(n)
    HBullD_Lower_Low_date_1 = np.empty(n, dtype=object)
    HBullD_Higher_Low_date_1 = np.empty(n, dtype=object)
    HBullD_Date_Gap_1 = np.zeros(n)
    HBullD_diffY_price_1 = np.zeros(n)
    HBullD_diffX_1 = np.zeros(n)

    # Initialize output arrays for HBullD_2 (Hidden Bullish Divergence - MACD)
    HBullD_2 = np.zeros(n)
    HBullD_Lower_Low_2 = np.zeros(n)
    HBullD_Higher_Low_2 = np.zeros(n)
    HBullD_Lower_Low_RSI_2 = np.zeros(n)
    HBullD_Higher_Low_RSI_2 = np.zeros(n)
    HBullD_Lower_Low_MACD_2 = np.zeros(n)
    HBullD_Higher_Low_MACD_2 = np.zeros(n)
    HBullD_Lower_Low_date_2 = np.empty(n, dtype=object)
    HBullD_Higher_Low_date_2 = np.empty(n, dtype=object)
    HBullD_Date_Gap_2 = np.zeros(n)
    HBullD_diffY_price_2 = np.zeros(n)
    HBullD_diffX_2 = np.zeros(n)

    # Initialize output arrays for HBullD_neg_MACD (Hidden Bullish Divergence - Negative MACD Minimas)
    HBullD_neg_MACD = np.zeros(n)
    HBullD_Lower_Low_neg_MACD = np.zeros(n)
    HBullD_Higher_Low_neg_MACD = np.zeros(n)
    HBullD_Lower_Low_RSI_neg_MACD = np.zeros(n)
    HBullD_Higher_Low_RSI_neg_MACD = np.zeros(n)
    HBullD_Lower_Low_MACD_neg_MACD = np.zeros(n)
    HBullD_Higher_Low_MACD_neg_MACD = np.zeros(n)
    HBullD_Lower_Low_date_neg_MACD = np.empty(n, dtype=object)
    HBullD_Higher_Low_date_neg_MACD = np.empty(n, dtype=object)
    HBullD_Date_Gap_neg_MACD = np.zeros(n)
    HBullD_diffY_price_neg = np.zeros(n)
    HBullD_diffX_neg = np.zeros(n)

    # Initialize for gen
    HBullD_diffY_gen = np.zeros(n)
    HBullD_diffX_gen = np.zeros(n)

    # Helper function to find the last non-zero indices before each index
    def get_last_nonzero_indices(indices, lm_low_window, extra_condition=None):
        last_nonzero = np.full(len(indices), -1, dtype=int)
        for i, idx in enumerate(indices):
            if extra_condition is None:
                nonzero_before = np.where(lm_low_window[:idx] != 0)[0]
            else:
                nonzero_before = np.where((lm_low_window[:idx] != 0) & extra_condition[:idx])[0]
            if len(nonzero_before) > 0:
                last_nonzero[i] = nonzero_before[-1]
        return last_nonzero

    # Process HBullD_1: Hidden Bullish Divergence based on Candlestick Lows
    valid_indices_1 = np.where(LM_Low_window_2_CS > 0)[0]
    valid_indices_1 = valid_indices_1[(valid_indices_1 >= window_1) & (valid_indices_1 < n - window_2)]
    last_nonzero_indices_1 = get_last_nonzero_indices(valid_indices_1, LM_Low_window_1_CS)
    valid_mask_1 = last_nonzero_indices_1 != -1
    valid_indices_1 = valid_indices_1[valid_mask_1]
    last_nonzero_indices_1 = last_nonzero_indices_1[valid_mask_1]

    if len(valid_indices_1) > 0:
        LM_Low_window_2_low = LM_Low_window_2_CS[valid_indices_1]
        LM_Low_window_1_low = LM_Low_window_1_CS[last_nonzero_indices_1]
        LM_Low_window_2_rsi = RSI[valid_indices_1]
        LM_Low_window_1_rsi = RSI[last_nonzero_indices_1]
        LM_Low_window_2_macd = macd_histogram[valid_indices_1]
        LM_Low_window_1_macd = macd_histogram[last_nonzero_indices_1]
        LM_Low_window_2_date = dates[valid_indices_1]
        LM_Low_window_1_date = dates[last_nonzero_indices_1]

        price_diff = LM_Low_window_2_low - LM_Low_window_1_low
        price_diff_percent = np.abs(100 * price_diff / LM_Low_window_1_low)
        rsi_diff = LM_Low_window_2_rsi - LM_Low_window_1_rsi
        macd_diff = LM_Low_window_2_macd - LM_Low_window_1_macd
        macd_sign_check = LM_Low_window_2_macd * LM_Low_window_1_macd

        price_condition = (price_diff > 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (
            (rsi_diff < 0) |
            (np.abs(rsi_diff) < RSI_tol) |
            ((macd_sign_check > 0) & (np.abs(rsi_diff) < 4 * RSI_tol))
        )
        macd_condition = macd_diff < 0

        bullish_divergence_1 = price_condition & rsi_condition & macd_condition

        HBullD_1[valid_indices_1[bullish_divergence_1]] = 1
        HBullD_Lower_Low_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_1_low[bullish_divergence_1]
        HBullD_Higher_Low_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_2_low[bullish_divergence_1]
        HBullD_Lower_Low_RSI_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_1_rsi[bullish_divergence_1]
        HBullD_Higher_Low_RSI_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_2_rsi[bullish_divergence_1]
        HBullD_Lower_Low_MACD_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_1_macd[bullish_divergence_1]
        HBullD_Higher_Low_MACD_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_2_macd[bullish_divergence_1]
        HBullD_Lower_Low_date_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_1_date[bullish_divergence_1]
        HBullD_Higher_Low_date_1[valid_indices_1[bullish_divergence_1]] = LM_Low_window_2_date[bullish_divergence_1]
        HBullD_Date_Gap_1[valid_indices_1[bullish_divergence_1]] = valid_indices_1[bullish_divergence_1] - last_nonzero_indices_1[bullish_divergence_1]

        # Store diffY and diffX for price slope
        HBullD_diffY_price_1[valid_indices_1[bullish_divergence_1]] = price_diff[bullish_divergence_1]
        HBullD_diffX_1[valid_indices_1[bullish_divergence_1]] = valid_indices_1[bullish_divergence_1] - last_nonzero_indices_1[bullish_divergence_1]

    # Process HBullD_2: Hidden Bullish Divergence based on MACD
    valid_indices_2 = np.where(LM_Low_window_2_MACD > 0)[0]
    valid_indices_2 = valid_indices_2[(valid_indices_2 >= window_1) & (valid_indices_2 < n - window_2)]
    last_nonzero_indices_2 = get_last_nonzero_indices(valid_indices_2, LM_Low_window_1_MACD)
    valid_mask_2 = last_nonzero_indices_2 != -1
    valid_indices_2 = valid_indices_2[valid_mask_2]
    last_nonzero_indices_2 = last_nonzero_indices_2[valid_mask_2]

    if len(valid_indices_2) > 0:
        LM_Low_window_2_low = LM_Low_window_2_MACD[valid_indices_2]
        LM_Low_window_1_low = LM_Low_window_1_MACD[last_nonzero_indices_2]
        LM_Low_window_2_rsi = RSI[valid_indices_2]
        LM_Low_window_1_rsi = RSI[last_nonzero_indices_2]
        LM_Low_window_2_macd = macd_histogram[valid_indices_2]
        LM_Low_window_1_macd = macd_histogram[last_nonzero_indices_2]
        LM_Low_window_2_date = dates[valid_indices_2]
        LM_Low_window_1_date = dates[last_nonzero_indices_2]

        # Compute price diffY for slope
        price_recent = CL[valid_indices_2]
        price_previous = CL[last_nonzero_indices_2]
        price_diff_price = price_recent - price_previous

        price_diff = LM_Low_window_2_low - LM_Low_window_1_low
        price_diff_percent = np.abs(100 * price_diff / LM_Low_window_1_low)
        rsi_diff = LM_Low_window_2_rsi - LM_Low_window_1_rsi
        macd_diff = LM_Low_window_2_macd - LM_Low_window_1_macd

        price_condition = (price_diff > 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (rsi_diff < 0) | (np.abs(rsi_diff) < RSI_tol)
        macd_condition = (
            ((LM_Low_window_2_macd < 0) & (LM_Low_window_1_macd < 0) & (macd_diff < 0)) |
            ((LM_Low_window_2_macd > 0) & (LM_Low_window_1_macd > 0) & (macd_diff < 0))
        )

        bullish_divergence_2 = price_condition & rsi_condition & macd_condition

        HBullD_2[valid_indices_2[bullish_divergence_2]] = 1
        HBullD_Lower_Low_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_1_low[bullish_divergence_2]
        HBullD_Higher_Low_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_2_low[bullish_divergence_2]
        HBullD_Lower_Low_RSI_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_1_rsi[bullish_divergence_2]
        HBullD_Higher_Low_RSI_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_2_rsi[bullish_divergence_2]
        HBullD_Lower_Low_MACD_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_1_macd[bullish_divergence_2]
        HBullD_Higher_Low_MACD_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_2_macd[bullish_divergence_2]
        HBullD_Lower_Low_date_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_1_date[bullish_divergence_2]
        HBullD_Higher_Low_date_2[valid_indices_2[bullish_divergence_2]] = LM_Low_window_2_date[bullish_divergence_2]
        HBullD_Date_Gap_2[valid_indices_2[bullish_divergence_2]] = valid_indices_2[bullish_divergence_2] - last_nonzero_indices_2[bullish_divergence_2]

        # Store diffY_price and diffX for price slope
        HBullD_diffY_price_2[valid_indices_2[bullish_divergence_2]] = price_diff_price[bullish_divergence_2]
        HBullD_diffX_2[valid_indices_2[bullish_divergence_2]] = valid_indices_2[bullish_divergence_2] - last_nonzero_indices_2[bullish_divergence_2]

    # Process HBullD_neg_MACD: Hidden Bullish Divergence based on Negative MACD Minimas
    valid_indices_3 = np.where((LM_Low_window_2_MACD > 0) & (macd_histogram < 0))[0]
    valid_indices_3 = valid_indices_3[(valid_indices_3 >= window_1) & (valid_indices_3 < n - window_2)]
    last_nonzero_indices_3 = get_last_nonzero_indices(valid_indices_3, LM_Low_window_1_MACD, macd_histogram < 0)
    valid_mask_3 = last_nonzero_indices_3 != -1
    valid_indices_3 = valid_indices_3[valid_mask_3]
    last_nonzero_indices_3 = last_nonzero_indices_3[valid_mask_3]

    if len(valid_indices_3) > 0:
        LM_Low_window_2_low = LM_Low_window_2_MACD[valid_indices_3]
        LM_Low_window_1_low = LM_Low_window_1_MACD[last_nonzero_indices_3]
        LM_Low_window_2_rsi = RSI[valid_indices_3]
        LM_Low_window_1_rsi = RSI[last_nonzero_indices_3]
        LM_Low_window_2_macd = macd_histogram[valid_indices_3]
        LM_Low_window_1_macd = macd_histogram[last_nonzero_indices_3]
        LM_Low_window_2_date = dates[valid_indices_3]
        LM_Low_window_1_date = dates[last_nonzero_indices_3]

        # Compute price diffY for slope
        price_recent = CL[valid_indices_3]
        price_previous = CL[last_nonzero_indices_3]
        price_diff_price = price_recent - price_previous

        price_diff = LM_Low_window_2_low - LM_Low_window_1_low
        price_diff_percent = np.abs(100 * price_diff / LM_Low_window_1_low)
        rsi_diff = LM_Low_window_2_rsi - LM_Low_window_1_rsi
        macd_diff = LM_Low_window_2_macd - LM_Low_window_1_macd
        macd_diff_percent = np.abs(100 * macd_diff / (0.000001 + LM_Low_window_1_macd))

        price_condition = (price_diff > 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (rsi_diff < 0) | (np.abs(rsi_diff) < 3.5 * RSI_tol)
        macd_condition = (macd_diff < 0) | (macd_diff_percent < MACD_tol)

        bullish_divergence_3 = price_condition & rsi_condition & macd_condition

        HBullD_neg_MACD[valid_indices_3[bullish_divergence_3]] = 1
        HBullD_Lower_Low_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_1_low[bullish_divergence_3]
        HBullD_Higher_Low_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_2_low[bullish_divergence_3]
        HBullD_Lower_Low_RSI_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_1_rsi[bullish_divergence_3]
        HBullD_Higher_Low_RSI_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_2_rsi[bullish_divergence_3]
        HBullD_Lower_Low_MACD_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_1_macd[bullish_divergence_3]
        HBullD_Higher_Low_MACD_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_2_macd[bullish_divergence_3]
        HBullD_Lower_Low_date_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_1_date[bullish_divergence_3]
        HBullD_Higher_Low_date_neg_MACD[valid_indices_3[bullish_divergence_3]] = LM_Low_window_2_date[bullish_divergence_3]
        HBullD_Date_Gap_neg_MACD[valid_indices_3[bullish_divergence_3]] = valid_indices_3[bullish_divergence_3] - last_nonzero_indices_3[bullish_divergence_3]

        # Store diffY_price and diffX for price slope
        HBullD_diffY_price_neg[valid_indices_3[bullish_divergence_3]] = price_diff_price[bullish_divergence_3]
        HBullD_diffX_neg[valid_indices_3[bullish_divergence_3]] = valid_indices_3[bullish_divergence_3] - last_nonzero_indices_3[bullish_divergence_3]

    # Combine HBullD_1 and HBullD_2 into HBullD_gen
    HBullD_gen = np.zeros(n)
    HBullD_Lower_Low_gen = np.zeros(n)
    HBullD_Higher_Low_gen = np.zeros(n)
    HBullD_Lower_Low_RSI_gen = np.zeros(n)
    HBullD_Higher_Low_RSI_gen = np.zeros(n)
    HBullD_Lower_Low_MACD_gen = np.zeros(n)
    HBullD_Higher_Low_MACD_gen = np.zeros(n)
    HBullD_Lower_Low_date_gen = np.empty(n, dtype=object)
    HBullD_Higher_Low_date_gen = np.empty(n, dtype=object)
    HBullD_Date_Gap_gen = np.zeros(n)

    indices = np.arange(window_1, n - window_2)
    condition_1 = HBullD_1[indices] == 1
    condition_2 = HBullD_2[indices] == 1
    combined_condition_gen = condition_1 | condition_2
    valid_indices_gen = indices[combined_condition_gen]

    for idx in valid_indices_gen:
        if HBullD_1[idx] == 1 or (HBullD_1[idx] == 1 and HBullD_2[idx] == 1):
            HBullD_gen[idx] = 1
            HBullD_Lower_Low_gen[idx] = HBullD_Lower_Low_1[idx]
            HBullD_Higher_Low_gen[idx] = HBullD_Higher_Low_1[idx]
            HBullD_Lower_Low_RSI_gen[idx] = HBullD_Lower_Low_RSI_1[idx]
            HBullD_Higher_Low_RSI_gen[idx] = HBullD_Higher_Low_RSI_1[idx]
            HBullD_Lower_Low_MACD_gen[idx] = HBullD_Lower_Low_MACD_1[idx]
            HBullD_Higher_Low_MACD_gen[idx] = HBullD_Higher_Low_MACD_1[idx]
            HBullD_Lower_Low_date_gen[idx] = HBullD_Lower_Low_date_1[idx]
            HBullD_Higher_Low_date_gen[idx] = HBullD_Higher_Low_date_1[idx]
            HBullD_Date_Gap_gen[idx] = HBullD_Date_Gap_1[idx]
            # Set diff for gen
            HBullD_diffY_gen[idx] = HBullD_diffY_price_1[idx]
            HBullD_diffX_gen[idx] = HBullD_diffX_1[idx]
        elif HBullD_2[idx] == 1:
            HBullD_gen[idx] = 1
            HBullD_Lower_Low_gen[idx] = HBullD_Lower_Low_2[idx]
            HBullD_Higher_Low_gen[idx] = HBullD_Higher_Low_2[idx]
            HBullD_Lower_Low_RSI_gen[idx] = HBullD_Lower_Low_RSI_2[idx]
            HBullD_Higher_Low_RSI_gen[idx] = HBullD_Higher_Low_RSI_2[idx]
            HBullD_Lower_Low_MACD_gen[idx] = HBullD_Lower_Low_MACD_2[idx]
            HBullD_Higher_Low_MACD_gen[idx] = HBullD_Higher_Low_MACD_2[idx]
            HBullD_Lower_Low_date_gen[idx] = HBullD_Lower_Low_date_2[idx]
            HBullD_Higher_Low_date_gen[idx] = HBullD_Higher_Low_date_2[idx]
            HBullD_Date_Gap_gen[idx] = HBullD_Date_Gap_2[idx]
            # Set diff for gen
            HBullD_diffY_gen[idx] = HBullD_diffY_price_2[idx]
            HBullD_diffX_gen[idx] = HBullD_diffX_2[idx]

    # Compute rolling normalization parameters for historical Yaxis
    bars = 600
    ratio = 2.40
    high_rolling_max = ohlc['high'].rolling(window=bars, min_periods=1).max()
    low_rolling_min = ohlc['low'].rolling(window=bars, min_periods=1).min()
    Yaxis_series = high_rolling_max - low_rolling_min
    height = bars / ratio

    # Compute MACD rolling normalization parameters
    ratio_macd = 4.80
    height_macd = bars / ratio_macd
    macd_high_rolling_max = pd.Series(macd_histogram).rolling(window=bars, min_periods=1).max()
    macd_low_rolling_min = pd.Series(macd_histogram).rolling(window=bars, min_periods=1).min()
    Yaxis_macd_series = macd_high_rolling_max - macd_low_rolling_min

    # Compute angles for gen
    HBullD_gen_Slope = np.zeros(n)
    gen_indices = np.where(HBullD_gen == 1)[0]
    for idx in gen_indices:
        Yaxis = Yaxis_series.iloc[idx]
        if Yaxis == 0:
            Yaxis = 1e-6
        diffY = HBullD_diffY_gen[idx]
        diffX = HBullD_diffX_gen[idx]
        abs_diffY = abs(diffY)
        if diffX == 0 or abs_diffY == 0:
            angle = 0.0
        else:
            diffY_to_Yaxis = Yaxis / abs_diffY
            normalised_slope = (height / diffY_to_Yaxis) / diffX
            angle = round(math.atan(normalised_slope) * 180 / math.pi, 2)
        HBullD_gen_Slope[idx] = angle

    # Compute angles for neg_MACD
    HBullD_neg_MACD_Slope = np.zeros(n)
    neg_indices = np.where(HBullD_neg_MACD == 1)[0]
    for idx in neg_indices:
        Yaxis = Yaxis_series.iloc[idx]
        if Yaxis == 0:
            Yaxis = 1e-6
        diffY = HBullD_diffY_price_neg[idx]
        diffX = HBullD_diffX_neg[idx]
        abs_diffY = abs(diffY)
        if diffX == 0 or abs_diffY == 0:
            angle = 0.0
        else:
            diffY_to_Yaxis = Yaxis / abs_diffY
            normalised_slope = (height / diffY_to_Yaxis) / diffX
            angle = round(math.atan(normalised_slope) * 180 / math.pi, 2)
        HBullD_neg_MACD_Slope[idx] = angle

    # Compute MACD angles for gen
    HBullD_gen_MACD_Slope = np.zeros(n)
    for idx in gen_indices:
        Yaxis_macd = Yaxis_macd_series.iloc[idx]
        if Yaxis_macd == 0:
            Yaxis_macd = 1e-6
        diffY_macd = HBullD_Higher_Low_MACD_gen[idx] - HBullD_Lower_Low_MACD_gen[idx]
        abs_diffY_macd = abs(diffY_macd)
        diffX = HBullD_diffX_gen[idx]
        if diffX == 0 or abs_diffY_macd == 0:
            angle = 0.0
        else:
            diffY_to_Yaxis = Yaxis_macd / abs_diffY_macd
            normalised_slope = (height_macd / diffY_to_Yaxis) / diffX
            angle = round(math.atan(normalised_slope) * 180 / math.pi, 2)
        HBullD_gen_MACD_Slope[idx] = angle

    # Compute MACD angles for neg_MACD
    HBullD_neg_MACD_MACD_Slope = np.zeros(n)
    for idx in neg_indices:
        Yaxis_macd = Yaxis_macd_series.iloc[idx]
        if Yaxis_macd == 0:
            Yaxis_macd = 1e-6
        diffY_macd = HBullD_Higher_Low_MACD_neg_MACD[idx] - HBullD_Lower_Low_MACD_neg_MACD[idx]
        abs_diffY_macd = abs(diffY_macd)
        diffX = HBullD_diffX_neg[idx]
        if diffX == 0 or abs_diffY_macd == 0:
            angle = 0.0
        else:
            diffY_to_Yaxis = Yaxis_macd / abs_diffY_macd
            normalised_slope = (height_macd / diffY_to_Yaxis) / diffX
            angle = round(math.atan(normalised_slope) * 180 / math.pi, 2)
        HBullD_neg_MACD_MACD_Slope[idx] = angle

    # Assign results to DataFrame
    ohlc['HBullD_gen'] = HBullD_gen
    ohlc['HBullD_Lower_Low_gen'] = HBullD_Lower_Low_gen
    ohlc['HBullD_Higher_Low_gen'] = HBullD_Higher_Low_gen
    ohlc['HBullD_Lower_Low_RSI_gen'] = HBullD_Lower_Low_RSI_gen
    ohlc['HBullD_Higher_Low_RSI_gen'] = HBullD_Higher_Low_RSI_gen
    ohlc['HBullD_Lower_Low_MACD_gen'] = HBullD_Lower_Low_MACD_gen
    ohlc['HBullD_Higher_Low_MACD_gen'] = HBullD_Higher_Low_MACD_gen
    ohlc['HBullD_Lower_Low_date_gen'] = HBullD_Lower_Low_date_gen
    ohlc['HBullD_Higher_Low_date_gen'] = HBullD_Higher_Low_date_gen
    ohlc['HBullD_Date_Gap_gen'] = HBullD_Date_Gap_gen
    ohlc['HBullD_gen_Slope'] = HBullD_gen_Slope
    ohlc['HBullD_gen_MACD_Slope'] = HBullD_gen_MACD_Slope

    ohlc['HBullD_neg_MACD'] = HBullD_neg_MACD
    ohlc['HBullD_Lower_Low_neg_MACD'] = HBullD_Lower_Low_neg_MACD
    ohlc['HBullD_Higher_Low_neg_MACD'] = HBullD_Higher_Low_neg_MACD
    ohlc['HBullD_Lower_Low_RSI_neg_MACD'] = HBullD_Lower_Low_RSI_neg_MACD
    ohlc['HBullD_Higher_Low_RSI_neg_MACD'] = HBullD_Higher_Low_RSI_neg_MACD
    ohlc['HBullD_Lower_Low_MACD_neg_MACD'] = HBullD_Lower_Low_MACD_neg_MACD
    ohlc['HBullD_Higher_Low_MACD_neg_MACD'] = HBullD_Higher_Low_MACD_neg_MACD
    ohlc['HBullD_Lower_Low_date_neg_MACD'] = HBullD_Lower_Low_date_neg_MACD
    ohlc['HBullD_Higher_Low_date_neg_MACD'] = HBullD_Higher_Low_date_neg_MACD
    ohlc['HBullD_Date_Gap_neg_MACD'] = HBullD_Date_Gap_neg_MACD
    ohlc['HBullD_neg_MACD_Slope'] = HBullD_neg_MACD_Slope
    ohlc['HBullD_neg_MACD_MACD_Slope'] = HBullD_neg_MACD_MACD_Slope

    return ohlc[['HBullD_gen', 'HBullD_neg_MACD', 'HBullD_gen_Slope', 'HBullD_neg_MACD_Slope', 'HBullD_gen_MACD_Slope', 'HBullD_neg_MACD_MACD_Slope']]
