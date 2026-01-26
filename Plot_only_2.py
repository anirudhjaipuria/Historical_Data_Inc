import finplot as fplt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Set Pandas display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Adjust width to avoid wrapping

csv_file_path = 'orig.csv'
# csv_file_path = r'whole.parquet'

# Read the CSV File
df = pd.read_csv(csv_file_path, low_memory=False)
# df = pd.read_parquet(csv_file_path)#.head(200)

fplt.background = fplt.odd_plot_background = '#242320'  # Adjust Plot Background colour
fplt.cross_hair_color = '#eefa'  # Adjust Crosshair colour

# Plotting Chart----------------------------------------------
# Plotting Candlesticks---------------------------------------
ax1, ax2, ax3 = fplt.create_plot('Chart', rows=3)
df['date'] = pd.to_datetime(df['date'], format='mixed')
candles = df[['date', 'open', 'close', 'high', 'low', 'macd_histogram']]
# candles = df[['date', 'open', 'close', 'high', 'low']]
fplt.candlestick_ochl(candles, ax=ax1)          # Plotting candlestick chart using

# Plotting RSI
fplt.plot(df.RSI, color='#000000', width=2, ax=ax2, legend='RSI')
fplt.set_y_range(0, 100, ax=ax2)  # Setting y-axis range
# fplt.add_horizontal_band(0, 100, color='#FFFFFF', ax=ax2)  # Changing background color to white
# fplt.add_horizontal_band(30, 70, color='#ffcccc', ax=ax2)  # Adding band for 30-70 RSI
fplt.add_horizontal_band(0, 1, color='#000000', ax=ax2)  # Dummy band to mark the ending of the plot
fplt.add_horizontal_band(99, 100, color='#000000', ax=ax2)  # Dummy band to mark the ending of the plot

# Plotting the MACD
fplt.volume_ocv(df[['date', 'open', 'close', 'macd_histogram']], ax=ax3, colorfunc=fplt.strength_colorfilter)

# # Plotting EMAs-----------------------------------------------
# df.EMA_20.plot(ax=ax1, legend='20-EMA') # Plotting exponential moving average period = 20
# df.EMA_50.plot(ax=ax1, legend='50-EMA')  # Plotting exponential moving average period = 50
# df.EMA_100.plot(ax=ax1, legend='100-EMA')  # Plotting exponential moving average period = 100
# df.EMA_200.plot(ax=ax1, legend='200-EMA')  # Plotting exponential moving average period = 200

percentage_range = 0.1

# for i in range(1, 17):
#     support = df[f'Support_{i}'].iloc[-1]
#     delta = percentage_range / 100 * support
#     fplt.add_horizontal_band(support - delta, support + delta, color='#ffcccc', ax=ax1)

for i in range(1, 17):
    if df[f'Fib_50_long_{i}'][len(df)-1] > 0:
        fplt.plot(pd.to_datetime(df.date[len(df)-1]), df[f'Fib_50_long_{i}'][len(df)-1], style='x', ax=ax1, color='yellow')
    if df[f'Fib_50_short_{i}'][len(df)-1] > 0:
        fplt.plot(pd.to_datetime(df.date[len(df)-1]), df[f'Fib_50_short_{i}'][len(df)-1], style='x', ax=ax1, color='yellow')
#     if df[f'Fib_618_{i}'][len(df) - 1] > 0:
#         fplt.plot(pd.to_datetime(df.date[len(df) - 1]), df[f'Fib_618_{i}'][len(df) - 1], style='x', ax=ax1, color='white')
#
# for i in range(1, 17):
#     if df[f'Fib_50_{i}'][len(df)-2] > 0:
#         fplt.plot(pd.to_datetime(df.date[len(df)-2]), df[f'Fib_50_{i}'][len(df)-2], style='x', ax=ax1, color='yellow')
#     if df[f'Fib_618_{i}'][len(df) - 2] > 0:
#         fplt.plot(pd.to_datetime(df.date[len(df) - 2]), df[f'Fib_618_{i}'][len(df) - 2], style='x', ax=ax1, color='white')

# for i in range(2, len(df)):
    # if df['Strategy_1'][i] == 1 or df['Strategy_2'][i] == 1:
    #     fplt.plot(pd.to_datetime(df.date[i]), df.low[i], style='o', ax=ax1, color='white')
    # if df['Strategy_2'][i] == 1:
    #     fplt.plot(pd.to_datetime(df.date[i]), 0.96*df.low[i], style='o', ax=ax1, color='blue')
    # if df['Strategy_3'][i] == 1:
    #     fplt.plot(pd.to_datetime(df.date[i]), 0.94 * df.low[i], style='o', ax=ax1, color='red')
    # if df['Strategy_4'][i] == 1:
    #     fplt.plot(pd.to_datetime(df.date[i]), 0.92 * df.low[i], style='o', ax=ax1, color='pink')
    # if df['Strategy_6'][i] == 1:
    #     fplt.plot(pd.to_datetime(df.date[i]), 1.0 * df.low[i], style='o', ax=ax1, color='yellow')

# for i in range(2, len(df)):
    # if df['Level_1_High_window_1_CS'][i] > 0:
    #     fplt.plot(pd.to_datetime(df['date'][i]), df['high'][i], style='x', ax=ax1, color='white')
    # if df['Level_1_Low_window_1_CS'][i] > 0:
    #     fplt.plot(pd.to_datetime(df['date'][i]), df['low'][i], style='x', ax=ax1, color='yellow')

    # if df['Level_2_High_window_1_CS'][i] > 0:
    #     fplt.plot(pd.to_datetime(df['date'][i]), 1.05*df['high'][i], style='x', ax=ax1, color='white')
    # if df['Level_2_Low_window_1_CS'][i] > 0:
    #     fplt.plot(pd.to_datetime(df['date'][i]), 0.95*df['low'][i], style='x', ax=ax1, color='yellow')

# for i in range(2, len(df)):
    # if df['HBearD_gen'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['HBearD_Higher_High_date_gen'][i]), df['HBearD_Higher_High_gen'][i], style='x', ax=ax1, color='blue')
    #     fplt.plot(pd.to_datetime(df['HBearD_Lower_High_date_gen'][i]), df['HBearD_Lower_High_gen'][i], style='x', ax=ax1, color='white')
    #
    #     fplt.plot(pd.to_datetime(df['HBearD_Higher_High_date_gen'][i]), df['HBearD_Higher_High_RSI_gen'][i], style='x', ax=ax2, color='blue')
    #     fplt.plot(pd.to_datetime(df['HBearD_Lower_High_date_gen'][i]), df['HBearD_Lower_High_RSI_gen'][i], style='x', ax=ax2, color='white')
    #
    #     fplt.plot(pd.to_datetime(df['HBearD_Higher_High_date_gen'][i]), df['HBearD_Higher_High_MACD_gen'][i], style='x', ax=ax3, color='blue')
    #     fplt.plot(pd.to_datetime(df['HBearD_Lower_High_date_gen'][i]), df['HBearD_Lower_High_MACD_gen'][i], style='x', ax=ax3, color='white')

    # if df['HBearD_pos_MACD'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['HBearD_Higher_High_date_pos_MACD'][i]), df['HBearD_Higher_High_pos_MACD'][i], style='x', ax=ax1, color='blue')
    #     fplt.plot(pd.to_datetime(df['HBearD_Lower_High_date_pos_MACD'][i]), df['HBearD_Lower_High_pos_MACD'][i], style='x', ax=ax1, color='white')
    #
    #     fplt.plot(pd.to_datetime(df['HBearD_Higher_High_date_pos_MACD'][i]), df['HBearD_Higher_High_RSI_pos_MACD'][i], style='x', ax=ax2, color='blue')
    #     fplt.plot(pd.to_datetime(df['HBearD_Lower_High_date_pos_MACD'][i]), df['HBearD_Lower_High_RSI_pos_MACD'][i], style='x', ax=ax2, color='white')
    #
    #     fplt.plot(pd.to_datetime(df['HBearD_Higher_High_date_pos_MACD'][i]), df['HBearD_Higher_High_MACD_pos_MACD'][i], style='x', ax=ax3, color='blue')
    #     fplt.plot(pd.to_datetime(df['HBearD_Lower_High_date_pos_MACD'][i]), df['HBearD_Lower_High_MACD_pos_MACD'][i], style='x', ax=ax3, color='white')

    # if df['CBearD_gen'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_gen'][i]), df['CBearD_Lower_High_gen'][i], style='x', ax=ax1, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_gen'][i]), df['CBearD_Higher_High_gen'][i], style='x', ax=ax1, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_gen'][i]), df['CBearD_Lower_High_RSI_gen'][i], style='x', ax=ax2, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_gen'][i]), df['CBearD_Higher_High_RSI_gen'][i], style='x', ax=ax2, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_gen'][i]), df['CBearD_Lower_High_MACD_gen'][i], style='x', ax=ax3, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_gen'][i]), df['CBearD_Higher_High_MACD_gen'][i], style='x', ax=ax3, color='blue')

    # if df['CBearD_gen'][i] == 1 and df['CBearD_pos_MACD'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_gen'][i]), df['CBearD_Lower_High_gen'][i], style='x', ax=ax1, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_gen'][i]), df['CBearD_Higher_High_gen'][i], style='x', ax=ax1, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_gen'][i]), df['CBearD_Lower_High_RSI_gen'][i], style='x', ax=ax2, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_gen'][i]), df['CBearD_Higher_High_RSI_gen'][i], style='x', ax=ax2, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_gen'][i]), df['CBearD_Lower_High_MACD_gen'][i], style='x', ax=ax3, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_gen'][i]), df['CBearD_Higher_High_MACD_gen'][i], style='x', ax=ax3, color='blue')

    # if df['CBearD_pos_MACD'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_pos_MACD'][i]), df['CBearD_Lower_High_pos_MACD'][i], style='x', ax=ax1, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_pos_MACD'][i]), df['CBearD_Higher_High_pos_MACD'][i], style='x', ax=ax1, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_pos_MACD'][i]), df['CBearD_Lower_High_RSI_pos_MACD'][i], style='x', ax=ax2, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_pos_MACD'][i]), df['CBearD_Higher_High_RSI_pos_MACD'][i], style='x', ax=ax2, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBearD_Lower_High_date_pos_MACD'][i]), df['CBearD_Lower_High_MACD_pos_MACD'][i], style='x', ax=ax3, color='red')
    #     fplt.plot(pd.to_datetime(df['CBearD_Higher_High_date_pos_MACD'][i]), df['CBearD_Higher_High_MACD_pos_MACD'][i], style='x', ax=ax3, color='blue')

    # if df['CBullD_neg_MACD'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['CBullD_Lower_Low_date_neg_MACD'][i]), df['CBullD_Lower_Low_neg_MACD'][i], style='x', ax=ax1, color='white')
    #     fplt.plot(pd.to_datetime(df['CBullD_Higher_Low_date_neg_MACD'][i]), df['CBullD_Higher_Low_neg_MACD'][i], style='x', ax=ax1, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBullD_Lower_Low_date_neg_MACD'][i]), df['CBullD_Lower_Low_RSI_neg_MACD'][i], style='x', ax=ax2, color='white')
    #     fplt.plot(pd.to_datetime(df['CBullD_Higher_Low_date_neg_MACD'][i]), df['CBullD_Higher_Low_RSI_neg_MACD'][i], style='x', ax=ax2, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['CBullD_Lower_Low_date_neg_MACD'][i]), df['CBullD_Lower_Low_MACD_neg_MACD'][i], style='x', ax=ax3, color='white')
    #     fplt.plot(pd.to_datetime(df['CBullD_Higher_Low_date_neg_MACD'][i]), df['CBullD_Higher_Low_MACD_neg_MACD'][i], style='x', ax=ax3, color='blue')

    # if df['HBullD_neg_MACD'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['HBullD_Lower_Low_date_neg_MACD'][i]), 0.95*df['HBullD_Lower_Low_neg_MACD'][i], style='x', ax=ax1, color='white')
    #     fplt.plot(pd.to_datetime(df['HBullD_Higher_Low_date_neg_MACD'][i]), df['HBullD_Higher_Low_neg_MACD'][i], style='x', ax=ax1, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['HBullD_Lower_Low_date_neg_MACD'][i]), df['HBullD_Lower_Low_RSI_neg_MACD'][i], style='x', ax=ax2, color='white')
    #     fplt.plot(pd.to_datetime(df['HBullD_Higher_Low_date_neg_MACD'][i]), df['HBullD_Higher_Low_RSI_neg_MACD'][i], style='x', ax=ax2, color='blue')
    #
    #     fplt.plot(pd.to_datetime(df['HBullD_Lower_Low_date_neg_MACD'][i]), df['HBullD_Lower_Low_MACD_neg_MACD'][i], style='x', ax=ax3, color='white')
    #     fplt.plot(pd.to_datetime(df['HBullD_Higher_Low_date_neg_MACD'][i]), df['HBullD_Higher_Low_MACD_neg_MACD'][i], style='x', ax=ax3, color='blue')

    # if df['CBullD_x2'][i] == 1:
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Lower_Low_date'][i]), 0.90*df['CBullD_x2_Lower_Low'][i], style='x', ax=ax1, color='white')
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Higher_Low_date'][i]), 0.95*df['CBullD_x2_Higher_Low'][i], style='x', ax=ax1, color='blue')
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Higher_Low_date_x2'][i]), df['CBullD_x2_Higher_Low_x2'][i], style='x', ax=ax1, color='yellow')
    #
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Lower_Low_date'][i]), df['CBullD_x2_Lower_Low_RSI'][i], style='x', ax=ax2, color='white')
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Higher_Low_date'][i]), df['CBullD_x2_Higher_Low_RSI'][i], style='x', ax=ax2, color='blue')
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Higher_Low_date_x2'][i]), df['CBullD_x2_Higher_Low_RSI_x2'][i], style='x', ax=ax2, color='yellow')
    #
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Lower_Low_date'][i]), df['CBullD_x2_Lower_Low_MACD'][i], style='x', ax=ax3, color='white')
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Higher_Low_date'][i]), df['CBullD_x2_Higher_Low_MACD'][i], style='x', ax=ax3, color='blue')
    #     fplt.plot(pd.to_datetime(df['CBullD_x2_Higher_Low_date_x2'][i]), df['CBullD_x2_Higher_Low_MACD_x2'][i], style='x', ax=ax3, color='yellow')


# Plotting trendlines using only the last row, extended to the last date in the CSV
last_row = df.iloc[-1]
last_date = df['date'].iloc[-1]  # last date in price data

for col in df.columns:
    if col.endswith('_Start_Price'):
        prefix = col.removesuffix('_Start_Price')  # or col[:-12]

        start_val_col = f'{prefix}_Start_Price'
        end_val_col = f'{prefix}_End_Price'
        start_date_col = f'{prefix}_Start_Date'
        end_date_col = f'{prefix}_End_Date'

        if all(c in df.columns for c in [start_val_col, end_val_col, start_date_col, end_date_col]):
            start_val = last_row[start_val_col]
            end_val = last_row[end_val_col]
            start_date = last_row[start_date_col]
            end_date = last_row[end_date_col]

            if (start_val != 0 and end_val != 0 and
                pd.notnull(start_date) and pd.notnull(end_date)):

                try:
                    x0 = pd.to_datetime(start_date)
                    x1 = pd.to_datetime(end_date)
                    x2 = pd.to_datetime(last_date)
                    y0 = float(start_val)
                    y1 = float(end_val)

                    # Calculate slope and projected y at x2
                    delta_days = (x1 - x0).total_seconds()
                    if delta_days == 0:
                        continue  # Avoid division by zero

                    slope = (y1 - y0) / delta_days
                    total_seconds_to_last = (x2 - x0).total_seconds()
                    y2 = y0 + slope * total_seconds_to_last

                    # Plot extended trendline from x0 to x2
                    fplt.add_line((x0, y0), (x2, y2), ax=ax1, color='white', width=2)
                except Exception as e:
                    print(f"Failed to extend and plot trendline {prefix}: {e}")

fplt.show()