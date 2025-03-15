import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
import os
import time
from sklearn.preprocessing import MinMaxScaler
import logging
from multiprocessing import get_context, pool
logger = logging.getLogger(__name__)


# Constants
TS_LENGTH = 81

# using volume weighted average price for now. could switch to open or close
PRICE_COL = 'open'

# theshold to use to determine whether stock is "flat" instead of up or down. Can change this later if we want
THRESHOLD = 0.1

T_COLS = ["t" + str(i) for i in range(TS_LENGTH)]


def is_valid_ts(candidate_ts, ts_length=TS_LENGTH):
    """"
    Check if the first and last entries of a time series are TS_LENGTH + 1 minutes apart
    :param candidate_ts: pandas dataframe - candidate time series to be checked
    :return: True if first and last entries are TS_LENGTH + 1 minutes apart. False if not
    """
    return ((candidate_ts.iloc[-1]['timestamp'] - candidate_ts.iloc[0]['timestamp']).total_seconds()) / 60 == ts_length

    
def make_row(ticker, df_slice, ts_length=TS_LENGTH, price_col=PRICE_COL, threshold=THRESHOLD):
    metadata = {"Symbol": ticker,
                "Start_Date": df_slice.iloc[0]['timestamp'],
                "End_Date": df_slice.iloc[-1]['timestamp']}
    
    min_price, max_price = df_slice[price_col].min(), df_slice[price_col].max()
    scaled_ts = (df_slice[price_col] - min_price) / (max_price - min_price + 1e-9)  # Avoid division by zero
    scaled_ts = np.asarray(scaled_ts)
    if scaled_ts[-2] == 0:
        return None
    percent_change = ((scaled_ts[-1] - scaled_ts[-2]) / scaled_ts[-2]) * 100
    
    if percent_change > threshold:
        two_class_label = 1
        three_class_label = 1
    elif abs(percent_change) <= threshold:
        two_class_label = 1
        three_class_label = 3
    elif percent_change <= -1 * threshold:
        two_class_label = 2
        three_class_label = 2

    ts_dic = dict(zip(T_COLS, scaled_ts))
    metadata.update(ts_dic)
    metadata.update({"two_class_label": two_class_label, "three_class_label":three_class_label})
    row = pd.DataFrame(metadata, index=[0])
    return row


def get_valid_ts_for_week(ticker, week_df_slice, ts_length=TS_LENGTH):
    """
    fill out later
    
    """
    i = 0
    rows = []
    while i <= len(week_df_slice) - ts_length:
        candidate_ts = week_df_slice.iloc[i: i + ts_length]
        if is_valid_ts(candidate_ts=candidate_ts, ts_length=ts_length):
            i = i +  ts_length
            row = make_row(ticker, candidate_ts, price_col=PRICE_COL, ts_length=TS_LENGTH)
            if row is not None:
                rows.append(row)
        i += 1

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_training_data_one_stock(data_file_path):

    training_data = []
    start_date = pd.to_datetime("2024-01-02 00:00:00")
    end_date = start_date + pd.Timedelta(7, 'd')
    df = pd.read_csv("/home/holden/github/OMSCS/DL/stock_data/" + data_file_path, parse_dates=['timestamp'])
    
    #df = df[df['time_diff_minutes'] == 1]
    #df = df.reset_index(drop=True)
    ticker = data_file_path[:-4]
    while end_date < pd.to_datetime("2024-12-27 23:59:00"):
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        df_slice = df[mask]
        if len(df_slice) > 80: 
            result = get_valid_ts_for_week(ticker, df_slice)
            if not result.empty:
                training_data.append(result)
        start_date = end_date
        end_date = start_date + pd.Timedelta(7, 'd')
    #print(training_data) 
    return pd.concat(training_data, ignore_index=True) if training_data else pd.DataFrame()


def process_stock(stock):
    print("pulling " + stock)
    return make_training_data_one_stock(stock)


def main():
    start = time.time()
    logging.basicConfig(filename="dataset_creation_logging.log", level=logging.INFO)
    logger.info("Started") 
    stock_list = os.listdir("stock_data")
    all_data = pd.DataFrame()
    stocks_per_batch = 20
    num_batches = len(stock_list) // stocks_per_batch
    last_batch_size = len(stock_list) % stocks_per_batch

    for i in range(num_batches + 1):
        print("processing batch " + str(i) + "/" + str(num_batches + 1) + "...")
        if i == num_batches:
            current_stocks = stock_list[i * stocks_per_batch: i * stocks_per_batch + last_batch_size]
        else:
            current_stocks = stock_list[i * stocks_per_batch: (i+1) * stocks_per_batch]
        print("processing stocks")
        print(current_stocks)
        with get_context('spawn').Pool(16) as p:
            result = p.map(process_stock, current_stocks)
        for item in result:
            if len(item) > 0:
                all_data = pd.concat([all_data, item]).reset_index(drop=True)
                
    #for stock in tqdm(stock_list[:5]):
    #    data_one_stock = make_training_data_one_stock(stock)
    #    logger.info("Pulled " + str(len(data_one_stock)) + " time series for stock: " + stock)
    #    if len(data_one_stock) > 0:
    #        all_data = pd.concat([all_data, data_one_stock])

    logger.info("Saving to CSV...")
    end = time.time()
    duration = (end - start) / 60
    logger.info("Took " + str(duration) + " minutes")
    all_data.to_csv("all_data_test.csv", index=False)

if __name__ == "__main__":
    main()
