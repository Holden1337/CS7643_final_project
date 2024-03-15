import pandas as pd
from polygon import RESTClient
from tqdm import tqdm 
import time
import logging
logger = logging.getLogger(__name__)

api_key = "BMd3CY940NNG3r9_reWCzTBbXc1cy2Oi"
client = RESTClient(api_key)

#from_date = "2024-11-29"
to_date = "2024-12-30"

constituents = pd.read_csv("constituents.csv")
tickers = list(constituents['Symbol'].values)
base_dir = '/home/holden/github/OMSCS/DL/stock_data/'

def get_price_df(ticker):
    """
    use polygon api to get stock data for year 2024.
    Different stocks have different amount of data with 
    some minutes being missing so we can't just uniformly
    pull all stocks and concat them into a single df. we can  
    just make a folder and save all the csvs there and pull them later
    :param ticker: string - stock ticker label
    :return df: dataframe with stock price info
    """

    multiplier = 1  # 1-minute intervals
    timespan = "minute"
    df = pd.DataFrame()
    multiplier = 1  # 1-minute intervals
    timespan = "minute"
    
    if len(df) == 0:
        from_date = "2024-01-01"
        to_date = "2024-12-31"
        aggs = client.get_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=from_date, to=to_date, limit=50000000)
        temp_df = pd.DataFrame(aggs)
        # Convert timestamp to datetime
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], unit="ms")
        df = pd.concat([df, temp_df])
    
    while df.iloc[-1]['timestamp'] < pd.to_datetime("2025-02-10 00:00:00"):
        from_date = df.iloc[-1]['timestamp'].strftime("%Y-%m-%d")
        aggs = client.get_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=from_date, to=to_date, limit=50000000)
        temp_df = pd.DataFrame(aggs)
        
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], unit="ms")
        temp_df_filtered = temp_df[~temp_df['timestamp'].isin(df['timestamp'])]
        if temp_df_filtered.iloc[-1]['timestamp'] > pd.to_datetime("2024-12-28 00:00:00"):
            temp_df_filtered = temp_df_filtered[temp_df_filtered['timestamp'] < pd.to_datetime("2024-12-28 00:00:00")]
            temp_df_filtered = temp_df_filtered.reset_index(drop=True)
        if len(temp_df_filtered) > 0:
            df = pd.concat([df, temp_df_filtered], ignore_index=True)
        else:
            return df
        

def main():
    logging.basicConfig(filename="stock_price_pulling_logging.log", level=logging.INFO)
    logger.info("Started")
    num_successes = 0
    num_fails = 0
    for ticker in tqdm(tickers):
        try:
            df = get_price_df(ticker)
            save_file_path = base_dir + ticker + ".csv"
            df.to_csv(save_file_path, index=False)
            logger.info("Finished pulling data for " + ticker)
            num_successes += 1
        except Exception as e:
            logger.info("Issue pulling data for " + ticker)
            num_fails += 1
            pass
        time.sleep(60)
    logger.info("Number of stocks successfully pulled: " + str(num_successes))
    logger.info("Number of stocks that failed to pull: " + str(num_fails))


if __name__ == '__main__':
    main()
    


