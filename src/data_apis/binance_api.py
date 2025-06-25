# Package Modules

from data_apis.interfaces import ICryptoExchangeProvider
from data_apis.utils import *
from data_apis.enums import *

# Other Modules

import polars as pl
from binance.spot import Spot
from binance.um_futures import UMFutures
from datetime import datetime
import sys
import time
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import tempfile
import shutil
import gzip

# Binance Provider Data API Class


class BinanceProvider(ICryptoExchangeProvider):

    def __init__(self, api_key: str = None, api_secret: str = None):

        self.api_key = api_key
        self.api_secret = api_secret

    def connect(self):
        """Establish connection with the exchange."""

        try:
            self.client_spot = Spot(api_key=self.api_key,
                                    api_secret=self.api_secret)
            response_spot = self.client_spot.ping()

            if response_spot == {}:
                print("\nSuccesfully connected to the Binance Spot API at",
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), "\n")
            else:
                print("\nUnexpected response from Binance Spot API at",
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), "\n")
                sys.exit(1)

            self.client_futures = UMFutures(key=self.api_key,
                                            secret=self.api_secret)

            response_futures = self.client_futures.ping()

            if response_futures == {}:
                print("Succesfully connected to the Binance Futures API at",
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), "\n")
            else:
                print("Unexpected response from Binance Futures API at",
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), "\n")
                sys.exit(1)

        except Exception as e:
            print(f"\nError during connection: {e}\n")
            sys.exit(1)

    def get_market_data(self, symbol: TokenPairs, freq: str, start: datetime, end: datetime, num_rows: int = 1000, futures_api: bool = False):
        """Fetch market data for a given symbol."""
        try:

            symb_str = self.get_pairs(symbol)

            client = self.client_futures if futures_api else self.client_spot

            columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                       'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

            k_lines = rate_limited_request(client.klines, symbol=symb_str, interval=freq,
                                           startTime=int(
                                               start.timestamp() * 1000),
                                           endTime=int(
                                               end.timestamp() * 1000),
                                           limit=num_rows)

            df = pl.DataFrame(
                k_lines, schema=columns, orient="row")
            df = df.with_columns([(pl.col("open_time") * 1000).cast(pl.Datetime),
                                  (pl.col("close_time") * 1000).cast(pl.Datetime)])
            df = df.with_columns(
                [pl.col(col).cast(pl.Float64) for col in numeric_cols])
            df = df.drop('ignore')

            return df

        except Exception as e:
            print(f"\nError fetching market data: {e}\n")
            sys.exit(1)

    def get_order_book(self, symbol: TokenPairs, depth: int, num_obs: int, freq: timedelta, directory: str, futures_api: bool = False):
        """Retrieve the order book for a given symbol. Binance does not offer historical order book data,
        so it is retrieved iteratively from the current moment up to certain future moment."""

        try:

            directory_path = os.path.join(directory, "data")
            os.makedirs(directory_path, exist_ok=True)

            symb_str = self.get_pairs(symbol)
            start_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            end_str = (datetime.now(
            ) + timedelta(seconds=freq.total_seconds())).strftime('%Y-%m-%d_%H-%M-%S')

            file_name = f"{symb_str}_{start_str}"+"_to_" + f"{end_str}" +\
                f"_{'futures' if futures_api else 'spot'}_order_book.csv.gz"

            client = self.client_futures if futures_api else self.client_spot

            gzip_file_path = os.path.join(directory_path, file_name)

            header = ["timestamp"] + [f"bid_price_{i+1}" for i in range(depth)] + \
                [f"bid_qty_{i+1}" for i in range(depth)] + \
                [f"ask_price_{i+1}" for i in range(depth)] + \
                [f"ask_qty_{i+1}" for i in range(depth)]

            with tempfile.TemporaryDirectory() as temp_dir:

                next_time = time.time()
                end_time = next_time + num_obs * freq.total_seconds()

                with tqdm(total=num_obs, desc="Retrieving Order Book Data") as pbar:
                    while next_time < end_time:

                        order_book = rate_limited_request(
                            client.depth, symbol=symb_str, limit=depth)

                        bids_df = pl.DataFrame(order_book.get(
                            'bids', []), schema=['bid_price', 'bid_qty'], orient="row")
                        asks_df = pl.DataFrame(order_book.get(
                            'asks', []), schema=['ask_price', 'ask_qty'], orient="row")
                        bids_df = bids_df.with_columns([pl.col('bid_price').cast(
                            pl.Float64), pl.col('bid_qty').cast(pl.Float64)])
                        asks_df = asks_df.with_columns([pl.col('ask_price').cast(
                            pl.Float64), pl.col('ask_qty').cast(pl.Float64)])

                        if len(bids_df) < depth:
                            bids_df = bids_df.extend_rows(pl.DataFrame(
                                [[0.0, 0.0]] * (depth - len(bids_df)), schema=["bid_price", "bid_qty"]))
                        if len(asks_df) < depth:
                            asks_df = asks_df.extend_rows(pl.DataFrame(
                                [[0.0, 0.0]] * (depth - len(asks_df)), schema=["ask_price", "ask_qty"]))

                        timestamp = datetime.now(
                            timezone.utc).timestamp()*1_000_000
                        row = [[timestamp] + bids_df.to_numpy().flatten().tolist() +
                               asks_df.to_numpy().flatten().tolist()]
                        snapshot_df = pl.DataFrame(
                            row, schema=header, orient="row")

                        temp_csv_path = os.path.join(
                            temp_dir, f"order_book_{int(timestamp)}.csv")

                        snapshot_df.write_csv(temp_csv_path)

                        with open(temp_csv_path, 'rt') as f_in:
                            with gzip.open(gzip_file_path, 'at', encoding='utf-8') as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        next_time += freq.total_seconds()
                        sleep_duration = next_time - time.time()
                        if sleep_duration > 0:
                            time.sleep(sleep_duration)

                        pbar.update(1)

            print(f"\nOrder book data saved to {gzip_file_path}\n")

        except Exception as e:
            print(f"\nError fetching order book data: {e}\n")
            sys.exit(1)

    def get_transactions(self, symbol: TokenPairs, start: datetime, end: datetime, chunk_freq: timedelta, directory: str, futures_api: bool = False):
        """Retrieve raw trade data for a given symbol within a time range and save it incrementally into a gzip-compressed CSV file."""
        try:
            directory_path = os.path.join(directory, "data")
            os.makedirs(directory_path, exist_ok=True)

            symb_str = self.get_pairs(symbol)
            start = start.replace(tzinfo=timezone.utc)
            end = end.replace(tzinfo=timezone.utc)
            start_str = start.strftime('%Y-%m-%d_%H-%M-%S')
            end_str = end.strftime('%Y-%m-%d_%H-%M-%S')
            file_name = f"{symb_str}_{start_str}_"+"to_"+f"{end_str}" + \
                f"_{'futures' if futures_api else 'spot'}_trades.csv.gz"
            gzip_file_path = os.path.join(directory_path, file_name)

            client = self.client_futures if futures_api else self.client_spot

            with tempfile.TemporaryDirectory() as temp_dir:

                current_start = start
                total_chunks = (end - start) // chunk_freq + 1
                from_id = None

                with tqdm(total=total_chunks, desc="Fetching trades") as pbar:

                    while current_start < end:

                        current_end = min(current_start + chunk_freq, end)

                        trades = []
                        while True:
                            new_trades = (
                                client.historical_trades(symbol=symb_str,
                                                         limit=1000, fromId=from_id)
                                if from_id else client.historical_trades(symbol=symb_str, limit=1000)
                            )

                            if not new_trades:
                                break

                            filtered_trades = []
                            for trade in new_trades:
                                trade_time = datetime.fromtimestamp(
                                    trade["time"] / 1000, tz=timezone.utc)

                                if trade_time > current_end:
                                    break
                                if trade_time >= current_start:
                                    filtered_trades.append(trade)

                            trades.extend(filtered_trades)
                            from_id = new_trades[-1]["id"] + 1

                            if trade_time > current_end:
                                break

                            time.sleep(0.5)

                        if trades:
                            trades_df = pl.DataFrame(trades)
                            temp_csv_path = os.path.join(
                                temp_dir, f"trades_{current_start.strftime('%Y%m%d%H%M%S')}.csv")
                            trades_df.write_csv(temp_csv_path)

                            with open(temp_csv_path, 'rt') as f_in:
                                with gzip.open(gzip_file_path, 'at', encoding='utf-8') as f_out:
                                    shutil.copyfileobj(f_in, f_out)

                        current_start = current_end
                        pbar.update(1)

            print(f"\nTrade data saved at {gzip_file_path}\n")

        except Exception as e:
            print(f"\nError retrieving trade data: {e}\n")
            sys.exit(1)

    @staticmethod
    def get_pairs(pair: TokenPairs):

        if pair == TokenPairs.btc_usdt:

            return "BTCUSDT"

        elif pair == TokenPairs.eth_usdt:

            return "ETHUSDT"

    @staticmethod
    def get_side(side: SideTrade):

        if side == SideTrade.buy:

            return "BUY"

        elif side == SideTrade.sell:

            return "SELL"
