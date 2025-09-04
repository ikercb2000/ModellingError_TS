# packages

import yfinance as yf
from datetime import datetime

# Yahoo Finance API Class


class YahooFinanceAPI:

    def __init__(self):
        self.df = None

    def download_data(self, ticker: str, start: datetime, end: datetime, progress: bool = True, auto_adjust: bool = True):

        start = self.datetime_to_string(start)
        end = self.datetime_to_string(end)

        self.df = yf.download(ticker, start=start, end=end,
                              progress=True, auto_adjust=True)

        if 'Adj Close' in self.df.columns:
            return self.df['Adj Close'].values
        elif 'Close' in self.df.columns:
            return self.df['Close'].values
        else:
            raise KeyError("No 'Adj Close' or 'Close' column found.")

    @staticmethod
    def datetime_to_string(date: datetime):
        return date.strftime("%Y-%m-%d")
