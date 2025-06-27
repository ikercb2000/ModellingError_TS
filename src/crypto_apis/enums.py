# Other Modules

from enum import Enum

# Enums

OrderType = Enum("OrderType", ["market", "limit"])
ExchangeNames = Enum("ExchangeNames", ["binance", "kraken"])
TokenPairs = Enum("TokenPairs", ["btc_usdt", "eth_usdt"])
SideTrade = Enum("SideTrade", ["buy", "sell"])
StockTickers = Enum(
    "StockTickers", ["AAPL", "NVDA", "MSFT", "META", "GOOG", "AMZN"])
FXPairs = Enum("FXPairs", ["EURUSD", "USDJPY", "AUDUSD",
               "USDCHF", "USDCNY", "USDCAD", "USDKRW", "EURGBP", "USDHKD"])
