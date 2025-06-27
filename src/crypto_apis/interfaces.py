# Other modules

from abc import ABC, abstractmethod

# Crypto Exchange Interface


class ICryptoExchangeProvider(ABC):
    @abstractmethod
    def connect(self):
        """Establish connection with the exchange API."""
        pass

    @abstractmethod
    def get_market_data(self, **kwargs):
        """Fetch market data for a given symbol."""
        pass

    @abstractmethod
    def get_order_book(self, **kwargs):
        """Retrieve the order book for a given symbol."""
        pass

    @abstractmethod
    def get_transactions(self, **kwargs):
        """Retrieve the trades for a given symbol."""
        pass


class IStockDataProvider(ABC):
    @abstractmethod
    def connect(self):
        """Establish connection with the exchange API."""
        pass

    @abstractmethod
    def get_market_data(self, **kwargs):
        """Fetch market data for a given symbol."""
        pass

    @abstractmethod
    def get_order_book(self, **kwargs):
        """Retrieve the order book for a given symbol."""
        pass

    @abstractmethod
    def get_transactions(self, **kwargs):
        """Retrieve the trades for a given symbol."""
        pass
