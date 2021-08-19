import os
import datetime as dt

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager


class Binance:
    def __init__(self) -> None:
        self._api_key = os.environ["BNB_API_KEY"]
        self._secret_key = os.environ["BNB_SECRET_KEY"]

        self._client = Client(self._api_key, self._secret_key)

    def api_key(self) -> str:
        return self._api_key

    def secret_key(self) -> str:
        return self._secret_key

    def order_book(self, symbol: str) -> None:
        print(self._client.get_order_book(symbol=symbol))

    def all_tickers(self):
        # get all symbol prices
        prices = self._client.get_all_tickers()
        print(prices)

    def client(self):
        return self._client


if __name__ == "__main__":
    bnb = Binance()
    # print(bnb.order_book(symbol='BNBBTC'))
    # bnb.all_tickers()
    for kline in bnb.client().get_historical_klines_generator(
        "BNBUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2019", "1 May, 2020"
    ):
        kline[0] = dt.datetime.fromtimestamp(kline[0] / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        kline[6] = dt.datetime.fromtimestamp(kline[6] / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(kline)
