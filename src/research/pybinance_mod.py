import pybinanceapi as pb
import pandas as pd
import os

from datetime import datetime

# from utils import stock_helper


def transform_data_to_list_of_olhcv(data: list) -> None:
    """
    Transform list of data to query ready data
    """
    return list(map(lambda x: parse_to_OHLCV(x), data))


def parse_to_OHLCV(data: list) -> dict:
    """
    Convert data to OHLCV.
    [1628807400000, '0.93890000', '0.94200000', '0.92220000', '0.93110000', '15134951.80000000', 1628809199999, '14107295.15935900', 20089, '7252452.80000000', '6760534.18742500', '0']
    """
    data[0] = datetime.utcfromtimestamp(data[0] / 1000).strftime("%Y-%m-%d %H:%M")
    return data[:6]


if __name__ == "__main__":
    recent_candles = pb.getKline(symbol="XRPUSDT", interval="30m")

    transformed_data = transform_data_to_list_of_olhcv(recent_candles)

    df = pd.DataFrame(transformed_data)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)
    df.set_index("Date", inplace=True)

    config_path = os.path.join(
        os.path.dirname("__file__"), "config/config-feedforward.txt"
    )
