from ast import parse
from multiprocessing import Array
import os
from os import close
from numpy.lib import type_check
from numpy.lib.function_base import average
import pandas as pd
import ta
import json
import time
import pandas_ta
import numpy as np
from collections import deque

from sklearn import preprocessing
from utils import indicators, feature_selection
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator, MACD
from ta.volume import AccDistIndexIndicator
from AlmaIndicator import ALMAIndicator
from utils.strategies import AverageCostMethod

# hide warnings
import warnings

warnings.filterwarnings("ignore")


def AnalyzeTrainedData(trade_log, reward_cat):
    """
    Reward is based on the equity size

    """
    trade_log["total_trades"] = (
        1 if trade_log["total_trades"] == 0 else trade_log["total_trades"]
    )

    pw = trade_log["win_count"] / trade_log["total_trades"]

    # Compute the win_ratio
    trade_log["win_ratio"] = trade_log["win_count"] / trade_log["total_trades"] * 100

    # Reward
    # trade_log['reward'] = trade_log['port_balance'] if (trade_log['position_days'] == 0) else trade_log['port_balance'] + trade_log['port_gain_loss']
    trade_log["reward"] = (
        trade_log["cumulative_profit"]
        if (trade_log["position_days"] == 0)
        else trade_log["cumulative_profit"] + trade_log["port_gain_loss"]
    )

    print(
        f'{trade_log["stock_name"]}\t'
        + f'{trade_log["win_count"]}\t'
        + f'{trade_log["loss_count"]}\t'
        + f'{trade_log["total_trades"]}'
        + "\tWinRat: {0:.2f}".format(trade_log["win_ratio"])
        + f'\tLastTrade: {trade_log["position_date"]}'
        + "\tReward: {0:.2f}".format(trade_log["reward"])
    )

    return trade_log


def save_appt(appt_file):
    with open(f'result/appt/appt_{time.strftime("%m%d%Y")}.json', "w") as json_file:
        json.dump(appt_file, json_file)


def load_all_stock_with_features(stocks, stocks_data):
    stock_data_list = {}
    print("Loading all stock including with features")
    for stock in stocks:
        if stock in stock_data_list:
            print(f"\tExtracting features for {stock} was done...")
            continue

        stock_data_list[stock] = feature_selection.GetTopFeatures(
            stock, stocks_data, max_feature=10
        )
        print(f"\tExtracting features for {stock} was done...")

    return stock_data_list


def get_all_stock_data():
    dfs = []
    for item in os.listdir("./data"):
        df = pd.read_csv(
            f"./data/{item}",
            header=None,
            names=[
                "stock code",
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Netforeign",
            ],
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.dropna(inplace=True)
        df.set_index("Date", inplace=True)

        # sort values by date
        df.sort_values("Date", inplace=True)
        dfs.append(df)

    main_df = pd.concat(dfs)

    main_df.drop("Netforeign", 1, inplace=True)
    main_df.tail()

    #################################################################################################
    # read tradeble stocks
    tradable = pd.read_csv("./tradable_psei.csv")

    # creating a new df of tradable stock
    tradable_stock_df = main_df[main_df["stock code"].isin(tradable["stock"])]
    tradable_stock_df.head()

    tradable_stock_list = tradable_stock_df["stock code"].unique()
    tradable_stock_list.sort()

    print(tradable_stock_list, len(tradable_stock_list))

    # group by tradable stock
    tradable_stock_df = tradable_stock_df.groupby("stock code")

    return tradable_stock_df

    # main_df.head()
    # print (main_df)
    # main_df = main_df.groupby('stock code')
    # print (main_df)
    # return main_df


def AddFeatures(data, category):
    """Add feature to the data.

    Args:
    --
    data : (DataFrame)
        The DataFrame of the stocks.
    category : (str)
        Composed of 5 arguments.
            "all" : Append all features.
            "trend": Append all trend-related features.
            "volatility": Append all volatile-related features.
            "volume": Append all volume-related features.
            "emaribbon": Append all ema + volume features.
            "oscillators": Append all oscillators + volume features.

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    # Store list of tradable stocks
    if len(data["stock code"]) > 0:
        data.drop("stock code", 1, inplace=True)  # drop stock code

    data = data.apply(pd.to_numeric)

    if category == "all" or category == "momentum":
        # Append all Momentum-related features
        data = ta.add_momentum_ta(
            data, high="High", low="Low", close="Close", volume="Volume"
        )
        # data.pta.indicators()
        print(f"data ${data}")

    if category == "all" or category == "trend":
        # Append all tren-related features
        data = ta.add_trend_ta(data, high="High", low="Low", close="Close")

    if category == "all" or category == "volatility":
        # Append all volatility-related features
        data = ta.add_volatility_ta(data, high="High", low="Low", close="Close")

    if category == "all" or category == "volume":
        # Append all volume-related features
        data = ta.add_volume_ta(
            data, high="High", low="Low", close="Close", volume="Volume"
        )

    elif category == "emaribbon":
        # Append user-specified features
        data = indicators.ema(data, period=5)
        data = indicators.ema(data, period=10)
        data = indicators.ema(data, period=20)
        data = indicators.ema(data, period=30)
        data = indicators.ema(data, period=50)
        data = indicators.ema(data, period=100)
        data = indicators.ema(data, period=200)
        data = indicators.volume(data, period=20)
        data = indicators.rsi(data)
        macd = MACD(close=data['Close'])
        data['macd'] = macd.macd()
        data['macd_diff'] = macd.macd_diff()
        data['macd_signal'] = macd.macd_signal()

    elif category == "almarsi":
        alma_indicator = ALMAIndicator(close=data["Close"])
        data["alma9"] = alma_indicator.alma()
        # data['alma20'] = alma_indicator.alma_weights(window=20)
        # data['alma50'] = alma_indicator.alma_weights(window=50)
        data = indicators.rsi(data, period=9)
        data = indicators.rsi(data, period=25)
        data = indicators.rsi(data, period=12)
    elif category == "hayahay":
        data = indicators.ema(data, period=9)
        data = indicators.ema(data, period=50)
        data = indicators.ema(data, period=20)
        data = indicators.ema(data, period=200)
        data["rsi14"] = RSIIndicator(close=data["Close"]).rsi()
        data["rsi25"] = RSIIndicator(close=data["Close"], n=25).rsi()
        data["alma15"] = ALMAIndicator(close=data["Close"], period=15).alma()
        data["alma9"] = ALMAIndicator(close=data["Close"], period=9).alma()
        data["adi"] = AccDistIndexIndicator(
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            volume=data["Volume"],
        ).acc_dist_index()
        data["chop"] = pandas_ta.chop(
            high=pandas_ta.ohlc4(
                open_=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            ),
            low=pandas_ta.ohlc4(
                open_=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            ),
            close=pandas_ta.ohlc4(
                open_=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            ),
            length=28,
        )

    # print (data)
    if category == "all":
        data.dropna(inplace=True, axis=1)
    else:
        data.dropna(inplace=True)

    # Scale the value
    for col in data.columns[:]:
        data[col] = preprocessing.scale(data[col].values)

    return data


def deque_sequence(data, sequence_length):
    # initialize deque .. sequence of
    # number of sequence by open ,high ,low ,close++
    sequence = deque(maxlen=sequence_length)

    # if sequence_length = 6 .. it looks like this.. a 6 by 5 matrix
    for _ in range(sequence_length):
        sequence.append([0 for _ in data.columns[:-1]])

    # print(f'Sequence: {sequence}')
    return sequence


def get_action(data, net, current_stock, sequence_length, backTest, trade_list=[]):

    # initialize deque sequence
    sequence = deque_sequence(data, sequence_length)

    trade_log = {
        "stock_name": current_stock,
        "position_days": 0,
        "total_trades": 0,
        "win_count": 0,
        "loss_count": 0,
        "cumulative_profit": 0,
        "position_date": "",
        "port_balance": 20000,
        "port_gain_loss": 0,
        "port_gain_loss_pct": 0,
        "actual_balance": 20000,
        "total_shares": 0,
        "current_position_ave": 0,  # Average price (include market charges)
        "current_position": 0,
        "market_value_entry": 0,
        "market_value": 0,
        "board_lot": 0,
        "capital_at_risk": -5,
    }


    avco = AverageCostMethod(
        stock_name=current_stock,
        initial_equity=100000,
        capital_per_trade=10000,
        capital_at_risk=-3,
        isBacktest=False,
        max_hold_period=20
    )

    i = 0
    # FEEDING THE NEURAL NET
    for vals in data.values:
        # append the values of data (open,high,low,close) to deque sequence
        sequence.append(vals[:-1])

        # convert deque function to a numpy array
        x = np.array(sequence)

        # flatten features
        x = x.flatten()

        #         #append positon_change and position days ... more feature
        #         x = np.append(x,[position_change,position_days])

        #         #feed features to neural network
        output = net.activate(x)

        #       #action recomended by the neural network
        action = np.argmax(output, axis=0)


        # Profit/loss ratio
        trade_log["current_date"] = data.index[i]
        trade_log["current_price"] = vals[-1]
        trade_log["board_lot"] = GetBoardLot(vals[-1])

        avco._take_action(action=action, current_date=data.index[i], market_price=vals[-1])

        # if backTest:
        #     trade_log, trade_list = profit_loss_action(
        #         action, trade_log, isBackTest=backTest, trade_list=trade_list
        #     )
        # else:
        #     trade_log = profit_loss_action(action, trade_log, isBackTest=backTest)

        i += 1

    # trade_log = AnalyzeTrainedData(trade_log, 9)

    reward = avco.current_equity
    if (avco.current_shares > 0):
        reward += avco.market_value

    print(avco.logs[-1])
    return reward
    # return float(trade_log["reward"])


def take_action(action, trade_log, max_hold_days=20, minimum_capital=10000, trade_list=[], isBackTest=False):
    """
    Strategies:
        1. Will use a fixed capital to trade per buy like tranching method
        2. When sell occurs, sell all positions and use accumulated_profit
    """

    if trade_log["position_days"] > 0:
        # Compute the current market value (include PSE Charges)
        trade_log = GetCurrentMarketValue(trade_log)

    # If action is BUY and has no position
    if (action == 1) and trade_log["position_days"] == 0:
        # Buy maximum allotted shares based on actual balance
        max_share = minimum_capital / trade_log["current_price"]
        max_share_per_lot = max_share - (max_share % trade_log["board_lot"])

        if max_share_per_lot == 0:
            return trade_log

        trade_log["position_days"] = 1
        trade_log["current_position"] = trade_log["current_price"]
        trade_log["position_date"] = trade_log["current_date"]

        trade_log["total_shares"] = max_share_per_lot

        # Apply PSE Charges
        trade_log = BuyTransaction(trade_log)

        # Computation does not include PSE Charges
        trade_log["actual_balance"] -= trade_log["market_value_entry"]

        if isBackTest:
            # Record the trade made
            trade_list.append(
                [
                    trade_log["current_date"],
                    trade_log["stock_name"],
                    trade_log["current_price"],
                    "BUY",
                ]
            )

    # If action is BUY and has position
    elif action == 1 and trade_log["position_days"] > 0:
        trade_log["position_days"] += 1

        # Sell if meet the condition
        #   1. Holding period is reach
        #   2. Capital at risk is reach
        if trade_log["position_days"] > max_hold_days:
            # or trade_log['port_gain_loss_pct'] <= trade_log['capital_at_risk']:
            trade_log, trade_list = SellTransaction(
                trade_log, isBackTest=isBackTest, trade_list=trade_list
            )

    # If action is SELL and has no position
    elif action == 2 and trade_log["position_days"] == 0:
        pass

    # If action is SELL and has position
    elif action == 2 and trade_log["position_days"] > 0:
        # Sell if meet the condition.
        #   1. Capital at risk is reach
        #   2. Percent change is equal or above 3%
        # if trade_log['port_gain_loss_pct'] <= trade_log['capital_at_risk'] \
        #     or trade_log['port_gain_loss_pct'] >= (cut_loss):
        trade_log, trade_list = SellTransaction(
            trade_log, isBackTest=isBackTest, trade_list=trade_list
        )
        # else:
        #     trade_log['position_days'] += 1

    # If action is hold and has no position
    elif action == 0 and trade_log["position_days"] == 0:
        pass

    #     """if action is hold and has position"""
    elif action == 0 and trade_log["position_days"] > 0:
        # Sell if meet the condition
        #   1. Holding period is reach
        #   2. Capital at risk is reach
        #   2. Take Profit is reach
        if (
            trade_log["position_days"] > max_hold_days
            or trade_log["port_gain_loss_pct"] <= trade_log["capital_at_risk"]
            or trade_log["port_gain_loss_pct"] >= take_profit
        ):

            trade_log, trade_list = SellTransaction(
                trade_log, isBackTest=isBackTest, trade_list=trade_list
            )
        else:
            trade_log["position_days"] += 1

    if isBackTest:
        return trade_log, trade_list
    else:
        return trade_log


def profit_loss_action(
    action,
    trade_log,
    cut_loss=2,
    take_profit=100,
    max_hold_days=30,
    trade_list=[],
    isBackTest=False,
):

    if trade_log["position_days"] > 0:

        # Compute the current market value (include PSE Charges)
        trade_log = GetCurrentMarketValue(trade_log)

    # If action is BUY and has no position
    if (action == 1) and trade_log["position_days"] == 0:
        # Buy maximum allotted shares based on actual balance
        max_share = trade_log["actual_balance"] / trade_log["current_price"]
        max_share_per_lot = max_share - (max_share % trade_log["board_lot"])

        if max_share_per_lot == 0:
            return trade_log

        trade_log["position_days"] = 1
        trade_log["current_position"] = trade_log["current_price"]
        trade_log["position_date"] = trade_log["current_date"]

        trade_log["total_shares"] = max_share_per_lot

        # Apply PSE Charges
        trade_log = BuyTransaction(trade_log)

        # Computation does not include PSE Charges
        trade_log["actual_balance"] -= trade_log["market_value_entry"]

        if isBackTest:
            # Record the trade made
            trade_list.append(
                [
                    trade_log["current_date"],
                    trade_log["stock_name"],
                    trade_log["current_price"],
                    "BUY",
                ]
            )

    # If action is BUY and has position
    elif action == 1 and trade_log["position_days"] > 0:
        trade_log["position_days"] += 1

        # Sell if meet the condition
        #   1. Holding period is reach
        #   2. Capital at risk is reach
        if trade_log["position_days"] > max_hold_days:
            # or trade_log['port_gain_loss_pct'] <= trade_log['capital_at_risk']:
            trade_log, trade_list = SellTransaction(
                trade_log, isBackTest=isBackTest, trade_list=trade_list
            )

    # If action is SELL and has no position
    elif action == 2 and trade_log["position_days"] == 0:
        pass

    # If action is SELL and has position
    elif action == 2 and trade_log["position_days"] > 0:
        # Sell if meet the condition.
        #   1. Capital at risk is reach
        #   2. Percent change is equal or above 3%
        # if trade_log['port_gain_loss_pct'] <= trade_log['capital_at_risk'] \
        #     or trade_log['port_gain_loss_pct'] >= (cut_loss):
        trade_log, trade_list = SellTransaction(
            trade_log, isBackTest=isBackTest, trade_list=trade_list
        )
        # else:
        #     trade_log['position_days'] += 1

    # If action is hold and has no position
    elif action == 0 and trade_log["position_days"] == 0:
        pass

    #     """if action is hold and has position"""
    elif action == 0 and trade_log["position_days"] > 0:
        # Sell if meet the condition
        #   1. Holding period is reach
        #   2. Capital at risk is reach
        #   2. Take Profit is reach
        if (
            trade_log["position_days"] > max_hold_days
            or trade_log["port_gain_loss_pct"] <= trade_log["capital_at_risk"]
            or trade_log["port_gain_loss_pct"] >= take_profit
        ):

            trade_log, trade_list = SellTransaction(
                trade_log, isBackTest=isBackTest, trade_list=trade_list
            )
        else:
            trade_log["position_days"] += 1

    if isBackTest:
        return trade_log, trade_list
    else:
        return trade_log


def GetCurrentMarketValue(trade_log):
    """ This will compute the current market value by computing the market charges when selling"""

    # PSE Charges
    #   Gross Transaction Amount    : share x price
    #   Broker's Commission         : 0.25% x GROSS_TRANSACTION_AMOUNT
    #   Broker's Commission VAT     : 12% x BROKER_COMMISION
    #   SCCP Fee                    : 0.01% x GROSS_TRANSACTION_AMOUNT
    #   PSE Transaction Fee         : 0.005% x GROSS_TRANSACTION_AMOUNT
    #   Sales Transaction Tax	    : 0.6% of the gross trade value

    gross_transaction_amount = (
        float(trade_log["current_price"]) * trade_log["total_shares"]
    )
    brokers_commision = 0.0025 * gross_transaction_amount
    brokers_commision = 20 if brokers_commision <= 20 else brokers_commision

    brokers_commision_vat = 0.12 * brokers_commision
    sccp_fee = 0.0001 * gross_transaction_amount
    pse_transaction_fee = 0.00005 * gross_transaction_amount
    sales_tax = 0.006 * gross_transaction_amount

    trade_log["market_value"] = gross_transaction_amount - (
        brokers_commision
        + brokers_commision_vat
        + sccp_fee
        + pse_transaction_fee
        + sales_tax
    )

    # Compute the difference of the entry market price than current
    trade_log["port_gain_loss"] = (
        trade_log["market_value"] - trade_log["market_value_entry"]
    )
    trade_log["port_gain_loss_pct"] = (
        trade_log["port_gain_loss"] / trade_log["port_balance"]
    ) * 100

    return trade_log


def BuyTransaction(trade_log):
    """
    Get average price and average market price by applying PSE Stock Fees.

    Returns:
        current_position_ave, market_value_ave
    """

    # PSE Charges
    #   Gross Transaction Amount    : share x price
    #   Broker's Commission         : 0.25% x GROSS_TRANSACTION_AMOUNT
    #   Broker's Commission VAT     : 12% x BROKER_COMMISION
    #   SCCP Fee                    : 0.01% x GROSS_TRANSACTION_AMOUNT
    #   PSE Transaction Fee         : 0.0005% x GROSS_TRANSACTION_AMOUNT

    gross_transaction_amount = trade_log["current_price"] * trade_log["total_shares"]
    brokers_commision = 0.0025 * gross_transaction_amount
    brokers_commision = 20 if brokers_commision <= 20 else brokers_commision

    brokers_commision_vat = 0.12 * brokers_commision
    sccp_fee = 0.0001 * gross_transaction_amount
    pse_transaction_fee = 0.00005 * gross_transaction_amount

    net_transaction_amount = (
        gross_transaction_amount
        + brokers_commision
        + brokers_commision_vat
        + sccp_fee
        + pse_transaction_fee
    )

    trade_log["current_position_ave"] = (
        net_transaction_amount / trade_log["total_shares"]
    )
    trade_log["market_value_entry"] = net_transaction_amount
    trade_log["market_value"] = net_transaction_amount

    return trade_log


def SellTransaction(trade_log, isBackTest=False, trade_list=[]):
    """
    Executing the trade.
    """

    if trade_log["port_gain_loss"] >= 0:
        trade_log["win_count"] += 1
    else:
        trade_log["loss_count"] += 1

    # Reset
    trade_log["cumulative_profit"] += trade_log["port_gain_loss"]
    trade_log["port_balance"] += trade_log["port_gain_loss"]
    trade_log["actual_balance"] = trade_log["port_balance"]
    trade_log["port_gain_loss"] = 0
    trade_log["port_gain_loss_pct"] = 0
    trade_log["total_shares"] = 0
    trade_log["position_days"] = 0
    trade_log["total_trades"] += 1

    if isBackTest:
        # Record the trade made
        trade_list.append(
            [
                trade_log["current_date"],
                trade_log["stock_name"],
                trade_log["current_price"],
                "SELL",
            ]
        )

    return trade_log, trade_list


def read_json(location_data="../result/appt"):
    """
    This function reads all the json inside a folder.
    """
    appt_list_per_hash = {}
    for item in os.listdir(location_data):
        # Load the json file
        with open(f"{location_data}/{item}") as f:
            data = json.load(f)

        # Get the list of the key
        for key in data:
            if key in appt_list_per_hash:
                pass
            else:
                appt_list_per_hash[key] = data[key]

    return appt_list_per_hash


def format_json(json_data, indent=4, sort_keys=True):
    return json.dumps(json_data, indent=indent, sort_keys=sort_keys)


def WriteJSONFile(filepath="./result/analysis/", filename="", data={}):
    with open(f"{filepath+filename}", "w") as fp:
        json.dump(data, fp)


def GetBoardLot(current_price):
    lot_size = 5
    current_price = float(current_price)
    # Price between 0.0001 and 0.0099
    if current_price <= 0.0099:
        lot_size = 1000000
    # Price between 0.01 and 0.049
    elif current_price <= 0.495:
        lot_size = 100000
    # Price between 0.05 and 0.495
    elif current_price <= 0.495:
        lot_size = 10000
    # Price between 0.5 and 4.99
    elif current_price <= 4.99:
        lot_size = 1000
    # Price between 5 and 49.95
    elif current_price <= 49.95:
        lot_size = 100
    # Price between 50 and 999.5
    elif current_price <= 999.5:
        lot_size = 10

    return lot_size
