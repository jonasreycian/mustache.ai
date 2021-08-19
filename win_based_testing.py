import pandas as pd
import numpy as np
import os
import neat
import pickle
import ta
from collections import deque
import random
import time
from statistics import mean
from sklearn import preprocessing

# hide warnings
import warnings

warnings.filterwarnings("ignore")

sequence_length = 30
trade_list = []
current_stock = ""
stock = "FLI"
total_profit = 0

# loading data
stock_data = load_data()

# reading all csv from data folder and combine them all
def load_data():

    dfs = []

    for item in os.listdir("data"):
        df = pd.read_csv(
            f"data/{item}",
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
    main_df.tail()

    ##################################################################################################
    # read tradeble stocks
    tradable = pd.read_csv("tradable.csv")

    # creating a new df of tradable stock
    tradable_stock_df = main_df[main_df["stock code"].isin(tradable["stock"])]
    tradable_stock_df.head()

    tradable_stock_list = tradable_stock_df["stock code"].unique()
    tradable_stock_list.sort()

    print(tradable_stock_list, len(tradable_stock_list))
    print("\n\n")

    # group by tradable stock
    tradable_stock_df = tradable_stock_df.groupby("stock code")

    return tradable_stock_df


def process_data(data):
    data.drop("Netforeign", 1, inplace=True)  # drop netforeign
    data.drop("stock code", 1, inplace=True)  # drop stock code
    data["Volume"].replace(0, 1, inplace=True)  # replace 0 value volume with 1
    data.interpolate(inplace=True)

    # adding technical indicators
    data = ta.add_all_ta_features(
        data,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=False,
    )
    #     print(data.tail())
    #     data.to_csv("dd.csv",index=False)

    col_list = [
        "Close",
        "Volume",
        "momentum_rsi",
        "momentum_wr",
        "momentum_ao",
        "momentum_stoch",
        "momentum_stoch_signal",
        "trend_trix",
        "trend_vortex_ind_pos",
        "trend_vortex_ind_neg",
        "trend_vortex_ind_diff",
        "trend_macd",
        "trend_macd_signal",
        "trend_macd_diff",
        "volatility_atr",
    ]

    data = data[col_list]

    # OPEN , HIGH, LOW, CLOSE, VOLUME
    data["Close"] = data["Close"]
    data["Volume"] = data["Volume"]
    data["momentum_rsi"] = data["momentum_rsi"]
    data["momentum_wr"] = data["momentum_wr"]
    data["momentum_stoch"] = data["momentum_stoch"]
    data["momentum_stoch_signal"] = data["momentum_stoch_signal"]
    data["Pure_Close"] = data["Close"]

    data.dropna(inplace=True)

    for col in data.columns[:-1]:
        data[col] = preprocessing.scale(data[col].values)

    return data


def deque_sequence(data):
    global sequence_length
    # initialize deque .. sequence of
    # number of sequence by open ,high ,low ,close++
    sequence = deque(maxlen=sequence_length)

    # if sequence_length = 6 .. it looks like this.. a 6 by 5 matrix
    for _ in range(sequence_length):
        sequence.append([0 for _ in data.columns[:-1]])

        """ 6(sequence) x 5(open,high,low,close)
                        0, 0, 0, 0, 0
                        0, 0, 0, 0, 0
                        0, 0, 0, 0, 0
                        0, 0, 0, 0, 0
                        0, 0, 0, 0, 0 """
    #     print(sequence)
    return sequence


def get_action(data, net):
    # 1stock
    global current_stock
    position_days = 0
    current_position = 0
    stock_reward = 0
    wins = 0
    loss = 0
    # initialize deque sequence
    sequence = deque_sequence(data)

    """ 6(sequence) x 5(open,high,low,close)
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0 """

    i = 0
    # FEEDING THE NEURAL NET
    for vals in data.values:
        #         print(vals[:-1])
        #         print(vals[-1])  ##pure_close
        current_price = vals[-1]
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

        #         #action recomended by the neural network
        action = np.argmax(output, axis=0)

        #         print(action)
        current_position, position_days, reward = do_action(
            data.index[i], current_price, action, position_days, current_position
        )
        stock_reward += reward

        i += 1
        if reward == 1:
            wins += 1
        elif reward == -1:
            loss += 1

    print(f"wins:{wins} loss:{loss}")
    current_stock = ""

    return stock_reward


def do_action(date, current_price, action, position_days, current_position):
    global current_stock
    global total_profit
    global trade_list
    reward = 0
    position_change = (current_price - current_position) / current_position

    #     """if action is BUY and has no position"""
    if action == 1 and position_days == 0:
        position_days = 1
        current_position = current_price
        trade_list.append([date, current_stock, current_price, 1])

    #     """if action is BUY and has position"""
    elif action == 1 and position_days > 0:
        position_days += 1

        # SELL if position is 60 days older or -10%
        if position_days >= 45 or position_change < -5 / 100:
            # SELL
            trade_list.append([date, current_stock, current_price, 2])
            # check trade if win or loss
            if position_change > 1.19 / 100:
                reward = 1
                total_profit += position_change * 100
            #                 print(f"profit:{position_change*100} days:{position_days}")

            else:
                reward = -1
                total_profit += position_change * 100
            #                 print(f"profit:{position_change*100} days:{position_days}")

            # RESET
            position_days = 0

    #     """if action is SELL and has no position"""
    elif action == 2 and position_days == 0:
        pass

    #     """if action is SELL and has position"""
    elif action == 2 and position_days > 0:
        position_days += 1
        trade_list.append([date, current_stock, current_price, 2])
        if position_change > 1.19 / 100:
            reward = 1
            total_profit += position_change * 100
        #             print(f"profit:{position_change*100} days:{position_days}")

        else:
            reward = -1
            total_profit += position_change * 100
        #             print(f"profit:{position_change*100} days:{position_days}")

        # RESET
        position_days = 0

    #     """if action is hold and has no position"""
    elif action == 0 and position_days == 0:
        pass

    #     """if action is hold and has position"""
    elif action == 0 and position_days > 0:
        position_days += 1

        # sell if position is 60 days older
        if position_days >= 45 or position_change < -5 / 100:
            # SELL
            trade_list.append([date, current_stock, current_price, 2])
            # check trade if win or loss
            if position_change > 1.19 / 100:
                reward = 1
                total_profit += position_change * 100
            #                 print(f"profit:{position_change*100} days:{position_days}")
            else:
                reward = -1
                total_profit += position_change * 100
            #                 print(f"profit:{position_change*100} days:{position_days}")

            # RESET
            position_days = 0

    return current_position, position_days, reward


def run(config_file):
    # globals
    global stock_data
    global current_stock
    global trade_list
    #     global stock
    total_reward = 0
    global total_profit
    p = 0

    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    winner = pickle.load(open("nets/winner_net0.pickle", "rb"))

    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    for stock, data in stock_data:
        current_stock = stock
        data = stock_data.get_group(stock)
        data.sort_values("Date", inplace=True)
        print(len(data), stock, data.index[-1])
        data = process_data(data)
        # data = data.loc['2020-03-01':]
        #       data = data.loc[:'2018-11-05']
        data = data.loc[:]
        if len(data) > 30:
            stock_reward = get_action(data, winner_net)
            total_reward += stock_reward

        print(f"{total_profit}")
        p += total_profit
        total_profit = 0

    print(p)
    trade_list = pd.DataFrame(trade_list, columns=["Date", "Stock", "Price", "Action"])
    trade_list.to_csv(
        ("result/trade_list_", time.strftime("%m%d%Y"), ".csv"), index=False
    )
    trade_list.to_csv(
        ("/mnt/c/daily_trade_result/trade_list_", time.strftime("%m%d%Y"), ".csv"),
        index=False,
    )


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
