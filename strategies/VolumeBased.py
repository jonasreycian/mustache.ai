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

# Global variables
sequence_length = 30
trade_list = []
current_stock = ""
stock = "JFC"
total_profit = 0

# reading all csv from data folder and combine them all
def load_data():

    dfs = []

    for item in os.listdir("../data"):
        df = pd.read_csv(
            f"../data/{item}",
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
    tradable = pd.read_csv("../tradable_psei.csv")

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
    data["trend_macd_diff"] = data["trend_macd_diff"]
    data["volatility_atr"] = data["volatility_atr"]
    data["Pure_Close"] = data["Close"]

    data.dropna(inplace=True)

    for col in data.columns[:-1]:
        data[col] = preprocessing.scale(data[col].values)

    return data


# Save the winner
def save_winner(winner):

    pickle_out = open(f"../result/winner_net_{strftime('%Y%m%d%H%M%S')}.pickle", "wb")
    pickle.dump(winner, pickle_out)
    pickle_out.close()


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
    return sequence


def get_action(data, net):
    # 1stock
    position_days = 0
    current_position = 0
    stock_reward = 0
    # initialize deque sequence
    sequence = deque_sequence(data)

    """ 6(sequence) x 5(open,high,low,close)
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0 """

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
            current_price, action, position_days, current_position
        )
        stock_reward += reward

    return stock_reward


def do_action(current_price, action, position_days, current_position):
    profit = 0
    reward = 0
    position_change = (current_price - current_position) / current_position

    #     """if action is BUY and has no position"""
    if action == 1 and position_days == 0:
        position_days = 1
        current_position = current_price

    #     """if action is BUY and has position"""
    elif action == 1 and position_days > 0:
        position_days += 1

        # SELL if position is 60 days older or -10%
        if position_days >= 45 or position_change < -5 / 100:
            # SELL

            # check trade if win or loss
            if position_change > 1.19 / 100:
                reward = 1
            #                 print(f"profit:{position_change*100} days:{position_days}")

            else:
                reward = -1
                profit = position_change * 100
            #                 print(f"profit:{position_change*100} days:{position_days}")

            # RESET
            position_days = 0

    #     """if action is SELL and has no position"""
    elif action == 2 and position_days == 0:
        pass

    #     """if action is SELL and has position"""
    elif action == 2 and position_days > 0:
        position_days += 1

        if position_change > 1.19 / 100:
            reward = 1
        #             print(f"profit:{position_change*100} days:{position_days}")

        else:
            reward = -1
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

            # check trade if win or loss
            if position_change > 1.19 / 100:
                reward = 1
            #                 print(f"profit:{position_change*100} days:{position_days}")
            else:
                reward = -1
            #                 print(f"profit:{position_change*100} days:{position_days}")

            # RESET
            position_days = 0

    return current_position, position_days, reward


def run(config_file):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    pe = neat.parallel.ParallelEvaluator(2, eval_genomes)
    # loading winner net
    # winner = pickle.load(open("winner_net7.pickle","rb"))

    # creating the winner net
    # winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # continue from save point
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-199')

    """
    neat-checkpoint-[0:199] was trained using only JFC data
    neat-checkpoint-[200] was trained using 
      ["JFC","ALI","AC","SMPH","BDO","SM","TEL","URC","MBT","BPI","GLO","JGS","PGOLD","ICT","MPI"]
    """

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(pe.evaluate, 100)
    save_winner(winner)

    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))


def eval_genomes(genome, config):
    # create genomes
    # eval genomes here
    start = time.time()
    # load stockdata
    global stock_data
    total_reward = 0

    # initialize genome
    genome.fitness = 0.0
    net = neat.nn.RecurrentNetwork.create(genome, config)

    # stock_list = ["JFC", "ALI", "AC", "SMPH", "BDO", "SM", "TEL", "URC",
    #               "MBT", "BPI", "GLO", "JGS", "PGOLD", "ICT", "MPI"]
    # stock_list = ["JFC"]

    stock_list = [stock for stock, data in stock_data]
    # stock list and shuffle
    random.shuffle(stock_list)

    # for stock in stock_list: #############################################
    data = stock_data.get_group(stock)
    data.sort_values("Date", inplace=True)
    data = process_data(data)
    data = data.loc[:"2016-01-01"]
    # if len(data) > 30:
    #           print(stock)
    #           print(f"trading days:{len(data)} start:{data.index[0]}")
    #           print(data['Pure_Close'].head())
    #           print(len(data))
    #           print(data.tail())

    stock_reward = get_action(data, net)
    # print(f"Stock {stock} reward: {stock_reward}")
    total_reward += stock_reward

    end = time.time()
    # print(f"time:{end-start}")

    print(f"Genome reward: {total_reward}")
    genome.fitness = total_reward
    return genome.fitness


# loading data
stock_data = load_data()

if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, "../config-feedforward.txt")
    run(config_path)
