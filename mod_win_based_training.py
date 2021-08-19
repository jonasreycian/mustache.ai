import pandas as pd
import numpy as np
import os
import neat
import pickle
from collections import deque
import multiprocessing
import statistics
import json
import uuid
import statistics
import time

from sklearn import preprocessing
from src.utils import stock_helper
from src.utils import feature_selection

# hide warnings
import warnings
import random

warnings.filterwarnings("ignore")

# Global variable
sequence_length = 10
current_stock = ""
current_stock_data = ""
stock_data = stock_helper.get_all_stock_data()
reward_log = {}
strategy = "momentum"


def save_winner(winner, stock):
    pickle_out = open(f"result/winner/{strategy}/{stock}.pickle", "wb")
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

    # print(f'Sequence: {sequence}')
    return sequence


def get_action(data, net):

    # initialize deque sequence
    sequence = deque_sequence(data)

    global current_stock

    trade_log = {}
    trade_log["stock_name"] = current_stock
    trade_log["position_days"] = 0
    trade_log["current_position"] = 0
    trade_log["total_trades"] = 0
    trade_log["profit_win"] = 0
    trade_log["profit_win_count"] = 0
    trade_log["profit_loss"] = 0
    trade_log["profit_loss_count"] = 0
    trade_log["cumulative_profit"] = 0
    trade_log["position_date"] = ""

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

        trade_log = stock_helper.profit_loss_action(action, trade_log)

        i += 1

    # Closed any open position
    trade_log["open_position"] = 0
    if trade_log["position_days"] > 0:
        position_change = (
            trade_log["current_price"] - trade_log["current_position"]
        ) / trade_log["current_position"]

        trade_log["open_position"] = position_change * 100

    trade_log = stock_helper.AnalyzeTrainedData(trade_log, 9)

    return float(trade_log["reward"])


def eval_genomes(genome, config):
    global current_stock
    global current_stock_data
    global reward_log

    # initialize genome
    genome.fitness = 0.0
    net = neat.nn.RecurrentNetwork.create(genome, config)

    log = get_action(current_stock_data, net)
    reward_log[current_stock] = log

    genome.fitness = log
    return genome.fitness


def run(config_file):
    global current_stock
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    pe = neat.parallel.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)

    # # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # continue from save point
    # p = neat.Checkpointer.restore_checkpoint('result/checkpoints/neat-checkpoint-149')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(pe.evaluate, 5)
    save_winner(winner, current_stock)

    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, "config/config-feedforward.txt")

    # For single train only
    # current_stock = 'JFC'
    # current_stock_data = feature_selection.GetTopFeatures(current_stock,
    #         stock_data,
    #         max_feature=15,
    #         isTrain=True,
    #         category=strategy)

    # run(config_path)

    ignore_stock = ["JFC"]

    for stock, data in stock_data:
        if stock in ignore_stock:
            continue

        current_stock = stock
        current_stock_data = feature_selection.GetTopFeatures(
            current_stock,
            stock_data,
            max_feature=15,
            isTrain=True,
            category=strategy,
        )
        run(config_path)
