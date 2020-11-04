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
from utils import stock_helper
from utils import feature_selection

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# Global variable
trade_list = []
sequence_length = 10
current_stock = ""
current_stock_data = ""
stock_data = stock_helper.get_all_stock_data()
fitness_analysis = {}
strategy = "momentum"

def deque_sequence(data):
    global sequence_length
    # initialize deque .. sequence of
    # number of sequence by open ,high ,low ,close++
    sequence = deque(maxlen=sequence_length)

    # if sequence_length = 6 .. it looks like this.. a 6 by 5 matrix
    for _ in range(sequence_length):
        sequence.append([0 for _ in data.columns[:-1]])

#     print(sequence)
    return sequence

def get_action(data,net):

    #initialize deque sequence
    sequence = deque_sequence(data)

    """ 6(sequence) x 5(open,high,low,close)
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0
            0, 0, 0, 0, 0 """

    global current_stock
    global trade_list

    trade_log = {}
    trade_log['stock_name'] = current_stock
    trade_log['position_days'] = 0
    trade_log['current_position'] = 0
    trade_log['total_trades'] = 0
    trade_log['profit_win'] = 0
    trade_log['profit_win_count'] = 0
    trade_log['profit_loss'] = 0
    trade_log['profit_loss_count'] = 0
    trade_log['cumulative_profit'] = 0

    i = 0
    #FEEDING THE NEURAL NET
    for vals in data.values:

        #append the values of data (open,high,low,close) to deque sequence
        sequence.append(vals[:-1])

        #convert deque function to a numpy array
        x = np.array(sequence)

        #flatten features
        x = x.flatten()

        #feed features to neural network
        output = net.activate(x)

        #action recomended by the neural network
        action = np.argmax(output, axis=0)
        # Profit/loss ratio

        trade_log['current_date'] = data.index[i]
        trade_log['current_price'] = vals[-1]

        trade_log, trade_list = stock_helper.profit_loss_action(
            action,
            trade_log,
            trade_list=trade_list,
            isBackTest=True)

        i += 1

    # Closed any open position
    trade_log['open_position'] = 0
    if trade_log['position_days'] > 0:
        position_change = (
            trade_log['current_price'] - trade_log['current_position']
            ) / trade_log['current_position']

        trade_log['open_position'] = position_change * 100

    trade_log = stock_helper.AnalyzeTrainedData(trade_log, 9)

    return trade_log['reward']

def run(config_file):
    global current_stock
    global current_stock_data

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    winner = pickle.load(open(f"result/winner/{strategy}/{current_stock}.pickle", "rb"))
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    fitness_reward = get_action(current_stock_data, winner_net)
    fitness_analysis[current_stock] = fitness_reward

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname('__file__')
    config_path = os.path.join(local_dir, 'config-feedforward.txt')

    #stock list and shuffle
    stock_list = [stock for stock,data in stock_data]
    # random.shuffle(stock_list)
    # stocks_with_features = stock_helper.load_all_stock_with_features(stock_list, stock_data)
    ignore_stock = ['JFC']

    for stock in stock_list:

        if stock in ignore_stock:
            current_stock = stock
            current_stock_data = feature_selection.GetTopFeatures(stock,
                    stock_data,
                    max_feature=15,
                    isTrain=False,
                    category=strategy)
            run(config_path)
            stock_helper.WriteJSONFile(filename=f'analysis_{time.strftime("%m%d%Y")}_20Gen.json', data=fitness_analysis)


    trade_list = pd.DataFrame(
        trade_list, columns=['Date', 'Stock', 'Price', 'Action'])
    trade_list.to_csv((f'result/backtest_result_{time.strftime("%m%d%Y")}.csv'), index=False)