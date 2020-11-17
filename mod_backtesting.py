import pandas as pd
import numpy as np
import os
import neat
import pickle
from collections import deque
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
strategy = "hayahay"

def run(config_file):
    global current_stock
    global current_stock_data
    global trade_list

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # winner = pickle.load(open(f"result/winner/{strategy}/{current_stock}.pickle", "rb"))
    winner = pickle.load(open(f"result/winner/{strategy}/{current_stock}.pickle", "rb"))
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    fitness_reward = stock_helper.get_action(current_stock_data, winner_net, current_stock, sequence_length, backTest=True, trade_list=trade_list)
    # print(trade_list)
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

    for stock in stock_list[:1]:
        current_stock = 'DITO'
        current_stock_data = feature_selection.GetTopFeatures('DITO',
                stock_data,
                max_feature=15,
                isTrain=False,
                category=strategy)
        run(config_path)
        stock_helper.WriteJSONFile(filename=f'analysis_{time.strftime("%m%d%Y")}_20Gen.json', data=fitness_analysis)

    trade_list = pd.DataFrame(
        trade_list, columns=['Date', 'Stock', 'Price', 'Action'])
    trade_list.sort_index(ascending=False)
    trade_list.to_csv((f'result/backtest_{strategy}_{time.strftime("%m%d%Y")}.csv'), index=False)