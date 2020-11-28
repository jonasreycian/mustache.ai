import os
import neat
import pickle
import multiprocessing

from utils import stock_helper
from utils import feature_selection

# hide warnings
import warnings

warnings.filterwarnings("ignore")

# Global variable
sequence_length = 10
current_stock = ""
current_stock_data = ""
stock_data = stock_helper.get_all_stock_data()
reward_log = {}
strategy = "emaribbon"


def save_winner(winner, stock):
    pickle_out = open(f"result/winner/{strategy}/{stock}.pickle", "wb")
    pickle.dump(winner, pickle_out)
    pickle_out.close()


def eval_genomes(genome, config):
    global current_stock
    global current_stock_data
    global reward_log

    # initialize genome
    genome.fitness = 0.0
    net = neat.nn.RecurrentNetwork.create(genome, config)

    log = stock_helper.get_action(current_stock_data,net, current_stock, sequence_length, backTest=False)
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
    winner = p.run(pe.evaluate, 20)
    save_winner(winner, current_stock)

    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    # stock list and shuffle
    stock_list = [stock for stock, data in stock_data]
    print(stock_list)

    # For single train only
    # current_stock = 'JFC'
    # current_stock_data = feature_selection.GetTopFeatures(current_stock,
    #         stock_data,
    #         max_feature=15,
    #         isTrain=True,
    #         category=strategy)

    # run(config_path)

    ignore_stock = ["DITO"]

    for stock in stock_list[:1]:

        # if stock in ignore_stock:
        current_stock = "DITO"
        current_stock_data = feature_selection.GetTopFeatures(
            "DITO", stock_data, max_feature=15, isTrain=True, category=strategy
        )
        run(config_path)
