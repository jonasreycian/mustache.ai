from collections import deque
import pickle
import multiprocessing
import neat
import pybinanceapi as pb
import pandas as pd
import os
import warnings

from datetime import datetime
from numpy import argmax, array

warnings.filterwarnings("ignore")


class Neat:
    # Global variable
    current_stock = ""
    current_stock_data = ""
    reward_log = {}

    def save_winner(self, winner, ticker):
        pickle_out = open(f"{ticker}.pickle", "wb")
        pickle.dump(winner, pickle_out)
        pickle_out.close()

    def deque_sequence(self, data):
        # initialize deque .. sequence of
        # number of sequence by open ,high ,low ,close++
        sequence = deque(maxlen=10)

        # if sequence_length = 6 .. it looks like this.. a 6 by 5 matrix
        for _ in range(10):
            sequence.append([0 for _ in data.columns[:-1]])

        # print(f'Sequence: {sequence}')
        return sequence

    def eval_genomes(self, genome, config):
        global current_stock
        global current_stock_data
        global reward_log

        # initialize genome
        genome.fitness = 0.0
        net = neat.nn.RecurrentNetwork.create(genome, config)

        print(f"current_stock_data: ")
        print(f"")
        log = self.get_action(self.current_stock_data, net)
        self.reward_log[self.current_stock] = log

        genome.fitness = log
        return genome.fitness

    def get_action(self, data, net):

        # initialize deque sequence
        sequence = self.deque_sequence(data)

        global current_stock

        trade_log = {}
        trade_log["stock_name"] = self.current_stock
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
            x = array(sequence)

            # flatten features
            x = x.flatten()

            #         #append positon_change and position days ... more feature
            #         x = np.append(x,[position_change,position_days])

            #         #feed features to neural network
            output = net.activate(x)

            #       #action recomended by the neural network
            action = argmax(output, axis=0)

            # Profit/loss ratio
            trade_log["current_date"] = data.index[i]
            trade_log["current_price"] = vals[-1]

            trade_log = self.profit_loss_action(action, trade_log)

            i += 1

        # Closed any open position
        trade_log["open_position"] = 0
        if trade_log["position_days"] > 0:
            position_change = (
                trade_log["current_price"] - trade_log["current_position"]
            ) / trade_log["current_position"]

            trade_log["open_position"] = position_change * 100

        trade_log = self.AnalyzeTrainedData(trade_log, 9)

        return float(trade_log["reward"])

    def AnalyzeTrainedData(self, trade_log, reward_cat):
        """
        Average Profitability Per Trade
            This will compute the average profit change per trade
        Formula:
            APPT = (PW×AW) − (PL×AL)
            where:
            PW = Probability of win
            AW = Average win
            PL = Probability of loss
            AL = Average loss

        Param:
            trade_log: Log of the current trade

        """
        trade_log["total_trades"] = (
            1 if trade_log["total_trades"] == 0 else trade_log["total_trades"]
        )

        pw = trade_log["profit_win_count"] / trade_log["total_trades"]
        aw = trade_log["profit_win"] / trade_log["total_trades"]
        pl = trade_log["profit_loss_count"] / trade_log["total_trades"]
        al = trade_log["profit_loss"] / trade_log["total_trades"]

        # Compute the appt
        trade_log["appt"] = (pw * aw) - (pl * al)
        # Compute the win_ratio
        trade_log["win_ratio"] = (
            trade_log["profit_win_count"] / trade_log["total_trades"] * 100
        )

        trade_log["pwaw"] = pw * aw

        # Compute the win_loss_ratio
        if trade_log["profit_loss_count"] == 0:
            trade_log["win_loss_ratio"] = trade_log["profit_win_count"]
        else:
            trade_log["win_loss_ratio"] = (
                trade_log["profit_win_count"] / trade_log["profit_loss_count"]
            )

        # Get reward
        trade_log = self.GetReward(reward_cat, trade_log)

        print(
            f'{trade_log["stock_name"]}\t'
            + f'{trade_log["profit_win_count"]}\t'
            + f'{trade_log["profit_loss_count"]}\t'
            + f'{trade_log["total_trades"]}'
            + "\tWLRatio: {0:.2f}".format(trade_log["win_loss_ratio"])
            + "\tAPPT: {0:.2f}".format(trade_log["appt"])
            + "\tWinRat: {0:.2f}".format(trade_log["win_ratio"])
            + "\tReward: {0:.2f}".format(trade_log["reward"])
            + "\tOpenPos: {0:.2f}".format(trade_log["open_position"])
            + "\tCProfit: {0:.2f}".format(trade_log["cumulative_profit"])
            + f'\tLastTrade: {trade_log["position_date"]}'
        )

        return trade_log

    def GetReward(self, reward_cat, trade_log):
        """
        Return reward based on given fitness
        """
        # win_loss_ratio * appt * #trades
        if reward_cat == 1:
            trade_log["reward"] = (
                trade_log["appt"]
                * trade_log["win_loss_ratio"]
                * trade_log["profit_win_count"]
            )
        # win_loss_ratio * appt * #trades (Modified)
        elif reward_cat == 2:
            wl_ratio = (
                3 if trade_log["win_loss_ratio"] > 3 else trade_log["win_loss_ratio"]
            )
            trade_log["reward"] = (
                trade_log["appt"] * wl_ratio * trade_log["profit_win_count"]
            )
        # profit_win_count win * appt
        elif reward_cat == 3:
            trade_log["reward"] = (
                (trade_log["profit_win"] ** 0.5)
                * trade_log["appt"]
                * (trade_log["win_loss_ratio"] ** 0.5)
            )
        # profit_win_count win * appt
        elif reward_cat == 4:
            trade_log["reward"] = (
                (trade_log["profit_win"] ** 0.5)
                * (trade_log["profit_win_count"] ** 0.5)
                * (trade_log["win_ratio"] ** 0.5)
            )
        elif reward_cat == 5:
            trade_log["reward"] = (
                (trade_log["win_loss_ratio"] ** 0.5)
                * (trade_log["profit_win_count"] ** 0.5)
                * (trade_log["win_ratio"] ** 0.5)
                * (trade_log["profit_win"] ** 0.5)
                * trade_log["appt"]
            )
        elif reward_cat == 6:
            trade_log["reward"] = (
                (trade_log["profit_win"] ** 0.5)
                * (trade_log["profit_win_count"] ** 0.5)
                * (trade_log["win_ratio"] ** 0.5)
                * trade_log["appt"]
            )
        elif reward_cat == 7:
            trade_log["reward"] = (
                (trade_log["profit_win"]) - trade_log["profit_loss"]
            ) * (trade_log["win_ratio"] ** 0.5)
        elif reward_cat == 8:
            trade_log["reward"] = (
                trade_log["cumulative_profit"] + trade_log["open_position"]
            ) * trade_log["win_loss_ratio"]
        elif reward_cat == 9:
            trade_log["reward"] = (
                trade_log["cumulative_profit"] + trade_log["open_position"]
            )

        return trade_log

    def profit_loss_action(
        self,
        action,
        trade_log,
        cut_loss=-5,
        take_profit=100,
        max_hold_days=90,
        trade_list=[],
        isBackTest=False,
    ):

        if trade_log["current_position"] == 0:
            trade_log["current_position"] = 1

        position_change = (
            (trade_log["current_price"] - trade_log["current_position"])
            / trade_log["current_position"]
            * 100
        )

        # If action is BUY and has no position
        if (action == 1) and trade_log["position_days"] == 0:
            trade_log["position_days"] = 1
            trade_log["current_position"] = trade_log["current_price"]
            trade_log["position_date"] = trade_log["current_date"]

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
            if (
                trade_log["position_days"] > max_hold_days
                or position_change <= cut_loss
                or position_change >= take_profit
            ):

                trade_log, trade_list = self.ExecuteTrade(
                    trade_log,
                    position_change,
                    isBackTest=isBackTest,
                    trade_list=trade_list,
                )

        # If action is SELL and has no position
        elif action == 2 and trade_log["position_days"] == 0:
            pass

        # If action is SELL and has position
        elif action == 2 and trade_log["position_days"] > 0:
            # Sell if meet the condition. 1:1 RRR
            if position_change <= cut_loss or position_change >= abs(cut_loss):
                trade_log, trade_list = self.ExecuteTrade(
                    trade_log,
                    position_change,
                    isBackTest=isBackTest,
                    trade_list=trade_list,
                )
            else:
                trade_log["position_days"] += 1

        # If action is hold and has no position
        elif action == 0 and trade_log["position_days"] == 0:
            pass

        #     """if action is hold and has position"""
        elif action == 0 and trade_log["position_days"] > 0:
            # Maximum holding period is only 30days and
            # has achieved -10% loss or 20% gain
            if (
                trade_log["position_days"] > max_hold_days
                or position_change <= cut_loss
                or position_change >= take_profit
            ):

                trade_log, trade_list = self.ExecuteTrade(
                    trade_log,
                    position_change,
                    isBackTest=isBackTest,
                    trade_list=trade_list,
                )
            else:
                trade_log["position_days"] += 1

        if isBackTest:
            return trade_log, trade_list
        else:
            return trade_log

    def ExecuteTrade(self, trade_log, position_change, isBackTest=False, trade_list=[]):
        """
        Executing the trade

        Parameters:
            trade_log: The trade logs for the current sessions
            position_change: dfsdfsdf
        """

        if position_change > 0:
            # trade_log['profit_win'] += position_change
            trade_log["profit_win"] += (
                trade_log["current_price"] - trade_log["current_position"]
            )
            trade_log["profit_win_count"] += 1

        else:
            # trade_log['profit_loss'] += abs(position_change)
            trade_log["profit_loss"] += abs(
                trade_log["current_price"] - trade_log["current_position"]
            )
            trade_log["profit_loss_count"] += 1

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

        trade_log["cumulative_profit"] += (
            trade_log["current_price"] - trade_log["current_position"]
        )
        trade_log["position_date"] = trade_log["current_date"]

        # Reset position_days
        trade_log["position_days"] = 0
        # Increment total_trades
        trade_log["total_trades"] += 1

        return trade_log, trade_list

    def transform_data_to_list_of_olhcv(self, data: list) -> None:
        """
        Transform list of data to query ready data
        """
        return list(map(lambda x: self.parse_to_OHLCV(x), data))

    def parse_to_OHLCV(self, data: list) -> dict:
        """
        Convert data to OHLCV.
        [1628807400000, '0.93890000', '0.94200000', '0.92220000', '0.93110000', '15134951.80000000', 1628809199999, '14107295.15935900', 20089, '7252452.80000000', '6760534.18742500', '0']
        """
        data[0] = datetime.utcfromtimestamp(data[0] / 1000).strftime("%Y-%m-%d %H:%M")
        return data[:6]

    def run(self, config_file):
        global current_stock
        # Load configuration.
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )

        pe = neat.parallel.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genomes)

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
        self.save_winner(winner, self.current_stock)

        # Display the winning genome.
        print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":
    process = Neat()
    recent_candles = pb.getKline(symbol="XRPUSDT", interval="30m")
    transformed_data = process.transform_data_to_list_of_olhcv(recent_candles)

    df = pd.DataFrame(transformed_data)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)
    df.set_index("Date", inplace=True)

    config_path = os.path.join(
        os.path.dirname("__file__"), "config/config-feedforward.txt"
    )

    process.current_stock_data = df
    process.current_stock = "XRPUSDT"

    # print(df.columns)
    process.run(config_file=config_path)
