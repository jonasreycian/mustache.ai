import os
import pandas as pd
import ta
import json
import time

from sklearn import preprocessing
from utils import indicators, feature_selection
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator, MACD
from AlmaIndicator import ALMAIndicator

# hide warnings
import warnings
warnings.filterwarnings('ignore')

def AnalyzeTrainedData(trade_log, reward_cat):
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
    trade_log['total_trades'] = 1 if trade_log['total_trades']==0 else trade_log['total_trades']

    pw = trade_log['profit_win_count'] / trade_log['total_trades']
    aw = trade_log['profit_win'] / trade_log['total_trades']
    pl = trade_log['profit_loss_count'] / trade_log['total_trades']
    al = trade_log['profit_loss'] / trade_log['total_trades']

    # Compute the appt
    trade_log['appt'] = (pw*aw) - (pl*al)
    # Compute the win_ratio
    trade_log['win_ratio'] = trade_log["profit_win_count"] / trade_log['total_trades'] * 100

    trade_log['pwaw'] = (pw*aw)

    # Compute the win_loss_ratio
    if trade_log['profit_loss_count'] == 0:
        trade_log['win_loss_ratio'] = trade_log["profit_win_count"]
    else:
        trade_log['win_loss_ratio'] = trade_log["profit_win_count"] / trade_log['profit_loss_count']

    # Get reward
    trade_log = GetReward(reward_cat, trade_log)

    print ( f'{trade_log["stock_name"]}\t' +
            f'{trade_log["profit_win_count"]}\t' +
            f'{trade_log["profit_loss_count"]}\t' +
            f'{trade_log["total_trades"]}' +
            '\tWLRatio: {0:.2f}'.format(trade_log["win_loss_ratio"]) +
            '\tAPPT: {0:.2f}'.format(trade_log["appt"]) +
            '\tWinRat: {0:.2f}'.format(trade_log["win_ratio"]) +
            '\tReward: {0:.2f}'.format(trade_log["reward"])+
            '\tOpenPos: {0:.2f}'.format(trade_log["open_position"])+
            '\tCProfit: {0:.2f}'.format(trade_log["cumulative_profit"])+
            f'\tLastTrade: {trade_log["position_date"]}'
            )

    return trade_log

def GetReward(reward_cat, trade_log):
    """
    Return reward based on given fitness
    """
    # win_loss_ratio * appt * #trades
    if reward_cat ==1:
        trade_log['reward'] = trade_log['appt'] * trade_log['win_loss_ratio'] * trade_log['profit_win_count']
    # win_loss_ratio * appt * #trades (Modified)
    elif reward_cat ==2:
        wl_ratio = 3 if trade_log['win_loss_ratio'] > 3 else trade_log['win_loss_ratio']
        trade_log['reward'] = trade_log['appt'] * wl_ratio * trade_log['profit_win_count']
    # profit_win_count win * appt
    elif reward_cat ==3:
        trade_log['reward'] = (trade_log['profit_win']**0.5) * trade_log['appt'] * (trade_log['win_loss_ratio']**0.5)
    # profit_win_count win * appt
    elif reward_cat ==4:
        trade_log['reward'] = (trade_log['profit_win']**0.5) * \
                              (trade_log['profit_win_count']**0.5)* \
                              (trade_log['win_ratio']**0.5)
    elif reward_cat ==5:
        trade_log['reward'] = (trade_log['win_loss_ratio']**0.5) * \
                            (trade_log['profit_win_count']**0.5) * \
                            (trade_log['win_ratio']**0.5) * \
                            (trade_log['profit_win']**0.5) * \
                            trade_log['appt']
    elif reward_cat ==6:
        trade_log['reward'] = (trade_log['profit_win']**0.5) * \
                                (trade_log['profit_win_count']**0.5)* \
                                (trade_log['win_ratio']**0.5) * \
                                trade_log['appt']
    elif reward_cat ==7:
        trade_log['reward'] = ((trade_log['profit_win']) - trade_log['profit_loss']) * (trade_log['win_ratio']**0.5)
    elif reward_cat ==8:
        trade_log['reward'] = (trade_log['cumulative_profit'] + trade_log['open_position']) * trade_log['win_loss_ratio']
    elif reward_cat ==9:
        trade_log['reward'] = trade_log['cumulative_profit'] + trade_log['open_position']

    return trade_log


def save_appt(appt_file):
    with open(f'result/appt/appt_{time.strftime("%m%d%Y")}.json', 'w') as json_file:
        json.dump(appt_file, json_file)

def load_all_stock_with_features(stocks, stocks_data):
    stock_data_list = {}
    print ('Loading all stock including with features')
    for stock in stocks:
        if stock in stock_data_list:
            print (f"\tExtracting features for {stock} was done...")
            continue

        stock_data_list[stock] = feature_selection.GetTopFeatures(stock, stocks_data, max_feature=10)
        print (f"\tExtracting features for {stock} was done...")

    return stock_data_list

def get_all_stock_data():
    dfs = []
    for item in os.listdir('./data'):
        df = pd.read_csv(f'./data/{item}',
                         header=None,
                         names=['stock code', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Netforeign'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.dropna(inplace=True)
        df.set_index('Date', inplace=True)

        # sort values by date
        df.sort_values('Date', inplace=True)
        dfs.append(df)

    main_df = pd.concat(dfs)
    main_df.drop('Netforeign', 1, inplace=True)
    main_df.tail()

    #################################################################################################
    # read tradeble stocks
    tradable = pd.read_csv('./tradable_psei.csv')

    # creating a new df of tradable stock
    tradable_stock_df = main_df[main_df['stock code'].isin(tradable['stock'])]
    tradable_stock_df.head()

    # group by tradable stock
    tradable_stock_df = tradable_stock_df.groupby('stock code')

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

    if category=="all" or category=="momentum":
        # Append all Momentum-related features
        print(f'data ${data}')
        data = ta.add_momentum_ta(data, high="High", low="Low", close="Close", volume="Volume")

    elif category=="all" or category=="trend":
        # Append all tren-related features
        data = ta.add_trend_ta(data, high="High", low="Low", close="Close")

    elif category=="all" or category=="volatility":
        # Append all volatility-related features
        data = ta.add_volatility_ta(data, high="High", low="Low", close="Close")

    elif category=="all" or category=="volume":
        # Append all volume-related features
        data = ta.add_volume_ta(data, high="High", low="Low", close="Close", volume="Volume")

    elif category=="emaribbon":
        # Append user-specified features
        data = indicators.ema(data, period=9)
        data = indicators.ema(data, period=20)
        data = indicators.ema(data, period=25)
        data = indicators.ema(data, period=30)
        data = indicators.ema(data, period=35)
        data = indicators.ema(data, period=40)
        data = indicators.ema(data, period=45)
        data = indicators.ema(data, period=50)
        data = indicators.ema(data, period=55)
        data = indicators.volume(data, period=20)

    elif category=='bottom_fish':
        # BUY:
        #   CCI > -100
        #   DCHL/DCHLI: Candle must touch the the lower indicators0
        data = ta.add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume')

        bottom_fish_list = ["Open","High","Low","Close", "Volume", \
            "volatility_dcl", "volatility_dch", "volatility_dcli", "volatility_dchi", "trend_cci"]
        bottom_fish = data[bottom_fish_list]
        bottom_fish['volatility_dcl'] = data['volatility_dcl']
        bottom_fish['volatility_dch'] = data['volatility_dch']
        bottom_fish['volatility_dcli'] = data['volatility_dcl']
        bottom_fish['volatility_dchi'] = data['volatility_dch']
        bottom_fish['trend_cci'] = data['trend_cci']
        bottom_fish = indicators.volume(bottom_fish, period=20)
        data = bottom_fish

    elif category=='macdo':
        indicator_macd = MACD(close=data['Close'], n_slow=20, n_fast=5, n_sign=5)
        data['trend_macd'] = indicator_macd.macd()
        data = indicators.ma(data, period=200)
        data = indicators.ema(data, period=62)
        data = indicators.ema(data, period=38)
        data = indicators.volume(data, period=20)


    elif category=='almarsi':
        alma_indicator = ALMAIndicator(close=data['Close'])
        data['alma9'] = alma_indicator.alma()
        # data['alma20'] = alma_indicator.alma_weights(window=20)
        # data['alma50'] = alma_indicator.alma_weights(window=50)
        data = indicators.rsi(data, period=9)
        data = indicators.rsi(data, period=25)
        data = indicators.rsi(data, period=12)

    # print (data)
    if category=="all":
        data.dropna(inplace=True, axis=1)
    else:
        data.dropna(inplace=True)

    # print (data)

    # Scale the value
    for col in data.columns[:]:
        data[col] = preprocessing.scale(data[col].values)

    # print (data)
    return data

def profit_loss_action(
        action,
        trade_log,
        cut_loss=-5,
        take_profit=40,
        max_hold_days=30,
        trade_list=[],
        isBackTest=False):

    if trade_log['current_position'] == 0:
        trade_log['current_position'] = 1

    position_change = (
        trade_log['current_price'] - trade_log['current_position']
        ) / trade_log['current_position'] * 100

    # If action is BUY and has no position
    if (action == 1) and trade_log['position_days'] == 0 :
        trade_log['position_days'] = 1
        trade_log['current_position'] = trade_log['current_price']
        trade_log['position_date'] = trade_log['current_date']

        if isBackTest:
            # Record the trade made
            trade_list.append([
                trade_log['current_date'],
                trade_log['stock_name'],
                trade_log['current_price'],
                'BUY'
            ])

    # If action is BUY and has position
    elif action == 1 and trade_log['position_days'] > 0 :
        trade_log['position_days'] += 1

        # Sell if meet the condition
        if trade_log['position_days'] > max_hold_days \
            or position_change <= cut_loss \
            or position_change >= take_profit:

            trade_log, trade_list = ExecuteTrade(trade_log, position_change, isBackTest=isBackTest, trade_list=trade_list)

    # If action is SELL and has no position
    elif action == 2 and trade_log['position_days'] == 0:
        pass

    # If action is SELL and has position
    elif action == 2 and trade_log['position_days'] > 0:
        # Sell if meet the condition. 1:1 RRR
        if position_change <= cut_loss or position_change >= abs(cut_loss):
            trade_log, trade_list = ExecuteTrade(trade_log, position_change, isBackTest=isBackTest, trade_list=trade_list)
        else:
            trade_log['position_days'] += 1


    # If action is hold and has no position
    elif action == 0 and trade_log['position_days'] == 0:
        pass

#     """if action is hold and has position"""
    elif action == 0 and trade_log['position_days'] > 0 :
        # Maximum holding period is only 30days and
        # has achieved -10% loss or 20% gain
        if trade_log['position_days'] > max_hold_days \
            or position_change <= cut_loss \
            or position_change >= take_profit:

            trade_log, trade_list = ExecuteTrade(trade_log, position_change, isBackTest=isBackTest, trade_list=trade_list)
        else:
            trade_log['position_days'] += 1

    if isBackTest:
        return trade_log, trade_list
    else:
        return trade_log

def ExecuteTrade(trade_log, position_change, isBackTest=False, trade_list=[]):
    """
    Executing the trade

    Parameters:
        trade_log: The trade logs for the current sessions
        position_change: dfsdfsdf
    """

    if position_change > 0:
        # trade_log['profit_win'] += position_change
        trade_log['profit_win'] += trade_log['current_price'] - trade_log['current_position']
        trade_log['profit_win_count'] += 1

    else:
        # trade_log['profit_loss'] += abs(position_change)
        trade_log['profit_loss'] += abs(trade_log['current_price'] - trade_log['current_position'])
        trade_log['profit_loss_count'] += 1

    if isBackTest:
        # Record the trade made
        trade_list.append([
            trade_log['current_date'],
            trade_log['stock_name'],
            trade_log['current_price'],
            'SELL'
        ])

    trade_log['cumulative_profit'] += (trade_log['current_price'] - trade_log['current_position'])
    trade_log['position_date'] = trade_log['current_date']

    # Reset position_days
    trade_log['position_days'] = 0
    # Increment total_trades
    trade_log['total_trades'] += 1

    return trade_log, trade_list

def read_json(location_data='../result/appt'):
    """
    This function reads all the json inside a folder.
    """
    appt_list_per_hash = {}
    for item in os.listdir(location_data):
        # Load the json file
        with open(f'{location_data}/{item}') as f:
            data = json.load(f)

        # Get the list of the key
        for key in data:
            if key in appt_list_per_hash:
                pass
            else:
                appt_list_per_hash[key] = data[key]

    return appt_list_per_hash

def format_json(json_data, indent = 4, sort_keys=True):
    return json.dumps(json_data, indent=indent, sort_keys=sort_keys)

def WriteJSONFile(filepath='./result/analysis/', filename="", data={}):
    with open(f'{filepath+filename}', 'w') as fp:
        json.dump(data, fp)