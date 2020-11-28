# In[1]:
import numpy as np
import pandas as pd
# import ta
import random
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import pandas_ta as ta

import gym
from gym import spaces
import os

from utils import stock_helper
from utils import feature_selection
#hide warnings
import warnings
warnings.filterwarnings('ignore')

# In[2]:
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

tradable_stock_list = tradable_stock_df['stock code'].unique()
tradable_stock_list.sort()

print(tradable_stock_list, len(tradable_stock_list))

# group by tradable stock
tradable_stock_df = tradable_stock_df.groupby('stock code')


class StockMarket(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, df , stocks, num_actions , window , cut_loss , max_period):
    super(StockMarket, self).__init__()

    self.df = df
    self.data = pd.DataFrame()
    self.scaler = MinMaxScaler()
    self.num_actions = num_actions
    self.window = window
    self.max_period = max_period
    self.cut_loss = cut_loss
    self.stock_list = stocks
    tax = -1.19
    self.tax = tax
    self.action_space = spaces.Discrete(num_actions)
    self.data = self._getRandomStock()
    # Example for using image as input:
    # self.observation_space = spaces.Box(low=np.full((self.window,len(self.data.columns)-1),[-math.inf ,float('-inf'),float('-inf'),0,-100,0,0]),
    #                                     high=np.full((self.window,len(self.data.columns)-1),[math.inf,float('inf'),float('inf'),100,0,100,100]),
    #                                     shape=(self.window,len(self.data.columns)-1), dtype=np.double)

    # self.observation_space = spaces.Box(low=float('-inf'),high=float('inf'),
    #                                     shape=((self.window*(len(self.data.columns)-1))+1,), dtype=np.float32)
    # self.observation_space = spaces.Box(low=0,high=1,
    #                                     shape=((self.window*(len(self.data.columns)))+1,), dtype=np.float32)

    self.observation_space = spaces.Box(low=float('-inf'),high=float('inf'),
                                    shape=(self.window*(len(self.data.columns))+1,), dtype = np.float32)

  def _getRandomStock(self):
    stock = random.choice(self.stock_list)

    print(stock)

    data = self.df.get_group(stock)

    data.ta.strategy(ta.CommonStrategy)

    # col_list = ['Close','Volume']
    # col_list = ['pct_Close','Volume','Close']
    # col_list = ['Open','High','Low','Close','Volume']
    # col_list = ['Open','High','Low','Close','Volume']
    # col_list = ['Open','High','Low','Close']
    col_list = ['open','high','low','close','SMA_10', 'SMA_20', 'SMA_50']

    # data.ta.strategy(ta.AllStrategy)
    # data.ta.sma(length=20, append=True)


    data = data[col_list]
    data.dropna(inplace=True)

    # print(data.head())
    # print(data.columns)

    # print(stock)
    # print(self.stock_list)

    return data

  def _getObservation(self):

    # col_list = ["volume_obv","volatility_atr","trend_macd_diff","momentum_rsi","momentum_wr","momentum_stoch","momentum_stoch_signal"]

    # col_list = ["pct_Close","Volume","momentum_rsi","momentum_wr","momentum_stoch","momentum_stoch_signal","trend_macd_diff"]

    # col_list = ['Close','Volume']

    # col_list = ['pct_Close','Volume']
    # col_list = ['Open','High','Low','Close','Volume']
    # col_list = ['Open','High','Low','Close']
    col_list = ['open','high','low','close','SMA_10', 'SMA_20', 'SMA_50']

    obs = self.data[col_list]

    # print(obs.head())

    observation = obs.iloc[self.current_step - self.window : self.current_step]

    # for col in observation.columns[1:]:
      # observation[col] = preprocessing.scale(observation[col].values)

    # observation['Volume'] =  preprocessing.scale(observation['Volume'].values)

    # print(observation)

    # observation = self.scaler.fit_transform(observation)

    # observation = preprocessing.scale(observation)

    # observation = preprocessing.MinMaxScaler().fit_transform(observation)

    observation = preprocessing.RobustScaler(quantile_range=(25, 75)).fit_transform(observation)

    observation = np.array(observation)

    observation = np.append(observation,[self._position])

    # print(observation)

    observation = observation.flatten()

    # print(observation.shape)

    return observation

  def _takeAction(self,action):
    reward = self.capital
    position_change = 0
    #BUY ACTION
    ###########################################################################
    #buy without position
    if action == 1 and self.period == 0 :
      self.position_price = self.data["close"].iloc[self.current_step - 1]
      self.period = 1
      reward = self.capital

    #buy with position
    elif action == 1 and self.period >= 1 :
      self.period += 1

      #check if position is cut_loss
      v2 = self.data["close"].iloc[self.current_step - 1]
      v1 = self.position_price

      position_change = ((v2-v1)/v1) * 100
      net_position = position_change - 1.19

      # print(net_position)
      reward = self.capital + (self.capital*(net_position/100))

      if position_change <= self.cut_loss :
        #if cut loss then sell the position
        self.position_price = 0
        self.period = 0

        self.capital = self.capital + (self.capital*(net_position/100))
        reward = self.capital

      #check if period is greater than max period
      #then sell the position
      if self.period >= self.max_period:
        self.position_price = 0
        self.period = 0

        # reward = net_position
        self.capital = self.capital + (self.capital*(net_position/100))
        reward = self.capital


    ###########################################################################
    #HOLD ACTION
    #hold without position -> go to next time_step

    #hold with position
    elif action == 0 and self.period >= 1:
      self.period += 1

      #check if position is cut_loss
      # v2 = self.data["Close"].iloc[self.current_step - 1]
      v2 = self.data["close"].iloc[self.current_step - 1]
      v1 = self.position_price

      position_change = ((v2-v1)/v1) * 100
      net_position = position_change - 1.19

      reward = self.capital + (self.capital*(net_position/100))
      # reward = self.capital

      if position_change <= self.cut_loss :
        #if cut loss then sell the position
        self.position_price = 0
        self.period = 0

        self.capital = self.capital + (self.capital*(net_position/100))
        reward = self.capital

      #check if period is greater than max period
      #then sell the position
      if self.period >= self.max_period:
        self.position_price = 0
        self.period = 0

        # reward = net_position

        self.capital = self.capital + (self.capital*(net_position/100))
        reward = self.capital

    ###########################################################################
    #SELL ACTION
    #sell without position -> then there is nothing to sell ->then next time step

    #sell with position
    elif action == 2 and self.period >= 1:
      #SELL
      v2 = self.data["close"].iloc[self.current_step - 1]
      v1 = self.position_price

      position_change = ((v2-v1)/v1) * 100
      net_position = position_change - 1.19

      self.position_price = 0
      self.period = 0

      # reward = net_position
      self.capital = self.capital + (self.capital*(net_position/100))
      reward = self.capital

    return reward,position_change


  def reset(self):
    self.period = 0
    self.position_price = 0
    # self._position = 0
    self.data = self._getRandomStock()
    self.current_step = random.randint(self.window +1,len(self.data)-100)
    # self.rewards = 0
    self.capital = 100
    self._position = self.capital
    print(self.current_step,self.data.iloc[self.current_step - 1])
    self.price_now = self.data.iloc[self.current_step - 1]
    ...
    return self._getObservation()  # reward, done, info can't be included

  def step(self, action):
    # self.price_now = self.data["Close"].iloc[self.current_step - 1]
    self.price_now = self.data["close"].iloc[self.current_step - 1]
    _reward,position_change = self._takeAction(action)
    # self.profits += _reward * (self.current_step/len(self.data))
    self._position = _reward
    self.current_step += 1
    observation = self._getObservation()

    done = False
    #check if current_step is the last data
    if self.current_step >= len(self.data):
      done = True
      # self.reset()

    info = {"action":action,"price position":self.position_price,"price now":self.price_now,"period":self.period,"reward":_reward ,"position":position_change}
    ...
    return observation, _reward , done, info

  def render(self, mode='human'):
    ...
  def close (self):
    ...

env = StockMarket(df=tradable_stock_df, stocks=tradable_stock_list, num_actions=3, window=20, cut_loss=-5, max_period=45)
env.reset()
print(env.observation_space.shape)
print(env.observation_space.sample())

from stable_baselines.common.env_checker import check_env
check_env(env)

from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy,MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2,A2C,ACKTR,ACER

# env = make_vec_env(env, n_envs=4)

env = DummyVecEnv([lambda: env])

model = ACER(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

env = StockMarket( df = tradable_stock_df,stocks = tradable_stock_list, num_actions = 3 , window = 20, cut_loss = -5, max_period = 45)
env = DummyVecEnv([lambda: env])
obs = env.reset()
totalrewards = 0
trades = 0
d = 0
profits = 0

for i in range(4000):
    d = d+1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(info)
    print(dones)
    # print(rewards)
    if info[0]["period"] == 0:
      trades += 1
      totalrewards = rewards


    if dones:
      # Note that the VecEnv resets automatically
      # when a done signal is encountered
      print("Goal reached!", "reward=", totalrewards,"trades=",trades)
      print("days=",d,"profits=",rewards)
      break

model.save(f"model_market-{datetime.datetime.now()}")