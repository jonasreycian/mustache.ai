import pandas as pd
import numpy as np
import datetime as dt
import ta

from utils import stock_helper as std

stock_cache = {}

def GetCorrelationMatixPerStock(stock_data, noOfFeature, category):

    # Data with features
    features = std.AddFeatures(stock_data, category=category)

    # Make a correlation matrix among  features and target variable. Finally show a Heat Map with the values of the correlation matrix.
    featuresCorr = features.corr()

    # Sort the value
    featuresCorr = featuresCorr['Close'].sort_values(ascending=False)
    # Remove OHLC
    # featuresCorr = featuresCorr[4:]

    # Get the only desired features
    if(len(featuresCorr) > noOfFeature):
        featuresCorr = featuresCorr[:noOfFeature]

    # Store the feature
    # print (featuresCorr.index.T.values)
    features = features[featuresCorr.index.T.values]

    # print (features)

    return features

def GetTopFeatures(stock: str, stock_data, max_feature=5, isTrain=True, category='all'):
    """
    This will return dataset including the top features.
    Feature was screen-out using the correlation matrix of each stock.\n
    Params:\n
        stock:          Name of the stock
        noOfFeature:    Number of maximum features to be train
    """

    # Store stock data
    stock_data = stock_data.get_group(stock).loc['2010-01-01':]

    data = []
    if stock in stock_cache:
        data = stock_cache[stock]
    else:
        data = GetCorrelationMatixPerStock(stock_data, max_feature, category=category)
        # Align data and stock data
        len_diff = len(stock_data) - len(data) # Starting point of pivot
        stock_data = stock_data[len_diff:]
        data['Pure_Close'] = stock_data['Close']
        stock_cache[stock] = data

    # print (data)


    # Get the 80% of data to be trained
    if isTrain:
        print (f"\nExtracting features for {stock} was done...")
        data_len = len(data)
        data = data[:int(data_len * 0.8)]
        print (data)

    return data