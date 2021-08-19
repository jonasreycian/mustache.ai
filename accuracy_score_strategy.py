import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ta

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
import seaborn as sns

# Get all tradable stocks
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

    # group by tradable stock
    tradable_stock_df = tradable_stock_df.groupby("stock code")

    return tradable_stock_df


def create_features(stock):
    # Store list of tradable stocks
    data = load_data().get_group(stock)
    data.drop("Netforeign", 1, inplace=True)  # drop netforeign
    data.drop("stock code", 1, inplace=True)  # drop stock code

    # Convert all columns data frame to numeric columns
    data = data.apply(pd.to_numeric)

    data = ta.add_momentum_ta(
        data, high="High", low="Low", close="Close", volume="Volume"
    )

    # Remove all NaN value
    data.dropna(inplace=True)

    # Standardize the data
    for col in data.columns[:]:
        data[col] = preprocessing.scale(data[col].values)

    # Create the target variable that take the values of 1 if the stock price go up or -1 if the stock price go down
    target = np.where(data["Close"].shift(-1) > data["Close"], 1, -1)

    return data, target


features, target = create_features("JFC")

# Validation Set approach : take 80% of the data as the training set and 20 % as the test set. X is a dataframe with  the input variable
X = features[
    [
        "momentum_rsi",
        "momentum_wr",
        "momentum_uo",
        "momentum_stoch",
        "momentum_stoch_signal",
    ]
]

# Y is the target or output variable
y = target

length_to_split = int(len(features) * 0.75)
# Splitting the X and y into train and test datasets
X_train, X_test = X[:length_to_split], X[length_to_split:]
y_train, y_test = y[:length_to_split], y[length_to_split:]

# Print the size of the train and test dataset
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

clf = tree.DecisionTreeClassifier(random_state=20)

print(f"CLF: {clf}")

# Create the model on train dataset
model = clf.fit(X_train, y_train)

print(f"Model\n{model}")

# Calculate the accuracy
print(accuracy_score(y_test, model.predict(X_test), normalize=True))

# KFold Cross Validation approach
kf = KFold(n_splits=5, shuffle=False)
kf.split(X)

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model = []

# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train the model
    model = clf.fit(X_train, y_train)
    # Append to accuracy_model the accuracy of the model
    accuracy_model.append(
        accuracy_score(y_test, model.predict(X_test), normalize=True) * 100
    )

# Print the accuracy
print(accuracy_model)

scores = pd.DataFrame(accuracy_model, columns=["Scores"])

sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=["Iter1", "Iter2", "Iter3", "Iter4", "Iter5"], y="Scores", data=scores)
plt.show()
sns.set()
