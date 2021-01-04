import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


normalize_data = {}
df = pd.read_csv('data/train_data.csv')

X = df.iloc[:, :9]
Y_sales = df.iloc[:, -2:-1]
Y_search = df.iloc[:, -1:]


# normalize
nor = MinMaxScaler()

def normalize(df, column):
    np_arr = df[column].to_numpy().reshape(-1, 1)
    np_arr = nor.fit_transform(np_arr).reshape(-1)
    normalize_data[column + '_max'] = nor.data_max_[0]
    normalize_data[column + '_min'] = nor.data_min_[0]
    df[column] = np_arr

normalize(X, 'last_month_search')
normalize(X, 'last_month_sales')
normalize(X, 'last_2month_sales_mean')
normalize(X, 'last_2month_sales_max')
normalize(X, 'last_2month_sales_min')
normalize(Y_sales, 'sales')
normalize(Y_search, 'search')

# pickle dump the normalize data
with open('models/normalize_data.pkl', mode='wb') as f:
    pickle.dump(normalize_data, f)


X_sales_train, X_sales_test, Y_sales_train, Y_sales_test = \
    train_test_split(X, Y_sales, test_size=0.2, random_state=42)
X_search_train, X_search_test, Y_search_train, Y_search_test = \
    train_test_split(X, Y_search, test_size=0.2, random_state=42)

# train data
dtrain_sales = xgb.DMatrix(X_sales_train, label=Y_sales_train)
dtest_sales = xgb.DMatrix(X_sales_test, label=Y_sales_test)
dtrain_search = xgb.DMatrix(X_search_train, label=Y_search_train)
dtest_search = xgb.DMatrix(X_search_test, label=Y_search_test)