import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from prepare_train_data import get_last_month_sales, get_last_month_search, get_window_info, get_sales_row


df = pd.read_csv('data/evaluation_public.csv')
df['popularity'] = None
df.rename(columns={'forecastVolum': 'salesVolume'})
train_sales = pd.read_csv('data/train_sales_data.csv')
train_search = pd.read_csv('data/train_search_data.csv')

f = open('models/normalize_data.pkl', 'rb')
normalize_data = pickle.load(f)
f.close()
f = open('models/car_model_encode.pkl', 'rb')
car_model_encode = pickle.load(f)
f.close()
f = open('models/province_encode.pkl', 'rb')
province_encode = pickle.load(f)
f.close()

xgb_model_sales = xgb.Booster()
xgb_model_search = xgb.Booster()
xgb_model_sales.load_model('models/sales.xgb')
xgb_model_search.load_model('models/search.xgb')

def transform(data, max_data, min_data):
    return (data - min_data) / (max_data - min_data)


def inverse(data, max_data, min_data):
    data = data * (max_data - min_data) + min_data
    if data < 0:
        data = (max_data + min_data) / 2
    
    return int(data)


def get_nor_last_month_sales(province, car_model, year, month):
    max_data = normalize_data['last_month_sales_max']
    min_data = normalize_data['last_month_sales_min']
    if month == 1:
        sales = get_last_month_sales(train_sales, province, car_model, year, month)
        return transform(float(sales), max_data, min_data)
    else:
        sales = get_last_month_sales(df, province, car_model, year, month)
        return transform(float(sales), max_data, min_data)


def get_nor_last_month_search(province, car_model, year, month):
    max_data = normalize_data['last_month_search_max']
    min_data = normalize_data['last_month_search_min']
    if month == 1:
        search = get_last_month_search(train_search, province, car_model, year, month)
        return transform(float(search), max_data, min_data)
    else:
        search = get_last_month_search(df, province, car_model, year, month)
        return transform(float(search), max_data, min_data)


def get_nor_window_info(province, car_model, year, month, window=2):
    max_data_mean = normalize_data['last_2month_sales_mean_max']
    min_data_mean = normalize_data['last_2month_sales_mean_min']
    max_data_max = normalize_data['last_2month_sales_max_max']
    min_data_max = normalize_data['last_2month_sales_max_min']
    max_data_min = normalize_data['last_2month_sales_min_max']
    min_data_min = normalize_data['last_2month_sales_min_min']
    if month == 1:
        sales_mean, sales_max, sales_min = get_window_info(train_sales, province, car_model, year, month, window)
        return (
            transform(sales_mean, max_data_mean, min_data_mean),
            transform(sales_max, max_data_max, min_data_max),
            transform(sales_min, max_data_min, min_data_min)
        )
    elif month == 2:
        sales = []
        sale = get_last_month_sales(df, province, car_model, year, month)
        sales.append(sale)
        sale = get_last_month_sales(train_sales, province, car_model, year, month - 1)
        sales.append(sale)
        return (
            transform(np.mean(sales), max_data_mean, min_data_mean),
            transform(np.max(sales), max_data_max, min_data_max),
            transform(np.min(sales), max_data_min, min_data_min)
        )
    else:
        sales_mean, sales_max, sales_min = get_window_info(df, province, car_model, year, month, window)
        return (
            transform(sales_mean, max_data_mean, min_data_mean),
            transform(sales_max, max_data_max, min_data_max),
            transform(sales_min, max_data_min, min_data_min)
        )


def get_province_code(province):
    return province_encode[province]


def get_car_model_code(car_model):
    return car_model_encode[car_model]


def get_input(province, car_model, year, month):
    province_code = get_province_code(province)
    car_model_code = get_car_model_code(car_model)
    last_month_search = get_nor_last_month_search(province, car_model, year, month)
    last_month_sales = get_nor_last_month_sales(province, car_model, year, month)
    last_2month_mean, last_2month_max, last_2month_min = get_nor_window_info(province, car_model, year, month)
    pred_input = np.array([province_code, car_model_code, last_month_search, 
                           last_month_sales, last_2month_mean, last_2month_max, last_2month_min]).reshape(1, -1)

    return xgb.DMatrix(pred_input, label=None)


def predict():
    max_sales = normalize_data['sales_max']
    min_sales = normalize_data['sales_min']
    max_search = normalize_data['search_max']
    min_search = normalize_data['search_min']
    for index, row in df.iterrows():
        province = row['adcode']
        car_model = row['model']
        year = row['regYear']
        month = row['regMonth']
        pred_input = get_input(province, car_model, year, month)
        pred_sales = xgb_model_sales.predict(pred_input)[0]
        pred_search = xgb_model_search.predict(pred_input)[0]
        pred_sales = inverse(pred_sales, max_sales, min_sales)
        pred_search = inverse(pred_search, max_search, min_search)

        pred_row = (df['adcode'] == province) & (df['model'] == car_model) & \
                      (df['regYear'] == year) & (df['regMonth'] == month)
        df.loc[pred_row, 'salesVolume'] = pred_sales
        df.loc[pred_row, 'popularity'] = pred_search

    df['salesVolume'] = df['salesVolume'].astype(int)
    df.to_csv('submit.csv', columns=['id', 'salesVolume'], index=False)


if __name__ == '__main__':
    predict()