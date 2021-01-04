import numpy as np
import pandas as pd
import csv
import os
import pickle
from sklearn.preprocessing import normalize


dir_name = os.path.dirname(os.path.abspath(__file__))

train_sales = pd.read_csv('data/train_sales_data.csv')
train_search = pd.read_csv('data/train_search_data.csv')

province_dict = {}
province_code = 0
car_model_dict = {}
car_model_code = 0

window = 2


def get_sales_row(df, province, car_model, year, month):
    row = df[(df['adcode'] == province) & (df['model'] == car_model) & 
                      (df['regYear'] == year) & (df['regMonth'] == month)]

    return row


def get_search_row(df, province, car_model, year, month):
    row = df[(df.adcode == province) & (df.model == car_model) &
                       (df.regYear == year) & (df.regMonth == month)]

    return row


def get_last_month_sales(df, province, car_model, year, month):
    if month == 1:
        year -= 1
        month = 13
    row = get_sales_row(df, province, car_model, year, month - 1)
    
    return int(row.salesVolume)


def get_last_month_search(df, province, car_model, year, month):
    if month == 1:
        year -= 1
        month = 13
    row = get_search_row(df, province, car_model, year, month - 1)

    return int(row.popularity)


def get_window_info(df, province, car_model, year, month, window):
    if year == 2016 and month <= window:
        return None, None, None
    sales = []
    for step in range(1, window + 1):
        m = month - step
        y = year
        if m <= 0:
            m += 12
            y -= 1
        row = get_sales_row(df, province, car_model, y, m)
        sales.append(int(row.salesVolume))
    return np.mean(sales), np.max(sales), np.min(sales)


def get_province_code(province):
    if province in province_dict.keys():
        return province_dict[province]
    else:
        global province_code
        province_code += 1
        province_dict[province] = province_code
        return province_code


def get_car_model_code(car_model):
    if car_model in car_model_dict.keys():
        return car_model_dict[car_model]
    else:
        global car_model_code
        car_model_code += 1
        car_model_dict[car_model] = car_model_code
        return car_model_code


def prepare_train_data():
    with open(os.path.join(dir_name, 'data/train_data_test.csv'), mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['province', 'car_model', 'year', 'month',
                         'last_month_search', 'last_month_sales', 'last_2month_sales_mean',
                         'last_2month_sales_max', 'last_2month_sales_min', 'sales', 'search'])

        for index, row in train_sales.iterrows():
            adcode = row['adcode']
            car_model = row['model']
            year = row['regYear']
            month = row['regMonth']
            if year == 2016 and month < 3:
                continue
            # feature
            last_month_sales = get_last_month_sales(train_sales, adcode, car_model, year, month)
            last_month_search = get_last_month_search(train_search, adcode, car_model, year, month)
            last_2month_sales_mean, last_2month_sales_max, last_2month_sales_min = \
                get_window_info(train_sales, adcode, car_model, year, month, window)
            province = get_province_code(adcode)
            car = get_car_model_code(car_model)
            # labels
            cur_sales = row['salesVolume']
            cur_sales = int(cur_sales)
            cur_search = get_search_row(train_search, adcode, car_model, year, month)['popularity']
            cur_search = int(cur_search)

            writer.writerow([province, car, year, month, last_month_search, last_month_sales, last_2month_sales_mean, 
                             last_2month_sales_max, last_2month_sales_min, cur_sales, cur_search])

    with open('models/province_encode.pkl', mode='wb') as f:
        pickle.dump(province_dict, f)
    with open('models/car_model_encode.pkl', mode='wb') as f:
        pickle.dump(car_model_dict, f)


if __name__ == '__main__':
    prepare_train_data()
    '''
    row = get_sales_row(310000, '3c974920a76ac9c1', 2016, 1)
    print(int(row.salesVolume))
    '''