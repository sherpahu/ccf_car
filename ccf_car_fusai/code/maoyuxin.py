import sys
import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import math



def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']


    #for col in tqdm(['label','popularity','area_sales_volume','pop_dist']):
    for col in tqdm(['label', 'popularity', 'area_sales_volume']):
        # shift
        for i in [1,2,3,4,5,6,7,8,9]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])

    return df, stat_feat

def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)

def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb'):
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2333,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7,
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              categorical_feature=cate_feat,
              early_stopping_rounds=100, verbose=100)
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                max_depth=5 , learning_rate=0.05, n_estimators=2000,
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9,
                                colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse'
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              early_stopping_rounds=100, verbose=100)
    return model

def get_train_model(df_, m, m_type='lgb'):
    df = df_.copy()
    # 数据集划分
    st = 13
    all_idx   = (df['mt'].between(st , m-1))
    train_idx = (df['mt'].between(st , m-5))
    valid_idx = (df['mt'].between(m-4, m-4))
    test_idx  = (df['mt'].between(m  , m  ))
    print('all_idx  :',st ,m-1)
    print('train_idx:',st ,m-5)
    print('valid_idx:',m-4,m-4)
    print('test_idx :',m  ,m  )
    # 最终确认
    train_x = df[train_idx][features]
    train_y = df[train_idx]['label']
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx]['label']
    # get model
    model = get_model_type(train_x,train_y,valid_x,valid_y,m_type)
    # offline
    df['pred_label'] = model.predict(df[features])
    best_score = score(df[valid_idx])
    # online
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][features], df[all_idx]['label'], categorical_feature=cate_feat)
    elif m_type == 'xgb':
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][features], df[all_idx]['label'])
    df['forecastVolum'] = model.predict(df[features])
    print('valid mean:',df[valid_idx]['pred_label'].mean())
    print('true  mean:',df[valid_idx]['label'].mean())
    print('test  mean:',df[test_idx]['forecastVolum'].mean())
    # 阶段结果
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    return sub,df[valid_idx]['pred_label']

if __name__ == '__main__':
    path = '../ccf_car/'
    train_sales = pd.read_csv(path + 'train_sales_data.csv')
    train_search = pd.read_csv(path + 'train_search_data.csv')
    train_user = pd.read_csv(path + 'train_user_reply_data.csv')
    evaluation_public = pd.read_csv(path + 'evaluation_public.csv')
    submit_example = pd.read_csv(path + 'submit_example.csv')
    data = pd.concat([train_sales, evaluation_public], ignore_index=True)
    data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
    data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
    data['label'] = data['salesVolume']
    data['id'] = data['id'].fillna(0).astype(int)
    data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
    # LabelEncoder
    for i in ['bodyType', 'model']:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
    data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']

    area_sales = {}
    for raw in data[['province', 'salesVolume','regMonth','regYear']].values:
        province = raw[0]
        sales = raw[1]
        if pd.isna(sales):
            continue
        regMonth = raw[2]
        regYear = raw[3]
        key = province + "_" + str(regYear) + "_" + str(regMonth)
        if area_sales.__contains__(key):
            area_sales[key] += sales
        else:
            area_sales[key] = sales
    new_column = []
    new_column1 = []
    for raw in data[['province', 'salesVolume','regMonth','regYear']].values:
        province = raw[0]
        sales = raw[1]
        if pd.isna(sales):
            new_column.append(None)
            new_column1.append(None)
            continue
        regMonth = raw[2]
        regYear = raw[3]
        key = province + "_" + str(regYear) + "_" + str(regMonth)
        new_column.append(area_sales[key])
        new_column1.append(sales/area_sales[key])
    data['area_sales_volume'] = new_column
    '''
    new_column2 = []
    temp = {}
    for raw in data[['adcode','model','regYear','regMonth','popularity','carCommentVolum','newsReplyVolum']].values:
        adcode = int(raw[0])
        model = int(raw[1])
        regYear = int(raw[2])
        regMonth = int(raw[3])
        id = str(adcode) + "_" + str(model) + "_" + str(regYear) + "_" + str(regMonth)
        if pd.isna(raw[4]) or pd.isna(raw[5]) or pd.isna(raw[6]):
            temp[id] = None
            continue
        value = [int(raw[4]),int(raw[5]),int(raw[6])]
        temp[id] = value

    for key in temp.keys():
        id = key
        value = temp[id]
        raw = key.split('_')
        adcode = raw[0]
        model = raw[1]
        regYear = raw[2]
        regMonth = raw[3]
        if value is None:
            new_column2.append(None)
            continue
        if regYear == '2016' and regMonth == '1':
            new_column2.append(None)
            continue
        if regMonth == '1':
            id2 =  str(adcode) + "_" + str(model) + "_" + str(int(regYear)-1) + "_" + "12"
            value2 = temp[id2]
            if value2 is None:
                new_column2.append(None)
                continue
            new_column2.append(math.sqrt((value[0]-value2[0]) * (value[0]-value2[0]) +
                                         (value[1]-value2[1]) * (value[1]-value2[1]) +
                                         (value[2]-value2[2]) * (value[2]-value2[2])))
        else:
            id2 = str(adcode) + "_" + str(model) + "_" + str(regYear) + "_" + str(int(regMonth)-1)
            value2 = temp[id2]
            if value2 is None:
                new_column2.append(None)
                continue
            new_column2.append(math.sqrt((value[0] - value2[0]) * (value[0] - value2[0]) +
                                         (value[1] - value2[1]) * (value[1] - value2[1]) +
                                         (value[2] - value2[2]) * (value[2] - value2[2])))
    #data['area_radio'] = new_column1
    data['pop_dist'] = new_column2
    '''



    for month in [25, 26, 27, 28]:
        m_type = 'xgb'

        data_df, stat_feat = get_stat_feature(data)

        num_feat = ['regYear'] + stat_feat
        cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']
        if m_type == 'lgb':
            for i in cate_feat:
                data_df[i] = data_df[i].astype('category')
        elif m_type == 'xgb':
            lbl = LabelEncoder()
            for i in tqdm(cate_feat):
                data_df[i] = lbl.fit_transform(data_df[i].astype(str))

        features = num_feat + cate_feat
        print(len(features), len(set(features)))

        sub, val_pred = get_train_model(data_df, month, m_type)
        data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
        data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values
    sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
    sub.columns = ['id', 'forecastVolum']
    sub[['id', 'forecastVolum']].round().astype(int).to_csv('car_prediction_sales.csv', index=False)