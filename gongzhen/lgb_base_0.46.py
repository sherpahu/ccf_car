# -*- coding: utf-8 -*-
# @Time    : 2019-08-23 17:27
# @Author  : Inf.Turing
# @Site    : 
# @File    : lgb_base.py
# @Software: PyCharm

# 随意看看就好

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb
#import catboost as ctb
import matplotlib.pyplot as plt

path = './ccf_car'

train_sales_data = pd.read_csv(path + '/train_sales_data.csv')
train_search_data = pd.read_csv(path + '/train_search_data.csv')
train_user_reply_data = pd.read_csv(path + '/train_user_reply_data.csv')

test = pd.read_csv(path + '/evaluation_public.csv')

data = pd.concat([train_sales_data, test], ignore_index=True)
data = data.merge(train_search_data, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user_reply_data, 'left', on=['model', 'regYear', 'regMonth'])

data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
del data['salesVolume'], data['forecastVolum']
data['bodyType'] = data['model'].map(train_sales_data.drop_duplicates('model').set_index('model')['bodyType'])

for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))

data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']

# 'popularity', 'carCommentVolum', 'newsReplyVolum','label'/

shift_feat = []

data['model_adcode'] = data['adcode'] + data['model']
data['model_adcode_mt'] = data['model_adcode'] * 100 + data['mt']
for i in [11]:
    i = i + 1
    shift_feat.append('shift_model_adcode_mt_label_{0}'.format(i))
    data['model_adcode_mt_{0}'.format(i)] = data['model_adcode_mt'] + i
    data_last = data[~data.label.isnull()].set_index('model_adcode_mt_{0}'.format(i))
    data['shift_model_adcode_mt_label_{0}'.format(i)] = data['model_adcode_mt'].map(data_last['label'])

num_feat = ['regYear'] + shift_feat
cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']

features = num_feat + cate_feat

# data['n_label'] = data['label'] / data.groupby('model')['label'].transform('mean')
train_idx = (data['mt'] <= 20)

valid_idx = (data['mt'].between(21, 24))

test_idx = (data['mt'] > 24)

data['model_weight'] = data.groupby('model')['label'].transform('mean')
data['n_label'] = data['label'] / data['model_weight']

train_x = data[train_idx][features]
train_y = data[train_idx]['n_label']

valid_x = data[valid_idx][features]
valid_y = data[valid_idx]['n_label']

# test_x = data[test_idx][features]

lgb_model = lgb.LGBMRegressor(
    num_leaves=32, reg_alpha=1, reg_lambda=0.1, objective='mse',
    max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=np.random.randint(1000),
    n_estimators=5000, subsample=0.8, colsample_bytree=0.8,
)

lgb_model.fit(train_x, train_y, eval_set=[
    (valid_x, valid_y),
], categorical_feature=cate_feat, early_stopping_rounds=100, verbose=100)

data['pred_label'] = lgb_model.predict(data[features]) * data['model_weight']


def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred: list,
        label: [list, 'mean'],

    }).reset_index()

    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)


best_score = score(data[valid_idx])
lgb_model.n_estimators = 666

lgb_model.fit(data[~test_idx][features], data[~test_idx]['n_label'], categorical_feature=cate_feat)
data['forecastVolum'] = lgb_model.predict(data[features]) * data['model_weight']
sub = data[test_idx][['id']]
sub['forecastVolum'] = data[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
sub.to_csv(path + 'lgb_base_0_46.csv', index=False)

#
#
#
#
# def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
#                   model_type='lgb'):
#     if 'sample_weight' not in data.keys():
#         data['sample_weight'] = 1
#     model.random_state = random_state
#     predict_label = 'predict_' + label
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     data[predict_label] = 0
#     test_index = (data[label].isnull()) | (data[label] == -1)
#     train_data = data[~test_index].reset_index(drop=True)
#     test_data = data[test_index]
#
#     for train_idx, val_idx in kfold.split(train_data):
#         model.random_state = model.random_state + 1
#
#         train_x = train_data.loc[train_idx][feature]
#         train_y = train_data.loc[train_idx][label]
#
#         test_x = train_data.loc[val_idx][feature]
#         test_y = train_data.loc[val_idx][label]
#         if model_type == 'lgb':
#             try:
#                 model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
#                           # eval_metric='mae',
#                           categorical_feature=cate_feature,
#                           sample_weight=train_data.loc[train_idx]['sample_weight'],
#                           verbose=100)
#             except:
#                 model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
#                           eval_metric='mae',
#                           # categorical_feature=cate_feature,
#                           sample_weight=train_data.loc[train_idx]['sample_weight'],
#                           verbose=100)
#         elif model_type == 'ctb':
#             model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
#                       # eval_metric='mae',
#                       cat_features=cate_feature,
#                       sample_weight=train_data.loc[train_idx]['sample_weight'],
#                       verbose=100)
#         train_data.loc[val_idx, predict_label] = model.predict(test_x)
#         if len(test_data) != 0:
#             test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature])
#     test_data[predict_label] = test_data[predict_label] / n_splits
#     print(mse(train_data[label], train_data[predict_label]), train_data[predict_label].mean(),
#           test_data[predict_label].mean())
#     return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label
#
#
# data, predict_label = get_predict_w(lgb_model, data, label='label',
#                                     feature=features, cate_feature=cate_feat,
#                                     random_state=2019, n_splits=5)
#
# # 自定义评价函数
# model = data[valid_idx]['model'].values
#
#
# def n_rmse(preds, labels):
#     valid_data = pd.DataFrame({
#         'label': labels,
#         'pred_label': preds,
#         'model': model
#     })
#     n_rmse = score(valid_data)
#     return 'n_rmse', n_rmse, True
#
# #
# # data['lgb'] = data[predict_label]
# #
# # data['forecastVolum'] = data['lgb'].apply(lambda x: 0 if x < 0 else x)
# # data[data.label.isnull()][['id', 'forecastVolum']].round().astype(int).to_csv(path + '/sub/sub.csv', index=False)
# #
# # ## 简单版本，预测下一个月。
# #
# #
# # #
# # # xgb_model = xgb.XGBRegressor(
# # #     max_depth=5,
# # #     learning_rate=0.01,
# # #     objective='reg:linear',
# # #     eval_metric='mae',
# # #     subsample=0.8, colsample_bytree=0.5,
# # #     n_estimators=3000,
# # #     reg_alpha=0.,
# # #     reg_lambda=0.001,
# # #     min_child_weight=50,
# # #     n_jobs=8,
# # #     seed=42,
# # # )
# # #
# # # data, predict_label = get_predict_w(xgb_model, data, label='temp_label',
# # #                                     feature=features, random_state=2018, n_splits=5)
# # #
# # # data['xgb'] = data[predict_label]
# # #
# # # ctb_params = {
# # #     'n_estimators': 10000,
# # #     'learning_rate': 0.02,
# # #     'random_seed': 4590,
# # #     'reg_lambda': 0.08,
# # #     'subsample': 0.7,
# # #     'bootstrap_type': 'Bernoulli',
# # #     'boosting_type': 'Plain',
# # #     'one_hot_max_size': 10,
# # #     'rsm': 0.5,
# # #     'leaf_estimation_iterations': 5,
# # #     'use_best_model': True,
# # #     'max_depth': 6,
# # #     'verbose': -1,
# # #     'thread_count': 4
# # # }
# # #
# # # ctb_model = ctb.CatBoostRegressor(**ctb_params)
# # #
# # # data, predict_label = get_predict_w(ctb_model, data, label='temp_label',
# # #                                     feature=features,
# # #                                     random_state=2019, n_splits=5, model_type='ctb')
# # #
# # # data['ctb'] = data[predict_label]
# # #
# # # data['t_label'] = data['lgb_mse'] * 0.4 + data['xgb_mse'] * 0.1 + data['ctb_mse'] * 0.4 + data['lgb_mae'] * 0.1
