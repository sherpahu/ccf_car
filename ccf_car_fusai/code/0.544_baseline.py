TRAIN_SALES_DATA_PATH = "../data/train_sales_data.csv"
TRAIN_SEARCH_DATA_PATH = "../data/train_search_data.csv"
TRAIN_USER_REPLY_DATA_PATH = "../data/train_user_reply_data.csv"
TEST_PATH = "../data/evaluation_public.csv"

import re
import gc
import os
import csv
import time
import math
import random
import pickle
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook, trange
from sklearn import preprocessing
from scipy import stats

from sklearn.tree import ExtraTreeRegressor
import lightgbm as lgb
import xgboost as xgb
# import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(df, data_type):
    if data_type == "train":
        return df[df.regYear == 2016]
    elif data_type == "test":
        return df[df.regYear == 2017]



train_sales = pd.read_csv(TRAIN_SALES_DATA_PATH)
train_search = pd.read_csv(TRAIN_SEARCH_DATA_PATH)
train_user = pd.read_csv(TRAIN_USER_REPLY_DATA_PATH)
test_data = pd.read_csv(TEST_PATH)

train_sales.salesVolume = train_sales.salesVolume.apply(lambda x: np.log(1+x))

# 特征工程
def cal_basic_fea(df:pd.DataFrame, cal_col:str, stat_dim:list, data_type:str) -> pd.DataFrame:
    """
    计算原始特征、周期特征、趋势特征
    """
    train_sales_data = get_data(train_sales, data_type)

    name_prefix = "_".join(stat_dim) + "_%s"%cal_col
    drop_name = "level_%d"%len(stat_dim)

    # 原始特征
    feature_data = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).unstack(level=-1)
    feature_data.columns = [name_prefix + "_%d"%x for x in feature_data.columns.ravel()]
    feature_data = feature_data.reset_index()

    # 周期特征
    ## shift_div
    tmp_df = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).apply(lambda x: x / x.shift(1)).reset_index()

    tmp_df = tmp_df.rename(columns={"salesVolume":"shift_div"})

    train_sales_data = pd.merge(train_sales_data, tmp_df, on=stat_dim, how="left")

    tmp_df = train_sales_data.dropna().groupby(stat_dim).shift_div.apply(lambda x: x.sum()).unstack(level=-1)

    tmp_df.columns = [name_prefix + "_shift_div_%d"%x for x in tmp_df.columns.ravel()]

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")

    ## shift_sub
    tmp_df = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).apply(lambda x: x - x.shift(1)).reset_index()

    tmp_df = tmp_df.rename(columns={"salesVolume":"shift_sub"})

    train_sales_data = pd.merge(train_sales_data, tmp_df, on=stat_dim, how="left")

    tmp_df = train_sales_data.dropna().groupby(stat_dim).shift_sub.apply(lambda x: x.sum()).unstack(level=-1)

    tmp_df.columns = [name_prefix + "_shift_sub_%d"%x for x in tmp_df.columns.ravel()]

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")

    # 趋势特征
    ## shift_div
    tmp_df = train_sales_data.groupby(stat_dim)["shift_div"].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).apply(lambda x: x / x.shift(1)).reset_index()

    tmp_df = tmp_df.rename(columns={"shift_div":"shift_2_div"})

    train_sales_data = pd.merge(train_sales_data, tmp_df, on=stat_dim, how="left")

    tmp_df = train_sales_data.dropna().groupby(stat_dim).shift_div.apply(lambda x: x.sum()).unstack(level=-1)

    tmp_df.columns = [name_prefix + "_shift_2_div_%d"%x for x in tmp_df.columns.ravel()]

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")

    ## shift_sub
    tmp_df = train_sales_data.groupby(stat_dim)["shift_sub"].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).apply(lambda x: x - x.shift(1)).reset_index()

    tmp_df = tmp_df.rename(columns={"shift_sub":"shift_2_sub"})

    train_sales_data = pd.merge(train_sales_data, tmp_df, on=stat_dim, how="left")

    tmp_df = train_sales_data.dropna().groupby(stat_dim).shift_sub.apply(lambda x: x.sum()).unstack(level=-1)

    tmp_df.columns = [name_prefix + "_shift_2_sub_%d"%x for x in tmp_df.columns.ravel()]

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")
    return feature_data

def cal_windows_fea(df:pd.DataFrame, cal_col:str, stat_dim:list, data_type:str) -> pd.DataFrame:
    """
    计算滑窗特征
    """
    train_sales_data = get_data(df, data_type)

    name_prefix = "_".join(stat_dim) + "_%s"%cal_col

    # 滑窗特征
    ## 均值
    feature_data = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).rolling(3).mean()

    feature_data = feature_data.dropna().unstack(level=-1)

    if len(stat_dim) == 3:
        feature_data.index = feature_data.index.droplevel(0)
        feature_data.index = feature_data.index.droplevel(0)
    elif len(stat_dim) == 2:
        feature_data.index = feature_data.index.droplevel(0)


    feature_data.reset_index(inplace=True)
    feature_data = feature_data.rename(columns={k:"%s_rolling_mean_%d"%(name_prefix, k) for k in range(13)})

    ## std
    tmp_df = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).rolling(3).std()

    tmp_df = tmp_df.dropna().unstack(level=-1)

    if len(stat_dim) == 3:
        tmp_df.index = tmp_df.index.droplevel(0)
        tmp_df.index = tmp_df.index.droplevel(0)
    elif len(stat_dim) == 2:
        tmp_df.index = tmp_df.index.droplevel(0)


    tmp_df.reset_index(inplace=True)
    tmp_df = tmp_df.rename(columns={k:"%s_rolling_std_%d"%(name_prefix, k) for k in range(13)})

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")

    ## sum
    tmp_df = train_sales_data.groupby(stat_dim)[cal_col].apply(lambda x: x.sum()).groupby(stat_dim[:-1]).rolling(3).sum()

    tmp_df = tmp_df.dropna().unstack(level=-1)

    if len(stat_dim) == 3:
        tmp_df.index = tmp_df.index.droplevel(0)
        tmp_df.index = tmp_df.index.droplevel(0)
    elif len(stat_dim) == 2:
        tmp_df.index = tmp_df.index.droplevel(0)

    tmp_df.reset_index(inplace=True)
    tmp_df = tmp_df.rename(columns={k:"%s_rolling_sum_%d"%(name_prefix, k) for k in range(13)})

    feature_data = pd.merge(feature_data, tmp_df, on=stat_dim[:-1], how="left")
    return feature_data


# cal_basic_fea
model2type = train_sales[["model", "bodyType"]].drop_duplicates().set_index("model").to_dict()["bodyType"]
# 城市+车
train_basic_fea = cal_basic_fea(train_sales, "salesVolume", ["adcode", "model", "regMonth"], "train")
# 城市
tmp_df = cal_basic_fea(train_sales, "salesVolume", ["adcode", "regMonth"], "train")
train_basic_fea = pd.merge(train_basic_fea, tmp_df, on="adcode", how="left")
# 车
tmp_df = cal_basic_fea(train_sales, "salesVolume", ["model", "regMonth"], "train")
train_basic_fea = pd.merge(train_basic_fea, tmp_df, on="model", how="left")
# 城市+车型
tmp_df = cal_basic_fea(train_sales, "salesVolume", ["adcode", "bodyType", "regMonth"], "train")
train_basic_fea["bodyType"] = train_basic_fea.model.apply(lambda x: model2type[x])
train_basic_fea = pd.merge(train_basic_fea, tmp_df, on=["adcode", "bodyType"], how="left")


# cal_windows_fea
# 城市+车
train_windows_fea = cal_windows_fea(train_sales, cal_col="salesVolume", stat_dim=["adcode", "model", "regMonth"], data_type="train")
# 城市
tmp_df = cal_windows_fea(train_sales, "salesVolume", ["adcode", "regMonth"], "train")
train_windows_fea = pd.merge(train_windows_fea, tmp_df, on="adcode", how="left")
# 车
tmp_df = cal_windows_fea(train_sales, "salesVolume", ["model", "regMonth"], "train")
train_windows_fea = pd.merge(train_windows_fea, tmp_df, on="model", how="left")
# 城市+车型
tmp_df = cal_windows_fea(train_sales, "salesVolume", ["adcode", "bodyType", "regMonth"], "train")
train_windows_fea["bodyType"] = train_windows_fea.model.apply(lambda x: model2type[x])
train_windows_fea = pd.merge(train_windows_fea, tmp_df, on=["adcode", "bodyType"], how="left")

# 合并特征
train_data = pd.merge(train_basic_fea, train_windows_fea, on=["adcode", "model", "bodyType"], how="left")
vaild_data = get_data(train_sales, "test").groupby(["adcode", "model"])["salesVolume"].apply(lambda x: pd.DataFrame(np.array(x)).T).reset_index().drop("level_2", axis=1)

# 模型
le_model = preprocessing.LabelEncoder()
le_bodyType = preprocessing.LabelEncoder()

le_model.fit(train_data.model)
le_bodyType.fit(train_data.bodyType)

lgb_params = {
    "num_leaves":32,
    "reg_alpha":1,
    "reg_lambda":0.1,
    "objective":'mse',
    "max_depth": 4,
    "learning_rate":0.01,
    "min_child_samples":5,
    "random_state":random.randint(100, 10000),
    "n_estimators":5000,
    "subsample":0.8,
    "colsample_bytree":0.8
}

df_train_columns = [c for c in train_data.columns if c not in []]
cate_fea = ["adcode", "model", "bodyType"]
train_data.model = le_model.transform(train_data.model)
train_data.bodyType = le_bodyType.transform(train_data.bodyType)
print(train_data.shape)
print(len(df_train_columns))


y_score = []    # 交叉验证
cv_pred = []    # 各折的预测值
predictions = 0
feature_importance_df = pd.DataFrame()
skf = KFold(n_splits=5, random_state=random.randint(100, 10000), shuffle=True)


label_0 = vaild_data[0]
label_1 = vaild_data[1]
label_2 = vaild_data[2]
label_3 = vaild_data[3]


for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_data, label_0)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_data.iloc[trn_idx][df_train_columns], label=label_0.iloc[trn_idx])
    val_data = lgb.Dataset(train_data.iloc[val_idx][df_train_columns], label=label_0.iloc[val_idx])

    result_df = train_data.iloc[val_idx][["adcode", "model"]]

    gbm_1 = lgb.train(lgb_params,
                    trn_data,
                    # init_model=gbm,
                    num_boost_round=150000,
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=200,
                    verbose_eval=200,
                    categorical_feature=cate_fea)     # 训练

    trn_data = lgb.Dataset(train_data.iloc[trn_idx][df_train_columns], label=label_1.iloc[trn_idx])
    val_data = lgb.Dataset(train_data.iloc[val_idx][df_train_columns], label=label_1.iloc[val_idx])

    gbm_2 = lgb.train(lgb_params,
                    trn_data,
                    # init_model=gbm,
                    num_boost_round=150000,
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=200,
                    verbose_eval=200,
                    categorical_feature=cate_fea)     # 训练

    trn_data = lgb.Dataset(train_data.iloc[trn_idx][df_train_columns], label=label_2.iloc[trn_idx])
    val_data = lgb.Dataset(train_data.iloc[val_idx][df_train_columns], label=label_2.iloc[val_idx])

    gbm_3 = lgb.train(lgb_params,
                    trn_data,
                    # init_model=gbm,
                    num_boost_round=150000,
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=200,
                    verbose_eval=200,
                    categorical_feature=cate_fea)     # 训练

    trn_data = lgb.Dataset(train_data.iloc[trn_idx][df_train_columns], label=label_3.iloc[trn_idx])
    val_data = lgb.Dataset(train_data.iloc[val_idx][df_train_columns], label=label_3.iloc[val_idx])

    gbm_4 = lgb.train(lgb_params,
                    trn_data,
                    # init_model=gbm,
                    num_boost_round=150000,
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=200,
                    verbose_eval=200,
                    categorical_feature=cate_fea)     # 训练

    result_df["y_pred_1"] = gbm_1.predict(train_data.iloc[val_idx][df_train_columns])
    result_df["y_pred_2"] = gbm_2.predict(train_data.iloc[val_idx][df_train_columns])
    result_df["y_pred_3"] = gbm_3.predict(train_data.iloc[val_idx][df_train_columns])
    result_df["y_pred_4"] = gbm_4.predict(train_data.iloc[val_idx][df_train_columns])
    # break

result_df["y_true_1"] = label_0
result_df["y_true_2"] = label_1
result_df["y_true_3"] = label_2
result_df["y_true_4"] = label_3

# 预测
# 基础特征
# 城市+车
test_basic_fea = cal_basic_fea(train_sales, "salesVolume", ["adcode", "model", "regMonth"], "test")
# 城市
tmp_df = cal_basic_fea(train_sales, "salesVolume", ["adcode", "regMonth"], "test")
test_basic_fea = pd.merge(test_basic_fea, tmp_df, on="adcode", how="left")
# 车
tmp_df = cal_basic_fea(train_sales, "salesVolume", ["model", "regMonth"], "test")
test_basic_fea = pd.merge(test_basic_fea, tmp_df, on="model", how="left")
# 城市+车型
tmp_df = cal_basic_fea(train_sales, "salesVolume", ["adcode", "bodyType", "regMonth"], "test")
test_basic_fea["bodyType"] = test_basic_fea.model.apply(lambda x: model2type[x])
test_basic_fea = pd.merge(test_basic_fea, tmp_df, on=["adcode", "bodyType"], how="left")

# 滑窗特征
# 城市+车
test_windows_fea = cal_windows_fea(train_sales, cal_col="salesVolume", stat_dim=["adcode", "model", "regMonth"], data_type="test")
# 城市
tmp_df = cal_windows_fea(train_sales, "salesVolume", ["adcode", "regMonth"], "test")
test_windows_fea = pd.merge(test_windows_fea, tmp_df, on="adcode", how="left")
# 车
tmp_df = cal_windows_fea(train_sales, "salesVolume", ["model", "regMonth"], "test")
test_windows_fea = pd.merge(test_windows_fea, tmp_df, on="model", how="left")
# 城市+车型
tmp_df = cal_windows_fea(train_sales, "salesVolume", ["adcode", "bodyType", "regMonth"], "test")
test_windows_fea["bodyType"] = test_windows_fea.model.apply(lambda x: model2type[x])
test_windows_fea = pd.merge(test_windows_fea, tmp_df, on=["adcode", "bodyType"], how="left")

# 合并
test_data = pd.merge(test_basic_fea, test_windows_fea, on=["adcode", "model", "bodyType"], how="left")

test_data.model = le_model.transform(test_data.model)
test_data.bodyType = le_bodyType.transform(test_data.bodyType)

y_pred_1 = gbm_1.predict(test_data[df_train_columns])
y_pred_2 = gbm_2.predict(test_data[df_train_columns])
y_pred_3 = gbm_3.predict(test_data[df_train_columns])
y_pred_4 = gbm_4.predict(test_data[df_train_columns])

y_pred_1 = (np.e ** y_pred_1 - 1).astype(int)
y_pred_2 = (np.e ** y_pred_2 - 1).astype(int)
y_pred_3 = (np.e ** y_pred_3 - 1).astype(int)
y_pred_4 = (np.e ** y_pred_4 - 1).astype(int)

result_df = test_basic_fea[["adcode", "model"]]
result_df["y_pred_1"] = y_pred_1
result_df["y_pred_2"] = y_pred_2
result_df["y_pred_3"] = y_pred_3
result_df["y_pred_4"] = y_pred_4

test_data = pd.read_csv(TEST_PATH)
test_data = test_data.drop("forecastVolum", axis=1)
test_data_1 = pd.merge(test_data.loc[test_data.regMonth == 1], result_df[["adcode", "model", "y_pred_1"]].rename(columns={"y_pred_1":"forecastVolum"}),\
                       how="left", on=["adcode", "model"])
test_data_2 = pd.merge(test_data.loc[test_data.regMonth == 2], result_df[["adcode", "model", "y_pred_2"]].rename(columns={"y_pred_2":"forecastVolum"}),\
                       how="left", on=["adcode", "model"])
test_data_3 = pd.merge(test_data.loc[test_data.regMonth == 3], result_df[["adcode", "model", "y_pred_3"]].rename(columns={"y_pred_3":"forecastVolum"}),\
                       how="left", on=["adcode", "model"])
test_data_4 = pd.merge(test_data.loc[test_data.regMonth == 4], result_df[["adcode", "model", "y_pred_4"]].rename(columns={"y_pred_4":"forecastVolum"}),\
                       how="left", on=["adcode", "model"])
result = pd.concat([test_data_1, test_data_2, test_data_3, test_data_4]).reset_index(drop=True)
result.forecastVolum = result.forecastVolum.astype(int)
result.loc[(result.forecastVolum < 0), "forecastVolum"] = 1
print((result.forecastVolum < 0 ).sum())
result[["id", "forecastVolum"]].to_csv("../submit/evaluation_public_20190916_lgb.csv", index=False)