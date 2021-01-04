import numpy as np
import pandas as pd
import os 
import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
import datetime
import time
import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

path  = './ccf_car/'

train_sales  = pd.read_csv(path+'train_sales_data.csv')
train_search = pd.read_csv(path+'train_search_data.csv')
train_user   = pd.read_csv(path+'train_user_reply_data.csv')

evaluation_public = pd.read_csv(path+'evaluation_public.csv')
submit_example    = pd.read_csv(path+'submit_example.csv')


##### 1月
m1_12    = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==12), 'salesVolume'].values
m1_11    = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==11), 'salesVolume'].values
m1_10    = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==10), 'salesVolume'].values
m1_09    = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==9) , 'salesVolume'].values
m1_08    = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==8) , 'salesVolume'].values
m1_07    = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==7) , 'salesVolume'].values
m1_06    = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==6) , 'salesVolume'].values

m1_12_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==12), 'salesVolume'].values * m1_12
m1_11_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==11), 'salesVolume'].values * m1_11
m1_10_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==10), 'salesVolume'].values * m1_10
m1_09_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==9) , 'salesVolume'].values * m1_09
m1_08_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==8) , 'salesVolume'].values * m1_08
m1_07_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==7) , 'salesVolume'].values * m1_07
m1_06_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==6) , 'salesVolume'].values * m1_06

evaluation_public.loc[evaluation_public.regMonth==1, 'forecastVolum'] =  m1_12_volum/2 + m1_11_volum/4 + m1_10_volum/8 + \
                                                      m1_09_volum/16 + m1_08_volum/32 + m1_07_volum/64 + m1_06_volum/64

##### 2月
m2_01 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1 ) , 'salesVolume'].values
m2_12 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==12) , 'salesVolume'].values
m2_11 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==11) , 'salesVolume'].values
m2_10 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==10) , 'salesVolume'].values
m2_09 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==9 ) , 'salesVolume'].values
m2_08 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==8 ) , 'salesVolume'].values
m2_07 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==7 ) , 'salesVolume'].values

m2_01_volum = evaluation_public.loc[(evaluation_public.regYear==2018)&(evaluation_public.regMonth==1 ), 'forecastVolum'].values * m2_01
m2_12_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==12), 'salesVolume'].values * m2_12
m2_11_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==11), 'salesVolume'].values * m2_11
m2_10_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==10), 'salesVolume'].values * m2_10
m2_09_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==9) , 'salesVolume'].values * m2_09
m2_08_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==8) , 'salesVolume'].values * m2_08
m2_07_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==7) , 'salesVolume'].values * m2_07

evaluation_public.loc[evaluation_public.regMonth==2, 'forecastVolum'] =  m2_01_volum/2 + m2_12_volum/4 + m2_11_volum/8 + \
                                                      m2_10_volum/16 + m2_09_volum/32 + m2_08_volum/64 + m2_07_volum/64

##### 3月
m3_02 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2 ) , 'salesVolume'].values
m3_01 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1 ) , 'salesVolume'].values
m3_12 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==12) , 'salesVolume'].values
m3_11 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==11) , 'salesVolume'].values
m3_10 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==10) , 'salesVolume'].values
m3_09 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==9 ) , 'salesVolume'].values
m3_08 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==8 ) , 'salesVolume'].values

m3_02_volum = evaluation_public.loc[(evaluation_public.regYear==2018)&(evaluation_public.regMonth==2 ), 'forecastVolum'].values * m3_02
m3_01_volum = evaluation_public.loc[(evaluation_public.regYear==2018)&(evaluation_public.regMonth==1 ), 'forecastVolum'].values * m3_01
m3_12_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==12), 'salesVolume'].values * m3_12
m3_11_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==11), 'salesVolume'].values * m3_11
m3_10_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==10), 'salesVolume'].values * m3_10
m3_09_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==9) , 'salesVolume'].values * m3_09
m3_08_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==8) , 'salesVolume'].values * m3_08

evaluation_public.loc[evaluation_public.regMonth==3, 'forecastVolum'] =  m3_02_volum/2 + m3_01_volum/4 + m3_12_volum/8 + \
                                                      m3_11_volum/16 + m3_10_volum/32 + m3_09_volum/64 + m3_08_volum/64

##### 4月
m4_03 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==4) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==3 ) , 'salesVolume'].values
m4_02 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==4) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==2 ) , 'salesVolume'].values
m4_01 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==4) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==1 ) , 'salesVolume'].values
m4_12 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==4) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==12) , 'salesVolume'].values
m4_11 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==4) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==11) , 'salesVolume'].values
m4_10 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==4) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==10) , 'salesVolume'].values
m4_09 = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==4) , 'salesVolume'].values / train_sales.loc[(train_sales.regYear==2016)&(train_sales.regMonth==9 ) , 'salesVolume'].values

m4_03_volum = evaluation_public.loc[(evaluation_public.regYear==2018)&(evaluation_public.regMonth==3 ), 'forecastVolum'].values * m4_03
m4_02_volum = evaluation_public.loc[(evaluation_public.regYear==2018)&(evaluation_public.regMonth==2 ), 'forecastVolum'].values * m4_02
m4_01_volum = evaluation_public.loc[(evaluation_public.regYear==2018)&(evaluation_public.regMonth==1 ), 'forecastVolum'].values * m4_01
m4_12_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==12), 'salesVolume'].values * m4_12
m4_11_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==11), 'salesVolume'].values * m4_11
m4_10_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==10), 'salesVolume'].values * m4_10
m4_09_volum = train_sales.loc[(train_sales.regYear==2017)&(train_sales.regMonth==9 ), 'salesVolume'].values * m4_09

evaluation_public.loc[evaluation_public.regMonth==4, 'forecastVolum'] =  m4_03_volum/2 + m4_02_volum/4 + m4_01_volum/8 + \
                                                      m4_12_volum/16 + m4_11_volum/32 + m3_10_volum/64 + m4_09_volum/64

evaluation_public[['id','forecastVolum']].round().astype(int).to_csv('ccf_car_sales.csv', index=False)