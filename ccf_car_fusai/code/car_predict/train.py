import os
import xgboost as xgb
import numpy as np

from params import params
from dataset import dtrain_sales, dtest_sales, dtrain_search, dtest_search



dir_name = os.path.dirname(os.path.abspath(__file__))

# train model
xgb_model_sales = xgb.train(params, dtrain_sales, 
                      num_boost_round=70, evals=[(dtest_sales, 'Test_sales')],
                      early_stopping_rounds=10)

xgb_model_search = xgb.train(params, dtrain_search, num_boost_round=70,
                             evals=[(dtest_search, 'Test_search')],
                             early_stopping_rounds=10)

# save model
xgb_model_sales.save_model(os.path.join(dir_name, 'models/sales.xgb'))
xgb_model_search.save_model(os.path.join(dir_name, 'models/search.xgb'))

# cv
'''
cv_results = xgb.cv(params, dtrain_sales, num_boost_round=50, nfold=5,
                    seed=42, metrics={'rmse'}, early_stopping_rounds=10)

print(cv_results)
'''