import numpy as np
import pandas as pd
import xgboost as xgb

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

Y = np.log(train['SalePrice'])

model = xgb.XGBRegressor(max_depth=3, n_estimators=10, max_leaves=0)

import pdb;pdb.set_trace()