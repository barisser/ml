import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb

finaltest = pd.read_csv('test.csv')
data = pd.read_csv('train.csv')

train = data.iloc[:1200]
test = data.iloc[1200:]


Ytrain = np.log(train['SalePrice'])
Ytest = np.log(test['SalePrice'])

class Model(object):
	def __init__(self):
		pass

	def fit(self, X, Y):
		self.linear = LinearRegression()
		self.linear.fit(X, Y)

	def predict(self, X):



model = xgb.XGBRegressor(max_depth=3, n_estimators=10, max_leaves=0)

feature_cols = ['LotFrontage', 'LotArea']
Xtrain = train[feature_cols]
Xtest = test[feature_cols]

def rmse(pred, Y):
	answer = (pred - Y)**2
	return answer.mean()

model.fit(Xtrain, Ytrain)
print("Benchmark: {}".format(rmse(Ytrain.mean(), Ytest)))
print(rmse(model.predict(Xtest), Ytest))

import pdb;pdb.set_trace()