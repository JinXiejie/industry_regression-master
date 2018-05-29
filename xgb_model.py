import pandas as pd
from sklearn import preprocessing
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error

# 800 rows x 5954 columns
train_data = pd.read_csv('Code/MetaData/train.csv')

# 300 rows x 5953 columns
test_data = pd.read_csv('Code/MetaData/test_A.csv')

# 800 rows x 5727 columns
train_data = train_data.dropna(how='all', axis=1)
# 300 rows x 5953 columns
test_data = test_data.dropna(how='all', axis=1)

train_data = train_data.fillna(train_data.median())
test_data = test_data.fillna(test_data.median())

date_tool_column = []
for index, row in train_data.iterrows():
    for column in train_data.columns:
        if row[column] >= 20170000000000:
            date_tool_column.append(column)

date_tool_feature = [x for x in train_data.columns if x not in date_tool_column]
# 800 rows x 5638 columns
train_date_null = train_data[date_tool_feature]

zero_var = []
var_val = pd.DataFrame(train_date_null.var())
for index, row in var_val.iterrows():
    if row[0] == 0:
        zero_var.append(index)

# 800 rows x 4963 columns
zero_var_feature = [x for x in train_date_null if x not in zero_var]
train_zero_var = train_date_null[zero_var_feature]

# mean_val = train_zero_var.mean()
# train_zero_mean = train_zero_var - mean_val
# train_data_normal = train_zero_mean / (train_zero_var.max() - train_zero_var.min())

# min_max_scaler = preprocessing.MinMaxScaler()
# train_data_scaled = pd.DataFrame(min_max_scaler.fit_transform(train_zero_var))

train_zero_var['Value'] = train_zero_var['Value'].apply(lambda x: math.log(x))
params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 7,
          'lambda': 100,
          'alpha': 10,
          'subsample': 0.75,
          'colsample_bytree': 0.75,
          'eta': 0.01,  # 0.002
          'seed': 1024,
          'silent': 1,
          }
feature = [x for x in train_zero_var.columns if x not in ['Value']]
# X_train, X_test, y_train, y_test = train_test_split(train_zero_var[feature], train_zero_var['Value'], test_size=0.2,
#                                                     random_state=0)
# xgbtrain = xgb.DMatrix(X_train, y_train)
# xgbeval = xgb.DMatrix(X_test, y_test)
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
# model = xgb.train(params, xgbtrain, num_boost_round=4000, early_stopping_rounds=50, evals=watchlist)
xgbtrain = xgb.DMatrix(train_zero_var[feature], train_zero_var['Value'])
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
model = xgb.train(params, xgbtrain, num_boost_round=1236, early_stopping_rounds=50, evals=watchlist)

# dump model
model.dump_model('dump.raw.txt')

train_test_columns = []
for column in train_zero_var[feature].columns:
    train_test_columns.append(column)
test_feature = [x for x in test_data if x in train_test_columns]
xgbtest = xgb.DMatrix(test_data[test_feature])
y_pred = pd.DataFrame(test_data['ID'])
y_pred['pred'] = model.predict(xgbtest)
y_pred['pred'] = y_pred['pred'].apply(lambda x: math.exp(x))
y_pred.to_csv('Code/result.csv', index=None, header=None)

