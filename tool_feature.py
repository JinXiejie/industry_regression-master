import pandas as pd
from sklearn import preprocessing
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
# from mlxtend.regressor import StackingRegressor

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
test_id = pd.DataFrame(test_data['ID'])
# 220x150 220x151
date_tool_column = ['220x150', '220x151']
for index, row in train_data.iterrows():
    for column in train_data.columns:
        if row[column] >= 20170000000000:
            date_tool_column.append(column)
    break
# 91


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

date_tool_feature = [x for x in train_data.columns if x in date_tool_column]
train_tool = train_data[date_tool_feature]
train_tool = train_tool[train_tool.columns.drop(list(train_tool.filter(regex='X')))]

tool_id = []
for column in train_tool.columns:
    tool_id.append(column)
tool_id.append('Value')

date_column = []
for x in date_tool_column:
    if x not in tool_id:
        date_column.append(x)

date_feature = [x for x in train_data if x not in date_column]
train_date_null = train_data[date_feature]

zero_var = []
var_val = pd.DataFrame(train_date_null.var())
for index, row in var_val.iterrows():
    if row[0] == 0:
        zero_var.append(index)

# 800 rows x 4963 columns
zero_var_feature = [x for x in train_date_null if x not in zero_var]
train_zero_var = train_date_null[zero_var_feature]

feature = [x for x in train_zero_var.columns if x not in ['Value']]
# dart_pred = pd.DataFrame(test_data['ID'])

train_test_columns = []
for column in train_zero_var[feature].columns:
    train_test_columns.append(column)
test_feature = [x for x in test_data if x in train_test_columns]
test_data = test_data[test_feature]

params = {'booster': 'dart',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 8,
          'lambda': 100,
          'subsample': 0.75,
          'colsample_bytree': 0.65,
          'eta': 0.02,
          'sample_type': 'uniform',
          'normalize': 'tree',
          'rate_drop': 0.15,  # 0.1
          'skip_drop': 0.85,  # 0.9
          'seed': 1024,
          'silent': 1,
          }

next_column = 2
data_column = []
df = pd.DataFrame(train_data['Value']).reset_index()
df_test = pd.DataFrame(test_id).reset_index()
for column_step in range(2, len(tool_id) - 1):
    next_column = 1
    data_column = []
    next_column += column_step
    for column in train_zero_var.columns:
        if column != tool_id[next_column]:
            data_column.append(column)
        else:
            train_feature1 = [x for x in train_zero_var.columns if x in data_column]
            train = train_zero_var[train_feature1]

            test_feature1 = [x for x in test_data.columns if x in data_column]
            test = test_data[test_feature1]

            train_feature2 = [x for x in train.columns if x not in tool_id]
            train = train[train_feature2]

            test_feature2 = [x for x in test.columns if x not in tool_id]
            test = test_data[test_feature2]

            train = train.reset_index()
            train = pd.concat((train, train_zero_var['Value']), axis=1)

            # print "generate the new train data"
            # dart_pred = pd.DataFrame()
            # for idx in range(0, 5):
            #     train_fold = train[train['index'] % 5 != idx]
            #     test_fold = train[train['index'] % 5 == idx]
            #
            #     stacking_feature = [x for x in train_fold.columns if x not in ['index', 'Value']]
            #     xgbtrain = xgb.DMatrix(train_fold[stacking_feature], train_fold['Value'])
            #     watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
            #     dart_model = xgb.train(params, xgbtrain, num_boost_round=1300, early_stopping_rounds=50,
            #                            evals=watchlist)
            #     y_pred = pd.DataFrame(test_fold['index'])
            #     y_pred['xgb_dart'] = dart_model.predict(xgb.DMatrix(test_fold[stacking_feature]))
            #     dart_pred = dart_pred.append(y_pred)
            # df = pd.merge(df, dart_pred, how='left', on='index')

            # test_data predict for Dart
            print "generate the new test data"
            dart_feature = [x for x in train.columns if x not in ['index', 'Value']]
            xgbtrain = xgb.DMatrix(train[dart_feature], train['Value'])
            watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
            dart_reg = xgb.train(params, xgbtrain, num_boost_round=1300, early_stopping_rounds=50, evals=watchlist)
            dart_test_pred = pd.DataFrame(df_test['index'])
            xgbtest = xgb.DMatrix(test)
            dart_test_pred['xgb_dart'] = dart_reg.predict(xgbtest)
            df_test = pd.merge(df_test, dart_test_pred, how='left', on='index')

            train = None
            # next_column += 1
            data_column = []
        if next_column >= len(tool_id):
            break
# 1328:0.179215, 772:0.165419, 1266:0.157488, 1361:0.144816, 798:0.181533
# 1185:0.1727, 779:0.165752, 1001:0.166599, 1025:0.160994, 689:0.170972

df.to_csv('Code/new_train_data_split_all.csv', index=None)
# df_test.to_csv('Code/new_test_data_split_all.csv', index=None)

new_train_data_split = pd.read_csv('Code/new_train_data_split.csv')
new_train_data_split_all = pd.read_csv('Code/new_train_data_split_all.csv')
new_train_data_split_all = new_train_data_split_all.drop(['Value'], axis=1)
new_test_data_split = pd.read_csv('Code/new_test_data_split.csv')
# new_test_data_split_all = pd.read_csv('Code/new_test_data_split_all.csv')

new_train = pd.merge(new_train_data_split, new_train_data_split_all, how='left', on='index')

new_train_feature = [x for x in new_train.columns if x not in ['index', 'Value']]
new_test_feature = [x for x in new_train.columns if x not in ['index', 'ID']]

X_train, X_test, y_train, y_test = train_test_split(new_train[new_train_feature],
                                                    new_train['Value'], test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
# 0.013906223536340767
# 0.011511905228264797 -- all columns

# from sklearn.linear_model import Ridge
# ridge_reg = Ridge(normalize=True)
# 0.01401852630765818

linear_reg.fit(X_train, y_train)
y_test = pd.DataFrame(y_test)
y_test.columns = ['Value']
y_test['pred'] = linear_reg.predict(X_test)
mean_squared_error(y_test['Value'], y_test['pred'])
