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
from mlxtend.regressor import StackingRegressor


# 800 rows x 5954 columns
train_data = pd.read_csv('Code/MetaData/train.csv')

# 300 rows x 5953 columns
test_data = pd.read_csv('Code/MetaData/test_A.csv')
test_data_id = pd.DataFrame(pd.read_csv('Code/MetaData/test_A.csv')['ID'])

# 800 rows x 5727 columns
train_data = train_data.dropna(how='all', axis=1)
# 300 rows x 5953 columns
test_data = test_data.dropna(how='all', axis=1)

train_data = train_data.fillna(train_data.median())
test_data = test_data.fillna(test_data.median())
# 220x150 220x151
date_tool_column = ['220x150', '220x151']
for index, row in train_data.iterrows():
    for column in train_data.columns:
        if row[column] >= 20170000000000:
            date_tool_column.append(column)
    break

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
feature = [x for x in train_zero_var.columns if x not in ['Value']]

train_test_columns = []
for column in train_zero_var[feature].columns:
    train_test_columns.append(column)
test_feature = [x for x in test_data if x in train_test_columns]
test_data = test_data[test_feature].reset_index()

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
X_train, X_test, y_train, y_test = train_test_split(train_zero_var[feature], train_zero_var['Value'], test_size=0.2,
                                                    random_state=0)
xgbtrain = xgb.DMatrix(X_train[feature], y_train)
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
model = xgb.train(params, xgbtrain, num_boost_round=2889, early_stopping_rounds=50, evals=watchlist)

rfreg = RandomForestRegressor(random_state=1, max_depth=15)
ridge_reg = Ridge(normalize=True)
lasso_reg = Lasso()
linear_reg = LinearRegression(normalize=True)
stacking_reg = StackingRegressor(regressors=[rfreg, ridge_reg, lasso_reg], meta_regressor=linear_reg)

feature = [x for x in train_zero_var.columns if x not in ['Value']]
# X_train, X_test, y_train, y_test = train_test_split(train_zero_var[feature], train_zero_var['Value'], test_size=0.2,
#                                                     random_state=0)
stacking_reg.fit(X_train, y_train)
stacking_test = pd.DataFrame(stacking_reg.predict(X_test))
stacking_test.columns = ['stacking_pred']
y_test = pd.DataFrame(y_test)
y_test.columns = ['Value']
mean_squared_error(stacking_test['stacking_pred'], y_test['Value'])

train_zero_var = train_zero_var.reset_index()

# predict for Random Forest
rf_pred = pd.DataFrame()
for idx in range(0, 5):
    train = train_zero_var[train_zero_var['index'] % 5 != idx]
    test = train_zero_var[train_zero_var['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    rfreg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Random Forest'] = rfreg.predict(test[stacking_feature])
    rf_pred = rf_pred.append(y_pred)
# rf_pred = rf_pred.dropna()
df = pd.merge(train_zero_var, rf_pred, how='left', on='index')

# test_data predict for Random Forest
rf_feature = [x for x in train_zero_var.columns if x not in ['index', 'Value']]
rfreg.fit(train_zero_var[rf_feature], train_zero_var['Value'])
rf_test_pred = pd.DataFrame(test_data['index'])
rf_test_pred['Random Forest'] = rfreg.predict(test_data[rf_feature])
df_test = pd.merge(test_data, rf_test_pred, how='left', on='index')

# predict for GBDT
from sklearn.ensemble import GradientBoostingRegressor
gbdt_reg = GradientBoostingRegressor(max_depth=7)
gbdt_pred = pd.DataFrame()
for idx in range(0, 5):
    train = train_zero_var[train_zero_var['index'] % 5 != idx]
    test = train_zero_var[train_zero_var['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    gbdt_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['GBDT'] = gbdt_reg.predict(test[stacking_feature])
    gbdt_pred = gbdt_pred.append(y_pred)
# rf_pred = rf_pred.dropna()
df = pd.merge(df, gbdt_pred, how='left', on='index')

# predict for Lasso
lasso_pred = pd.DataFrame()
for idx in range(0, 5):
    train = train_zero_var[train_zero_var['index'] % 5 != idx]
    test = train_zero_var[train_zero_var['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    lasso_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Lasso'] = lasso_reg.predict(test[stacking_feature])
    lasso_pred = lasso_pred.append(y_pred)
# lasso_pred = lasso_pred.dropna()
df = pd.merge(df, lasso_pred, how='left', on='index')

# test_data predict for Lasso
lasso_feature = [x for x in train_zero_var.columns if x not in ['index', 'Value']]
lasso_reg.fit(train_zero_var[lasso_feature], train_zero_var['Value'])
lasso_test_pred = pd.DataFrame(test_data['index'])
lasso_test_pred['Lasso'] = lasso_reg.predict(test_data[lasso_feature])
df_test = pd.merge(df_test, lasso_test_pred, how='left', on='index')

# predict for Ridge
ridge_pred = pd.DataFrame()
for idx in range(0, 5):
    train = train_zero_var[train_zero_var['index'] % 5 != idx]
    test = train_zero_var[train_zero_var['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    ridge_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Ridge'] = ridge_reg.predict(test[stacking_feature])
    ridge_pred = ridge_pred.append(y_pred)
# lasso_pred = lasso_pred.dropna()
df = pd.merge(df, ridge_pred, how='left', on='index')


# predict for Dart
dart_pred = pd.DataFrame()
for idx in range(0, 5):
    train = train_zero_var[train_zero_var['index'] % 5 != idx]
    test = train_zero_var[train_zero_var['index'] % 5 == idx]

    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    xgbtrain = xgb.DMatrix(train[stacking_feature], train['Value'])
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
    dart_model = xgb.train(params, xgbtrain, num_boost_round=2889, early_stopping_rounds=50, evals=watchlist)
    y_pred = pd.DataFrame(test['index'])
    y_pred['xgb_dart'] = dart_model.predict(xgb.DMatrix(test[stacking_feature]))
    dart_pred = dart_pred.append(y_pred)
# dart_pred = dart_pred.dropna()
df = pd.merge(df, dart_pred, how='left', on='index')

# test_data predict for Dart
dart_feature = [x for x in train_zero_var.columns if x not in ['index', 'Value']]
xgbtrain = xgb.DMatrix(train_zero_var[dart_feature], train_zero_var['Value'])
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
dart_reg = xgb.train(params, xgbtrain, num_boost_round=2889, early_stopping_rounds=50, evals=watchlist)
dart_test_pred = pd.DataFrame(test_data['index'])
dart_test_pred['xgb_dart'] = dart_reg.predict(xgb.DMatrix(test_data[dart_feature]))
df_test = pd.merge(df_test, dart_test_pred, how='left', on='index')

new_train_data = df[['index', 'Random Forest', 'Lasso', 'xgb_dart', 'Ridge', 'Value']]
new_train_data.to_csv('Code/new_train_data.csv', index=None)

new_test_data = df_test[['index', 'Random Forest', 'Lasso', 'xgb_dart']]
new_test_data.to_csv('Code/new_test_data.csv', index=None)

new_params = {'booster': 'dart',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'max_depth': 8,
              'lambda': 10,
              'subsample': 0.75,
              'eta': 0.02,
              'sample_type': 'uniform',
              'normalize': 'tree',
              'rate_drop': 0.1,  # 0.1
              'skip_drop': 0.9,  # 0.9
              'seed': 1024,
              'silent': 1,
              }

new_train_data = pd.read_csv('Code/new_train_data.csv')
new_test_data = pd.read_csv('Code/new_test_data.csv')
new_feature = [x for x in new_train_data.columns if x not in ['index', 'Value']]
X_train, X_test, y_train, y_test = train_test_split(new_train_data[new_feature], new_train_data['Value'], test_size=0.2,
                                                    random_state=0)
xgbtrain = xgb.DMatrix(X_train, y_train)
xgbeval = xgb.DMatrix(X_test, y_test)
watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
new_dart_model = xgb.train(new_params, xgbtrain, num_boost_round=4000, early_stopping_rounds=50, evals=watchlist)
# 0.011500632081


# xgbtrain = xgb.DMatrix(new_train_data[new_feature], new_train_data['Value'])
# watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
# new_dart_model = xgb.train(params, xgbtrain, num_boost_round=1023, early_stopping_rounds=50, evals=watchlist)
# stacking_result = pd.DataFrame(test_data['ID'])
# stacking_result['stacking_result'] = new_dart_model.predict(xgb.DMatrix(new_test_data[new_feature]))
# stacking_result.to_csv('Code/stacking_result.csv', index=None, header=None)


X_train, X_test, y_train, y_test = train_test_split(new_train_data[new_feature], new_train_data['Value'], test_size=0.2,
                                                    random_state=0)
linear_reg.fit(X_train[new_feature], y_train)
y_test = pd.DataFrame(y_test)
y_test.columns = ['Value']
y_test['pred'] = linear_reg.predict(X_test[new_feature])
mean_squared_error(y_test['Value'], y_test['pred'])
# 0.010146288510991228

# generate the .csv result of stacking_linear
linear_reg.fit(new_train_data[new_feature], new_train_data['Value'])
stacking_linear = pd.DataFrame(test_data_id)
stacking_linear['pred'] = linear_reg.predict(new_test_data[new_feature])
stacking_linear.to_csv('Code/stacking_linear.csv', index=None, header=None)

df2 = pd.DataFrame(new_train_data['index'])
# predict for Lasso-2
lasso_reg2 = Lasso(normalize=True, random_state=0)
lasso_pred2 = pd.DataFrame()
for idx in range(0, 5):
    train = new_train_data[new_train_data['index'] % 5 != idx]
    test = new_train_data[new_train_data['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    lasso_reg2.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Lasso'] = lasso_reg2.predict(test[stacking_feature])
    lasso_pred2 = lasso_pred2.append(y_pred)
# lasso_pred = lasso_pred.dropna()
df2 = pd.merge(df2, lasso_pred2, how='left', on='index')


# predict for Ridge2
ridge_reg = Ridge(normalize=True)
ridge_pred2 = pd.DataFrame()
for idx in range(0, 5):
    train = new_train_data[new_train_data['index'] % 5 != idx]
    test = new_train_data[new_train_data['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    ridge_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Ridge'] = ridge_reg.predict(test[stacking_feature])
    ridge_pred2 = ridge_pred2.append(y_pred)
# lasso_pred = lasso_pred.dropna()
df2 = pd.merge(df2, ridge_pred2, how='left', on='index')

new_feature = [x for x in df2.columns if x not in ['index']]
X_train, X_test, y_train, y_test = train_test_split(df2[new_feature], new_train_data['Value'], test_size=0.2,
                                                    random_state=0)
linear_reg.fit(X_train[new_feature], y_train)
y_test = pd.DataFrame(y_test)
y_test.columns = ['Value']
y_test['pred'] = linear_reg.predict(X_test[new_feature])
mean_squared_error(y_test['Value'], y_test['pred'])

# predict for Dart2
dart_pred2 = pd.DataFrame()
for idx in range(0, 5):
    train = new_train_data[new_train_data['index'] % 5 != idx]
    test = new_train_data[new_train_data['index'] % 5 == idx]

    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    xgbtrain = xgb.DMatrix(train[stacking_feature], train['Value'])
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
    dart_model = xgb.train(params, xgbtrain, num_boost_round=1100, early_stopping_rounds=50, evals=watchlist)
    y_pred = pd.DataFrame(test['index'])
    y_pred['xgb_dart'] = dart_model.predict(xgb.DMatrix(test[stacking_feature]))
    dart_pred2 = dart_pred2.append(y_pred)
# dart_pred = dart_pred.dropna()
df2 = pd.merge(df2, dart_pred2, how='left', on='index')

new_train_data_sample = pd.DataFrame()
new_train_data_sample['weight'] = abs(new_train_data['xgb_dart'] - new_train_data['Value'])
new_train_data_sample = new_train_data_sample['weight'].apply(lambda x: int(20 * x)).reset_index()
new_train_data_sample = pd.merge(new_train_data_sample, train_zero_var, how='left', on='index')

new_train_data_sample = new_train_data_sample.drop(702)

data_plus = pd.DataFrame()
for index, row in new_train_data_sample.iterrows():
    idx = int(index)
    print idx
    for weight in range(0, int(row['weight'])):
        data_plus = pd.concat([data_plus, new_train_data_sample[idx: idx + 1]], axis=0)


df_new = data_plus.drop(['weight'], axis=1)
df_new = pd.concat((train_zero_var, df_new))
df_new = df_new.drop(['index'], axis=1)

df_new = df_new.reset_index().drop(['index'], axis=1)

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
          'rate_drop': 0.25,  # 0.1
          'skip_drop': 0.75,  # 0.9
          'seed': 1024,
          'silent': 1,
          }
new_feature = [x for x in train_zero_var.columns if x not in ['index', 'Value']]
X_train, X_test, y_train, y_test = train_test_split(train_zero_var[new_feature], train_zero_var['Value'], test_size=0.2,
                                                    random_state=0)
xgbtrain = xgb.DMatrix(X_train, y_train)
xgbeval = xgb.DMatrix(X_test, y_test)
watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
new_model = xgb.train(params, xgbtrain, num_boost_round=7000, early_stopping_rounds=50, evals=watchlist)


new_params = {'booster': 'dart',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'max_depth': 8,
              'lambda': 100,
              'subsample': 0.65,
              'eta': 0.02,
              'sample_type': 'uniform',
              'normalize': 'tree',
              'rate_drop': 0.15,  # 0.1
              'skip_drop': 0.85,  # 0.9
              'seed': 1024,
              'silent': 1,
              }
new_feature = [x for x in df_new.columns if x not in ['Value']]
X_train, X_test, y_train, y_test = train_test_split(df_new[new_feature], df_new['Value'], test_size=0.2,
                                                    random_state=0)
xgbtrain = xgb.DMatrix(X_train, y_train)
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
df_new_model = xgb.train(new_params, xgbtrain, num_boost_round=700, early_stopping_rounds=50, evals=watchlist)
y_test = pd.DataFrame(y_test)
y_test.columns = ['Value']
y_test['pred'] = df_new_model.predict(xgb.DMatrix(X_test))
mean_squared_error(y_test['Value'], y_test['pred'])
pri_train = pd.read_csv('Code/train1.csv')
pri_test = pd.read_csv('Code/testA_answer.csv')
pri_train = pd.concat((pri_train, pri_test), axis=0)
final = pd.merge(train_zero_var, pri_train, how='inner')


answer_a = pd.read_csv('Code/[new] fusai_answer_a_20180127.csv', header=None)
answer_a.columns = ['ID', 'Value']
mean_squared_error(answer_a['Value'], stacking_linear['pred'])

test_data_b = pd.read_csv('Code/MetaData/test_B.csv')


