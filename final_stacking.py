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


def calculate_mse(temp, y1, y2):
    real_temp = temp[y1] - temp[y2]
    realrealtemp = real_temp * real_temp
    mse = realrealtemp.sum() / len(temp)
    print(mse)
    return mse

final_train = pd.read_csv('Code/final_train.csv')
date_tool_column = []
for index, row in final_train.iterrows():
    for column in final_train.columns:
        if row[column] >= 20170000000000:
            date_tool_column.append(column)
    break

date_tool_feature = [x for x in final_train if x not in date_tool_column]
final_train = final_train[date_tool_feature]
final_train.drop(['Unnamed: 0'], axis=1, inplace=True)
final_train = final_train.reset_index()
df = pd.DataFrame(final_train[['index', 'Value']])

final_test = pd.read_csv('Code/final_test.csv')
test_feature = [x for x in final_train.columns if x not in ['index', 'Value']]
final_test = final_test[test_feature].reset_index()
final_test = final_test.fillna(final_test.median())
df_test = pd.DataFrame(final_test['index'])

final_train.drop(['220X223', '220X224'], axis=1, inplace=True)
# a = set(final_train.columns)
# b = set(final_test.columns)
# a.difference(b)
# predict for Random Forest
rf_reg = RandomForestRegressor(random_state=1, max_depth=15)
rf_pred = pd.DataFrame()
for idx in range(0, 5):
    train = final_train[final_train['index'] % 5 != idx]
    test = final_train[final_train['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    rf_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Random Forest'] = rf_reg.predict(test[stacking_feature])
    rf_pred = rf_pred.append(y_pred)
# rf_pred = rf_pred.dropna()
df = pd.merge(df, rf_pred, how='left', on='index')

# final_test predict for Random Forest
rf_feature = [x for x in final_train.columns if x not in ['index', 'Value']]
rf_reg.fit(final_train[rf_feature], final_train['Value'])
rf_test_pred = pd.DataFrame(final_test['index'])
rf_test_pred['Random Forest'] = rf_reg.predict(final_test[rf_feature])
df_test = pd.merge(df_test, rf_test_pred, how='left', on='index')
# final_test = final_test.dropna(how='all', axis=1)

# predict for Lasso
lasso_reg = Lasso(alpha=1.3)
lasso_pred = pd.DataFrame()
for idx in range(0, 5):
    train = final_train[final_train['index'] % 5 != idx]
    test = final_train[final_train['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    lasso_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Lasso'] = lasso_reg.predict(test[stacking_feature])
    lasso_pred = lasso_pred.append(y_pred)
df = df.drop(['Lasso'], axis=1)
df = pd.merge(df, lasso_pred, how='left', on='index')

# test_data predict for Lasso
lasso_feature = [x for x in final_train.columns if x not in ['index', 'Value']]
lasso_reg.fit(final_train[lasso_feature], final_train['Value'])
lasso_test_pred = pd.DataFrame(final_test['index'])
lasso_test_pred['Lasso'] = lasso_reg.predict(final_test[lasso_feature])
df_test = pd.merge(df_test, lasso_test_pred, how='left', on='index')


# predict for Ridge
ridge_reg = Ridge(normalize=True, alpha=62)
ridge_pred = pd.DataFrame()
for idx in range(0, 5):
    train = final_train[final_train['index'] % 5 != idx]
    test = final_train[final_train['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    ridge_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['Ridge'] = ridge_reg.predict(test[stacking_feature])
    ridge_pred = ridge_pred.append(y_pred)
df = pd.merge(df, ridge_pred, how='left', on='index')

# final_test predict for Ridge
ridge_feature = [x for x in final_train.columns if x not in ['index', 'Value']]
ridge_reg.fit(final_train[ridge_feature], final_train['Value'])
ridge_test_pred = pd.DataFrame(final_test['index'])
ridge_test_pred['Ridge'] = ridge_reg.predict(final_test[ridge_feature])
df_test = pd.merge(df_test, ridge_test_pred, how='left', on='index')


# predict for Dart
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
dart_pred = pd.DataFrame()
for idx in range(0, 5):
    train = final_train[final_train['index'] % 5 != idx]
    test = final_train[final_train['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    xgbtrain = xgb.DMatrix(train[stacking_feature], train['Value'])
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
    dart_model = xgb.train(params, xgbtrain, num_boost_round=1420, early_stopping_rounds=50, evals=watchlist)
    y_pred = pd.DataFrame(test['index'])
    y_pred['xgb_dart'] = dart_model.predict(xgb.DMatrix(test[stacking_feature]))
    dart_pred = dart_pred.append(y_pred)
# dart_pred = dart_pred.dropna()
df = pd.merge(df, dart_pred, how='left', on='index')

# final_test predict for Dart
dart_feature = [x for x in final_train.columns if x not in ['index', 'Value']]
xgbtrain = xgb.DMatrix(final_train[dart_feature], final_train['Value'])
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
dart_reg = xgb.train(params, xgbtrain, num_boost_round=1420, early_stopping_rounds=50, evals=watchlist)
dart_test_pred = pd.DataFrame(final_test['index'])
dart_test_pred['xgb_dart'] = dart_reg.predict(xgb.DMatrix(final_test[dart_feature]))
df_test = pd.merge(df_test, dart_test_pred, how='left', on='index')

# predict for GBDT
from sklearn.ensemble import GradientBoostingRegressor
gbdt_reg = GradientBoostingRegressor(max_depth=7)
gbdt_pred = pd.DataFrame()
for idx in range(0, 5):
    train = final_train[final_train['index'] % 5 != idx]
    test = final_train[final_train['index'] % 5 == idx]
    stacking_feature = [x for x in train.columns if x not in ['index', 'Value']]
    gbdt_reg.fit(train[stacking_feature], train['Value'])
    y_pred = pd.DataFrame(test['index'])
    y_pred['GBDT'] = gbdt_reg.predict(test[stacking_feature])
    gbdt_pred = gbdt_pred.append(y_pred)
# rf_pred = rf_pred.dropna()
df = pd.merge(df, gbdt_pred, how='left', on='index')

# final_test predict for GBDT
gbdt_feature = [x for x in final_train.columns if x not in ['index', 'Value']]
gbdt_reg.fit(final_train[gbdt_feature], final_train['Value'])
gbdt_test_pred = pd.DataFrame(final_test['index'])
gbdt_test_pred['GBDT'] = gbdt_reg.predict(final_test[gbdt_feature])
df_test = pd.merge(df_test, gbdt_test_pred, how='left', on='index')

df_test.to_csv("Code/df_test.csv")
# linear model in stacking
df1 = pd.read_csv('Code/df1.csv')
linear_reg = LinearRegression(normalize=True)
new_feature = [x for x in df.columns if x not in ['index', 'Value', 'Lasso', 'Ridge']]
mse = 0.0
for idx in range(0, 5):
    train = df[final_train['index'] % 5 != idx]
    test = df[final_train['index'] % 5 == idx]
    linear_reg.fit(train[new_feature], train['Value'])
    y_test = pd.DataFrame(test['Value'])
    y_test.columns = ['Value']
    y_test['pred'] = linear_reg.predict(test[new_feature])
    mse += mean_squared_error(y_test['Value'], y_test['pred'])
print mse / 5.0
# 0.0190598141474
# 0.0191681274591
df = pd.read_csv('Code/df.csv')
test_data_b = pd.read_csv('Code/MetaData/test_B.csv')
linear_feature = [x for x in df.columns if x not in ['index', 'Value']]
linear_reg.fit(df[linear_feature], df['Value'])
stacking_linear = pd.DataFrame(test_data_b['ID'])
stacking_linear['pred'] = linear_reg.predict(df_test[linear_feature])
stacking_linear.to_csv('Code/stacking_linear.csv', index=None, header=None)
calculate_mse(df, 'Value', 'Random Forest')

dart_feature = [x for x in final_train.columns if x not in ['index', 'Value', 'Lasso', 'Ridge']]
X_train, X_test, y_train, y_test = train_test_split(final_train[dart_feature], final_train['Value'], test_size=0.2,
                                                    random_state=0)
xgbtrain = xgb.DMatrix(X_train, y_train)
xgbeval = xgb.DMatrix(X_test, y_test)
watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
new_dart_model = xgb.train(params, xgbtrain, num_boost_round=7000, early_stopping_rounds=50, evals=watchlist)
# 0.011500632081





