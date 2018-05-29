import pandas as pd
from sklearn import preprocessing
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
import operator
from matplotlib import pylab as plt

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

# train_zero_var['Value'] = train_zero_var['Value'].apply(lambda x: math.log(x))
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
feature = [x for x in train_zero_var.columns if x not in ['TOOL', 'Value']]
train = pd.concat((train_data['TOOL'], train_zero_var), axis=1)
train = train.drop(702)
tool_id = list(set(train_data['TOOL']))
mse = []
for tool in tool_id:
    train_iter = train[train['TOOL'] == tool]
    X_train, X_test, y_train, y_test = train_test_split(train_iter[feature], train_iter['Value'], test_size=0.2,
                                                        random_state=0)
    xgbtrain = xgb.DMatrix(X_train, y_train)
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
    model = xgb.train(params, xgbtrain, num_boost_round=2889, early_stopping_rounds=50, evals=watchlist)
    xgbtest = xgb.DMatrix(X_test)
    y_pred = pd.DataFrame()
    y_pred['pred'] = model.predict(xgbtest)
    y_test = pd.DataFrame(y_test)
    y_test.columns = ['Value']
    print mean_squared_error(y_test['Value'], y_pred['pred'])
    mse.append(mean_squared_error(y_test['Value'], y_pred['pred']))
    # 0.022157213724337512
    # 0.027298981771017208
    # 0.021027423632476189
    # 0.016701836076272902
    # 0.020138875325864287

# X_train, X_test, y_train, y_test = train_test_split(train[feature], train['Value'], test_size=0.2, random_state=0)
# xgbtrain = xgb.DMatrix(X_train, y_train)
# xgbeval = xgb.DMatrix(X_test, y_test)
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
# model = xgb.train(params, xgbtrain, num_boost_round=4000, early_stopping_rounds=50, evals=watchlist)

feature = [x for x in train_zero_var.columns if x not in ['Value']]
X_train, X_test, y_train, y_test = train_test_split(train_zero_var[feature], train_zero_var['Value'], test_size=0.2,
                                                    random_state=0)
xgbtrain = xgb.DMatrix(X_train[feature], y_train)
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
model = xgb.train(params, xgbtrain, num_boost_round=2889, early_stopping_rounds=50, evals=watchlist)

xgbtest = xgb.DMatrix(X_test)
y_pred = pd.DataFrame()
y_pred['pred'] = model.predict(xgbtest)
y_test = pd.DataFrame(y_test)
y_test.columns = ['Value']
mean_squared_error(y_test['Value'], y_pred['pred'])

train_test_columns = []
for column in train_zero_var[feature].columns:
    train_test_columns.append(column)
test_feature = [x for x in test_data if x in train_test_columns]
xgbtest = xgb.DMatrix(test_data[test_feature])
y_pred = pd.DataFrame(test_data['ID'])
y_pred['pred'] = model.predict(xgbtest)
# y_pred['pred'] = y_pred['pred'].apply(lambda x: math.exp(x))
y_pred.to_csv('Code/result.csv', index=None, header=None)


# plot the map of feature importance
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


ceate_feature_map(feature)
# importance = model.get_fscore(fmap='xgb.fmap')
importance = model.get_score(importance_type='gain')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'score'])
df.to_csv('Code/feature_importance_dart.csv', index=None)
df['score'] = df['score'] / df['score'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(16, 30))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('Code/feature_importance_dart.png')

train_feature_score = pd.read_csv('Code/feature_importance_dart.csv')
train_feature_score = train_feature_score[train_feature_score['score'] >= 0.01500]
# feature_select = train_feature_score['feature']
feature_select = [x for x in train.columns if x in list(set(train_feature_score['feature']))]
train_feature_select = train[feature_select]
train_feature_select = pd.concat((train[['TOOL', 'Value']], train_feature_select), axis=1)

tool_id = list(set(train_feature_select['TOOL']))
mse = []
feature1 = [x for x in train_feature_select.columns if x not in ['TOOL', 'Value']]
for tool in tool_id:
    train_iter = train_feature_select[train_feature_select['TOOL'] == tool]
    X_train, X_test, y_train, y_test = train_test_split(train_iter[feature1], train_iter['Value'], test_size=0.2,
                                                        random_state=0)
    xgbtrain = xgb.DMatrix(X_train, y_train)
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
    model = xgb.train(params, xgbtrain, num_boost_round=2889, early_stopping_rounds=50, evals=watchlist)
    xgbtest = xgb.DMatrix(X_test)
    y_pred = pd.DataFrame()
    y_pred['pred'] = model.predict(xgbtest)
    y_test = pd.DataFrame(y_test)
    y_test.columns = ['Value']
    print mean_squared_error(y_test['Value'], y_pred['pred'])
    mse.append(mean_squared_error(y_test['Value'], y_pred['pred']))
for tool in tool_id:
    train_iter = train_feature_select[train_feature_select['TOOL'] == tool]
    X_train, X_test, y_train, y_test = train_test_split(train_iter[feature1], train_iter['Value'], test_size=0.2,
                                                        random_state=0)
    xgbtrain = xgb.DMatrix(X_train, y_train)
    xgbeval = xgb.DMatrix(X_test, y_test)
    watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
    model = xgb.train(params, xgbtrain, num_boost_round=4000, early_stopping_rounds=50, evals=watchlist)
