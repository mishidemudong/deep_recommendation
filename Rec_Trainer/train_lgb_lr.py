#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:46:02 2021

@author: liang
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import time
import pickle, json

fea_category = {
    "Currency":["DAY_PRICE_CHANGE_RATE","DAY_PRICE_CHANGE_RATE_RANK",
                "WEEK_PRICE_CHANGE_RATE",
                "WEEK_PRICE_CHANGE_RATE_RANK",
                "DAY_VOLUME_CHANGE_RATE",
                "DAY_VOLUME_CHANGE_RATE_RANK",
                "WEEK_VOLUME_CHANGE_RATE",
                "WEEK_VOLUME_CHANGE_RATE_RANK",
                "WEEK_CMC_SEARCH_HOT_RANK",
                "CURRENCY_LIKES",
                "CURRENCY_LIKES_RANKING",
                "CURRENCY_LIKES_RANKING_RISES_24_HOUR",
                "CURRENCY_LIKES_RANKING_RISES_24_HOUR_RANK",
                "CURRENCY_CAP",
                "CURRENCY_CAP_RANK",
                "CURRENCY_CATEGORY"],
    "InvestPortfolio":["INVEST_PROFIT_RATE_24_HOUR",
                        "INVEST_PROFIT_RATE_24_HOUR_RANK",
                        "INVEST_PROFIT_RATE_1_WEEK",
                        "INVEST_PROFIT_RATE_1_WEEK_RANK",
                        "INVEST_PROFIT_RATE_12_MONTH",
                        "INVEST_PROFIT_RATE_12_MONTH_RANK",
                        "AVERAGE_COLLECT",
                        "AVERAGE_COLLECT_RANK",
                        "INVEST_PROFIT_USED",
                        "INVEST_PROFIT_USED_RANK"],
    "Plate":[           "PLATE_CAP_CHANGE_RATE_1_DAY",
                        "PLATE_CAP_CHANGE_RATE_1_DAY_RANK",
                        "PLATE_CAP_CHANGE_RATE_1_WEEK",
                        "PLATE_CAP_CHANGE_RATE_1_WEEK_RANK",
                        "PLAT_COLLECT",
                        "PLAT_COLLECT_RANK"]
}


def time2stamp(tss1):
    #    print(tss1)
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))

    return timeStamp


modelpath = './WDLModel'
config_path = '{}/best_hyperparams.json'.format(modelpath)

config = json.load(open(config_path, 'r'))

fea_config = pickle.load(open('./featuremodel/fea_config.pkl', 'rb'))
target = [fea_config['map_eng_name'][fea_config['target'][0]]]
target_name = fea_config['target_name']
train_data_path = './data/train_eng_fea.csv'
data = pd.read_csv(train_data_path)  # .sample(frac=0.1, replace=False, random_state=5, axis=0)

data[target] = [target_name[item[0]][1] for item in list(data[target].values)]

fea_model = pickle.load(open(config['fea_model_savepath'], 'rb'))
data[fea_model['sparse_features']] = data[fea_model['sparse_features']].fillna('-1', )
data[fea_model['dense_features']] = data[fea_model['dense_features']].fillna(0, )

sparse_fea = {}
for feat in fea_model['sparse_features']:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
    lbe.classes_ = np.append(lbe.classes_, '<unknow>')
    sparse_fea[feat] = lbe

time_fea = ["INTERACTION_CREATE_TIME", "USER_CREATE_TIME", "ITEM_CREATE_TIME"]

for fea in time_fea:
    data[fea] = data[fea].map(lambda x: time2stamp(x))

mms = MinMaxScaler(feature_range=(0, 1))
data[fea_model['dense_features']] = mms.fit_transform(data[fea_model['dense_features']])
train, test = train_test_split(data, test_size=0.2, random_state=2020)

y_train = train[target]  # training label
y_test = test[target]  # testing label
X_train = train.drop(target, axis=1)  # training dataset
X_test = test.drop(target, axis=1)  # testing dataset

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

# specify your configurations as a dict
depth = 2
num_leaves = 2 ** depth - 1

params = {'boosting_type': 'dart',
          'objective': 'regression',
          'metric': 'l2',
          'num_leaves': num_leaves,
          'max_depth': -1,
          'learning_rate': 0.02,
          'n_estimators': 1000,
          'min_split_gain': 0.05,
          'min_child_weight': 0.5,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'reg_alpha': 0.2,
          'reg_lambda': 0.2,
          'drop_rate': 0.2,
          'skip_drop': 0.8,
          'max_drop': 200,
          'seed': 100,
          'silent': False
          }

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)


print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Calculate feature importances...')


importance = pd.DataFrame()
importance['Feature'] = X_train.columns.values
importance['ImportanceWeight'] = gbm.feature_importance(importance_type='split')
importance['ImportanceGain'] = gbm.feature_importance(importance_type='gain')

importance.sort_values(by='ImportanceWeight', ascending=False, inplace=True)
importance.head()


fea_importance = {'Currency':[],
                  'InvestPortfolio':[],
                  'Plate':[]
                  }

for index, item in importance.iterrows():
    if item['Feature'] in fea_category['Currency']:
        # fea_category['Currency'].append((item['Feature'], item['ImportanceGain']))
        fea_importance['Currency'].append(item['Feature'])

    if item['Feature'] in fea_category['InvestPortfolio']:
        # fea_category['Currency'].append((item['Feature'], item['ImportanceGain']))
        fea_importance['InvestPortfolio'].append(item['Feature'])

    if item['Feature'] in fea_category['Plate']:
        # fea_category['Currency'].append((item['Feature'], item['ImportanceGain']))
        fea_importance['Plate'].append(item['Feature'])


config_path = './recmodel/my_model_best_0.0025.json'
config = json.load(open(config_path,'r'))
config['fea_importance'] = fea_importance

with open(config_path,'w') as file_obj:
    json.dump(config,file_obj)
file_obj.close()

