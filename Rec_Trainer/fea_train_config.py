#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:00:24 2021

@author: liang
"""
#############################
#task = 'regression'
#sparse_features = ["movie_id", "user_id",
#                   "gender", "age", "occupation", "zip"]
#target = ['rating']
from fea_config import user_fea_type,item_fea_type,interaction_fea_type


############################
task = 'regression' #regression
sparse_features = []
dense_features = []


for key, value in user_fea_type:
    if value == 'int' or 'id' in key:
        sparse_features.append(key)
    elif value == 'string':
        dense_features.append(key)


for key, value in item_fea_type:
    if value == 'int' or 'id' in key:
        sparse_features.append(key)
    elif value == 'string':
        dense_features.append(key)
        
for key, value in interaction_fea_type:
    if value == 'int' or 'id' in key:
        sparse_features.append(key)
    elif value == 'string':
        dense_features.append(key)
        
target = ['交互类型']
