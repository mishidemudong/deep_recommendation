#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:28:29 2021

@author: liang
"""

import pandas as pd
import pickle
import time

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 


item_fea_df = pd.read_csv('./feamap/item_map.csv')
item_fea_map = {}
for key, value, value2 in zip(item_fea_df['label_name'], item_fea_df['label_code'], item_fea_df['data_type']):
    item_fea_map[key] = (value.upper(), value2)


user_fea_df = pd.read_csv('./feamap/user_map.csv')
user_fea_map = {}
for key, value, value2 in zip(user_fea_df['label_name'], user_fea_df['label_code'], user_fea_df['data_type']):
    user_fea_map[key] = (value.upper(), value2)


interaction_fea_df = pd.read_csv('./feamap/interaction_map.csv')
interaction_fea_map = {}
for key, value, value2 in zip(interaction_fea_df['label_name'], interaction_fea_df['label_code'], interaction_fea_df['data_type']):
    interaction_fea_map[key] = (value.upper(), value2)
    
target_name = {
            'ITEM_SHOW':('SHOW',0),
            'ITEM_CLICK':('CLICK',1),
            'LIST_CLICK':('CLICK',1),
            'FOLLOW':('FOLLOW',2),
            'FAVOURATE':('FAVOURATE',2),
            'REVIEW':('REVIEW',2),
            'COIN_BUY':('BUY_CLICK',3)
        }    


############################
task = 'regression' #regression
sparse_features = set()
dense_features = set()


for label,value  in (Merge(Merge(user_fea_map,interaction_fea_map), item_fea_map)).items():
    if 'float' in value[1]:
        dense_features.add(label)
    elif 'int' in value[1]:
        sparse_features.add(label)
        
    elif 'ID' in label:
        sparse_features.add(label)
        
    elif value[1] == 'string_type':
        sparse_features.add(label)
        
        

map_eng_name = {}
for k,v in item_fea_map.items():
    k,value = k.strip(), v[0].strip()
    map_eng_name[k] = value
    
for k,v in user_fea_map.items():
    k,value = k.strip(), v[0].strip()
    map_eng_name[k] = value
    
for k,v in interaction_fea_map.items():
    k,value = k.strip(), v[0].strip()
    map_eng_name[k] = value

target = ['????????????']
fea_config = {}
fea_config['item_fea_map'] = item_fea_map
fea_config['user_fea_map'] = user_fea_map
fea_config['interaction_fea_map'] = interaction_fea_map
fea_config['map_eng_name'] = map_eng_name
fea_config['sparse_features'] = list(sparse_features)
fea_config['dense_features'] = list(dense_features)
fea_config['target'] = target
fea_config['target_name'] = target_name
fea_config['task'] = task



output = open('./featuremodel/fea_config.pkl', 'wb')
pickle.dump(fea_config, output)
output.close()



