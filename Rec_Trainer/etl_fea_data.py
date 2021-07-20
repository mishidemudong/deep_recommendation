#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:49:52 2021

@author: liang
"""

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import json,pickle

def file_name(file_dir, suffix_name):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == suffix_name:  
                L.append(os.path.join(root, file))  
    return L  

def saveTrainData(fea_config, ori_data_path, fea_model_savepath, train_data_savepath):
    
    sparse_features = [fea_config['map_eng_name'][item] for item in fea_config['sparse_features']]
    dense_features = [fea_config['map_eng_name'][item] for item in fea_config['dense_features']]
    target = [fea_config['map_eng_name'][ fea_config['target'][0]]]
    target_name = fea_config['target_name']

    data = pd.read_csv(ori_data_path)#.sample(frac=0.1, replace=False, random_state=5, axis=0)
    # 3.generate input data for model
    data[target] = [target_name[item[0]][1] for item in list(data[target].values)]
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    fea_model = {}
    fea_model['sparse_features'] = sparse_features
    fea_model['dense_features'] = dense_features
    
    sparse_fea = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        lbe.classes_ = np.append(lbe.classes_, '<unknow>')
        sparse_fea[feat] = lbe
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
#    mms.classes_ = np.append(mms.classes_, '<unknow>')
    
    fea_model['sparse_fea_model'] = sparse_fea
    fea_model['dense_fea_model'] = mms

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 2,embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    
    fea_model['dnn_feature_columns'] = dnn_feature_columns
    fea_model['linear_feature_columns'] = linear_feature_columns
    
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    fea_model['feature_names'] = feature_names
    
    output = open(fea_model_savepath, 'wb')
    pickle.dump(fea_model, output)
    output.close()
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    
    
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    x_train, y_train = train_model_input, train[target].values
    x_test, y_test = test_model_input, test[target].values
    
    split_data = {}
    split_data['x_train'] = x_train
    split_data['y_train'] = y_train
    split_data['x_test'] = x_test
    split_data['y_test'] = y_test
    
    output = open(train_data_savepath, 'wb')
    pickle.dump(split_data, output)
    output.close()

if __name__ == "__main__":    
    
    ##config
    config_path = './config.json'
    config = {}
    ori_data_path = './data/train_eng_fea.csv'
    fea_path      = './featuremodel/fea_config.pkl'
    fea_model_savepath = './featuremodel/fea_model.pkl'
    train_data_savepath = './data/train_test_split.pkl'
   
    config['ori_data_path'] = ori_data_path
    config['fea_path'] = fea_path
    config['fea_model_savepath'] = fea_model_savepath
    config['train_data_savepath'] = train_data_savepath
   
    with open(config_path,'w') as file_obj:
        json.dump(config,file_obj)
    fea_config = pickle.load(open(fea_path, 'rb')) 
    
#     ##make train dev data
    saveTrainData(fea_config, ori_data_path, fea_model_savepath, train_data_savepath)