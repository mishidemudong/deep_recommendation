#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:38:55 2021

@author: liang
"""

import os
import pickle
import keras
from deepctr.models import AFM, DeepFM, WDL, xDeepFM, DSIN
from deepctr.feature_column import SparseFeat,get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
import pandas as pd
import json

class RecPredictHandler():
    """
    """
    def __init__(self, config_path):
        self.session = keras.backend.get_session()
        
        if os.path.exists(config_path):
            self.config = json.load(open(config_path,'r'))
            self.buildfeaturemodel()
            self.buildmodel()
        else:
            print('Build Handler Faild!!')
            
    def buildfeaturemodel(self):

        self.fea_model = pickle.load(open(self.config['fea_model_path'], 'rb')) 
#        self.dense_feamodel = pickle.load(open(self.fea_model['dense_feamoel_path'], 'rb')) 
        
        print("load fea_model success")
                
    def buildmodel(self):
        
        if self.config['model_type'] == 'wdl':
            self.model = WDL(self.fea_model['linear_feature_columns'], self.fea_model['dnn_feature_columns'],
                            dnn_hidden_units=(self.config['all_params']['dnn_him'], self.config['all_params']['dnn_him']),
                            l2_reg_linear=0.00001,
                            l2_reg_embedding=0.00001, 
                            l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                            dnn_activation='relu', task='binary'
                            )
            
        elif self.config['model_type'] == 'xdeepfm':
            self.model = xDeepFM(self.fea_model['linear_feature_columns'], self.fea_model['dnn_feature_columns'], 
                                dnn_hidden_units=(self.config['all_params']['dnn_him'], self.config['all_params']['dnn_him)']),
                                cin_layer_size=(self.config['all_params']['cin_size'], self.config['all_params']['cin_size']), 
                                cin_split_half=True, 
                                cin_activation='relu', 
                                l2_reg_linear=0.00001,
                                l2_reg_embedding=0.00001, 
                                l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
                                dnn_activation='relu', dnn_use_bn=False, task='binary')
        print("build model success")
        self.model.load_weights(self.config['saved_model_path'])
        print("load model success")

    
    def preprocess(self, data):
        
        trans_data = {}
        
        sparse_features = self.fea_model['sparse_features']
        dense_features = self.fea_model['dense_features']
    
        trans_data[sparse_features] = data[sparse_features].fillna('-1', )
        trans_data[dense_features] = data[dense_features].fillna(0, )
    
        #
        for feat in sparse_features:
            trans_data[feat] = self.fea_model['sparse_feamodel'][feat].transform(data[feat])
        
        trans_data[dense_features] = self.fea_model['dense_feamodel'].transform(trans_data[dense_features])
        
        return trans_data
    

    def predict(self, pred_data):
        res = {'user_id': pred_data['user_id'][0]}
        
        with self.session.graph.as_default():
            with self.session.as_default():
                test_model_input = self.preprocess(pred_data)
                pred = self.model.predict(test_model_input, batch_size=256)
                res['result'] = [(it_id, score[1]) for it_id, score in zip(pred_data['item_id'], pred)]
                
        return res

    def predict_test(self, pred_data):
        res = {'user_id': pred_data['user_id'][0]}
        res['result'] = [(it_id, random.random()) for it_id in pred_data['item_id']]

        return res




if __name__ == '__main__':
    
    
    config_path = './recmodel/my_model_best_0.7408.json'
    rec_object = RecPredictHandler(config_path)
    
    item_list = ['i_001','i_002','i_003']
    pred_df = pd.DataFrame()
    pred_df['user_id'] = ['0001'] * len(item_list)
    pred_df['item_id'] = item_list
    
    result = rec_object.predict_test(pred_df)
    print(result)