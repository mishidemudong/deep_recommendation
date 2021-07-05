#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:38:55 2021

@author: liang
"""
import tensorflow as tf
import os
import pickle
import keras
from deepctr.models import AFM, DeepFM, WDL, xDeepFM, DSIN
from deepctr.feature_column import SparseFeat,get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
import pandas as pd
import json
import numpy as np

class RecPredictHandler():
    """
    """
    def __init__(self, config_path):
        
        if os.path.exists(config_path):
            self.config = json.load(open(config_path,'r'))
            self.buildfeaturemodel()
            self.buildmodel()
        else:
            print('Build Handler Faild!!')
        
        ###important , must after buildmodel
        init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(init)
            
    def buildfeaturemodel(self):

        self.fea_model = pickle.load(open(self.config['fea_model_savepath'], 'rb')) 
#        self.dense_feamodel = pickle.load(open(self.fea_model['dense_feamoel_path'], 'rb')) 
        
        print("load fea_model success")
                
    def buildmodel(self):
        
        if self.config['model_type'] == 'wdl':
            self.model = WDL(self.fea_model['linear_feature_columns'], self.fea_model['dnn_feature_columns'],
                            dnn_hidden_units=(self.config['all_params']['dnn_him'], self.config['all_params']['dnn_him']),
                            l2_reg_linear=0.00001,
                            l2_reg_embedding=0.00001, 
                            l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                            dnn_activation='relu', task=self.config['task']
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
                                dnn_activation='relu', dnn_use_bn=False, task=self.config['task'])
            

        print("build model success")
        self.model.load_weights(self.config['saved_model_path'])
        print("load model success")

    
    def preprocess(self, data):
                
        sparse_features = self.fea_model['sparse_features']
        dense_features = self.fea_model['dense_features']
    
        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )
    
        #
        for feat in sparse_features:
#            print(feat)
            class_ = self.fea_model['sparse_fea_model'][feat].classes_
            data[feat] = data[feat].map(lambda s: '<unknow>' if s not in class_ else s)
            data[feat] = self.fea_model['sparse_fea_model'][feat].transform(data[feat])
        
            
        data[dense_features] = self.fea_model['dense_fea_model'].transform(data[dense_features])
        
    
        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )
    
        pred_model_input = {name:data[name] for name in self.fea_model['feature_names']}
        
        return pred_model_input
    

    def predict(self, user_list, pred_data):
        
        res = []
        
        with self.session.graph.as_default():
            with self.session.as_default():
                for user_id in user_list:
                    result = {}
                    result['user_id'] = user_id
                    data = pred_data[pred_data['USER_ID'] == user_id]
                    test_model_input = self.preprocess(data)
#                    print(len(test_model_input))
                    pred = self.model.predict(test_model_input, batch_size=256)
#                    print(pred)
#                    result['item_score_list'] = {it_id:str(score[0]) for it_id, score in zip(pred_data['item_id'], pred)}
                    result['item_score_list'] = [(it_id,str(score[0])) for it_id, score in zip(pred_data['ITEM_ID'], pred)]
                    result['model_type'] = self.config['model_type']
                    res.append(result)
                
        return res

    def feature_importance_choose(self, pred_data, category):
        tag_array = []
        item_tag_array = self.feature_importance_choose(pred_data)
        return tag_array

    def predict2(self, user_list, pred_data):

        res = []

        with self.session.graph.as_default():
            with self.session.as_default():
                for user_id in user_list:
                    result = {}
                    result['USER_ID'] = user_id
                    data = pred_data[pred_data['USER_ID'] == user_id] 
                    test_model_input = self.preprocess(data)
                    #                    print(len(test_model_input))
                    pred = self.model.predict(test_model_input, batch_size=256)
                    #                    print(pred)
                    #                    result['item_score_list'] = {it_id:str(score[0]) for it_id, score in zip(pred_data['item_id'], pred)}
                    print("ITEM_CATEGORY", pred_data['ITEM_CATEGORY'])
                    result['ITEM_SCORE_LIST'] = sorted([(it_id, str(score[0]), category) for it_id, score, category in
                                                 zip(pred_data['ITEM_ID'], pred, pred_data['ITEM_CATEGORY'])], key=lambda x:x[1], reverse=True )
                    result['MODEL_TYPE'] = self.config['model_type']
                    res.append(result)

        return res, self.config['fea_importance']

    def predict_test(self, pred_data):
        res = {'USER_ID': pred_data['USER_ID'][0]}
        res['result'] = [(it_id, random.random()) for it_id in pred_data['ITEM_ID']]

        return res




if __name__ == '__main__':
    
    
    config_path = './recmodel/my_model_best_0.0007.json'
    rec_object = RecPredictHandler(config_path)
    
#    item_list = ['i_001','i_002','i_003']
    
#    pred_df = pd.DataFrame()
#    pred_df['user_id'] = ['0001'] * len(item_list)
#    pred_df['item_id'] = item_list
    
#    result = rec_object.predict_test(pred_df)
    
    from utils.load_fea_main import loadalldata
    from utils.db_utils import build_redis_connect
        
    redis_curse = build_redis_connect()
    
    
    sample = {"user_list":["20200326092704","20200326091413","20200326092046","20200326092191","20200326093210","20200326094712","20200326096604","20200326096737","20200326098290","20200326095767"],  "item_list":["BTC","TRX","TUSD","BTT","DAI","BNB","TEL","SNX","DCR","LTC","DOT","SNT","XLM","DEGO","ETC","ETH","KNC","VET","DRGN","SOLVE","ROOBEE","AION","CRPT","RBTC","AERGO","TRTL","IOTX","OPEN","VIDT","TOKO","KICK","UTK","KAT" ]}
    
    item_list = sample['item_list']
    user_list = sample['user_list']
    
    pred_data = loadalldata(redis_curse, user_list, item_list)
    import copy
    
    preprocess_data = rec_object.preprocess(copy.copy(pred_data))
    result = rec_object.predict(sample, pred_data)
#    print(result)