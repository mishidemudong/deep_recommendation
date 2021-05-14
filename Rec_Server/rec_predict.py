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

import json

class RecPredictHandler():
    """
    """
    def __init__(self, config_path):
        self.session = keras.backend.get_session()
        
        if os.path.exists(config_path):
            self.config = json.loads(open())
            self.buildfeaturemodel()
            self.buildmodel(self.config)
        else:
            print('Build Handler Faild!!')
            
    def buildfeaturemodel(self):

        self.sparse_feamodel = pickle.load(open(self.config['sparse_feamoel_path'], 'rb')) 
        self.dense_feamodel = pickle.load(open(self.config['dense_feamoel_path'], 'rb')) 
        
        print("load fea_model success")
                
    def buildmodel(self):
        
        if self.config['model_type'] == 'wdl':
            self.model = WDL(self.config['linear_feature_columns'], self.config['dnn_feature_columns'],
                            dnn_hidden_units=(self.config['dnn_him'], self.config['dnn_him']),
                            l2_reg_linear=0.00001,
                            l2_reg_embedding=0.00001, 
                            l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                            dnn_activation='relu', task='binary'
                            )
            
        elif self.config['model_type'] == 'xdeepfm':
            self.model = xDeepFM(self.config['linear_feature_columns'], self.config['dnn_feature_columns'], 
                                dnn_hidden_units=(self.config['dnn_him'], self.config['dnn_him)']),
                                cin_layer_size=(self.config['cin_size'], self.config['cin_size']), 
                                cin_split_half=True, 
                                cin_activation='relu', 
                                l2_reg_linear=0.00001,
                                l2_reg_embedding=0.00001, 
                                l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
                                dnn_activation='relu', dnn_use_bn=False, task='binary')
            
        self.model.load_weights(self.config['saved_model_path'])
        print("load model success")
        print("build model success")

    
    def preprocess(self, data):
        
        trans_data = {}
        
        sparse_features = self.config['sparse_features']
        dense_features = self.config['dense_features']
    
        trans_data[sparse_features] = data[sparse_features].fillna('-1', )
        trans_data[dense_features] = data[dense_features].fillna(0, )
    
        #
        for feat in sparse_features:
            trans_data[feat] = self.sparse_feamodel[feat].transform(data[feat])
        
        trans_data[dense_features] = self.dense_feamodel.transform(trans_data[dense_features])
        
        return trans_data
    

    def predict(self, pred_data):
        res = {}
        
        with self.session.graph.as_default():
            with self.session.as_default():
                
                res['user_id'] = pred_data['user_id']
                test_model_input = self.preprocess(pred_data['item_id'])
                pred = self.model.predict(test_model_input, batch_size=256)
                res['result'] = [(it_id, score) for it_id, score in zip(pred_data['item_id'], pred)]
                
        return res




if __name__ == '__main__':
    
    
    config_path = ''
    chatbot = RecPredictHandler(config_path)
    
    
    pred =  {"profile": {}, "subtrack": "knowledge", "goal": [["START", "海清", "北京遇上西雅图"], ["北京遇上西雅图", "主演", "海清"]], "situation": "", "knowledge": [["海清", "评论", "长 得 有点像 刘若英 和 徐静蕾 的 合体"], ["海清", "毕业 院校", "北京电影学院 表演系"], ["海清", "出生 日期", "1978 - 1 - 12"], ["海清", "性别", "女"], ["海清", "职业", "演员"], ["海清", "领域", "明星"], ["海清", "代表作", "北京遇上西雅图"], ["北京遇上西雅图", "获奖", "香港电影金像奖 _ ( 2014 ； 第33届 ) _ 提名 _ 金像奖 - 最佳女主角 _ 汤唯 Wei Tang"], ["北京遇上西雅图", "主演", "海清"], ["北京遇上西雅图", "口碑", "口碑 一般"], ["北京遇上西雅图", "类型", "喜剧"], ["北京遇上西雅图", "领域", "电影"], ["海清", "评论", "中国 的 媳妇 专业户"], ["海清", "血型", "O型"], ["海清", "获奖", "北京遇上西雅图_提名 _ ( 2014 ； 第32届 ) _ 大众电影百花奖 _ 百花奖 - 最佳 女 配角"], ["海清", "描述", "美女"], ["海清", "描述", "女星"]], "history": ["说起 一位 女星 我 都会 竖起 大拇指 ， 很 欣赏 她 。", "你 说 的 是 哪位 美女 呢 ？", "看 过 双面胶 吗 ， 里面 的 女 主角 就是 海清 。", "牙齿 伶俐 ， 能说会道 的 一个人 。", "饰演 的 角色 都 有 一股 女 强人 的 味道 。", "确实 是 一个 好 演员 。"]}
    
    
    result = chatbot.response(pred)
    print(result)