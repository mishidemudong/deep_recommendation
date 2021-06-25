#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:11:24 2021

@author: liang
"""
import os
#os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import tensorflow as tf

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM, WDL, xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import pickle

from tensorflow.keras.optimizers import Adam

#from bert4keras.optimizers import Adam
#from bert4keras.optimizers import extend_with_weight_decay
#from bert4keras.optimizers import extend_with_gradient_accumulation
import json


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def buildmodel(config):
        
    if config['model_type'] == 'wdl':
        model = WDL(fea_model['linear_feature_columns'], fea_model['dnn_feature_columns'],
                        dnn_hidden_units=(config['all_params']['dnn_him'], config['all_params']['dnn_him']),
                        l2_reg_linear=0.00001,
                        l2_reg_embedding=0.00001, 
                        l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                        dnn_activation='relu', task=config['task']
                        )
        
    elif config['model_type'] == 'xdeepfm':
        model = xDeepFM(fea_model['linear_feature_columns'], fea_model['dnn_feature_columns'], 
                            dnn_hidden_units=(config['all_params']['dnn_him'], config['all_params']['dnn_him)']),
                            cin_layer_size=(config['all_params']['cin_size'], config['all_params']['cin_size']), 
                            cin_split_half=True, 
                            cin_activation='relu', 
                            l2_reg_linear=0.00001,
                            l2_reg_embedding=0.00001, 
                            l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
                            dnn_activation='relu', dnn_use_bn=False, task=config['task'])
        
#    model.load_weights(config['saved_model_path'])

    print("build model success")
    
    return model

class EvaluatorB(tf.keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, evaconfig):
        self.auc = 0.0
        self.save_model_path = evaconfig['best_model_path']
        self.config = evaconfig
        
        if not os.path.isdir(self.save_model_path):
            os.makedirs(self.save_model_path)

    def on_epoch_end(self, epoch, logs=None):
#        if epoch > 3:
        pred_ans = model.predict(x_test, batch_size=256)
        
        metrics = {}
        metrics["test LogLoss"] = round(log_loss(y_test, pred_ans), 4)
        metrics["test AUC"] = round(roc_auc_score(y_test, pred_ans), 4)
        
        if metrics['test AUC'] > self.auc:
            self.auc = metrics['test AUC']
            self.config['metrics'] = metrics
            self.config['saved_model_path'] = '{}/my_model_best_{}.weights'.format(self.save_model_path, self.auc)
            model.save_weights('{}/my_model_best_{}.weights'.format(self.save_model_path, self.auc))  # 保存模型 #my_model_nezhaunilm_kg_best   best_bleu 0.169
            filename='{}/my_model_best_{}.json'.format(self.save_model_path, self.auc)
            with open(filename,'w') as file_obj:
                json.dump(self.config,file_obj)
        
        print('valid_data:', metrics)
        model.save_weights('{}/my_model_last.weights'.format(self.save_model_path))  # 保存模型

class EvaluatorRe(tf.keras.callbacks.Callback):
        """评估与保存
        """
        def __init__(self, config):
            self.mse = 99999999
            self.save_model_path = config['best_model_path']
            self.config = config
            
            if not os.path.isdir(self.save_model_path):
                os.makedirs(self.save_model_path)
    
        def on_epoch_end(self, epoch, logs=None):
    #        if epoch > 3:
            pred_ans = model.predict(x_test, batch_size=256)
            
            metrics = {}
            metrics["test mse"] = round(mean_squared_error(y_test, pred_ans), 4)
            
            if metrics['test mse'] < self.mse:
                self.mse = metrics['test mse']
                self.config['metrics'] = metrics
                self.config['saved_model_path'] = '{}/my_model_best_{}.weights'.format(self.save_model_path, self.mse)
                model.save_weights(self.config['saved_model_path'])  # 保存模型 #my_model_nezhaunilm_kg_best   best_bleu 0.169
                filename='{}/my_model_best_{}.json'.format(self.save_model_path, self.mse)
                with open(filename,'w') as file_obj:
                    json.dump(self.config,file_obj)
            
            print('valid_data:', metrics)
            model.save_weights('{}/my_model_last.weights'.format(self.save_model_path))  # 保存模型

if __name__ == '__main__':
    
    
    ''''''


    modelpath = './WDLModel'
    config_path='{}/best_hyperparams.json'.format(modelpath)
    config = json.load(open(config_path,'r'))
    
    
    fea_model = pickle.load(open(config['fea_model_savepath'], 'rb')) 
    train_data_savepath = './data/train_test_split.pkl'
    split_data = pickle.load(open(train_data_savepath, 'rb'))
    x_train, y_train, x_test, y_test = split_data['x_train'], split_data['y_train'], split_data['x_test'], split_data['y_test']
    
    bushu_keys = ['model_type','all_params','fea_model_savepath', 'task']
    
    best_config = {}
    for key in bushu_keys:
        if key in config.keys():
            best_config[key] = config[key]
    best_config['best_model_path'] = './recmodel'
    model = buildmodel(best_config)
#    
#    print("===================================")
#    print(config)
#    if best_config['task'] == 'binary':
#        evaluator = EvaluatorB(best_config)
#    elif best_config['task'] == 'regression':
#        evaluator = EvaluatorRe(best_config)
#    
#    save_dir = './tensorboard'
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(
#        log_dir=os.path.join(save_dir, 'tf_logs'), histogram_freq=0, write_graph=False,
#        write_grads=False, update_freq=320)
    
    
#    AdamW = extend_with_weight_decay(Adam, 'AdamW')
#    AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
#    
#    optimizer = AdamWG(
#        lr=config['all_params']['lrate'],
#        weight_decay_rate=0.01,
#        exclude_from_weight_decay=['Norm', 'bias'],
##        grad_accum_steps=128
#    )
    
#    optimizer = Adam(best_config['all_params']['lrate'])
#
#    if config['task'] == 'binary':
#        model.compile(optimizer, "binary_crossentropy",
#                      metrics=['binary_crossentropy', 'accuracy'], 
#                      )
#    elif config['task'] == 'regression':
#        model.compile(optimizer, "mse", metrics=['mse'], )
#    
#    
#    history = model.fit(x_train, y_train,
#                        batch_size=best_config['all_params']['batch_size'],
#                        epochs=10, verbose=2, validation_split=0.2, 
#                        callbacks=[
#                        evaluator,
#                        tensorboard_callback,
#                        ])
    
    #predict and evaluate
#    modelpath = './WDLModel'
##    model.save_weights('./model/my_model.weights')
#    model.load_weights('{}/my_model_best_0.0016.weights'.format(modelpath))
#    pred_ans = model.predict(x_test, batch_size=256)
#    
##    pred_ans_train = model(test_model_input, training=True)
#    if config['task'] == 'binary':
#        print("test LogLoss", round(log_loss(y_test, pred_ans), 4))
#        print("test AUC", round(roc_auc_score(y_test, pred_ans), 4))
#        
#    elif config['task'] == 'regression':
#        print("test MSE", round(mean_squared_error(y_test, pred_ans), 4))
    