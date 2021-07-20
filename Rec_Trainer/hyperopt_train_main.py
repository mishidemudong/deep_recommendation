#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:49:52 2021

@author: liang
"""

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM, WDL, xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import pickle

from hyperas import optim
from hyperas.distributions import choice, uniform

from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation


import json
import numpy as np
import tensorflow as tf                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def makedata():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    train_data_savepath = './data/train_test_split.pkl'

    split_data = pickle.load(open(train_data_savepath, 'rb'))
    
    x_train, y_train, x_test, y_test = split_data['x_train'], split_data['y_train'], split_data['x_test'], split_data['y_test']
    
    
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    global modelpath
    
    class EvaluatorB(tf.keras.callbacks.Callback):
        """评估与保存
        """
        def __init__(self, config):
            self.auc = 0.0
            self.save_model_path = config['modelpath']
            self.config = config
            
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
                model.save_weights(self.config['saved_model_path'])  # 保存模型 #my_model_nezhaunilm_kg_best   best_bleu 0.169
                filename='{}/my_model_best_{}.json'.format(self.save_model_path, self.auc)
                with open(filename,'w') as file_obj:
                    json.dump(self.config,file_obj)
            
            print('valid_data:', metrics)
            model.save_weights('{}/my_model_last.weights'.format(self.save_model_path))  # 保存模型
            
    class EvaluatorRe(tf.keras.callbacks.Callback):
        """评估与保存
        """
        def __init__(self, config):
            self.mse = 9999999999
            self.save_model_path = config['modelpath']
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
    
    config = json.load(open('./config.json'))
    
    fea_model_savepath = './featuremodel/fea_model.pkl'
    train_data_savepath = './data/train_test_split.pkl'
    
    feature_config = pickle.load(open(fea_model_savepath, 'rb'))
    
    linear_feature_columns = feature_config['linear_feature_columns']
    dnn_feature_columns = feature_config['dnn_feature_columns']
    
    split_data = pickle.load(open(train_data_savepath, 'rb'))
    
    x_train, y_train, x_test, y_test = split_data['x_train'], split_data['y_train'], split_data['x_test'], split_data['y_test']
    
    dnn_him = {{choice([128, 256, 512])}}
    cin_size = {{choice([64, 128, 192])}}
    
#    config['model_type'] = 'xdeepfm'
#    config['task'] = 'binary'
#    model = xDeepFM(linear_feature_columns, dnn_feature_columns, 
#                    dnn_hidden_units=(dnn_him, dnn_him),
#                    cin_layer_size=(cin_size, cin_size,), 
#                    cin_split_half=True, 
#                    cin_activation='relu', 
#                    l2_reg_linear=0.00001,
#                    l2_reg_embedding=0.00001, 
#                    l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
#                    dnn_activation='relu', dnn_use_bn=False, task='binary')
    config['model_type'] = 'wdl'
    config['task'] = 'regression'
    model = WDL(linear_feature_columns, dnn_feature_columns,
                dnn_hidden_units=(dnn_him, dnn_him),
                l2_reg_linear=0.00001,
                l2_reg_embedding=0.00001, 
                l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                dnn_activation='relu', task=config['task']
                )
    
    # 5. checkpoint
    
    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
    
    lrate = {{choice([0.01, 0.001, 0.0005])}}
    optimizer = AdamWG(
        lr=lrate,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
#        grad_accum_steps=128
    )
    
    if config['task'] == 'binary':
        model.compile(optimizer, "binary_crossentropy",
                      metrics=['binary_crossentropy', 'accuracy'], 
                      )
    elif config['task'] == 'regression':
        model.compile(optimizer, "mse", metrics=['mse'], )
    
    
    batch_size = {{choice([64, 128, 256])}}
    
    
    modelpath = './WDLModel'
    config['modelpath'] = modelpath 
    config['dnn_him'] = dnn_him
    config['lrate'] = lrate
    config['batch_size'] = batch_size
    
    config['all_params'] = {}
    config['all_params']['dnn_him'] = [128, 256, 512] 
    config['all_params']['batch_size'] = [64, 128, 256] 
    config['all_params']['lrate'] = [0.01, 0.001, 0.0005]
    
    print("===================================")
    print(config)
    if config['task'] == 'binary':
        evaluator = EvaluatorB(config)
    elif config['task'] == 'regression':
        evaluator = EvaluatorRe(config)
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, 
                        epochs=1, 
                        verbose=2, 
                        validation_split=0.2, 
                        callbacks=[
                        evaluator,
#                        tensorboard_callback,
                        ]
                        )
    
    #get the highest validation accuracy of the training epochs
    if config['task'] == 'binary':
        validation_acc = np.amax(history.history['val_acc']) 
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
    
    elif config['task'] == 'regression':
        validation_mse = np.amin(history.history['val_mean_squared_error']) 
        print('Best validation mse of epoch:', validation_mse)
        return {'loss': validation_mse, 'status': STATUS_OK, 'model': model}
    



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


    #train model    
    fea_model_savepath = './featuremodel/fea_model.pkl'
    output = open(fea_model_savepath, 'rb')
    fea_model = pickle.load(output)

    best_run, best_model, space = optim.minimize(model=create_model,
                                          data=makedata,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
                                          return_space=True)
    
    X_train, Y_train, X_test, Y_test = makedata()
 
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    
    modelpath = './WDLModel'
    filename='{}/best_hyperparams.json'.format(modelpath)
    
    high_name  = file_name('./WDLModel', '.json')
    
    if fea_config["task"] == 'binary':
        config = json.load(open(sorted(high_name)[-1],'r'))
    elif fea_config["task"] == 'regression':
        config = json.load(open(sorted(high_name)[0],'r'))
    
    best_config = config
    for key,value in best_run.items():
        if key in best_config['all_params'].keys():
            print(key)
            print(best_config['all_params'][key])
            best_config['all_params'][key] = best_config['all_params'][key][value]
        
        
    with open(filename,'w') as file_obj:
        json.dump(best_config,file_obj)
#        
    '''
    [0.552747669549783, 0.52751887, 0.7616]
    Best performing model chosen hyper-parameters:
    {'batch_size': 0, 'cin_size': 1, 'dnn_him': 2}
    注意最佳参数是ID号
    
    Best validation acc of epoch:                                                    
        0.7775625   
    
    '''
