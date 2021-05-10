#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:49:52 2021

@author: liang
"""
import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM, WDL, xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

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

        
def make_features(data, sparse_features, dense_features):


    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    return feature_names, linear_feature_columns, dnn_feature_columns

def makedata():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    data = pd.read_csv('/media/liang/Nas/推荐系统数据集/criteo_sampled_data.csv')

    # 3.generate input data for model
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
    
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    x_train, y_train = train_model_input, train[target].values
    x_test, y_test = test_model_input, test[target].values
    
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
            pred_ans = model.predict(test_model_input, batch_size=256)
            
            metrics = {}
            metrics["test LogLoss"] = round(log_loss(test[target].values, pred_ans), 4)
            metrics["test AUC"] = round(roc_auc_score(test[target].values, pred_ans), 4)
            
            if metrics['test AUC'] > self.auc:
                self.auc = metrics['test AUC']
                self.config['metrics'] = metrics
                model.save_weights('{}/my_model_best_{}.weights'.format(self.save_model_path, self.auc))  # 保存模型 #my_model_nezhaunilm_kg_best   best_bleu 0.169
                filename='{}/my_model_best_{}.json'.format(self.save_model_path, self.auc)
                with open(filename,'w') as file_obj:
                    json.dump(self.config,file_obj)
            
            print('valid_data:', metrics)
            model.save_weights('{}/my_model_last.weights'.format(self.save_model_path))  # 保存模型
            
    
    data = pd.read_csv('/media/liang/Nas/推荐系统数据集/criteo_sampled_data.csv')

    # 3.generate input data for model
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
    
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    x_train, y_train = train_model_input, train[target].values
    x_test, y_test = test_model_input, test[target].values
    
    dnn_him = {{choice([128, 256, 512])}}
    cin_size = {{choice([64, 128, 192])}}
#    model = xDeepFM(linear_feature_columns, dnn_feature_columns, 
#                    dnn_hidden_units=(dnn_him, dnn_him),
#                    cin_layer_size=(cin_size, cin_size,), 
#                    cin_split_half=True, 
#                    cin_activation='relu', 
#                    l2_reg_linear=0.00001,
#                    l2_reg_embedding=0.00001, 
#                    l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
#                    dnn_activation='relu', dnn_use_bn=False, task='binary')
    
    model = WDL(linear_feature_columns, dnn_feature_columns,
                dnn_hidden_units=(dnn_him, dnn_him),
                l2_reg_linear=0.00001,
                l2_reg_embedding=0.00001, 
                l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                dnn_activation='relu', task='binary'
                )
    
    # 5. checkpoint
#    modelpath = './xDeepFMmodel'
#    evaluator = EvaluatorB(modelpath)
#    
#    save_dir = './tensorboard'
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(
#        log_dir=os.path.join(save_dir, 'tf_logs'), histogram_freq=0, write_graph=False,
#        write_grads=False, update_freq=320)
    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
    
    lrate = {{choice([0.01, 0.001, 0.0005])}}
    optimizer = AdamWG(
        lr=lrate,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
#        grad_accum_steps=128
    )

    model.compile(optimizer, "binary_crossentropy",
                  metrics=['binary_crossentropy', 'accuracy'], 
                  )
    
    
    batch_size = {{choice([64, 128, 256])}}
    config = {}
    
    modelpath = './WDLModel'       
    config['modelpath'] = modelpath 
    config['dnn_him'] = dnn_him
    config['lr'] = lrate
    config['batch_size'] = batch_size
    
    config['all_params'] = {}
    config['all_params']['dnn_him'] = [128, 256, 512] 
    config['all_params']['batch_size'] = [64, 128, 256] 
    config['all_params']['lr'] = [0.01, 0.001, 0.0005]
    
    evaluator = EvaluatorB(config)
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, 
                        epochs=5, 
                        verbose=2, 
                        validation_split=0.2, 
                        callbacks=[
                        evaluator,
#                        tensorboard_callback,
                        ]
                        )
    
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(history.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}



if __name__ == "__main__":
    
    datapath = '/media/liang/Nas/推荐系统数据集/criteo_sampled_data.csv'
    
    best_run, best_model = optim.minimize(model=create_model,
                                          data=makedata,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    
    X_train, Y_train, X_test, Y_test = makedata()
#
#    
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    filename='./best_hyperparams.json'
    with open(filename,'w') as file_obj:
        json.dump(best_run,file_obj)
        
    '''
    [0.552747669549783, 0.52751887, 0.7616]
    Best performing model chosen hyper-parameters:
    {'batch_size': 0, 'cin_size': 1, 'dnn_him': 2}
    注意最佳参数是ID号
    
    Best validation acc of epoch:                                                    
        0.7775625   
    
    '''
