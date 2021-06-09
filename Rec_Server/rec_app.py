#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:36:07 2021

@author: liang
"""


from flask import Flask,request,make_response
from flask_cors import CORS
import tensorflow as tf
import json
from rec_predict import RecPredictHandler
#from recall_main import get_itemidlist
from utils.load_fea_main import loadalldata
from utils.db_utils import build_redis_connect


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#sess = tf.compat.v1.get_default_session(config=config)


print("loading model...")
config_path = './recmodel/my_model_best_0.0007.json'
rec_model = RecPredictHandler(config_path)
print("load REC model success !")


redis_curse = build_redis_connect()
print("build redis curse success !")

print("start rec server success !")



app = Flask(__name__)
@app.route('/rec_score_predict', methods=['POST', 'GET'])
def rec_score_func():

    print(request.get_json())
    sample = request.get_json()
    
    #1 召回item_id 
#    itemid_list = get_itemidlist('all')
    itemid_list = sample['item_list']
    user_list = sample['user_list']
    #2 load user_data, item_data, interaction_data
    #  or load user_fea, item_data, interaction_fea
    pred_data = loadalldata(redis_curse, user_list, itemid_list)

    #3predict
    response = rec_model.predict(sample, pred_data)
#    response = rec_model.predict_test(pred_data)

    if response: 
        res = {
                'errorcode': 0,
                'response': response
            }
    else:
        res = {
                'errorcode': 400,
                'error': 'no result'
            }
    
    result_json = make_response(json.dumps(res,ensure_ascii=False))
    result_json.headers['Access-Control-Allow-Methods'] = '*'
    
    return result_json

CORS(app, supports_credentials=True)
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
       "please wait until server has fully started"))
    
    app.run(debug=False, host='0.0.0.0', port=8008)