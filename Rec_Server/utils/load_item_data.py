#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:57:42 2021

@author: liang
"""

import pandas as pd
import requests


def get_data_redis(curse_redis, id_list):
    
    #####get data from redis
    dta = curse_redis.mget(id_list)
#    columns_list = list(eval(dta[0]).keys())
    columns_list = eval(curse_redis.get('COLUMNS'))
    data_array = []
    for item in dta:
        
        if item != None:
            value_list = list(eval(item).values())
        else:
            value_list = [''] * len(columns_list)
            
        data_array.append(value_list)
    result_df = pd.DataFrame(data_array, columns= columns_list)  
    
    return result_df

def get_item_api(itemid_list):
    
    data_array = []
    
    url = "http://test"
    data = '{"key":"value"}'
    res = requests.post(url=url,data=data)
    columns_list = list(res.text.keys())
    
    
    for item_id in itemid_list:
        data = '{"key":"value"}'
        #字符串格式
        res = requests.post(url=url,data=data)
        print(res.text)
        value_list = list(res.text.values())
        data_array.append(value_list)
        
    itemdf = pd.DataFrame(data_array, columns= columns_list) 
    itemdf['item_id'] = itemid_list    
    
    return itemdf
    
if __name__ == "__main__":  
    
    import redis
    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    redis_curse = redis.Redis(connection_pool=pool)
    
    
    sample = {"user_list":["20200326092704","20200326091413","20200326092046","20200326092191","20200326093210","20200326094712","20200326096604","20200326096737","20200326098290","20200326095767"],  "item_list":["BTC","TRX","TUSD","BTT","DAI","BNB","TEL","SNX","DCR","LTC","DOT","SNT","XLM","DEGO","ETC","ETH","KNC","VET","DRGN","SOLVE","ROOBEE","AION","CRPT","RBTC","AERGO","TRTL","IOTX","OPEN","VIDT","TOKO","KICK","UTK","KAT" ]}
    
    itemid_list = sample['item_list']
    
    
    item_df = get_data_redis(redis_curse, itemid_list)
    