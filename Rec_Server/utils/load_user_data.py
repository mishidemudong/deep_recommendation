#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:55:57 2021

@author: liang
"""

'''
use redis for online mode

'''

import pandas as pd

def get_user_data_redshift(curse, user_id):
    
    sql = '''
    
    select user_id,max(create_time)
            from 表
            group by user_id
            where user_id in ({})
    
    '''.format(user_id)
        
    user_df = curse.redshift_to_pandas(sql)    
    
    return user_df

def get_data_redis(curse_redis, id_list, id_name):
    
    #####get data from redis
    dta = curse_redis.mget(id_list)
#    columns_list = list(eval(dta[0]).keys())
    columns_list = eval(curse_redis.get('COLUMNS'))
    data_array = []
    for item in dta:
        if item != None:
            value_list = list(eval(item).values())
        else:
            value_list = [None] * len(columns_list)
            
        data_array.append(value_list)
#    print("*****",len(columns_list))
    result_df = pd.DataFrame(data_array, columns= columns_list)  
    result_df[id_name] = id_list
    
    return result_df

if __name__ == "__main__":  
    
    import redis
    redis_user_config = { 
            'host':'localhost', 
            'port': 6379, 
            'db':1,
            'decode_responses':True
            }
    
    pool = redis.ConnectionPool(**redis_user_config)
    redis_curse = redis.Redis(connection_pool=pool)
    
    
#    sample = {"user_list":["20200326092704","20200326091413","20200326092046","20200326092191","20200326093210","20200326094712","20200326096604","20200326096737","20200326098290","20200326095767"],  "item_list":["BTC","TRX","TUSD","BTT","DAI","BNB","TEL","SNX","DCR","LTC","DOT","SNT","XLM","DEGO","ETC","ETH","KNC","VET","DRGN","SOLVE","ROOBEE","AION","CRPT","RBTC","AERGO","TRTL","IOTX","OPEN","VIDT","TOKO","KICK","UTK","KAT" ]}
#    user_list = sample['user_list']
    
    user_list = ['60b4cdc47789d200098dc87d', '60b4cc777789d200098dc879']
    
    user_df = get_data_redis(redis_curse, user_list, 'user_id')