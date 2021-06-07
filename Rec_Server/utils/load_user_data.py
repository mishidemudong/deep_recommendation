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

def get_data_redis(curse_redis, id_list):
    
    #####get data from redis
    dta = curse_redis.mget(id_list)
    columns_list = list(eval(dta[0]).keys())
    data_array = []
    for item in dta:
        value_list = list(eval(item).values())
        data_array.append(value_list)
    result_df = pd.DataFrame(data_array, columns= columns_list)  
    
    
    return result_df
