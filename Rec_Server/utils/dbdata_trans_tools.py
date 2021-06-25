#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:29:29 2021

@author: liang
"""
from tqdm import tqdm
import pandas as pd
import pandas_redshift as pr
from db_utils import dbconfig, s3config, redis_interaction_config, redis_item_config, redis_user_config 
import redis


def transpose_2d(data):
    # transposed = list(zip(*data))
    # [(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]
    # 注意 zip 本身返回的数据类型为 tuple 元组
    # 其中符号 * 号可以对元素进行解压或展开

    transposed = list(map(list, zip(*data)))
    return transposed

def get_sqldata(curse, columns, tablename, user_id=None):
    
    
    if user_id == None:
        sql = '''
            
            select {}
                    from 表
                    group by user_id
            
            '''.format(columns)
        
    else:
        sql = '''
            
            select {}
                    from {}
                    group by user_id
                    where user_id in ({})
            
            '''.format(columns, tablename, user_id)
            
    user_df = curse.redshift_to_pandas(sql)  
    
    
    return user_df

def interaction_data_func(curse, columns, tablename, user_id=None):
    
    sql = '''
    

    select tcfi.* from t_cust_face_identify tcfi where tcfi.cust_no = '2004081422003482994' order by tcfi.create_time desc limit 1;


    
    '''
    
    user_df = curse.redshift_to_pandas(sql)  
    
    
    return user_df

####redis

#r.set('user_fea', bytes_)

def writedf2redis(r, df, key):
    print('start write data to redis...')
    keys = []
    for index, item in tqdm(df.iterrows()):
        keys.append(item[key])
        r.set(item[key], str(item.to_dict()))
        
    ####set columns    
    r.set('COLUMNS', str(df.columns.tolist()))
    print('write data to redis done!')
    
    return keys
    
def writedf2redis2keys(r, df, key1, key2):
    print('start write data to redis...')
    keys = []
    for index, item in tqdm(df.iterrows()):
        key = str(item[key1]) + '_' + str(item[key2])
        r.set(key, str(item.to_dict()))
        
        keys.append(key)
    print('write data to redis done!')
    r.set('COLUMNS', str(df.columns.tolist()))
    return keys
    

    
    
if __name__ == "__main__":    
    
#    curse = pr.connect_to_redshift(**dbconfig)
    
    ###############################################user##########################################################
    ######read data
#    user_df = pd.read_csv('../../data/user_eng_fea.csv').drop_duplicates(['user_id'])
##    columns = ''
##    user_df =  get_sqldata(curse, columns)
#    
#    #####write
#    pool = redis.ConnectionPool(**redis_user_config)
#    redis_curse = redis.Redis(connection_pool=pool)
#    
#    writedf2redis(redis_curse, user_df, 'user_id')
#    redis_curse.set('COLUMNS', str(user_df.columns.tolist()))
    ###############################################item##########################################################
    #####read data
#    item_df = pd.read_csv('../../data/item_eng_fea.csv').drop_duplicates(['item_id'])#.drop('UUID', 1)
#    columns = ''
#    item_df =  get_sqldata(curse, columns)
    #####write
#    pool = redis.ConnectionPool(**redis_item_config)
#    redis_curse = redis.Redis(connection_pool=pool)
#    writedf2redis(redis_curse, item_df, 'item_id')
#    redis_curse.set('COLUMNS', str(item_df.columns.tolist()))
    
#    ###############################################interaction##########################################################
#    #####read data
    interaction_df = pd.read_csv('../../data/interaction_eng_fea.csv').drop_duplicates('user_id')
#    columns = ''
#    interaction_df =  get_sqldata(curse, columns)

    #####write
    pool = redis.ConnectionPool(**redis_interaction_config)
    redis_curse = redis.Redis(connection_pool=pool)
    interaction_keys = writedf2redis2keys(redis_curse, interaction_df, 'user_id', 'item_id')
    redis_curse.set('COLUMNS', str(interaction_df.columns.tolist()))
    
    