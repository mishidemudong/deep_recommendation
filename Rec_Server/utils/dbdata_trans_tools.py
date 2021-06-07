#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:29:29 2021

@author: liang
"""
from tqdm import tqdm
import pandas as pd
import pandas_redshift as pr
from db_utils import dbconfig, s3config, redis_config
import redis


def transpose_2d(data):
    # transposed = list(zip(*data))
    # [(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]
    # 注意 zip 本身返回的数据类型为 tuple 元组
    # 其中符号 * 号可以对元素进行解压或展开

    transposed = list(map(list, zip(*data)))
    return transposed

def get_sqldata(curse, columns, user_id=None):
    
    
    if user_id == None:
        sql = '''
            
            select user_id,max(user_create_time)
                    from 表
                    group by user_id
            
            '''
        
    else:
        sql = '''
            
            select user_id,max(user_create_time)
                    from 表
                    group by user_id
                    where user_id in ({})
            
            '''.format(user_id)
            
    user_df = curse.redshift_to_pandas(sql)  
    
    
    return user_df


####redis

#r.set('user_fea', bytes_)

def writedata2redis(r, df, keys):
    print('start write data to redis...')
    for index, item in tqdm(user_df.iterrows()):
        r.set(item['user_id'], str(item.to_dict()))
    print('write data to redis done!')
    

    
    
if __name__ == "__main__":    
    
    curse = pr.connect_to_redshift(**dbconfig)
    pool = redis.ConnectionPool()
    redis_curse = redis.Redis(connection_pool=pool)
    user_df = pd.read_csv('/media/liang/Project2/推荐系统/git_code/deep_recommendation/data/user_eng_fea.csv')
    
    user_df =  get_sqldata(curse, user_id)
    
    writedata2redis(redis_curse, user_df, )
    
    
    
    