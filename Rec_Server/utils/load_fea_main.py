#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:57:00 2021

@author: liang
"""
from utils.load_user_data import get_user_data
from utils.load_item_data import get_item_data
from utils.load_interaction_data import get_interaction_data

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import pandas_redshift as pr
import redis

from db_utils import dbconfig, s3config 

def left_joinFunc(df1,df2,colname):
    return pd.merge(df1, df2,how='left', on=colname)# 

def joinRunc(df1,df2,colname1,colname2):
    return pd.merge(df1,df2,how='left',left_on=colname1, right_on=colname2)

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 

def loadalldata(user_id, itemid_list):
    
    pr_curse= pr.connect_to_redshift(**dbconfig)
    
    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    redis_curse = redis.Redis(connection_pool=pool)

    user_df = get_user_data(redis_curse, user_id)
    
    item_df = get_item_data(pr, itemid_list)
    
    interation_df = get_interaction_data(redis_curse, user_id, itemid_list)
    
#    pred_df = left_joinFunc(interactordata, user_data, 'user_id')
#    pred_df = left_joinFunc(pred_df, item_data, 'item_id')
    
    
    pred_df = left_joinFunc(interation_df, user_df, 'user_id')
    #train_df = left_joinFunc(interation_df, user_df)
    
    colname1 = 'item_id'
    colname2 = 'item_create_time'
    
    new_df = pd.DataFrame()
    
    for index, item in tqdm(pred_df.iterrows()):
        choose_list = item_df[item_df[colname1] == item[colname1]]
        choose_list = choose_list[choose_list['item_create_time'] <= item['interaction_create_time']]
        
        if choose_list.shape[0] > 0:
            latest_one = choose_list.sort_values(by=colname2).head(1)
        
            join_dict = Merge(item.to_dict(), latest_one.to_dict())
        
            new_df = new_df.append(pd.DataFrame.from_dict(join_dict))
        

    pred_df['pred_time'] = [int(time.time())] * len(itemid_list)
     
    return pred_df