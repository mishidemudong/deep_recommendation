#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:57:00 2021

@author: liang
"""
from utils.load_user_data import get_data_redis
from utils.load_interaction_data import get_interaction_redis, makeinteraction_id

#from load_user_data import get_data_redis
#from load_interaction_data import get_interaction_redis, makeinteraction_id

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

#from db_utils import dbconfig, s3config 

def left_joinFunc(df1,df2,colname):
    return pd.merge(df1, df2,how='left', on=colname)# 

def joinRunc(df1,df2,colname1,colname2):
    return pd.merge(df1,df2,how='left',left_on=colname1, right_on=colname2)

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 



def loadalldata(redis_curse, user_list, item_list):

    user_df = get_data_redis(redis_curse['user'], user_list, 'user_id')
    
    item_df = get_data_redis(redis_curse['item'], item_list, 'item_id')
    
    interact_id = makeinteraction_id(user_list, item_list)
    interation_df = get_interaction_redis(redis_curse['interaction'], interact_id)
    
    
    pred_df = left_joinFunc(interation_df, user_df, 'user_id')
    pred_df = left_joinFunc(pred_df, item_df, 'item_id')
    
    pred_df[['user_id', 'item_id']] = pred_df[['user_id', 'item_id']].astype('str')
    
#    colname1 = 'item_id'
#    colname2 = 'item_create_time'
#    new_df = pd.DataFrame()
#    for index, item in tqdm(pred_df.iterrows()):
#        choose_list = item_df[item_df[colname1] == item[colname1]]
#        choose_list = choose_list[choose_list['item_create_time'] <= item['interaction_create_time']]
#        if choose_list.shape[0] > 0:
#            latest_one = choose_list.sort_values(by=colname2).head(1)
#            join_dict = Merge(item.to_dict(), latest_one.to_dict())
#            new_df = new_df.append(pd.DataFrame.from_dict(join_dict))
        
#    pred_df['pred_time'] = [int(time.time())] * len(itemid_list)
     
    return pred_df

if __name__ == "__main__":  
    
    from db_utils import build_redis_connect
    
    redis_curse = build_redis_connect()
    
    sample = {"user_list":["20200326092704","20200326091413","20200326092046","20200326092191","20200326093210","20200326094712","20200326096604","20200326096737","20200326098290","20200326095767"],  "item_list":["BTC","TRX","TUSD","BTT","DAI","BNB","TEL","SNX","DCR","LTC","DOT","SNT","XLM","DEGO","ETC","ETH","KNC","VET","DRGN","SOLVE","ROOBEE","AION","CRPT","RBTC","AERGO","TRTL","IOTX","OPEN","VIDT","TOKO","KICK","UTK","KAT" ]}
    
    itemid_list = sample['item_list']
    user_list = sample['user_list']
    
    pred_data = loadalldata(redis_curse, user_list, itemid_list)