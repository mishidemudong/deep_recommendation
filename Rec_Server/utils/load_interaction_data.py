#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:58:56 2021

@author: liang
"""

import pandas as pd

def makeinteraction_id(user_list, itemid_list):
    interact_id = []
    
    for user_id in user_list:
        for itemid in itemid_list:
            interact_id.append(user_id + '_' + itemid)
            
    return interact_id


def get_interaction_data(user_id, itemid_list):
    interaction_df = pd.DataFrame()
    interaction_df['user_id'] = [user_id]*len(itemid_list)
    interaction_df['item_id'] = itemid_list
    
    return interaction_df


def get_interaction_redis(curse_redis, id_list):
    
    #####get data from redis
    dta = curse_redis.mget(id_list)
#    print(dta[0])
    columns_list = curse_redis.keys()
#    data = next(item for item in dta if item is not None)
#    columns_list = list(eval(data).keys())
    
    data_array = []
    for item in dta:
        if item != None:
            value_list = list(eval(item).values())
        else:
            value_list = [''] * len(columns_list)
        data_array.append(value_list)
    result_df = pd.DataFrame(data_array, columns = columns_list)  
    
    
    return result_df


if __name__ == "__main__":  
    
    import redis
    pool = redis.ConnectionPool(host='localhost', port=6379, db=2, decode_responses=True)
    redis_curse = redis.Redis(connection_pool=pool)
    
    
    sample = {"user_list":["20200326092704"],  "item_list":["BTC","TRX" ]}
    
    itemid_list = sample['item_list']
    user_list = sample['user_list']
    
    
#    id_list = makeinteraction_id(user_list, itemid_list)
    
    id_list = ['20200326097953_TEST-BOT-ID 6',
 '20200326093938_TEST-BOT-ID 10',
 '20200326098480_TEST-BOT-ID 2',
 '20200326104105_TEST-BOT-ID 5',
 '20200326096592_TEST-BOT-ID 7',
 '20200326093873_TEST-BOT-ID 9',
 '20200326091914_TEST-BOT-ID 3',
 '20200326097180_TEST-BOT-ID 1',
 '20200326098542_OCEAN',
 '20200326099149_3']
    
    df = get_interaction_redis(redis_curse, id_list)