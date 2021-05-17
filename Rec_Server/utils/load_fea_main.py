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



def left_joinFunc(df1,df2,colname):
    return pd.merge(df1, df2,how='left', on=colname)# 

def joinRunc(df1,df2,colname1,colname2):
    return pd.merge(df1,df2,how='left',left_on=colname1, right_on=colname2)


def loadalldata(user_id, itemid_list):
    
    user_data = get_user_data(user_id)
    
    item_data = get_item_data(itemid_list)
    
    interactordata = get_interaction_data(user_id, itemid_list)
    
    pred_df = left_joinFunc(interactordata, user_data, 'user_id')
    pred_df = left_joinFunc(pred_df, item_data, 'item_id')

    pred_df['pred_time'] = [int(time.time())] * len(itemid_list)
     
    return pred_df