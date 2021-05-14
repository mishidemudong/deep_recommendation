#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:57:00 2021

@author: liang
"""
from load_user_data import get_user_data
from load_item_data import get_item_data
from load_interaction_data import get_interaction_data

import pandas as pd
import numpy as np



def left_joinFunc(df1,df2,colname):
    return pd.merge(df1, df2,how='left', on=colname)# 

def joinRunc(df1,df2,colname1,colname2):
    return pd.merge(df1,df2,how='left',left_on=colname1, right_on=colname2)


def loadalldata(user_id, itemid_list):
    
    alldata = []
    user_data = get_user_data(user_id)
    
    item_data = get_item_data(itemid_list)
    
    interactordata = get_interaction_data(user_id, itemid_list)
    
    alldata = left_joinFunc(interactordata, user_data, 'user_id')
    alldata = left_joinFunc(alldata, item_data, 'item_id')
     
    return alldata