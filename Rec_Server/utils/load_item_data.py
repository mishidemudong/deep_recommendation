#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:57:42 2021

@author: liang
"""
import pandas as pd


def get_item_data(itemid_list):
    
    itemdf = pd.DataFrame()
    itemdf['item_id'] = itemid_list
    
    sql = ''' select {} from {} where item_id in {} and order by '''
    
    
    return itemdf
    
    
    
    
    