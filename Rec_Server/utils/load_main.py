#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:57:00 2021

@author: liang
"""
from load_user_data import get_user_data
from load_item_data import get_item_data
from load_interaction_data import get_interaction_data

def loadalldata(user_id):
    
    
    alldata = []
    user_data = get_user_data(user_id)
    
    
    return alldata