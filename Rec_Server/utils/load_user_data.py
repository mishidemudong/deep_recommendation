#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:55:57 2021

@author: liang
"""
import pandas as pd

def get_user_data(user_id):
    
    
    user_df = pd.DataFrame()
    
    user_df['user_id'] = user_id
    
    
    return user_df