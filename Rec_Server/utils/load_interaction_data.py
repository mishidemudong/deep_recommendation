#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:58:56 2021

@author: liang
"""

import pandas as pd

def get_interaction_data(user_id, itemid_list):
    interaction_df = pd.DataFrame()
    interaction_df['user_id'] = [user_id]*len(itemid_list)
    interaction_df['item_id'] = itemid_list
    
    
    return interaction_df