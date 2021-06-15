#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:33:48 2021

@author: liang
"""
from utils.load_fea_main import loadalldata
from utils.db_utils import build_redis_connect
    
redis_curse = build_redis_connect()

sample = {"user_list":["20200326092704"],  "item_list":["KAT" ]}

itemid_list = sample['item_list']
user_list = sample['user_list']

pred_data = loadalldata(redis_curse, user_list, itemid_list)