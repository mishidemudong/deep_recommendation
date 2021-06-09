#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:07:00 2021

@author: liang
"""

import redis
dbconfig = {}

s3config = {}

redis_item_config = { 
            'host':'localhost', 
            'port': 6379, 
            'db' : 0,
            'decode_responses':True
            }

redis_user_config = { 
            'host':'localhost', 
            'port': 6379, 
            'db':1,
            'decode_responses':True
            }

redis_interaction_config = { 
            'host':'localhost', 
            'port': 6379, 
            'db' : 2,
            'decode_responses':True
            }
names = ['item','user','interaction']
redis_config = [redis_item_config, redis_user_config, redis_interaction_config]

def build_redis_connect():
    redis_curse_object = {}
    
    for name, config in zip(names,redis_config):
        pool = redis.ConnectionPool(**config)
        redis_curse = redis.Redis(connection_pool=pool)
        
        redis_curse_object[name] = redis_curse
        
    return redis_curse_object