#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:59:28 2021

@author: liang
"""



import random
import pandas as pd
from fea_config import user_fea_map, item_fea_map, interaction_fea_map, target_name
from make_random_date import make_random_time

user_id = ['user_{}'.format(str(no)) for no in range(50000)]
item_id = ['item_{}'.format(str(no)) for no in range(300)]



def random_int(num, decimal):
    array = []
    
    for i in range(num):
        array.append(random.randint(0, decimal))
        
    return array

def random_float(num,decimal):
    array = []
    
    for i in range(num):
        array.append(random.uniform(0, decimal))
        
    return array

def random_float_1(num):
    array = []
    
    for i in range(num):
        array.append(random.uniform(0, 1))
        
    return array


######make user fea
multi = 10
enum_list = ['KYC等级', 'KYC国家']
KYC_level = ['A','B','C']
kyc_country = ['USA', 'JAPAN', 'ITERLY', 'UK']
rank = 600
user_df = pd.DataFrame()
user_df['user_id'] = user_id * multi
all_num = len(user_id) * multi

for label,value  in user_fea_map.items():
    if label == 'KYC等级':
        array = [random.choice(KYC_level) for _ in range(all_num)]
        user_df[label] = array
    elif label == 'KYC国家':
        array = [random.choice(kyc_country) for _ in range(all_num)]
        user_df[label] = array
    
    else:
    
        if value[1] == 'float01':
            array = random_float_1(all_num)
            user_df[label] = array
        elif value[1] == 'float':
            array = random_float(all_num,1000000)
            user_df[label] = array
        elif value[1] == 'int':
            array = random_int(all_num, rank)
            user_df[label] = array

a1=(2021,5,1,0,0,0,0,0,0)        #设置开始日期时间元组（2021-05-01 00：00：00）
a2=(2021,6,30,23,59,59,0,0,0)    #设置结束日期时间元组（2021-12-31 23：59：59）
user_df['user_create_time'] = make_random_time(all_num, a1, a2, False)
user_df.to_csv('../data/user_fea.csv', mode='w', index=False)

######make item fea
multi = 1000
item_df = pd.DataFrame()
item_df['item_id'] = item_id * multi        
all_num = len(item_id) * multi

for label,value  in item_fea_map.items():
    if value[1] == 'float01':
        array = random_float_1(all_num)
        item_df[label] = array
    elif value[1] == 'float':
        array = random_float(all_num,1000000)
        item_df[label] = array
    elif value[1] == 'int':
        array = random_int(all_num, rank)
        item_df[label] = array
        
    elif value[1] == 'string_type':
        array = [random.choice(['101','100','111','102']) for _ in range(all_num)]
        item_df[label] = array
        
a1=(2021,5,1,0,0,0,0,0,0)        #设置开始日期时间元组（2021-05-01 00：00：00）
a2=(2021,6,30,23,59,59,0,0,0)    #设置结束日期时间元组（2021-12-31 23：59：59）
item_df['item_create_time'] = make_random_time(all_num, a1, a2, False)
item_df.to_csv('../data/item_fea.csv', mode='w', index=False)


######make interaction fea
multi = 100
interation_df = pd.DataFrame()
interation_df['item_id'] = item_id * multi
all_num = len(item_id) * multi
interation_df['user_id'] = [random.choice(user_id) for _ in range(all_num)]

for label,value  in interaction_fea_map.items():
    if value[1] == 'float01':
        array = random_float_1(all_num)
        interation_df[label] = array
    elif value[1] == 'float':
        array = random_float(all_num,1000000)
        interation_df[label] = array
    elif value[1] == 'int':
        array = random_int(all_num, rank)
        interation_df[label] = array
        
    elif value[1] == 'string_type':
        target = list(target_name.keys())
        array = [random.choice(target) for _ in range(all_num)]
        interation_df[label] = array
        
a1=(2021,5,1,0,0,1,0,0,0)        #设置开始日期时间元组（2021-05-01 00：00：00）
a2=(2021,6,30,23,59,59,0,0,0)    #设置结束日期时间元组（2021-12-31 23：59：59）
interation_df['interaction_create_time'] = make_random_time(all_num, a1, a2, False)
interation_df.to_csv('../data/interaction_fea.csv', mode='w', index=False)



    

