#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:59:28 2021

@author: liang
"""



import random
import pandas as pd

from make_random_date import make_random_time
from tqdm import tqdm
import pickle
import os
import numpy as np
from multiprocessing import Pool
import time

fea_config = pickle.load(open('../fea_config.pkl', 'rb'))


sparse_features = fea_config['sparse_features']
dense_features = fea_config['dense_features']
target = fea_config['target']
task = fea_config['task']
target_name = fea_config['target_name']
item_fea_map = fea_config['item_fea_map']  
user_fea_map = fea_config['user_fea_map'] 
interaction_fea_map = fea_config['interaction_fea_map'] 


userid_df = pd.read_csv('../data/user_id.csv')[['user_id', 'u_id']]
user_id = list(userid_df['user_id'].unique())

item_df = pd.read_csv('../data/item_id.csv')['resource']
item_id = list(item_df.unique())
# user_id = ['user_{}'.format(str(no)) for no in range(50000)]
# item_id = ['item_{}'.format(str(no)) for no in range(300)]

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 

def left_joinFunc2(df1,df2,colname1,colname2):
    return df1.join(df2, df1[colname1] == df2[colname2],'left').drop(colname2, axis=1)

def left_joinFunc(df1,df2,colname):
    return pd.merge(df1, df2,how='left', on=colname)#

def findlatestandleftjoin(df1,df2,colname1,colname2):
    new_df = pd.DataFrame()

    for index, item in df1.iterrows():
        choose_list = df2[df2[colname1] == item[colname1] and df2[colname2] >= item[colname2]]
        latest_one = choose_list.sort_values(by=colname2)[0]

        join_df = left_joinFunc(item, latest_one, colname1)

        new_df = new_df.append(join_df)

    return new_df



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
#multi = 10
#enum_list = ['KYC等级', 'KYC国家']
#KYC_level = ['A','B','C']
#kyc_country = ['USA', 'JAPAN', 'ITELY', 'UK']
#if_test = [True, False]
#rank = 600
#user_df = pd.DataFrame()
#user_df['用户ID'] = user_id
#user_df = left_joinFunc2(user_df, userid_df, '用户ID', 'user_id')
#all_num = len(user_id)

#label = 'KYC等级'
#array = [random.choice(KYC_level) for _ in range(all_num)]
#user_df[label] = array
#label = 'KYC国家'
#array = [random.choice(kyc_country) for _ in range(all_num)]
#user_df[label] = array
#label = '测试用户'
#array = [random.choice(if_test) for _ in range(all_num)]
#user_df[label] = array
#user_df.to_csv('../data/user_base_info.csv', index=False)

#user_df = pd.read_csv('../data/user_base_info.csv')
##
#for label,value  in user_fea_map.items():
#    if value[1] == 'float01':
#        array = random_float_1(all_num)
#        user_df[label] = array
#    elif value[1] == 'float':
#        array = random_float(all_num,1000000)
#        user_df[label] = array
#    elif value[1] == 'int':
#        array = random_int(all_num, rank)
#        user_df[label] = array
##
#a1=(2021,5,1,0,0,0,0,0,0)        #设置开始日期时间元组（2021-05-01 00：00：00）
#a2=(2021,6,30,23,59,59,0,0,0)    #设置结束日期时间元组（2021-12-31 23：59：59）
#user_df['user_create_time'] = make_random_time(all_num, a1, a2, False)
#user_df.to_csv('../data/user_fea.csv', mode='w', index=False)
#eng = user_df.rename(columns=fea_config['map_eng_name'], inplace=False) 
#if os.path.exists('../data/user_eng_fea.csv'):
#    eng.to_csv('../data/user_eng_fea.csv', mode ='a',header=False, index= False)
#else:    
#    eng.to_csv('../data/user_eng_fea.csv', mode='w', index=False)
#    
#######make item fea
#multi = 1000
#item_df = pd.DataFrame()
#item_df['物品ID'] = item_id * multi        
#all_num = len(item_id) * multi
#
#for label,value  in item_fea_map.items():
#    if value[1] == 'float01':
#        array = random_float_1(all_num)
#        item_df[label] = array
#    elif value[1] == 'float':
#        array = random_float(all_num,1000000)
#        item_df[label] = array
#    elif value[1] == 'int':
#        array = random_int(all_num, rank)
#        item_df[label] = array
#        
#    elif value[1] == 'string_type':
#        array = [random.choice(['101','100','111','102']) for _ in range(all_num)]
#        item_df[label] = array
##        
#a1=(2021,5,1,0,0,0,0,0,0)        #设置开始日期时间元组（2021-05-01 00：00：00）
#a2=(2021,6,30,23,59,59,0,0,0)    #设置结束日期时间元组（2021-12-31 23：59：59）
#item_df['item_create_time'] = make_random_time(all_num, a1, a2, False)
#item_df.to_csv('../data/item_fea.csv', mode='w', index=False)
#eng = item_df.rename(columns=fea_config['map_eng_name'], inplace=False) 
#if os.path.exists('../data/item_eng_fea.csv'):
#    eng.to_csv('../data/item_eng_fea.csv', mode ='a',header=False, index= False)
#else:    
#    eng.to_csv('../data/item_eng_fea.csv', mode='w', index=False)
#
#
######make interaction fea
multi = 100
interation_df = pd.DataFrame()
#interation_df['物品ID'] = item_id * multi
#all_num = len(item_id) * multi
#interation_df['用户ID'] = [random.choice(user_id) for _ in range(all_num)]
user_id_array = []
item_id_array = []
for user in tqdm(user_id):
    item_id_array.extend(item_id)
    user_id_array.extend([user] * len(item_id))
interation_df['用户ID'] = user_id_array
interation_df['物品ID'] = item_id_array        
all_num = len(item_id_array)    


for label,value in tqdm(interaction_fea_map.items()):

    if value[1] == 'float01':
        array = random_float_1(all_num)
        interation_df[label] = array
    elif value[1] == 'float':
        array = random_float(all_num,1000000)
        interation_df[label] = array

    elif 'int' in value[1]:
        array = random_int(all_num, int(value[1][-2:]))
        interation_df[label] = array
        
    elif value[1] == 'string_type':
        target = list(target_name.keys())
        array = [random.choice(target) for _ in range(all_num)]
        interation_df[label] = array
        
a1=(2021,5,1,0,0,1,0,0,0)        #设置开始日期时间元组（2021-05-01 00：00：00）
a2=(2021,6,30,23,59,59,0,0,0)    #设置结束日期时间元组（2021-12-31 23：59：59）
interation_df['interaction_create_time'] = make_random_time(all_num, a1, a2, False)
interation_df = interation_df[['用户ID','物品ID','交互类型','物品所在顺序','历史点击次数','历史购买次数','interaction_create_time']]
interation_df.to_csv('../data/interaction_fea.csv', mode='w', index=False)
eng = interation_df.rename(columns=fea_config['map_eng_name'], inplace=False) 
#if os.path.exists('../data/interaction_eng_fea.csv'):
#    eng.to_csv('../data/interaction_eng_fea.csv', mode ='a',header=False, index= False)
#else:    
eng.to_csv('../data/interaction_eng_fea.csv', mode='w', index=False)
#
#
#
#user_df = pd.read_csv('../data/user_fea.csv')
#userbase_df = pd.read_csv('../data/user_id.csv')[['user_id', 'u_id']]

#user_df = left_joinFunc2(user_df, userbase_df, '用户ID', 'u_id')


#item_df = pd.read_csv('../data/item_fea.csv')
#interation_df = pd.read_csv('../data/interaction_fea.csv')
#train_df = left_joinFunc(interation_df, user_df, '用户ID').sample(1000)
##train_df = left_joinFunc(interation_df, user_df)
#
#colname1 = '物品ID'
#colname2 = 'item_create_time'
#
#t1= time.time()
#new_df = pd.DataFrame()
#for index, item in tqdm(train_df.iterrows()):
#    choose_list = item_df[item_df[colname1] == item[colname1]]
#    choose_list = choose_list[choose_list['item_create_time'] <= item['interaction_create_time']]
#    
#    if choose_list.shape[0] > 0:
#        latest_one = choose_list.sort_values(by=colname2).head(1)
#    
#        join_dict = Merge(item.to_dict(), latest_one.to_dict())
#    
#        new_df = new_df.append(pd.DataFrame.from_dict(join_dict))
#t2 = time.time()
#print("Serial time =",t2-t1)
#print(new_df.head())
#    
#new_df.to_csv('../data/train_fea.csv', mode ='w',index= False)
#new_df.rename(columns=fea_config['map_eng_name'], inplace=True) 
#if os.path.exists('../data/train_eng_fea.csv'):
#    new_df.to_csv('../data/train_eng_fea.csv', mode ='a',header=False, index= False)
#else:
#    new_df.to_csv('../data/train_eng_fea.csv', mode ='w', index= False)
#
#def find_func(dist_df):
#    new_df = pd.DataFrame()
#
#    for index, item in tqdm(dist_df.iterrows()):
#        choose_list = item_df[item_df[colname1] == item[colname1]]
#        choose_list = choose_list[choose_list['item_create_time'] <= item['interaction_create_time']]
#        
#        if choose_list.shape[0] > 0:
#            latest_one = choose_list.sort_values(by=colname2).head(1)
#        
#            join_dict = Merge(item.to_dict(), latest_one.to_dict())
#        
#            new_df = new_df.append(pd.DataFrame.from_dict(join_dict))
#            
#    return new_df
        

#
#df_parts=np.array_split(train_df,20)
#print(len(df_parts),type(df_parts[0]))
## with Pool(processes=8,initializer=init_process,initargs=(a,)) as pool:  
#with Pool(processes=4) as pool:        
#    result_parts = pool.map(find_func,df_parts)
#    # pool.map(MainRange,df_parts)
#result_parallel= pd.concat(result_parts)
#t3 = time.time()
#print("Parallel time =",t3-t2)
#print(result_parallel.head())
#
#
#result_parallel.to_csv('../data/train_fea.csv', mode ='w',index= False)
#result_parallel.rename(columns=fea_config['map_eng_name'], inplace=True) 
#if os.path.exists('../data/train_eng_fea.csv'):
#    result_parallel.to_csv('../data/train_eng_fea.csv', mode ='a',header=False, index= False)
#else:
#    result_parallel.to_csv('../data/train_eng_fea.csv', mode ='w', index= False)