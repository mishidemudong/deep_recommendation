#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:21:50 2021

@author: liang
"""

import pandas as pd
from main_mock_data import *

user_df = pd.read_csv('../data/user_fea.csv')
item_df = pd.read_csv('../data/item_fea.csv')
interation_df = pd.read_csv('../data/interaction_fea.csv')
train_df = left_joinFunc(interation_df, user_df, '用户ID')
#train_df = left_joinFunc(interation_df, user_df)

colname1 = '物品ID'
colname2 = 'item_create_time'

new_df = pd.DataFrame()

for index, item in tqdm(train_df.iterrows()):
    choose_list = item_df[item_df[colname1] == item[colname1]]
    choose_list = choose_list[choose_list['item_create_time'] <= item['interaction_create_time']]
    
    if choose_list.shape[0] > 0:
        latest_one = choose_list.sort_values(by=colname2).head(1)
    
        join_dict = Merge(item.to_dict(), latest_one.to_dict())
    
        new_df = new_df.append(pd.DataFrame.from_dict(join_dict))
        
new_df.to_csv('../data/train_fea.csv', mode ='w',index= False)
new_df.rename(columns=fea_config['map_eng_name'], inplace=True) 
if os.path.exists('../data/train_eng_fea.csv'):
    new_df.to_csv('../data/train_eng_fea.csv', mode ='a',header=False, index= False)
else:
    new_df.to_csv('../data/train_eng_fea.csv', mode ='w', index= False)

df_parts=np.array_split(trained_df,20)
print(len(df_parts),type(df_parts[0]))
# with Pool(processes=8,initializer=init_process,initargs=(a,)) as pool:  
with Pool(processes=8) as pool:        
    result_parts = pool.map(MainRange,df_parts)
    # pool.map(MainRange,df_parts)
for item in result_parts:
    train_token_ids.extend(item)