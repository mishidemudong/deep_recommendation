#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:59:28 2021

@author: liang
"""
import os
import pandas_redshift as pr
import pandas as pd

pr.connect_to_redshift(dbname = 'dev',
                        host = 'redshift-cluster-machinelearning.coyzrp7jdx52.ap-northeast-1.redshift.amazonaws.com',
                        port = 5439,
                        user = 'awsuser',
                        password = 'Kucoin2021rec')


###fea map
item_fea_df = pd.read_csv('./feamap/item_map.csv')
item_fea_map = {}
for key, value, value2 in zip(item_fea_df['label_name'], item_fea_df['label_code'], item_fea_df['data_type']):
    item_fea_map[key] = (value.upper(), value2)


user_fea_df = pd.read_csv('./feamap/user_map.csv')
user_fea_map = {}
for key, value, value2 in zip(user_fea_df['label_name'], user_fea_df['label_code'], user_fea_df['data_type']):
    user_fea_map[key] = (value.upper(), value2)


interaction_fea_df = pd.read_csv('./feamap/interaction_map.csv')
interaction_fea_map = {}
for key, value, value2 in zip(interaction_fea_df['label_name'], interaction_fea_df['label_code'], interaction_fea_df['data_type']):
    interaction_fea_map[key] = (value.upper(), value2)
    
target_name = {
            '曝光':('SHOW',0),
            '点击':('CLICK',1),
            '关注':('FOLLOW',2),
            '收藏':('FAVOURATE',2),
            '评论':('REVIEW',2),
            '购买':('BUY_CLICK',3)
        }    

map_eng_name = {}
for k,v in item_fea_map.items():
    k,value = k.strip(), v[0].strip()
    map_eng_name[k] = value
    
for k,v in user_fea_map.items():
    k,value = k.strip(), v[0].strip()
    map_eng_name[k] = value
    
for k,v in interaction_fea_map.items():
    k,value = k.strip(), v[0].strip()
    map_eng_name[k] = value



##user data

user_table_name = 'dev.algorithm_kucoin_dw_v2_app.app_user_profile_s1_daily'

print("start read user data from redshift...")
user_df = pr.redshift_to_pandas("select user_id,uid,test_user,kyc_level,kyc_region,register_date,last_30_avg_balance_usdt,last_30_fee_usdt,last_trade_interval,last_60_trade_cnt,last_60_trade_funds_usdt from {} where dt>='2021-07-18' ".format(user_table_name))

user_df.columns = user_df.columns.map(lambda x:x.upper())
if os.path.exists('./data/user_eng_fea.csv'):
    user_df.to_csv('./data/user_eng_fea.csv', mode ='a',header=False, index= False)
else:    
    user_df.to_csv('./data/user_eng_fea.csv', mode='w', index=False)
print("read user data from redshift done!")


# ##item data
item_fea = ['uid', 'create_time'] + [v[0] for k,v in item_fea_map.items()].remove('ITEM_CATEGORY') 

item_table_name = 'dev.algorithm_kucoin_event.label_v3_item_log_30051441'

item_df = pr.redshift_to_pandas("select {} from {} where create_time>='2021-06-01' ".format(','.join(item_fea), item_table_name))

item_df.columns = user_df.columns.map(lambda x:x.upper())
if os.path.exists('./data/item_eng_fea.csv'):
    item_df.to_csv('./data/item_eng_fea.csv', mode ='a',header=False, index= False)
else:    
    item_df.to_csv('./data/item_eng_fea.csv', mode='w', index=False)
print("read item data from redshift done!")


# ## interaction data
iteraction_table_name = 'dev.algorithm_kucoin_label.label_v3_interaction'
iteraction_fea = ['uid', 'create_time'] + [v[0] for k,v in item_fea_map.items()].remove('ITEM_CATEGORY') 

iteraction_df = pr.redshift_to_pandas("select {} from {} where create_time>='2021-06-01' ".format(','.join(iteraction_fea), iteraction_table_name))

iteraction_df.columns = iteraction_df.columns.map(lambda x:x.upper())
if os.path.exists('./data/iteraction_eng_fea.csv'):
    iteraction_df.to_csv('./data/iteraction_eng_fea.csv', mode ='a',header=False, index= False)
else:    
    iteraction_df.to_csv('./data/iteraction_eng_fea.csv', mode='w', index=False)
print("read iteraction data from redshift done!")




def find_func(dist_df):
    new_df = pd.DataFrame()

    for index, item in tqdm(dist_df.iterrows()):
        choose_list = item_df[item_df[colname1] == item[colname1]]
        choose_list = choose_list[choose_list['item_create_time'.upper()] <= item['interaction_create_time'.upper()]]
        
        if choose_list.shape[0] > 0:
            latest_one = choose_list.sort_values(by=colname2).head(1)
        
            join_dict = Merge(item.to_dict(), latest_one.to_dict())
        
            new_df = new_df.append(pd.DataFrame.from_dict(join_dict))
            
    return new_df
        


df_parts=np.array_split(train_df,20)
print(len(df_parts),type(df_parts[0]))
# with Pool(processes=8,initializer=init_process,initargs=(a,)) as pool:  
with Pool(processes=4) as pool:        
    result_parts = pool.map(find_func,df_parts)
    # pool.map(MainRange,df_parts)
result_parallel= pd.concat(result_parts)
t3 = time.time()
print("Parallel time =",t3-t1)
print(result_parallel.head())


result_parallel.to_csv('../data/train_fea.csv', mode ='w',index= False)
result_parallel.rename(columns=fea_config['map_eng_name'], inplace=True) 
if os.path.exists('../data/train_eng_fea.csv'):
    result_parallel.to_csv('../data/train_eng_fea.csv', mode ='a',header=False, index= False)
else:
    result_parallel.to_csv('../data/train_eng_fea.csv', mode ='w', index= False)