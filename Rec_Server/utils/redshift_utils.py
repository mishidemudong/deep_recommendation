#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:15:51 2021

@author: liang
"""

import pandas as pd
import pandas_redshift as pr

from db_utils import dbconfig, s3config 

#config = {
#        'dbname' :'dev',
#        'host':'',
#        'port':5439,
#        'user':'',
#        'password':''
#            
#        }

pr.connect_to_redshift(**config)



save_df = pd.read_csv('../data/train_eng_fea.csv')

s3config = {
        'aws_access_key_id' : ,
        'aws_secret_access_key' : ,
        'bucket': ,
        'subdirectory' :
        }

# Connect to S3
pr.connect_to_s3(**s3config)

pr.pandas_to_redshift(data_frame = save_df,
                      redshift_table_name = 'train_eng_fea_test')

