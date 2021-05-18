#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:00:24 2021

@author: liang
"""
#############################
#task = 'regression'
#sparse_features = ["movie_id", "user_id",
#                   "gender", "age", "occupation", "zip"]
#target = ['rating']


############################
task = 'regression' #regression
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]