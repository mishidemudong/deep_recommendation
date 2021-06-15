#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:01:24 2021

@author: liang
"""

'''
AWS personalize model
'''
import boto3
import json
import os

class PersonalizePredictHandler():
    """
    """

    def __init__(self, config_path):
        if os.path.exists(config_path):
            self.config = json.load(open(config_path, 'r'))
            self.campaignArn = self.config['campaignArn'] #'arn:aws:personalize:us-east-2:005527976057:campaign/movielens_campaign'
            self.personalizeRt = boto3.client('personalize-runtime')
        else:
            print('Build Personalize Handler Faild!!')

    def predict(self, user_list, item_list):

        res = []

        for user_id in user_list:
            result = {}
            result['user_id'] = user_id
            response = self.personalizeRt.get_personalized_ranking(
                campaignArn=self.campaignArn,
                userId=user_id,
                inputList=item_list,
            )

            result['item_score_list'] = [(it_id, index) for index, it_id in enumerate(response['personalizedRanking'])]
            result['model_type'] = 'personalize'
            res.append(result)

        return res