#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:01:24 2021

@author: liang
"""

'''
AWS personalize model
'''

class PersonalizePredictHandler():
    """
    """
    def __init__(self, config_path):
        
        
        print("build model success")
    
        
    def predict(self, user_list, pred_data):
        
        res = []
        
        for user_id in user_list:
            result = {}
            result['user_id'] = user_id
            data = pred_data[pred_data['user_id'] == user_id]
            test_model_input = self.preprocess(data)
#            print(len(test_model_input))
            pred = self.model.predict(test_model_input, batch_size=256)
#            print(pred)
#            result['item_score_list'] = {it_id:str(score[0]) for it_id, score in zip(pred_data['item_id'], pred)}
            result['item_score_list'] = [(it_id,str(score[0])) for it_id, score in zip(pred_data['item_id'], pred)]
            
            result['model_type'] = 'personalize'
            res.append(result)
                
        return res