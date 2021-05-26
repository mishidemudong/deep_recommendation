#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:28:29 2021

@author: liang
"""

#import pandas as pd
#df = pd.read_csv('./interaction_fea.csv')
#iteraction_fea = {}
#for key, value in zip(df['字段名称'], df['英文标识']):
#    iteraction_fea[key] = value

item_fea = {'物品ID': 'item_id',
             '物品类别': 'item_type',
             '24 小时趋势': '24Htrend',
             '24 小时内上涨下跌比例': '24Htrendnum',
             '24 小时内上涨下跌排名': '24Htrendindex',
             '一周趋势': '1Wtrend',
             '一周内上涨下跌比例': '1Wtrendnum',
             '一周内上涨下跌排名': '1Wtrendindex',
             '市值': 'marketcap',
             '市值排名': 'marketcapindex',
             '24 小时内交易量': '24Hvolume',
             '一周内交易量': '1Wvolume',
             '24 小时内搜索排名': '24Hsearchindex',
             '24 小时内搜索排名变化': '24Hsearchchange',
             '一周内搜索排名': '1Wsearchindex',
             '一周内搜索排名变化': '1Wsearchchange',
             '归属的概念板块': 'category',
             '关注量': 'follownum',
             '关注量排行': 'followindex',
             '关注量排行变化': 'followindexchange',
             '三月内趋势': '3Mtrend',
             '三月内上涨下跌比例': '3Mtrendnum',
             '三月内上涨下跌排名': '3Mtrendindex',
             '成分币种数量': 'compositionnum',
             '详情描述': 'description',
             '评论内容': 'comment',
             '创建时间':'create_time'
        }
item_table_name = ''


user_fea = {
         '测试用户': 'if_test',
         '用户长id': 'user_id?',
         '用户短id': '?',
         'KYC等级': 'kyc_level',
         'KYC国家': 'kyc_country',
         '注册时间': 'register_time',
         '30天持仓均值': '30days_avg_holdshares',
         '30天手续费收入': '30days_fee',
         '最后一次交易时间间隔': 'interval_of_last_trade',
         '60天内交易次数': 'tradetimes_in_60days',
         '60天内交易金额': 'trademoney_in_60days',
         '创建时间':'create_time'
        }
user_table_name = ''


iteraction_fea = {
        '用户ID': 'user_id',
         '物品ID': 'item_id',
         '交互类型': 'interaction_type',
         '交互时间': 'interaction_time',
         '物品所在顺序': 'item_order_index',
         '历史点击次数': 'click_times',
         '历史购买次数': 'purchase_times',
         '历史交互记录': 'history_interaction_list',
         '上次点击时间': 'lasttime_click_date',
         '上次购买时间': 'lasttime_purchase_date',
         '上次卖出时间': 'lasttime_sold_date',
         '上次评论时间': 'lasttime_comment_date',
         '评论内容': 'comment',
         '创建时间':'create_time'
        }
iteraction_table_name = ''
