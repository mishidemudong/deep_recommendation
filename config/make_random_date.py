#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:09:09 2021

@author: liang
"""

import time
import random

#a1=(2021,5,1,0,0,0,0,0,0)              #设置开始日期时间元组（2021-05-01 00：00：00）
#a2=(2021,6,30,23,59,59,0,0,0)    #设置结束日期时间元组（2021-12-31 23：59：59）



def make_random_time(num, a1, a2, if_stamp):
    array = []
    start=time.mktime(a1)    #生成开始时间戳
    end=time.mktime(a2)      #生成结束时间戳
    #随机生成num个日期字符串
    for i in range(num):
        t=random.randint(start,end)    #在开始和结束时间戳中随机取出一个
        date_touple=time.localtime(t)          #将时间戳生成时间元组
        date=time.strftime("%Y-%m-%d %H:%M:%S",date_touple)  #将时间元组转成格式化字符串（1976-05-21）
        print(date)
        if if_stamp:
            timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
            timeStamp = int(time.mktime(timeArray))
            array.append(timeStamp)
        else:
            array.append(date)
        
    return array
        
if __name__ == "__main__":
    result = make_random_time(10, False)