#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:10:27 2021

@author: liang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 15:47:24 2018

@author: ldk
"""

import traceback
from scipy import spatial


def ParserFunction(parser):
    parser.add_option("-A","--APPname",type = "string",dest="AppName",
                      help="required app name")    
    
    parser.add_option("--key_table",type = "string",dest="key_table",
                      help="required key_table")

    parser.add_option("--value_table",type = "string",dest="value_table",
                      help="required value_table")
    
    parser.add_option("--disMode",type = "string",dest="disMode",
                      help="required disMode")    
    
    parser.add_option("-o","--outtable",type = "string",dest="outputtable",
                      help="required outputtable")
    
    (options,args) = parser.parse_args()
    return options  

def saveMap(Map,outputpath):
    Map.saveAsTextFile(outputpath)

def saveasTable(df,outputtable):
    '''
    params:
        df：待保存的DataFrame
        outputtable :输出表hive表名
    '''    
    try:
        df.write.mode('overwrite').saveAsTable(outputtable)
    except Exception:
        traceback.print_exc()
        raise Exception('df save failed!')
    finally:
        print (" saved %s success"%outputtable)

def left_joinFunc(df1,df2,colname1,colname2):
    '''
    params:
        df1,df2：左右DataFrame
        colname1,colname2：左右表关联字段        
    '''    
    try:
        tmpdf = df1.join(df2, df1[colname1] == df2[colname2],'left').drop(df2[colname2])
    except Exception:
        traceback.print_exc()
        raise Exception('left join failed!')
    finally:
        return tmpdf

def distanceFunc(x,y,mode):
    '''
    params:
        x,y：向量x，y
        mode：距离计算方法        
    '''    
    try:
        if mode == 'eucla':
            #计算欧几里德距离：
            distance = spatial.distance.euclidean(x,y)
        elif mode == 'pearson':
            #计算皮尔逊相关度：
            distance = spatial.distance.correlation(x,y)
        
        elif mode == 'cosine':
            #计算余弦距离：
            distance = spatial.distance.cosine(x,y)
        
        elif mode == 'jaccard':
            #计算jaccard 距离：
            distance = spatial.distance.jaccard(x,y)
    except Exception:
        traceback.print_exc()
        raise Exception('cal dis failed!')
        
    return  distance  


def sortfunc(x):
    return sorted(x,key=lambda a:a[1])

def extractfunc(x):
    tmp =[]
    for i in x:
        tmp.append(i[0])
#   去重 
    extractlist = sorted(set(tmp),key = tmp.index)
    return extractlist

def filterFunc(item):
    '''限制区域过滤'''
    if item[1][0]==1 and item[3][0] ==1:
        return False
    #成年人限制
    elif item[1][1]==0 and item[3][1] ==1:
        return False
    else:
        return True

def calProDuctList(session,keyDF,valueDF,disMode):
    '''
    params:
        keyDF,valueDF:待推荐用户DF，产品信息DF
        mode：距离计算方法        
    ''' 
    createCombiner = (lambda a:a)
    mergeval = (lambda a,b:a+b)  
    mergecomb = (lambda a,b:a+b)
    
    keyRDD = keyDF.rdd.map(lambda x:(x[0],(x[1:])))
    valueRDD = valueDF.rdd.map(lambda x:(x[0],(x[1:])))
    
    newkeyDF=session.createDataFrame(keyRDD,["phone_no","usertaglist"])
    newvalueDF=session.createDataFrame(valueRDD,["pro_id","protaglist"])

    print (newkeyDF.show(1))
    print (newvalueDF.show(1))    
    
	#笛卡尔积
    try:
        joinDF = newkeyDF.crossJoin(newvalueDF)#,how="inner"
    except Exception:
        traceback.print_exc()
        raise Exception('joinDF failed!')
#    joinDF = keyDF.join(valueDF)
    
    print ("joinDF success")
    print (joinDF.show(1))
    
    #to joinRDD
    RDD = joinDF.rdd
    
    
    #条件过滤
    RDD = RDD.filter(lambda x:filterFunc(x))
    RDD.cache()
    
    #计算 disRDD
    #disRDD = joinRDD.map(lambda x:[x[0],x[3],caldis(x)] )
    try:
        RDD = RDD.map(lambda x:(x[0],[(x[2],distanceFunc(x[1],x[3],disMode))]))
        RDD = RDD.combineByKey(createCombiner,mergeval,mergecomb)
    except Exception:
        traceback.print_exc()
        raise Exception('combine RDD failed!')
    
    print(RDD.collect()[:1])    
    print("combine success")  
     
    #sort and extract
    try:
        RDD = RDD.map(lambda x:(x[0],sortfunc(x[1])))   
#        print RDD.collect()[:1]    
        print("sortfunc success" )  
        
        RDD = RDD.map(lambda x:(x[0],extractfunc(x[1])))
#        print RDD.collect()[:1]    
        print("extractfunc success")
        
    except Exception:
        traceback.print_exc()
        raise Exception('sort and extract failed!')
        
    print("disRDD_sort success")
    
    try:
        maplistDF = session.createDataFrame(RDD,['phone_no','recommendlist'])
    except Exception:
        traceback.print_exc()
        raise Exception('maplist df failed!')    
        
    print("maplist_df success")
    
    return maplistDF

def run(session,disMode,inputtable,outputtable):
    '''
    params:
        disMode:"eucla","cosin"...
        inputtable: dict
            user_table	:待推荐用户数据
            product_table	:行为轨迹标签hive表名
        outputtable: 输出表hive表名
        
    '''
    user_df = session.sql("select * from %s limit 20"%inputtable["user_table"])
    
    product_df = session.sql("select * from %s "%inputtable["product_table"])
    
    maplist_rdd = calProDuctList(session,user_df,product_df,disMode)
    
        
    saveasTable(maplist_rdd,outputtable)
    
        
if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    from pyspark.sql import SparkSession
    from optparse import OptionParser
    
    #parsering the args[]
    parser = OptionParser()
    options = ParserFunction(parser)

    #define sparksession    
    session = SparkSession.builder.appName(options.AppName)    
    session=session.enableHiveSupport()
    session=session.getOrCreate()
    
    inputtable = {'user_table' : options.key_table,
                  'product_table' : options.value_table}

    outputtable = options.outputtable
    disMode = options.disMode

    run(session,disMode,inputtable,outputtable)