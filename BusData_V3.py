#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022.03.12

@author: Da Yezi
"""
import re
import numpy as np
import pandas as pd
import math
from geopy.distance import geodesic
import os
import time
import geopandas as gpd
from shapely.geometry import LineString
from scipy.spatial import cKDTree

PI=math.pi
a = 6378245.0
ee = 0.00669342162296594323

#获取目录下 文件名
def file_name(user_dir):
    file_list=list()
    for root,dirs,files in os.walk(user_dir):
        for file in files:
            if file.split('.')[-1]=='csv':
                file_list.append(os.path.join(root,file))
    return file_list

#%%时间操作

#时间转为等长字符串
def datetime_str(x): 
    a = [m for m in re.split(r'/| |:',x)] #根据时间字段可能包括的分隔符，对字符串进行分割
    return '-'.join([a[2], a[1].rjust(2,'0'), a[0].rjust(2,'0'), a[3].rjust(2,'0'), a[4].rjust(2,'0'), a[5].rjust(2,'0')])

#计算同一个月内的时间差值
def dt_time(x,y):
    dt = x.str.slice(8,10).astype('float')*3600*24+\
    x.str.slice(11,13).astype('float')*3600+\
    x.str.slice(14,16).astype('float')*60+\
    x.str.slice(17,19).astype('float')-\
    y.str.slice(8,10).astype('float')*3600*24-\
    y.str.slice(11,13).astype('float')*3600-\
    y.str.slice(14,16).astype('float')*60-\
    y.str.slice(17,19).astype('float')
    return dt

#%%坐标操作

#坐标转换
def coordinate_transto_gcj(data):
    df=addr_gcj(data)
    return df
  
#添加新的坐标定位
def addr_gcj(data):
    lon=list(data.loc[:,"LONGITUDE"])
    lat=list(data.loc[:,"LATITUDE"])
    addr=[[lat[i],lon[i]] for i in range(data.shape[0])]
    temp=np.array(wgs_to_gcj(addr)).T
    data.loc[:,"lat"]=temp[0]
    data.loc[:,"lon"]=temp[1]
    return data

#定义函数进行坐标系转换  WGS-84—>GCJ-02
def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0
    return ret

def transform_lon(x, y):

    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0
    return ret

def wgs_to_gcj(addr_list):
    gcj_addr=[]
    for i in range(len(addr_list)):
        lat=addr_list[i][0]
        lon=addr_list[i][1]
        
        dLat = transform_lat(float(lon) - 105.0, float(lat) - 35.0)
        dLon = transform_lon(float(lon) - 105.0, float(lat) - 35.0)
        
        radLat = lat / 180.0 * PI
        
        magic = math.sin(radLat)
        magic = 1 - ee * magic * magic
        sqrtMagic = math.sqrt(magic)
        
        dLat = (dLat * 180.0)/((a * (1 - ee))/(magic * sqrtMagic) * PI)
        dLon = (dLon * 180.0)/(a/sqrtMagic * math.cos(radLat) * PI)
        mgLat = lat + dLat
        mgLon = lon + dLon
        gcj_addr.append([mgLat,mgLon])
    return gcj_addr 

#每个站点与该GPS点的距离
def stat_match(lat1,lon1, data_route):
    data_route['distance'] = data_route.apply(lambda x: get_distance(lat1,lon1,x['lat'],x['lon']),axis=1)

    #返回最小距离的index
    min_distance_idx = data_route['distance'].idxmin()
    #最小距离是否满足筛选条件
    if data_route.loc[min_distance_idx,'distance']<=100:
        #满足条件，则匹配到站点名称
        return data_route.loc[min_distance_idx,['站名','上下行','线路']]
    else:
        #不满足条件，则返回0
        return pd.DataFrame({'站名':['0'],'上下行':['0'],'线路':['0']})

#%% 距离计算
def get_distance(lat1,lon1,lat2,lon2):
    return geodesic((lat1,lon1),(lat2,lon2)).m

def getdistance(lat1,lon1,lat2,lon2):
    return [geodesic((lat1[i],lon1[i]),(lat2[i],lon2[i])).m for i in range(len(lat1))]

#%% ckdtree
R = 6367
N = 10000

def changedata(data):
    # numpy弧度制和角度制转换deg2rad, rad2deg
    phi = np.deg2rad(data[:, 1])  # LAT 
    theta = np.deg2rad(data[:, 0])  # LON
    # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等.
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等.
    data = np.c_[data, R * np.cos(phi) * np.cos(theta), R * np.cos(phi) * np.sin(theta), R * np.sin(phi)]
    return data

# Convert Euclidean chord length to great circle arc length 将欧几里得弦长度转换为大圆弧长度
def dist_to_arclength(chord_length):
    central_angle = 2 * np.arcsin(chord_length / (2.0 * R))
    arclength = R * central_angle
    return arclength

def using_kdtree(ref_data, que_Data):
    ref_data = changedata(ref_data)
    que_Data = changedata(que_Data) #转为弧度制
    tree = cKDTree(ref_data[:, 2:5]) #生成tree
    distance, index = tree.query(que_Data[:, 2:5])  #获取最近节点的距离，和索引
    return dist_to_arclength(distance), index

#%% 数据预处理

#公交车gps数据预处理
def bus_gps_preprocess(data):
    columns = ['ROUTEID', 'PRODUCTID', 'STATIONSEQNUM','ISARRLFT', 
                'ACTDATETIME','LONGITUDE','LATITUDE','GPSSPEED']  
    data = data[columns] #取有用字段

    #ROUTEID字段 筛选线路号113
    data = data[data['ROUTEID'].isin([113])]

    data['ISARRLFT'] = data['ISARRLFT'].fillna(5).copy() #填充缺失的到离站信息
    data.dropna(how='any',inplace=True)  #其他字段存在缺失的，删除该行
    data.reset_index(drop=True,inplace=True) #重置索引

    #将时间处理为等长字符串，0-1为天，3-4为月，6-9为年，11-12为小时，14-15为分钟，17-18为秒
    data['ACTDATETIME'] = data['ACTDATETIME'].apply(datetime_str)

    #数据去重,保留重复的最后一行数据
    data.drop_duplicates(subset=['PRODUCTID','ACTDATETIME'],keep='last',inplace=True)
    data.reset_index(drop=True,inplace=True)
    data.sort_values(by = ['PRODUCTID','ACTDATETIME'],inplace = True)
    return data

#公交车站点数据预处理
def route_preprocess(data):
    data.rename(columns={'经度':'station_lon','纬度':'station_lat',
    '站名':'station_name','上下行':'up_down_line','线路':'line_id'},inplace=True)
    print(f'公交车线路数据当前字段名：{data.columns}')
    return data

#%% 轨迹点与站点信息匹配
def gps_station_match(data,route):
    
    #提取经纬度信息
    Aname = ['LONGITUDE','LATITUDE']
    Bname = ['station_lon','station_lat']
    bus_gps = data[Aname].values
    station_points = route[Bname].values

    dist_ckdn, indexes_ckdn = using_kdtree(station_points,bus_gps) #省略kdtree的距离

    data['stop_index'] = indexes_ckdn  #每个gps点都对应其最近站点在树中的索引
    data['dist'] = dist_ckdn* 1000
    route['stop_index'] = range(len(route))  #站点树的索引
    gdf = pd.merge(data,route,on = 'stop_index')  #将gps点与其最近站点的所有信息进行匹配

    gdf = gdf[gdf['dist']<=100]  #筛选与站点距离100m以内的轨迹点
    gdf.reset_index(drop=True,inplace=True)
    gdf.sort_values(by=['PRODUCTID','ACTDATETIME'],inplace=True)

    col = ['PRODUCTID','up_down_line','stop_index']
    gdf1 = gdf.groupby(col).agg({'dist':'min'}).reset_index()
    gdf = pd.merge(gdf,gdf1,on =  ['PRODUCTID','up_down_line','stop_index','dist'])
    gdf.drop_duplicates(subset={'PRODUCTID','up_down_line','stop_index','dist'},keep='last',inplace=True)
    gdf.reset_index(drop=True,inplace=True)
    gdf.sort_values(by=['PRODUCTID','ACTDATETIME'],inplace=True)

    gdf['index1'] = gdf['stop_index'].shift(1)
    gdf['PRODUCTID1'] = gdf['PRODUCTID'].shift(1)    
    gdf.dropna(how='any',inplace=True)
    gdf.reset_index(drop=True,inplace=True)

    #本站与上一站连续
    gdf['last_stop'] = gdf['station_name'].shift(1)
    gdf['last_stop_time'] = gdf['ACTDATETIME'].shift(1)    
    gdf = gdf[((gdf['stop_index']-gdf['index1'])==1) & ((gdf['PRODUCTID']-gdf['PRODUCTID1'])==0)]
    gdf.dropna(how='any',inplace=True)
    gdf.reset_index(drop=True,inplace=True)
    #计算连续站点的行程时间
    gdf['travel_time'] = dt_time(gdf['ACTDATETIME'],gdf['last_stop_time'])
    gdf['currstop-laststop'] = gdf.apply(lambda x: '-'.join([x['last_stop'],x['station_name']]),axis=1)
    
    gdf.drop(['index1','PRODUCTID1'],axis=1, inplace=True)

    return gdf

#%% 主函数

def main():

    data_path = r"D:\研究生\研究\公交线网优化\全量数据\公交GPS全量数据\公交GPS数据_1.csv"
    route_path = r"D:\研究生\研究\公交线网优化\全量数据\公交线路信息.csv"

    print("读取gps数据中……")
    data = pd.read_csv(data_path,sep=',',encoding='utf-8_sig')
    print("读取线路数据中……")
    route = pd.read_csv(route_path,sep=',',encoding='utf-8_sig')

    print("gps数据预处理……")
    data = bus_gps_preprocess(data)
    print("线路数据预处理……")
    route = route_preprocess(route)

    #站点匹配
    print('站点匹配中……')
    start = time.time()
    gdf = gps_station_match(data,route)
    end = time.time()
    print(f'匹配用时：{end-start}')
    return gdf

if __name__ == '__main__':
    gdf = main()
    gdf.to_csv('test.csv',sep=',', index =False, encoding='utf-8_sig')