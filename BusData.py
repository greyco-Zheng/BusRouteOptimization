import re
import numpy as np
import pandas as pd
import math
from geopy.distance import geodesic
import os

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

# #计算经纬度点的距离
def get_distance(lat1,lon1,lat2,lon2):
    return geodesic((lat1,lon1),(lat2,lon2)).m

#每个站点与该GPS点的距离
def stat_match(lat1,lon1, data_route):
    data_route['distance'] = data_route.apply(lambda x: get_distance(lat1,lon1,x['LATITUDE'],x['LONGITUDE']),axis=1)

    #返回最小距离
    min_distance_idx = data_route['distance'].idxmin(axis = 0)
    #最小距离是否满足筛选条件
    if data_route.loc[min_distance_idx,'distance']<=100:
        #满足条件，则匹配到站点名称
        return data_route.loc[min_distance_idx,['站名','上下行','线路']]
    else:
        #不满足条件，则返回0
        return pd.DataFrame({'站名':['0'],'上下行':['0'],'线路':['0']})


#时间转为等长字符串
def datetime_str(x): 
    a = x.split('-')
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

def Bus(data,data_route):
    columns = ['ROUTEID', 'PRODUCTID', 'STATIONSEQNUM','ISARRLFT', 
               'ACTDATETIME','LONGITUDE','LATITUDE','GPSSPEED']
    data = data[columns]
    data.dropna(how='any',inplace=True)
    data.reset_index(drop=True,inplace=True)
    # data = coordinate_transto_gcj(data)   #坐标系转换 lon lat


    #将时间处理为等长字符串，0-1为天，3-4为月，6-9为年，11-12为小时，14-15为分钟，17-18为秒
#     data['ACTDATETIME'] =  pd.to_datetime(data['ACTDATETIME'])
    data['ACTDATETIME'] = data['ACTDATETIME'].apply(lambda x: '-'.join([m.rjust(2,'0') for m in re.split(r'/| |:',x)]))
    data = data.loc[(data['ACTDATETIME'].str.slice(8,10)!='')
                    &(data['ACTDATETIME'].str.slice(11,13)!='')
                    &(data['ACTDATETIME'].str.slice(14,16)!='')
                    &(data['ACTDATETIME'].str.slice(17,19)!=''),:] 
    data['ACTDATETIME'] = data['ACTDATETIME'].apply(datetime_str)  #调整为正常顺序 
    #数据去重,保留重复的最后一行数据
    data.drop_duplicates(subset=['PRODUCTID','ACTDATETIME'],keep='last',inplace=True)   
    data.sort_values(by = ['PRODUCTID','ACTDATETIME'],inplace = True)       

    ###整理为 到离站数据
    #ROUTEID字段 筛选线路号122,113，
    #ISARRLFT字段 筛选到站、离站的数据 (1-到站，2-离站)
    #PRODUCTID1 保证为同一辆车的到离站
    data = data[(data['ROUTEID'].isin([113])) & (data['ISARRLFT'].isin([1,2]))]
    data.reset_index(drop=True,inplace=True)

    data_route.rename(columns={'经度':'LONGITUDE','纬度':'LATITUDE'},inplace=True)
    data_route = coordinate_transto_gcj(data_route)   #坐标系转换 lon lat

    #先按照3位小数初步筛选GPS点
    data['LON'] = round(data['LONGITUDE'],3).copy()
    data['LAT'] = round(data['LATITUDE'],3).copy()
    data_route['LON'] = round(data_route['LONGITUDE'],3).copy()
    data_route['LAT'] = round(data_route['LATITUDE'],3).copy()

    data_new = pd.merge(data,data_route,on=['LON','LAT'])
    print(data_new.head())
    data_new = data_new[['PRODUCTID','ACTDATETIME','ISARRLFT','LATITUDE_x','LONGITUDE_x']]
    data_new.rename(columns={'LATITUDE_x':'LATITUDE','LONGITUDE_x':'LONGITUDE'},inplace=True)

    #按照距离精细匹配
    data_new[['站名','上下行','线路']] = data_new[['LATITUDE','LONGITUDE']].apply(lambda x: stat_match(x['LATITUDE'], x['LONGITUDE'],data_route),axis=1)  #为每一个GPS点 匹配站点
    data_new = data_new[data_new['站名']!='0']
    data_new.reset_index(drop=True,inplace=True)
    data_new.drop_duplicates(subset={'PRODUCTID','ACTDATETIME'},keep='last',inplace=True)
    
    data_new.sort_values(by=['PRODUCTID','上下行','ACTDATETIME'],inplace=True)
    data_new.reset_index(drop=True,inplace=True)
    

    return data_new,data_route


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


def main():
    folder = r"D:\研究生\研究\公交线路优化\聊城2-创意赛-公交线网优化\公交线网优化\全量数据"
    IC_path = folder+"\\IC交易数据.csv"
    BusGPS_paths = file_name(folder+"\\公交GPS全量数据")
    BusRoute_path = folder+"\\公交线路信息.csv"
    Station_path = folder+"\\公交站点分布.csv"

    #读取IC卡交易数据，routecode字段筛选线路13，22
    data_IC = pd.read_csv(IC_path,sep=',')
    data_IC = data_IC[data_IC['routecode'].isin([13,22])]
    data_IC.reset_index(drop=True,inplace=True)

    #读取公交车线路信息（13路，包括站点名称、上下行、经纬度）
    data_route = pd.read_csv(BusRoute_path, sep=',')

    #读取公交车站点分布（好像没啥用）
    data_stat = pd.read_csv(Station_path, sep=',')

    for i in range(1):
        #读取公交车轨迹数据，选取有用列
        data = pd.read_csv(BusGPS_paths[i],sep=',',nrows=10000)
        data1,route1=Bus(data,data_route)
    print(data1.head(20))

if __name__ == "__main__":
    main()

