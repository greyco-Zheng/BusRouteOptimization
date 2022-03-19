#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022.03.19
第四版说明：对站点名称及经纬度进行补充，原站点位置与水城通e行上的站点存在偏差，
down_line表示上下行=1的列，站北花园→聊大花园，共计28站；
up_line表示上下行=2的列，聊大花园→站北花园，共计29站，新增闸口一站；
考虑到站点匹配中若同时用上下行的经纬度坐标，容易出现上下行改变的情况，
因此本版本只考虑用up_line的经纬度坐标进行匹配,stop_index = range(29)
后续再区分上下行的方法：data.direction = data.stop_index - data.stop_index(-1) = -1 属于up_line中stop_index到下一站，同理计算行程时间
data.direction = data.stop_index - data.stop_index(-1) = 1  属于down_line中stop_index到下一站，同理计算行程时间
行程时间分布：groupby(stop_index + direction)按站台和上下行进行聚合

generate_travel_time函数：将gps文件夹下的所有数据进行站点匹配，并计算相邻站点的行程时间
week_hour函数：添加两列标签，weekday标签代表一周中的第几天（0-6），hour代表第几个小时（0-23）
distribution函数：对特定站点的行程时间进行分布拟合，并输出最佳分布参数（没有考虑时间维度）
draw_plot函数：画箱型图
draw_tarvel_time函数：画出每个站点，每个上/下行，在工作日、休息日、总体场景下的行程时间分布
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
R = 6371
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
    data['ISARRLFT'] = data['ISARRLFT'].copy().fillna(5) #填充缺失的到离站信息
    data.dropna(how='any',inplace=True)  #其他字段存在缺失的，删除该行
    data.reset_index(drop=True,inplace=True) #重置索引

    #将时间处理为等长字符串，0-1为天，3-4为月，6-9为年，11-12为小时，14-15为分钟，17-18为秒
    data['ACTDATETIME'] = data['ACTDATETIME'].copy().apply(datetime_str)
    # data['ACTDATETIME'] = pd.to_datetime(data.ACTDATETIME,format('%Y-%m-%d %H:%M:%S'))

    # 数据清洗
    # 筛选运营时间在06：00：00-19：00：00之间的数据
    data = data[(data['ACTDATETIME'].str.slice(11,13).astype('float')>=6)&(data['ACTDATETIME'].str.slice(11,13).astype('float')<=19)]
    #ROUTEID字段 筛选线路号113
    data = data[data['ROUTEID'].isin([113])]
    #数据去重,删除车号和时间重复的数据
    data.drop_duplicates(subset=['PRODUCTID','ACTDATETIME'],keep='last',inplace=True)
    data.sort_values(by = ['PRODUCTID','ACTDATETIME'],inplace = True)
    data.reset_index(drop=True,inplace=True)
    return data

#公交车站点数据预处理
def route_preprocess(data):
    data.rename(columns={'经度':'station_lon','纬度':'station_lat',
    '站名':'station_name','上下行':'up_down_line','线路':'line_id'},inplace=True)
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
    route['stop_index'] = range(len(route))  #站点树的索引
    gdf = pd.merge(data,route,on = 'stop_index')  #将gps点与其最近站点的所有信息进行匹配

    gdf['dist'] = dist_ckdn* 1000
    # 以防数据不太好，因此在此步骤筛选时先将条件放宽
    gdf = gdf[(gdf['dist']<=200)|(gdf['ISARRLFT'].isin([1,2]))].copy()
    # 聚合最短距离按理来说应该在同一班次内进行，但是数据原因没有班次数据信息，因此这里只能考虑
    # 在日期+小时内进行聚合，但是一班次的车程经过验算为40min左右，且存在一辆车一直行驶的情况，故该方法仍有可能存在误差
    gdf['date_time'] = gdf['ACTDATETIME'].copy().str.slice(0,13)
    col = ['PRODUCTID','date_time','stop_index']
    gdf1 = gdf.groupby(col).agg({'dist':'min'}).reset_index()
    gdf2 = pd.merge(gdf1,gdf,on =  ['PRODUCTID','date_time','stop_index','dist'])
    # 按理说这里除去出发前的数据和终到后的数据，keep = 'last'只能保证出发前的数据删除正确，
    # 若终到站之后车辆前往停车场的距离小于200m，则会出现误差
    gdf2.drop_duplicates(subset={'PRODUCTID','stop_index','dist'},keep='last',inplace=True)
    gdf2.sort_values(by=['PRODUCTID','ACTDATETIME'],inplace=True)
    gdf2.reset_index(drop=True,inplace=True)
    
    gdf3 = gdf2.copy()
    columns = ['ROUTEID', 'PRODUCTID','ACTDATETIME','date_time','stop_index','LONGITUDE','LATITUDE','GPSSPEED','dist','station_name']  
    gdf3 = gdf3[columns] #取有用字段
    # 区分上下行的方法：data.direction = data.stop_index - data.stop_index(-1) = -1 属于up_line中stop_index到下一站，同理计算行程时间
    # data.direction = data.stop_index - data.stop_index(-1) = 1  属于down_line中stop_index到下一站，同理计算行程时间
    # 行程时间分布：groupby(stop_index + direction)按站台和上下行进行聚合
    gdf3['next_stop_index'] = gdf3['stop_index'].shift(-1)
    gdf3['stop_gap'] = gdf3['stop_index'] - gdf3['next_stop_index']
    gdf3['travel_time'] = dt_time(gdf3['ACTDATETIME'].shift(-1),gdf3['ACTDATETIME']) # 本站到下一站的行程时间
    # 删除不合理的时间
    gdf3 = gdf3[((gdf3['stop_gap']==1)|(gdf3['stop_gap']==-1))&(gdf3['travel_time']<=600)&(gdf3['travel_time']>0)]
    # 对各站点的上下行行程时间进行统计
    # 考虑到上下行的站点数不同，在down_line中是不存在闸口一站的,
    # 因此需要将down_line中 =11的行的行程时间加上其下面一行的行程时间，然后再把闸口站的down_line行数据删除
    gdf3.loc[(gdf3['stop_index']==11)&(gdf3['stop_gap']==1), 'travel_time'] = gdf3.loc[(gdf3['stop_index']==11)&(gdf3['stop_gap']==1), 'travel_time']  + gdf3.loc[(gdf3['stop_index']==11)&(gdf3['stop_gap']==1), 'travel_time'].shift(-1)
    gdf3 = gdf3[-((gdf3['stop_index']==10)&(gdf3['stop_gap']==1))]
    # gdf3是一个GPS表里面上下行所有站点前往下一个站点的行程时间的总表
    return gdf3

#%% 生成行程时间
def generate_travel_time(folder,output_folder):
    # spyder使用：直接选中工作文件夹 ~/聊城2-创意赛-公交线网优化/公交线网优化
    # IC_path = folder+"/IC交易数据.csv"
    BusGPS_paths = file_name(folder+"/公交GPS全量数据")
    print("读取线路数据中……")
    BusRoute_path = folder+"/公交线路信息new.xlsx"
    route = pd.read_excel(BusRoute_path,sheet_name='up_line')
    print("读取gps数据中……")

    for i in range(len(BusGPS_paths)):
        data = pd.read_csv(BusGPS_paths[i],sep=',')
        suffix = re.split(r'[\\.]',BusGPS_paths[i])[-2] #后缀名为gps文件名

        print("gps数据预处理……")
        data = bus_gps_preprocess(data)
        print("线路数据预处理……")
        route = route_preprocess(route)

        #站点匹配
        print('站点匹配中……')
        start = time.time()
        result= gps_station_match(data,route)
        result.to_csv(f'{output_folder}/result_{suffix}.csv',sep=',', index =False, encoding='utf-8_sig')
        end = time.time()
        print(f'进度：{i+1}/{len(BusGPS_paths)} \t 匹配用时：{end-start} \n')
    return True

#按星期、小时对行程时间求均值
def week_hour(data):
    data = data[['date_time','travel_time']]
    data['date_time']  =pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['weekday'] = data['date_time'].apply(lambda x: x.weekday())
    # data = data.groupby(by=['weekday','hour'])['travel_time'].mean().reset_index()
    print(data.head())
    return data

#%% 行程时间分布拟合
#1. 先画一周内平均行程时间分布
#2. 行程时间的最优分布（该站点的所有数据？某个时间点的数据？）
#3. 行程时间预测模型（ARIMA模型，SVM模型）

def distribution(data):
    from fitter import Fitter
    start = time.time()
    f = Fitter(data['travel_time'].values,timeout=10)
    f.fit()
    suffix = '_'.join(['StopIndex',str(data['stop_index'].values[0]),'StopGap',str(data['stop_gap'].values[0])])
    #输出所有拟合分布质量
    all_distribution = pd.DataFrame(f.summary())
    all_distribution.to_csv(f'all_distribution_{suffix}.csv',sep=',',encoding='utf-8')
    #输出拟合效果最好的分布及其参数
    best_param = pd.DataFrame(f.get_best(method='sumsquare_error'))
    best_param.to_csv(f'best_distribution_param_{suffix}.csv',sep=',',encoding='utf-8')
    print('best distribution: \n',f.get_best(method='sumsquare_error'))
    end = time.time()
    print(f'分布拟合用时：{end-start}')
    return True

#%%
import pandas as pd
def draw_plot(ax,data,title):
    import numpy as np
    import seaborn as sns
    sns.boxplot(x='hour',y='travel_time',
            data=data,orient='v',ax=ax)

    # ax.scatter(data['hour'].values,data['travel_time'].values)
    ax.set(title=title) #设置axes及子图标题
    # ax.set_xlabel('时间（小时）',color='g')#设置x轴刻度标签
    # ax.set_ylabel('行程时间（秒）',color='g')#设置x轴刻度标签    
    # ax.set_xlim(6,21)#设置x轴刻度范围
    # ax.set_ylim(0,max(data['travel_time'])*1.2)

    # ax.set_xticks(np.arange(0,24))
    # ax.set_xticklabels(np.arange(0,24))
    ax.grid()

    # ax.set_yticks(np.arange(0,300,60))
    # ax.set_yticklabels(np.arange(0,300,60))
    return True
#行程时间分布图
def draw_tarvel_time(data,stop_index,stop_gap):
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
    plt.rcParams['axes.unicode_minus']=False  #显示负号
    fig, axs = plt.subplots(3,1,figsize=(12,12),dpi=600,
                        sharex = False,#x轴刻度值共享开启
                        sharey = False,#y轴刻度值共享关闭                        
                        )
    #工作日按小时分布
    data_wkday = data[data['weekday'].isin([0,1,2,3,4])]
    draw_plot(axs[0],data_wkday,'工作日')
    #周末按小时分布
    data_wkends = data[data['weekday'].isin([5,6])]
    draw_plot(axs[1],data_wkends,'休息日')
    #不区分工作日休息日，按小时分布
    axs[2].scatter(data['hour'].values,data['travel_time'].values)    
    draw_plot(axs[2],data,'总体')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4) # 调整子图间距   
    fig.suptitle(f'StopIndex_{stop_index}_StopGap_{stop_gap}',color='r')#设置fig即整整张图的标题

    plt.savefig(f'StopIndex_{stop_index}_StopGap_{stop_gap}.png',dpi=600)
    return True

#%% 主函数
def main():
    folder = r"D:\研究生\研究\公交线网优化\全量数据"
    #处理folder文件夹下所有的gps数据，输出行程时间csv
    #在当前文件夹手动新建一个travel_time文件夹
    generate_travel_time(folder,'travel_time')

    #合并所有行程时间
    travel_time_folder = r"D:\JupyterFiles\公交数据处理\code\travel_time"
    time_paths = file_name(travel_time_folder)
    data_list = [pd.read_csv(x,sep=',',encoding='utf-8_sig') for x in time_paths]
    time_data = pd.concat(data_list,ignore_index=True)
    #输出合并以后的行程时间
    #时间范围：2020.12.31 - 2021.08.31
    time_data.sort_values(by=['PRODUCTID','ACTDATETIME','date_time'],inplace=True)
    time_data.to_csv('all_travel_time.csv',sep=',',encoding='utf-8_sig')
    time_data = pd.read_csv('all_travel_time.csv',sep=',',encoding='utf-8_sig')
    #按站点索引、上下行 绘制行程时间分布图
    #每个站点24小时
    stops = time_data['stop_index'].unique()
    for i,stop in enumerate(stops):
        data1 = time_data[(time_data['stop_index']==stop) & (time_data['stop_gap']==1)]
        data2 = time_data[(time_data['stop_index']==stop) & (time_data['stop_gap']==-1)]
        if len(data1)>0:
            data1.reset_index(drop=True,inplace=True)
            data1 = week_hour(data1)
            #画出该站的行程时间分布图
            draw_tarvel_time(data1,stop,'1')
        if len(data2)>0:
            data2.reset_index(drop=True,inplace=True)
            data2 = week_hour(data2)
            draw_tarvel_time(data2,stop,'-1')
        print(f'进度：{i+1}/{len(stops)}')

    #按站点索引对
    return True
#%%
if __name__ == '__main__':
    main()
# %%
