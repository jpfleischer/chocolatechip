import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import logging
logging.basicConfig(level=logging.INFO)
import pymysql
from joblib import Parallel, delayed
import datetime
import sys
import json
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import norm
import time
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#from pandas.core.common import SettingWithCopyWarning
#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
num_cores = mp.cpu_count()-20


def get_times(iid: int):
    
    if iid == 3287:
        times = [
            '2024-02-26 07:00:00.000', '2024-02-26 19:00:00.000',
            '2024-02-27 07:00:00.000', '2024-02-27 19:00:00.000',
            '2024-02-28 07:00:00.000', '2024-02-28 19:00:00.000',
            '2024-03-07 07:00:00.000', '2024-03-07 19:00:00.000',
            '2024-03-08 07:00:00.000', '2024-03-08 19:00:00.000',
            '2024-03-09 07:00:00.000', '2024-03-09 19:00:00.000',
            '2024-03-10 07:00:00.000', '2024-03-10 19:00:00.000',
            '2024-03-12 07:00:00.000', '2024-03-12 19:00:00.000',
            '2024-03-13 07:00:00.000', '2024-03-13 19:00:00.000',
            '2024-03-14 07:00:00.000', '2024-03-14 19:00:00.000',
            '2024-03-15 07:00:00.000', '2024-03-15 19:00:00.000',
            '2024-03-16 07:00:00.000', '2024-03-16 19:00:00.000'
            ]
    
    elif iid == 3248:
        times = [
         '2024-02-26 07:00:00.000', '2024-02-26 19:00:00.000',
         '2024-02-27 07:00:00.000', '2024-02-27 19:00:00.000',
         '2024-02-28 07:00:00.000', '2024-02-28 19:00:00.000',
         '2024-03-06 07:00:00.000', '2024-03-06 19:00:00.000',
         '2024-03-07 07:00:00.000', '2024-03-07 19:00:00.000',
         '2024-03-08 07:00:00.000', '2024-03-08 19:00:00.000',
         '2024-03-09 07:00:00.000', '2024-03-09 19:00:00.000',
         '2024-03-10 07:00:00.000', '2024-03-10 19:00:00.000',
         '2024-03-12 07:00:00.000', '2024-03-12 19:00:00.000',
         '2024-03-13 07:00:00.000', '2024-03-13 19:00:00.000',
         '2024-03-14 07:00:00.000', '2024-03-14 19:00:00.000',
         '2024-03-16 07:00:00.000', '2024-03-16 19:00:00.000',
         '2024-03-20 07:00:00.000', '2024-03-20 19:00:00.000',
         '2024-03-21 07:00:00.000', '2024-03-21 19:00:00.000',
         '2024-03-22 07:00:00.000', '2024-03-22 19:00:00.000',
         '2024-03-23 07:00:00.000', '2024-03-23 19:00:00.000',
         '2024-03-24 07:00:00.000', '2024-03-24 19:00:00.000',
         '2024-03-25 07:00:00.000', '2024-03-25 19:00:00.000',
         '2024-03-26 07:00:00.000', '2024-03-26 19:00:00.000',
         '2024-04-06 07:00:00.000', '2024-04-06 19:00:00.000',
         '2024-04-07 07:00:00.000', '2024-04-07 19:00:00.000',
         '2024-04-08 07:00:00.000', '2024-04-08 19:00:00.000'
         ]
        
    elif iid == 3032:
        times = [
        '2024-04-03 07:00:00.000', '2024-04-03 19:00:00.000',
        '2024-04-04 07:00:00.000', '2024-04-04 19:00:00.000',
        '2024-04-05 07:00:00.000', '2024-04-05 19:00:00.000',
        '2024-04-06 07:00:00.000', '2024-04-06 19:00:00.000',
        '2024-04-07 07:00:00.000', '2024-04-07 19:00:00.000',
        '2024-04-08 07:00:00.000', '2024-04-08 19:00:00.000',
        '2024-04-10 07:00:00.000', '2024-04-10 19:00:00.000',
        '2024-04-11 07:00:00.000', '2024-04-11 19:00:00.000',
        '2024-04-13 07:00:00.000', '2024-04-13 19:00:00.000',
        '2024-04-14 07:00:00.000', '2024-04-14 19:00:00.000',
        '2024-04-15 07:00:00.000', '2024-04-15 19:00:00.000',
        '2024-04-16 07:00:00.000', '2024-04-16 19:00:00.000',
        '2024-04-17 07:00:00.000', '2024-04-17 19:00:00.000',
        '2024-04-19 07:00:00.000', '2024-04-19 19:00:00.000',
        '2024-04-20 07:00:00.000', '2024-04-20 19:00:00.000',
        '2024-04-21 07:00:00.000', '2024-04-21 19:00:00.000',
        '2024-04-22 07:00:00.000', '2024-04-22 19:00:00.000',
        '2024-04-23 07:00:00.000', '2024-04-23 19:00:00.000',
            ]
        
    elif iid == 3265:
        times = [  #3265 University
         '2024-04-03 07:00:00.000', '2024-04-03 19:00:00.000',
         '2024-04-04 07:00:00.000', '2024-04-04 19:00:00.000',
         '2024-04-05 07:00:00.000', '2024-04-05 19:00:00.000',
         '2024-04-06 07:00:00.000', '2024-04-06 19:00:00.000',
         '2024-04-07 07:00:00.000', '2024-04-07 19:00:00.000',
         '2024-04-08 07:00:00.000', '2024-04-08 19:00:00.000',
         '2024-04-09 07:00:00.000', '2024-04-09 19:00:00.000',
         '2024-04-10 07:00:00.000', '2024-04-10 19:00:00.000',
         '2024-04-11 07:00:00.000', '2024-04-11 19:00:00.000',
         '2024-04-12 07:00:00.000', '2024-04-12 19:00:00.000',
         '2024-04-13 07:00:00.000', '2024-04-13 19:00:00.000',
         '2024-04-14 07:00:00.000', '2024-04-14 19:00:00.000',
         '2024-04-15 07:00:00.000', '2024-04-15 19:00:00.000',
         '2024-04-16 07:00:00.000', '2024-04-16 19:00:00.000',
         '2024-04-17 07:00:00.000', '2024-04-17 19:00:00.000',
         '2024-04-20 07:00:00.000', '2024-04-20 19:00:00.000',
         '2024-04-21 07:00:00.000', '2024-04-21 19:00:00.000',
         '2024-04-22 07:00:00.000', '2024-04-22 19:00:00.000',
         '2024-04-23 07:00:00.000', '2024-04-23 19:00:00.000'
         ]
        
    elif iid == 3334:
        times = [  #3334 61st
         '2024-03-05 07:00:00.000', '2024-03-05 19:00:00.000',
         '2024-03-06 07:00:00.000', '2024-03-06 19:00:00.000',
         '2024-03-08 07:00:00.000', '2024-03-08 19:00:00.000',
         '2024-03-09 07:00:00.000', '2024-03-09 19:00:00.000',
         '2024-03-12 07:00:00.000', '2024-03-12 19:00:00.000',
         '2024-04-02 07:00:00.000', '2024-04-02 19:00:00.000',
         '2024-04-03 07:00:00.000', '2024-04-03 19:00:00.000',
         '2024-04-04 07:00:00.000', '2024-04-04 19:00:00.000',
         '2024-04-05 07:00:00.000', '2024-04-05 19:00:00.000',
         '2024-04-06 07:00:00.000', '2024-04-06 19:00:00.000',
         '2024-04-07 07:00:00.000', '2024-04-07 19:00:00.000',
         '2024-04-08 07:00:00.000', '2024-04-08 19:00:00.000',
         '2024-04-09 07:00:00.000', '2024-04-09 19:00:00.000',
         '2024-04-10 07:00:00.000', '2024-04-10 19:00:00.000',
         '2024-04-11 07:00:00.000', '2024-04-11 19:00:00.000',
         '2024-04-12 07:00:00.000', '2024-04-12 19:00:00.000',
         '2024-04-13 07:00:00.000', '2024-04-13 19:00:00.000',
         '2024-04-14 07:00:00.000', '2024-04-14 19:00:00.000',
         '2024-04-15 07:00:00.000', '2024-04-15 19:00:00.000',
         '2024-04-16 07:00:00.000', '2024-04-16 19:00:00.000',
         '2024-04-17 07:00:00.000', '2024-04-17 19:00:00.000',
         '2024-04-18 07:00:00.000', '2024-04-18 19:00:00.000',
         '2024-04-19 07:00:00.000', '2024-04-19 19:00:00.000',
         '2024-04-20 07:00:00.000', '2024-04-20 19:00:00.000',
         '2024-04-21 07:00:00.000', '2024-04-21 19:00:00.000',
         '2024-04-22 07:00:00.000', '2024-04-22 19:00:00.000',
         '2024-04-26 07:00:00.000', '2024-04-26 19:00:00.000',
         '2024-04-27 07:00:00.000', '2024-04-27 19:00:00.000',
         '2024-04-28 07:00:00.000', '2024-04-28 19:00:00.000',
         '2024-04-29 07:00:00.000', '2024-04-29 19:00:00.000',
         '2024-04-30 07:00:00.000', '2024-04-30 19:00:00.000',
         '2024-05-01 07:00:00.000', '2024-05-01 19:00:00.000',
         '2024-05-02 07:00:00.000', '2024-05-02 19:00:00.000',
         '2024-05-08 07:00:00.000', '2024-05-08 19:00:00.000',
         '2024-05-09 07:00:00.000', '2024-05-09 19:00:00.000',
         '2024-05-10 07:00:00.000', '2024-05-10 19:00:00.000',
         '2024-05-11 07:00:00.000', '2024-05-11 19:00:00.000',
         '2024-05-12 07:00:00.000', '2024-05-12 19:00:00.000',
         '2024-05-13 07:00:00.000', '2024-05-13 19:00:00.000'
         ]
    else:
        raise ValueError('Invalid intersection ID')

    return times

# ########################################### Loading vehicle and ped tracks ########################################### #
def speeds_parallel(df_t, pixel2meter, conversion=True, window=5):
    df_t = df_t.sort_values(by='timestamp')
    df_t = df_t.drop_duplicates(subset='timestamp', keep="first")
    df_t['diffx'] = df_t['center_x'].diff()/pixel2meter
    df_t['diffy'] = df_t['center_y'].diff()/pixel2meter
    df_t['td'] = df_t['timestamp'].diff().dt.total_seconds()

    df_t['diffx'] = df_t['diffx'].fillna(0)
    df_t['diffy'] = df_t['diffy'].fillna(0)
    df_t['td'] = df_t['td'].fillna(.1)

    df_t['speed_x'] = df_t['diffx']/df_t['td']
    df_t['speed_y'] = df_t['diffy']/df_t['td']
    avg_spx = np.mean(df_t['speed_x'])
    avg_spy = np.mean(df_t['speed_y'])
    std_spx = np.std(df_t['speed_x'])
    std_spy = np.std(df_t['speed_y'])
    df_t.loc[abs(df_t.speed_x) > (abs(avg_spx) + std_spx), 'speed_x'] = avg_spx
    df_t.loc[abs(df_t.speed_y) > (abs(avg_spy) + std_spy), 'speed_y'] = avg_spy
    df_t.loc[abs(df_t.speed_x) < (abs(avg_spx) - std_spx), 'speed_x'] = avg_spx
    df_t.loc[abs(df_t.speed_y) < (abs(avg_spy) - std_spy), 'speed_y'] = avg_spy
    #win = 5
    #beg_avg_spx = np.mean(df_t['speed_x'][1:win+1])
    #beg_avg_spy = np.mean(df_t['speed_y'][1:win+1])
    #df_t['speed_x'].iloc[0] = beg_avg_spx
    #df_t['speed_y'].iloc[0] = beg_avg_spy
    df_t['speed_x'] = df_t['speed_x'].rolling(window=5).mean().fillna(avg_spx)
    df_t['speed_y'] = df_t['speed_y'].rolling(window=5).mean().fillna(avg_spy)
    #Correct speeds where difference is > 1m/s^2
    df_t.speed_x = correctLocalAberration(df_t.speed_x, df_t.td, 0)
    df_t.speed_y = correctLocalAberration(df_t.speed_y, df_t.td, 0)
#     # Make another with x and y acceleration
    df_t['accel_x'] = df_t['speed_x'].diff()/df_t['td'].fillna(0)
    df_t['accel_y'] = df_t['speed_y'].diff()/df_t['td'].fillna(0)
    df_t['inst_speed'] =  np.sqrt((df_t['speed_x'])*(df_t['speed_x']) + (df_t['speed_y'])*(df_t['speed_y']))
#     if window is not None:
    df_t['absspeed'] = df_t['inst_speed'].fillna(0)
    df_t['speeddiff'] = df_t['absspeed'].diff()
    df_t['speeddiff'] = df_t['speeddiff'].fillna(0)
    #acceleration
    df_t['acceleration'] = (df_t['speeddiff']/df_t['td']).clip(upper=3.3528, lower=-3.3528) #max acceleration on a typical car is 60mph in 8 seconds
    if conversion: # Convert speeds to mph
        df_t['inst_speed'] = df_t['inst_speed']*2.23694 # Meters/second ->Miles/hour
        df_t['absspeed'] = df_t['absspeed']*2.23694 # Meters/second ->Miles/hour
    return df_t

def speeds_parallel2(df_t, factors, conversion=True, window=5):
    from scipy import stats
    factor = factors[factors['intersection_id']==df_t.intersection_id.unique()[0]]['pixels2meter'].values[0]
    df_t = df_t.sort_values(by='timestamp')
    df_t['diffx'] = df_t['center_x'].diff()/factor
    df_t['diffy'] = df_t['center_y'].diff()/factor
    df_t['td'] = df_t['timestamp'].diff().dt.total_seconds()
    df_t['diffx'] = df_t['diffx'].fillna(0)
    df_t['diffy'] = df_t['diffy'].fillna(0)
    df_t['td'] = df_t['td'].fillna(.1)
    #From coordinates compute speed (x and y components)
    df_t['speed_x'] = df_t['diffx']/df_t['td']
    df_t['speed_y'] = df_t['diffy']/df_t['td']
    #Compute median speed
    med_spx = df_t['speed_x'].median()
    med_spy = df_t['speed_y'].median()
    #Compute Inter Quartile Range (IQR)
    iqrx = stats.iqr(df_t.speed_x, interpolation = 'midpoint')
    iqry = stats.iqr(df_t.speed_y, interpolation = 'midpoint')
    #For all points that are further from the median by
    #2.22 * IQD distance are assigned the median value
    df_t.loc[df_t.speed_x > med_spx + 2.22*iqrx, 'speed_x'] = med_spx
    df_t.loc[df_t.speed_x < med_spx - 2.22*iqrx, 'speed_x'] = med_spx
    df_t.loc[df_t.speed_y > med_spy + 2.22*iqry, 'speed_y'] = med_spy
    df_t.loc[df_t.speed_y < med_spy - 2.22*iqry, 'speed_y'] = med_spy
    #Compute acceleration
    #Clip acceleration values above 3m/s and below 10m/s
    df_t['accx'] = (df_t['speed_x'].diff()/df_t['td']).clip(upper=3, lower=-10)
    df_t['accy'] = (df_t['speed_y'].diff()/df_t['td']).clip(upper=3, lower=-10)
    df_t.iloc[0, df_t.columns.get_loc('accx')] = 0
    df_t.iloc[0, df_t.columns.get_loc('accy')] = 0
    #Run a rolling average on the acceleration (window size 5)
    df_t['accx'] = df_t['accx'].rolling(window=window).mean().fillna(0)
    df_t['accy'] = df_t['accy'].rolling(window=window).mean().fillna(0)
    #Adjust speeds from acceleration
    df_t.speed_x = (df_t['speed_x'] + (df_t['accx']*df_t['td']).shift(-1)).shift(1)
    df_t.speed_y = (df_t['speed_y'] + (df_t['accy']*df_t['td']).shift(-1)).shift(1)
    #Compute speed begin_mean from values at timestamps (1,2,3)
    #and set to the first value.
    begin_window = 3
    if begin_window > df_t.shape[0]:
        begin_window = df_t.shape[0]
    beg_avg_spx = np.mean(df_t['speed_x'][1:begin_window+1])
    beg_avg_spy = np.mean(df_t['speed_y'][1:begin_window+1])
    df_t.iloc[0, df_t.columns.get_loc('speed_x')] = beg_avg_spx
    df_t.iloc[0, df_t.columns.get_loc('speed_y')] = beg_avg_spy
    #Run a rolling average on the speed
    df_t['speed_x'] = df_t['speed_x'].rolling(window=3).mean().fillna(beg_avg_spx)
    df_t['speed_y'] = df_t['speed_y'].rolling(window=3).mean().fillna(beg_avg_spy)
    df_t.iloc[0, df_t.columns.get_loc('speed_x')] = beg_avg_spx
    df_t.iloc[0, df_t.columns.get_loc('speed_y')] = beg_avg_spy
    #Do one round of local aberration correction
    correctLocalAberration(df_t, df_t.td, 'speed_x')
    correctLocalAberration(df_t, df_t.td, 'speed_y')
    df_t['inst_speed'] =  np.sqrt((df_t['speed_x'])*(df_t['speed_x']) + (df_t['speed_y'])*(df_t['speed_y']))
    df_t['absspeed'] = df_t['inst_speed'].fillna(0)
    df_t['speeddiff'] = df_t['absspeed'].diff()
    df_t['speeddiff'] = df_t['speeddiff'].fillna(0)
    #accleration
    df_t['acceleration'] = (df_t['speeddiff']/df_t['td']).clip(upper=3, lower=-10) #2.78 -9.7536
    if conversion: # Convert speeds to mph
        df_t['inst_speed'] = df_t['inst_speed']*2.23694 # Meters/second ->Miles/hour
        df_t['absspeed'] = df_t['absspeed']*2.23694 # Meters/second ->Miles/hour
    return df_t
def correctLocalAberration(speed, td, iteration):
    pos = np.where(abs(speed.diff()/td) > 4)[0]
    for index in pos:
        start = index - 1
        end = index + 2
        if start < 0:
            start = 0
        if end > speed.shape[0]:
            end = speed.shape[0]
            start = end - 3
        speed.iloc[index] = np.mean(speed[start:end])
    pos1 = np.where(abs(speed.diff()/td) > 4)[0]
    if np.array_equal(pos, pos1):
        return speed
    else:
        if len(pos1) > 0 and iteration < 5:
            speed = correctLocalAberration(speed, td, iteration+1)
    return speed
def count_speeders(tracks, speed_limits):
    num_speeders = 0
    for (name, track) in tracks.groupby('unique_ID'):
        intersection_id = track.intersection_id
        approach = track['cluster'][0]
        if approach not in ['N', 'S', 'E', 'W']:  # Pedestrian
            continue
        speed_limit = speed_limits[(speed_limits['intersection_id']==intersection_id) &
                                   (speed_limits['approach']==approach.lower())].speedlimit
        if track['absspeed'].max() > float(speed_limit):
            num_speeders += 1
    return num_speeders

iid = 3334
start_time = pd.to_datetime("2024-03-05 07:00:00.000")
end_time = pd.to_datetime("2024-05-13 19:00:00.000")

iidlist = []
stimelist = []
ftimelist = []
# Append times within the start and end bounds

times = get_times(iid)
for i in range(0, len(times), 2):
    current_start = pd.to_datetime(times[i])
    current_end = pd.to_datetime(times[i+1])
    if current_start >= start_time and current_end <= end_time:
        iidlist.append(iid)
        stimelist.append(current_start)
        ftimelist.append(current_end)

sets = [x for x in zip(iidlist, stimelist, ftimelist)]
speeds_df = pd.DataFrame()
for intersection_id, start_time, end_time in sets:
    mydb = pymysql.connect(host='who_knows', user='who_knows', password='who_knows', database='who_knows')
    conversion = pd.read_sql("select distinct intersection_id, pixels2meter from IntersectionProperties;", con=mydb)
    pixels2meter = conversion[conversion['intersection_id']==int(intersection_id)]['pixels2meter'].values[0]
    #Prepare tracks dataframe
    from datetime import date, timedelta

    for n in range(0, 1):
        start_date = pd.to_datetime(start_time) + datetime.timedelta(n)
        end_date = pd.to_datetime(end_time) + datetime.timedelta(n)
        print ("Processing trajetories for intersection {}, start time {} and end time {}".format(intersection_id, start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S")))
        sql_query  = "SELECT * FROM RealDisplayInfo where timestamp >= '{}' AND timestamp <= '{}' AND intersection_id = {}".format(start_date.strftime("%Y-%m-%d %H:%M:%S"),end_date.strftime("%Y-%m-%d %H:%M:%S"),intersection_id)
        tracks = pd.read_sql(sql_query, con=mydb)
        tracks['timestamp'] = pd.to_datetime(tracks['timestamp'])
        sql_query  = "SELECT * FROM RealTrackProperties where timestamp >= '{}' AND timestamp <= '{}' AND intersection_id = {} and isAnomalous = 0".format(start_date.strftime("%Y-%m-%d %H:%M:%S"),end_date.strftime("%Y-%m-%d %H:%M:%S"),intersection_id)
        track_p = pd.read_sql(sql_query, con=mydb)
        track_p['timestamp'] = pd.to_datetime(track_p['timestamp'])
        temp = tracks.join(track_p[['unique_ID', 'phase', 'cluster', 'lane', 'movement', 'redJump','lanechange','nearmiss', 'city']].set_index('unique_ID'), on='unique_ID', how='inner')
        if (temp.empty == True):
            continue
        temp = temp[(temp['skip_begin']!=1) &
                            (temp['skip_end']!=1) &
                            (temp['class']!='pedestrian')]
        #processed_list = Parallel(n_jobs=num_cores)(delayed(speeds_parallel)(i, conversion) for s,i in temp.groupby('unique_ID'))
        processed_list = Parallel(n_jobs=num_cores)(delayed(speeds_parallel)(i, pixels2meter) for s,i in temp.groupby('unique_ID'))
        tracks1 = pd.concat(processed_list)
        '''
        gpby = temp.groupby('unique_ID')
        processed_list = []
        for s,i in gpby:
            processed_list.append(speeds_parallel(i, pixels2meter))
        '''
        stats = pd.DataFrame([])
        hourlist = []
        movelist = []
        phaselist = []
        lanelist = []
        speedlist = []
        uidlist = []
        vehicletype = []
        for name, frame in tracks1.groupby('unique_ID'):
            hourlist.append(frame.iloc[0].timestamp.hour)
            movelist.append(frame.iloc[0].movement)
            phaselist.append(frame.iloc[0].phase)
            lanelist.append(frame.iloc[0].lane)
            speedlist.append(frame.absspeed.mean())
            uidlist.append(frame.iloc[0].unique_ID)
            vehicletype.append(frame.iloc[0]['class'])
        stats['hour'] = hourlist
        stats['movement'] = movelist
        stats['phase'] = phaselist
        stats['lane'] = lanelist
        stats['average_speed'] = speedlist
        stats['unique_ID'] = uidlist
        stats['class'] = vehicletype
        speeds_df = pd.concat([speeds_df, stats])
light_speeds = speeds_df[((speeds_df['class'] == 'car') | (speeds_df['class'] == 'motorbike'))]
heavy_speeds = speeds_df[((speeds_df['class'] == 'bus') | (speeds_df['class'] == 'truck'))]
print ("Average speed of light vehicles")
print (light_speeds.groupby(['movement', 'phase', 'lane']).mean().average_speed)
print ("Average speed of heavy vehicles")
print (heavy_speeds.groupby(['movement', 'phase', 'lane']).mean().average_speed)
print (speeds_df.groupby(['movement', 'phase', 'lane']).mean().average_speed)
print (light_speeds.groupby(['movement', 'phase']).mean().average_speed)
print (heavy_speeds.groupby(['movement', 'phase']).mean().average_speed)
speeds_df.to_csv("{}/{}_avgspeeds.csv".format(intersection_id,intersection_id), index=False)

