import pymysql
import pandas as pd
import sys
import numpy as np

from chocolatechip.MySQLConnector import MySQLConnector

# instantiate and connect
connector   = MySQLConnector()
myoutputdb  = connector._connect()
mycursor    = myoutputdb.cursor()

iid = sys.argv[1]
select_ttc_df = pd.DataFrame()
for i in range(2, len(sys.argv), 2):
    start_time = pd.to_datetime(sys.argv[i])
    end_time = pd.to_datetime(sys.argv[i+1])

    # ttc_query = 'select * from TTCTable where intersection_id=%(name)s and timestamp between  and %(end_time)s;'

    ttc_query  = "SELECT * FROM TTCTable where intersection_id = '{}'and include_flag = 1 and timestamp between '{}' and '{}' and p2v=0".format(iid, start_time, end_time)
    #ttc_query  = "SELECT * FROM TTCTable where intersection_id = '{}'and timestamp between '{}' and '{}'".format(iid, start_time, end_time)
    select_ttc_df = pd.concat([select_ttc_df, pd.read_sql(ttc_query, myoutputdb)])

select_ttc_df = select_ttc_df[select_ttc_df['time'] < 3.5]

normalized_time = (select_ttc_df['time'] - select_ttc_df['time'].min()) / (select_ttc_df['time'].max() - select_ttc_df['time'].min())
normalized_speed1 = (select_ttc_df['speed1'] - select_ttc_df['speed1'].min()) / (select_ttc_df['speed1'].max() - select_ttc_df['speed1'].min())

select_ttc_df['normalized_time'] = normalized_time
select_ttc_df['normalized_speed1'] = normalized_speed1
select_ttc_df = select_ttc_df.drop_duplicates('unique_ID1', keep='first')

ttc_window = select_ttc_df



def y_parametric(t, intercept):
    return t + intercept

arbitrary_row = ttc_window.iloc[0]
intercept = arbitrary_row['normalized_speed1'] - arbitrary_row['normalized_time']

distances = []
for index, row in ttc_window.iterrows():
    distance = abs(row['normalized_speed1'] - row['normalized_time'] - intercept) / np.sqrt(2)
    px = row['normalized_time']
    py = row['normalized_speed1']
    line_y = y_parametric(px, intercept)
    if py >= line_y:
        print("Point is above or on the line")
        distances.append(-distance)
    elif py < line_y:
        print("Point is below the line")
        distances.append(distance)
    
ttc_window['ranking'] = distances
sorted_df = ttc_window.sort_values(by='ranking')
print (sorted_df.groupby(['cluster1', 'cluster2'])['camera_id'].count())
sorted_df=sorted_df.reset_index(drop=True)
sorted_df.to_csv(str(iid) + '/sorted_ranking.csv', index=True, index_label='index')
