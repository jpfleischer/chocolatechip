import os
import pandas as pd
import glob
import sys
import pymysql
import json
import datetime

def loadenv(envfile):
    with open(envfile) as f:
        for line in f:
            # Skip comments and empty lines
            if line.strip() and not line.startswith('#'):
                # Split the line into key and value
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

def getYMDHM(row):
    ru = str(row.unique_ID1)
    index = ru.find('21')
    moi = index+2
    di = index+4
    hi = index+6
    mi = index+8
    year = str(20)+ru[index:index+2]
    month = ru[moi:moi+2]
    day = ru[di:di+2]
    hour = ru[hi:hi+2]
    minute = ru[mi:mi+2]
    return year, month, day, hour, minute

def getOffset(row):
    loadenv('./.env')
    host = os.getenv('DB_HOST')    
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    db = os.getenv('DB_NAME')
    
    myoutputdb = pymysql.connect(host=host, \
            user=user, passwd=password, \
            db=db, port=3306)

    year, month, day, hour, minute = getYMDHM(row)
    offsetq = 'select * from OffsetTable where intersection_id={} and camera_id={} and year={} and month={} and day={} and hour={} and minute={}'.format(row.intersection_id, row.camera_id, year, month, day, hour, minute)
    offsettable = pd.read_sql(offsetq, myoutputdb)
    tsize = len(offsettable)
    offset = 0
    if (tsize > 0):
        offset = offsettable.offset.iloc[tsize-1]
    myoutputdb.close()
    return pd.Timedelta(offset)


def do_main(row, conflict_type):
    index = row['index'] #create a new column with name index in the TTC.CSV file
    delta = 0 #getOffset(row)
    stime = datetime.datetime.strptime(row.timestamp, "%Y-%m-%d %H:%M:%S.%f")
    stime = stime# + delta
    iid = row.intersection_id
    camid = str(row.unique_ID1)[0:2]
    minute = stime.minute - stime.minute%15
    #fbase from timestamp
    fbaseft = camid.zfill(2) + '_' + str(stime.year) + '-' + str(stime.month).zfill(2) + '-' + str(stime.day).zfill(2) + '_' + str(stime.hour).zfill(2) + '-' + str(minute).zfill(2)
    #fbase from unique_ID
    year, month, day, hour, minu = getYMDHM(row)
    fbasefu = camid.zfill(2) + '_' + str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2) + '_' + str(hour).zfill(2) + '-' + str(minu).zfill(2)
    fname = '/mnt/hdd/gvideo/{}*.mp4'.format(fbaseft)
    fname = '/mnt/hdd/data/video_pipeline/tracking/{}*.mp4'.format(fbaseft)
    fnamelist = glob.glob(fname)
    if not fnamelist:
        return
    f = fnamelist[0]
    fileprefix = './'+str(row.intersection_id)+'/' +conflict_type+'/'+str(index).zfill(3)+ '_' + fbaseft+'_'+str(row.unique_ID1)+'_'+str(row.unique_ID2)
    ofv = fileprefix+'.mp4'
    oftxt = fileprefix+'.txt'
    original_stdout = sys.stdout # Save a reference to the original standard output

    with open(oftxt, 'w') as ftxt:
        sys.stdout = ftxt # Change the standard output to the file we created.
        print(row)
        sys.stdout = original_stdout # Reset the standard output to its original value
    sseconds = (stime.minute % 15)*60 + stime.second
    st = 5
    tt = 5
    if row.p2v == 1:
        st = 15
        tt = 15
    sseconds = sseconds - st
    if (sseconds < 0):
        sseconds = 0
    dur = st + tt
    cmd = 'ffmpeg -i {} -ss {} -t {} -c mpeg4 {}'.format(f, sseconds, dur, ofv)
    print (cmd)
    os.system(cmd)


def main(input, p2v):
    argdf = pd.read_csv(input)

    for i, row in argdf.iterrows():
        do_main(row, 'v2v')

    argdf = pd.read_csv(p2v)

    for i, row in argdf.iterrows():
        do_main(row, 'p2v')



