import os, sys

#This script converts gridsmart file names to pipeline compatible names
#Sample gridsmart files: 2023-08-04_09-27-01-rtsp_Stirling-61Av_0.mp4
#Sample miovision files: 2977eb33-a029-4a21-87dd-ab86816e8a61-Jul_07_2023_09_45_10_00.mp4
def get_new_filename(ofile):
    intersec_dict = {'SR7_0.': 21, 'SR7_1.': 22, '68Av_0': 24, '66Av_0': 25, '66Av_1': 26, 'Univ_0': 28,  'Univ_1': 29, '61Av_0': 31, '61Av_1': 32}
    if (ofile[4]=='-'):     # gridsmart file format
        date_time = ofile[0:19]
        date_hour = ofile[0:14]
        minute_sec = ofile[14:19]
 
        intersec = ofile[34:40]
 
        camera_id=str(intersec_dict[intersec])
        mfile = str(camera_id) + '_' + date_time + '.000.mp4'
        return (mfile)
    else:                   # miovision file format
        if (ofile[0:8] == '2e747114'):
            camera_id = '34'
        else:
            camera_id = '35'
        millisecs = ofile[-6:-4]
        #print (millisecs)
        secs = ofile[-9:-7]
        #print (secs)
        mins = ofile[-12:-10]
        #print (mins)
        hours = ofile[-15:-13]
        #print (hours)
        year = ofile[-20:-16]
        #print (year)
        date = ofile[-23:-21]
        #print (date)
        month = ofile[-27:-24]
        #print (month)
        month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month_str = str(month_dict[month])
        mfile = camera_id + '_' + year + '-' + month_str.zfill(2) + '-' + date + '_' + hours + '-' + mins + '-' + secs + '.' + millisecs.zfill(3) + '.mp4'
        return (mfile)


def main(ofile: str):
    f = get_new_filename(ofile)
    destination='/mnt/hdd/data/video_pipeline/' + f
    cmd = 'sudo cp {} {}'.format(ofile, destination)
    print (cmd)
    os.system(cmd)