from chocolatechip.MySQLConnector import MySQLConnector
import numpy as np
from yaspin import yaspin
from yaspin.spinners import Spinners
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import time
import io
import base64


def heatmap_generator(df_type: str,
                      mean: bool,
                      intersec_id: int,
                      p2v: bool = None,
                      conflict_type: str = None,
                      pedestrian_counting: bool = False,
                      return_agg: bool = False   # <-- NEW
                      ):
    """
    Generate a heatmap (and lineplot) for either 'track' or 'conflict' data.
    If return_agg=True, return a summarized DataFrame grouped by day_of_week
    for possible downstream normalizing.
    """

    if df_type not in ['track', 'conflict']:
        raise ValueError('df_type must be "track" or "conflict"')
    
    if df_type == 'conflict' and p2v is None:
        raise ValueError('p2v must be True or False when df_type is "conflict"')
    
    # if df_type == 'conflict' and conflict_type is None:
        # raise ValueError('conflict_type must be left right or thru when df_type is "conflict" and p2v true')
    
    if p2v is False and conflict_type in ['left turning', 'right turning', 'thru']:
        raise ValueError('try commenting the three lines and uncommenting the one, or make p2v true')

    #5&6 is for through
    #3&4 is for left turn
    #1&2 are for right turn


    intersec_lookup = {
        3287: "Stirling Road and N 68th Avenue",
        3248: "Stirling Road and N 66th Avenue",
        3032: "Stirling Road and SR-7",
        3265: "Stirling Road and University Drive",
        3334: "Stirling Road and Carriage Hills Drive/SW 61st Avenue",
        3252: "Stirling Road and Davie Road Extension",
    }

    cam_lookup = {
        3287: 24,
        3248: 27,
        3032: 23,
        3265: 30,
        3334: 33,
        3252: 36,
    }

    params = {
        'start_date': '2024-02-26 07:00:00',
        'end_date': '2024-02-27 00:00:00',
        'intersec_id': intersec_id,
        'cam_id': cam_lookup[intersec_id],
        'p2v': 0 if p2v is False else 1
    }


    if params['intersec_id'] == 3287:
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
    
    elif params['intersec_id'] == 3248:
        times = [
        #  '2024-02-26 07:00:00.000', '2024-02-26 19:00:00.000',
        #  '2024-02-27 07:00:00.000', '2024-02-27 19:00:00.000',
        #  '2024-02-28 07:00:00.000', '2024-02-28 19:00:00.000',
        #  '2024-03-06 07:00:00.000', '2024-03-06 19:00:00.000',
        #  '2024-03-07 07:00:00.000', '2024-03-07 19:00:00.000',
        #  '2024-03-08 07:00:00.000', '2024-03-08 19:00:00.000',
        #  '2024-03-09 07:00:00.000', '2024-03-09 19:00:00.000',
        #  '2024-03-10 07:00:00.000', '2024-03-10 19:00:00.000',
        #  '2024-03-12 07:00:00.000', '2024-03-12 19:00:00.000',
        #  '2024-03-13 07:00:00.000', '2024-03-13 19:00:00.000',
        #  '2024-03-14 07:00:00.000', '2024-03-14 19:00:00.000',
        #  '2024-03-16 07:00:00.000', '2024-03-16 19:00:00.000',
        #  '2024-03-20 07:00:00.000', '2024-03-20 19:00:00.000',
        #  '2024-03-21 07:00:00.000', '2024-03-21 19:00:00.000',
        #  '2024-03-22 07:00:00.000', '2024-03-22 19:00:00.000',
        #  '2024-03-23 07:00:00.000', '2024-03-23 19:00:00.000',
        #  '2024-03-24 07:00:00.000', '2024-03-24 19:00:00.000',
        #  '2024-03-25 07:00:00.000', '2024-03-25 19:00:00.000',
        #  '2024-03-26 07:00:00.000', '2024-03-26 19:00:00.000',
        #  '2024-04-06 07:00:00.000', '2024-04-06 19:00:00.000',
        #  '2024-04-07 07:00:00.000', '2024-04-07 19:00:00.000',
        #  '2024-04-08 07:00:00.000', '2024-04-08 19:00:00.000'
        ####
        '2024-10-29 07:00:00.000', '2024-10-29 19:00:00.000', 
        '2024-10-30 07:00:00.000', '2024-10-30 19:00:00.000', 
        '2024-10-31 07:00:00.000', '2024-10-31 19:00:00.000', 
        '2024-11-01 07:00:00.000', '2024-11-01 19:00:00.000', 
        '2024-11-02 07:00:00.000', '2024-11-02 19:00:00.000', 
        '2024-11-03 07:00:00.000', '2024-11-03 19:00:00.000', 
        '2024-11-06 07:00:00.000', '2024-11-06 19:00:00.000',
        '2024-11-07 07:00:00.000', '2024-11-07 19:00:00.000', 
        '2024-11-08 07:00:00.000', '2024-11-08 19:00:00.000', 
        '2024-11-09 07:00:00.000', '2024-11-09 19:00:00.000', 
        '2024-11-10 07:00:00.000', '2024-11-10 19:00:00.000', 
        '2024-11-11 07:00:00.000', '2024-11-11 19:00:00.000', 
        '2024-11-12 07:00:00.000', '2024-11-12 19:00:00.000',
         ]
        
    elif params['intersec_id'] == 3032:
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
        
    elif params['intersec_id'] == 3265:
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
        
    elif params['intersec_id'] == 3334:
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
    elif params['intersec_id'] == 3252:
        times = [
            '2024-07-26 07:00:00.000', '2024-07-26 19:00:00.000',
            '2024-07-27 07:00:00.000', '2024-07-27 19:00:00.000',
            '2024-07-28 07:00:00.000', '2024-07-28 19:00:00.000',
            '2024-08-05 07:00:00.000', '2024-08-05 19:00:00.000',
            '2024-08-06 07:00:00.000', '2024-08-06 19:00:00.000',
            '2024-08-07 07:00:00.000', '2024-08-07 19:00:00.000',
            '2024-08-08 07:00:00.000', '2024-08-08 19:00:00.000',
            '2024-08-26 07:00:00.000', '2024-08-26 19:00:00.000',
            '2024-08-27 07:00:00.000', '2024-08-27 19:00:00.000', 
            '2024-08-28 07:00:00.000', '2024-08-28 19:00:00.000', 
            '2024-08-29 07:00:00.000', '2024-08-29 19:00:00.000', 
            '2024-08-30 07:00:00.000', '2024-08-30 19:00:00.000', 
            '2024-08-31 07:00:00.000', '2024-08-31 19:00:00.000', 
            '2024-09-01 07:00:00.000', '2024-09-01 19:00:00.000', 
            '2024-09-02 07:00:00.000', '2024-09-02 19:00:00.000', 
            '2024-09-05 07:00:00.000', '2024-09-05 19:00:00.000',

        ]
    else:
        raise ValueError('Invalid intersection ID')

    omega = pd.DataFrame()

    for i in range(0, len(times), 2):
        params['start_date'] = times[i]
        params['end_date'] = times[i+1]
        params['start_date_datetime_object'] = pd.to_datetime(params['start_date'])
        params['end_date_datetime_object'] = pd.to_datetime(params['end_date'])

        my = MySQLConnector()

        with yaspin(Spinners.pong, text=f"Fetching data from MySQL starting at {times[i]}") as sp:
            df = my.handleRequest(params, df_type)

            
        df['day_of_week'] = df['timestamp'].dt.day_name()


        # print(df.to_string())
        # describe df
        # Convert 'day_of_week' to categorical to maintain order of days in the heatmap
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
        ordered=True)

        # print(df)
        # print(params['start_date'], params['end_date'])
        # time.sleep(2)

        # print(df.head())
        # print(len(df))
        omega = pd.concat([omega, df])

    # print(omega['timestamp'].unique())
    # time.sleep(1000)

    if df_type == 'track':
        if pedestrian_counting:
            omega = omega[omega['class'] == 'pedestrian']
        else:
            omega = omega[omega['class'] != 'pedestrian']
    else:
        if p2v:
            if conflict_type == 'left turning':
                omega = omega[((omega['p2v']==1)&((omega['conflict_type'] == 3)|(omega['conflict_type'] == 4)))]
            elif conflict_type == 'right turning':
                omega = omega[((omega['p2v']==1)&((omega['conflict_type'] == 1)|(omega['conflict_type'] == 2)))]
            elif conflict_type == 'thru':
                omega = omega[((omega['p2v']==1)&((omega['conflict_type'] == 5)|(omega['conflict_type'] == 6)))]
            elif conflict_type == 'all':
                omega = omega[(omega['p2v']==1)]

    column_name = 'track_id' if df_type == 'track' else 'unique_ID1'

    # Convert 'timestamp' to date only
    omega['date'] = omega['timestamp'].dt.date

    print('!')
    print(omega['date'].unique())

    # Create a new column 'track_count' that represents the count of unique conflicts for each unique combination of 'day_of_week' and 'hour_digit'
    omega['track_count'] = omega.groupby(['day_of_week', 'hour_digit'])[column_name].transform('nunique')

        
    if df_type == "track":
        if pedestrian_counting:
            title = (f"Heatmap of {'average' if mean else 'total'} pedestrian counts "
                    f"by day of week and hour\n{intersec_lookup[params['intersec_id']]}")
        else:
            title = (f"Heatmap of {'average' if mean else 'total'} vehicle counts "
                    f"by day of week and hour\n{intersec_lookup[params['intersec_id']]}")

        # For track data: peds vs vehs
        track_label = "peds" if pedestrian_counting else "vehs"
        name = (f"heatmap_{params['intersec_id']}_{track_label}_{df_type}_"
                f"{'mean' if mean else 'sum'}"
                f"_{conflict_type.replace(' ', '_') if conflict_type else 'all'}")

    elif df_type == "conflict":
        if p2v:
            title = (f"Heatmap of {'average' if mean else 'total'} pedestrian-vehicle conflicts "
                    f"by day of week and hour\n{intersec_lookup[params['intersec_id']]} "
                    f"- {conflict_type.title()} Vehicles")
        else:
            title = (f"Heatmap of {'average' if mean else 'total'} vehicle-vehicle conflicts "
                    f"by day of week and hour\n{intersec_lookup[params['intersec_id']]}")

        # For conflict data: p2v vs v2v
        conflict_label = "p2v" if p2v else "v2v"
        name = (f"heatmap_{params['intersec_id']}_{conflict_label}_{df_type}_"
                f"{'mean' if mean else 'sum'}"
                f"_{conflict_type.replace(' ', '_') if conflict_type else 'all'}")

    if not mean:
        # Convert 'timestamp' to date only and create a new column 'date_weekday_short' that combines date and abbreviated weekday

        
        omega['date'] = omega['timestamp'].dt.date
        omega['weekday_short'] = omega['timestamp'].dt.day_name().str[:3]
        omega['date_weekday_short'] = omega['date'].astype(str) + ' (' + omega['weekday_short'] + ')'
        
        # Create a pivot table
        pivot_table = omega.pivot_table(values=column_name, index='date_weekday_short', columns='hour_digit', aggfunc='count', fill_value=0)

        # Create a heatmap
        plt.figure(figsize=(10, 6))
        if not p2v:
            vmax = 20 if df_type == 'conflict' else 4200
            cmap = 'YlGnBu'
        else:
            vmax = 40 if df_type == 'conflict' else 4200
            cmap = 'inferno'

        if df_type == 'track':
            cmap = 'viridis'

        sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt='.0f', vmin=0, vmax=vmax, annot_kws={"size": 10},
                    cbar_kws={'format': plt.FuncFormatter(lambda x, pos: f'{int(x)}')})

        # title = f'Heatmap of {"average" if mean else "total"} track counts by date and hour\n{intersec_lookup[params["intersec_id"]]}' if df_type == 'track' else f'Heatmap of {"average" if mean else "total"} {"V2V" if not p2v else "P2V"} conflict counts by date and hour\n{intersec_lookup[params["intersec_id"]]}{' - ' + conflict_type.title() + ' Vehicles' if conflict_type else ''}'
        plt.title(title, fontsize=14)
        plt.xlabel('Hour of Day')
        plt.ylabel('Date (Weekday)')

        #change here 1
        #plt.savefig(f'{name}.pdf', bbox_inches='tight')
        #plt.savefig(f'{name}.png', bbox_inches='tight')

        # Create a line plot
        plt.figure(figsize=(10, 6))
        for day in pivot_table.index:
            plt.plot(pivot_table.columns, pivot_table.loc[day], label=day)

        plt.title(title.replace('Heatmap', 'Lineplot'))
        plt.xlabel('Hour of Day')
        plt.ylabel('Track Count')
        plt.legend(title='Day of Week', loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        # Set x-ticks to all hours
        plt.xticks(pivot_table.columns)
        # Set y-ticks to integer format
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
        plt.ylim(0, vmax)

        #plt.savefig(f'{name.replace('heatmap', 'lineplot')}_lineplot.pdf', bbox_inches='tight')
        #plt.savefig(f'{name.replace('heatmap', 'lineplot')}_lineplot.png', bbox_inches='tight')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        
        base64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return base64_img


    omega['hour'] = omega['timestamp'].dt.hour
    
    # Filter data to keep only non-zero track counts
    non_zero_tracks = omega.groupby(['hour', 'day_of_week', 'date']).size().reset_index(name='track_count')
    non_zero_tracks = non_zero_tracks[non_zero_tracks['track_count'] > 0]
    # non_zero_tracks = non_zero_tracks

    # Calculate the number of unique days for each hour and weekday with non-zero track counts
    unique_days_per_hour_and_weekday = non_zero_tracks.groupby(['hour', 'day_of_week'])['date'].nunique().reset_index(name='unique_day_count')

    # Create a DataFrame with the number of tracks per hour and per weekday
    tracks_per_hour_and_weekday = omega.groupby(['hour', 'day_of_week']).size().reset_index(name='track_count')

    # Merge the unique days DataFrame with the track counts DataFrame
    merged_df = pd.merge(tracks_per_hour_and_weekday, unique_days_per_hour_and_weekday, on=['hour', 'day_of_week'], how='left')

    # Fill NaN values in unique_day_count with 1 to handle cases where there are no non-zero track counts
    merged_df['unique_day_count'] = merged_df['unique_day_count'].fillna(1)

    # Calculate the average track count
    merged_df['average_track_count'] = merged_df['track_count'] / merged_df['unique_day_count']

    # Create a pivot table with 'average_track_count' as values, 'day_of_week' as index, and 'hour' as columns
    pivot_table = merged_df.pivot_table(
        values='average_track_count', 
        index='day_of_week', 
        columns='hour', 
        fill_value=0
    )
    # Ensure that the pivot table includes all hours from 7 to 18
    for hour in range(7, 19):
        if hour not in pivot_table.columns:
            pivot_table[hour] = 0

    # Sort the columns to ensure hours are in order
    pivot_table = pivot_table.sort_index(axis=1)


    print(pivot_table)

    vmax = 35 if df_type == 'conflict' else 150

    if not p2v:        
        cmap = 'YlGnBu'
    else:
        cmap = 'inferno'

    if df_type == 'track':
        if not pedestrian_counting:
            vmax = 7000
        cmap = 'YlGnBu'

    # Create a heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt='.0f', vmin=0, vmax=vmax, annot_kws={"size": 10},
            cbar_kws={'format': plt.FuncFormatter(lambda x, pos: f'{int(x)}')})

    if df_type == "track":
        if pedestrian_counting:
            print('number of pedestrians:', len(omega[omega['class'] == 'pedestrian']))
            title = f'Heatmap of average pedestrian counts by day of week and hour\n{intersec_lookup[params["intersec_id"]]}' 
        else:
            print('number of vehicles:', len(omega[omega['class'] != 'pedestrian']))
            title = f'Heatmap of average vehicle counts by day of week and hour\n{intersec_lookup[params["intersec_id"]]}'


    #change here 2 
    
    else:
        if p2v:
            title = f'Heatmap of average pedestrian-vehicle conflicts by day of week and hour\n{intersec_lookup[params["intersec_id"]]} - {conflict_type.title()} Vehicles' 
        else:
            title = f'Heatmap of average vehicle-vehicle conflicts by day of week and hour\n{intersec_lookup[params["intersec_id"]]}' 
    # plt.title(title, fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    # name = f'heatmap_{params['intersec_id']}_{"peds" if pedestrian_counting else "vehs"}_{df_type}_{"mean" if mean else "sum"}_{"p2v" if p2v else "v2v"}_{conflict_type.replace(" ", "_") if conflict_type else "all"}'
    
    #plt.savefig(f'{name}.pdf', bbox_inches='tight')
    #plt.savefig(f'{name}.png', bbox_inches='tight', dpi=600)

    # Create a line plot with hours on the x-axis, average track count on the y-axis, and different colors for each day of the week
    plt.figure(figsize=(6, 5))
    for day in pivot_table.index:
        plt.plot(pivot_table.columns, pivot_table.loc[day], label=f'{day}')

    if df_type == 'conflict':
        vmax = 40
    else:
        if pedestrian_counting:
            vmax = 120
        else:
            vmax = 7000
    # vmax = 25 if df_type == 'conflict' else 150
    
        
    # plt.title(title.replace('Heatmap', 'Lineplot'), fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    if df_type == 'track':
        plt.ylabel('Average Track Count', fontsize=12)
    else:
        plt.ylabel('Average Conflict Count', fontsize=12)
    # plt.ylabel('Average Track Count', fontsize=12)
    plt.legend(title='Day of Week', loc='upper left', bbox_to_anchor=(1, 1))
    # Set x-ticks to all hours
    plt.xticks(pivot_table.columns, fontsize=10)
    plt.yticks(fontsize=10)
    # Set y-ticks to integer format
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    plt.ylim(0, vmax)
    plt.grid(True)

    #plt.savefig(f'{name.replace('heatmap', 'lineplot')}_lineplot.pdf', bbox_inches='tight')
    #plt.savefig(f'{name.replace('heatmap', 'lineplot')}_lineplot.png', bbox_inches='tight', dpi=600)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
        
    base64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # === NEW: Optionally return some aggregated data for normalization ===
    if return_agg:
        # Example: group by day_of_week and count unique IDs
        # (You can adapt or extend this as needed)
        agg_df = (omega.groupby(['day_of_week', 'date'])[column_name]
                  .nunique()
                  .reset_index(name='count'))
        
        # Summarize by day_of_week across all dates in this dataset
        agg_day = agg_df.groupby('day_of_week')['count'].sum().reset_index(name='total_count')
        # Number of distinct days used
        days_per_dw = agg_df.groupby('day_of_week')['date'].nunique().reset_index(name='num_days')
        
        # Merge to get total_count and num_days side by side
        merged = pd.merge(agg_day, days_per_dw, on='day_of_week', how='left')
        # Example: an average count
        merged['average_count'] = merged['total_count'] / merged['num_days']

        return merged  # <--- Return your aggregated data

    # If you do NOT want to return data, just let the function exit
    # after plotting
    return base64_img


def calculate_conflict_rates(conflict_counts_df, volume_counts_df, volume_type='vehicle'):
    """
    Takes two aggregated DataFrames that each have:
      - 'day_of_week'
      - 'total_count' (or some numeric count)
      - 'num_days'
      - 'average_count' (if you want)
    and calculates conflicts per 1,000 <volume_type>.
    
    conflict_counts_df: aggregated DataFrame of conflict data
      columns: [day_of_week, total_count, num_days, average_count, ...]
    volume_counts_df: aggregated DataFrame of volume data (vehicles/peds)
      columns: [day_of_week, total_count, num_days, average_count, ...]
    volume_type: str, e.g. 'vehicle' or 'pedestrian'
    
    Returns a merged DataFrame with new columns:
      - conflicts_per_1000_<volume_type>
      - average_conflicts_per_1000_<volume_type>
    """
    # Rename to avoid collisions
    conflict_df = conflict_counts_df.rename(
        columns={'total_count': 'conflict_count', 'average_count': 'avg_conflict_count'}
    )
    volume_df = volume_counts_df.rename(
        columns={'total_count': f'{volume_type}_count', 'average_count': f'avg_{volume_type}_count'}
    )

    # Merge on day_of_week
    merged = pd.merge(
        conflict_df[['day_of_week', 'conflict_count', 'avg_conflict_count']],
        volume_df[['day_of_week', f'{volume_type}_count', f'avg_{volume_type}_count']],
        on='day_of_week',
        how='outer'  # or 'inner' if you only want matching days
    )

    # Calculate conflicts per 1,000 <volume_type>
    merged[f'conflicts_per_1000_{volume_type}'] = np.where(
        merged[f'{volume_type}_count'] > 0,
        (merged['conflict_count'] / merged[f'{volume_type}_count']) * 1000,
        0
    )

    merged[f'avg_conflicts_per_1000_{volume_type}'] = np.where(
        merged[f'avg_{volume_type}_count'] > 0,
        (merged['avg_conflict_count'] / merged[f'avg_{volume_type}_count']) * 1000,
        0
    )

    return merged

    
def plot_normalized_conflicts(
    df: pd.DataFrame,
    day_col: str,
    rate_col: str,
    intersection_id: int,
    plot_title: str,
    output_filename: str
):
    """
    Plot the specified 'rate_col' of 'df' against 'day_col', 
    saving the figure to 'output_filename'.

    :param df: DataFrame that includes day_of_week and a 
               conflicts-per-1,000 column (or average conflicts).
    :param day_col: Name of the column representing the day of week (e.g., "day_of_week").
    :param rate_col: Name of the column in 'df' that has the conflict rate 
                     (e.g., "avg_conflicts_per_1000_pedestrian").
    :param intersection_id: Intersection ID, used for labeling.
    :param plot_title: Title for the plot.
    :param output_filename: File path/name to save the figure.
    """
    if df.empty:
        print(f"[WARNING] No data to plot for intersection {intersection_id}: '{plot_title}'.")
        return

    # Ensure the day_col is a categorical in a sensible order
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df[day_col] = pd.Categorical(df[day_col], categories=day_order, ordered=True)

    # Sort DataFrame by that day_col
    df.sort_values(day_col, inplace=True)

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        df[day_col],
        df[rate_col],
        marker='o',
        label=plot_title
    )
    plt.title(plot_title, fontsize=14)

    plt.ylim(0, 10)

    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel(rate_col.replace('_', ' ').title(), fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Optionally add a legend
    plt.legend()

    # Save the plot

    #change here 3

    #plt.savefig(output_filename, dpi=300)
    #print(f"Saved normalized plot: {output_filename}")
    #plt.close()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
        
    base64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return base64_img



##################### main prg #######################
##
## first, p2v
##
# conflict or track
# df_type = "track"
df_type = "conflict"

# mean or sum?
mean = True

p2v = False


# conflict_types = ['left turning', 'right turning', 'thru', 'all']
# for conflict_type in conflict_types:
    # heatmap_generator(df_type, mean, inter, p2v, conflict_type)

##
## v2v
##

# p2v = False

# heatmap_generator(df_type, mean, inter, p2v)


# df_type = "track"

# for inter in [3032, 3265, 3334, 3248, 3287]:
for inter in [3252, ]:  # or your list of IDs
    print(f"\n=== Processing intersection: {inter} ===")

    # (1) Gather P2V conflict + Ped Volume aggregates
    p2v_conflicts_agg = heatmap_generator(
        df_type="conflict",
        mean=True,
        intersec_id=inter,
        p2v=True,
        conflict_type='all',
        pedestrian_counting=False,
        return_agg=True
    )
    ped_volume_agg = heatmap_generator(
        df_type="track",
        mean=True,
        intersec_id=inter,
        p2v=False,
        conflict_type=None,
        pedestrian_counting=True,
        return_agg=True
    )

    # (2) Calculate P2V per 1,000 Pedestrians
    p2v_per_1000_peds = calculate_conflict_rates(
        conflict_counts_df=p2v_conflicts_agg,
        volume_counts_df=ped_volume_agg,
        volume_type='pedestrian'
    )

    print(f"\n=== Intersection {inter}: P2V Conflicts per 1,000 Pedestrians ===")
    print(p2v_per_1000_peds)

    # (3) Plot P2V per 1,000 Pedestrians
    plot_normalized_conflicts(
        df=p2v_per_1000_peds,
        day_col="day_of_week",
        rate_col="avg_conflicts_per_1000_pedestrian",  # or conflicts_per_1000_pedestrian
        intersection_id=inter,
        plot_title=f"Intersection {inter}: P2V Conflicts per 1k Pedestrians",
        output_filename=f"intersection_{inter}_p2v_per_1000_peds.png"
    )

    # (4) Gather V2V conflict + Vehicle Volume aggregates
    v2v_conflicts_agg = heatmap_generator(
        df_type="conflict",
        mean=True,
        intersec_id=inter,
        p2v=False,
        conflict_type='all',
        pedestrian_counting=False,
        return_agg=True
    )
    vehicle_volume_agg = heatmap_generator(
        df_type="track",
        mean=True,
        intersec_id=inter,
        p2v=False,
        conflict_type=None,
        pedestrian_counting=False,
        return_agg=True
    )

    # (5) Calculate V2V per 1,000 Vehicles
    v2v_per_1000_vehicles = calculate_conflict_rates(
        conflict_counts_df=v2v_conflicts_agg,
        volume_counts_df=vehicle_volume_agg,
        volume_type='vehicle'
    )

    print(f"\n=== Intersection {inter}: V2V Conflicts per 1,000 Vehicles ===")
    print(v2v_per_1000_vehicles)

    # (6) Plot V2V per 1,000 Vehicles
    plot_normalized_conflicts(
        df=v2v_per_1000_vehicles,
        day_col="day_of_week",
        rate_col="avg_conflicts_per_1000_vehicle",  # or conflicts_per_1000_vehicle
        intersection_id=inter,
        plot_title=f"Intersection {inter}: V2V Conflicts per 1k Vehicles",
        output_filename=f"intersection_{inter}_v2v_per_1000_vehicles.png"
    )

#if the heatmap() does not have return_agg = True, then binary image data should be returned
#to find changes if you ever want to revert, ctrl f and then look up change. 
#Remove lines relating to base64 and uncomment other lines