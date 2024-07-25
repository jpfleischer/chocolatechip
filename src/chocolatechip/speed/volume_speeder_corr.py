from chocolatechip.MySQLConnector import MySQLConnector
from yaspin import yaspin
from yaspin.spinners import Spinners
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import pearsonr

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
        '2024-04-03 07:00:00.000', '2024-04-03 19:00:00.000']
        ''',
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
            ]'''
        
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

def fetch_or_cache_data(my, iid, start_time, end_time, df_type='track'):
    cache_filename = f"cache_{iid}"
    if df_type == 'track':
        cache_filename += "_track"
    else:
        cache_filename += "_conflict"
    cache_filename += f"_{start_time.replace(':', '').replace('-', '').replace(' ', '_')}_{end_time.replace(':', '').replace('-', '').replace(' ', '_')}.csv"
    cache_filename = os.path.join('cache', cache_filename)

    if not os.path.isdir('cache'):
        os.mkdir('cache')

    if os.path.exists(cache_filename):
        if df_type == 'track':
            df = pd.read_csv(cache_filename, parse_dates=['start_timestamp', 'end_timestamp'])
        else:
            df = pd.read_csv(cache_filename)
    else:
        if df_type == 'track':
            df = my.query_tracksreal(iid, start_time, end_time)
        else:
            params = {
                'intersec_id': iid,
                'start_date': start_time,
                'end_date': end_time
            }
            df = my.handleRequest(params, 'speedcorr')
        df.to_csv(cache_filename, index=False)
        print(f"Data cached to file: {cache_filename}")
    
    return df

def speed_plot(iid: int):
    my = MySQLConnector()
    mega_df = pd.DataFrame()
    times = get_times(iid)  # Fetch times based on intersection ID
    ttc_df = pd.DataFrame()
    for i in range(0, len(times), 2):
        start_time = times[i]
        end_time = times[i+1]
        with yaspin(Spinners.earth, text=f"Fetching data from MySQL starting at {start_time}") as sp:
            df = fetch_or_cache_data(my, iid, start_time, end_time)
            ttc_df = pd.concat([ttc_df, fetch_or_cache_data(my, iid, start_time, end_time, 'conflict')])
            mega_df = pd.concat([mega_df, df])

    # Ensure 'start_timestamp' is a datetime type
    mega_df['start_timestamp'] = pd.to_datetime(mega_df['start_timestamp'])

    # Extract the hour of day from the 'start_timestamp' column
    mega_df['hour_of_day'] = mega_df['start_timestamp'].dt.hour

    # Create a dictionary to map military hours to standard time with AM/PM
    hour_mapping = {
        0: '12 AM', 1: '1 AM', 2: '2 AM', 3: '3 AM', 4: '4 AM', 5: '5 AM', 6: '6 AM', 7: '7 AM', 8: '8 AM', 9: '9 AM', 10: '10 AM', 11: '11 AM',
        12: '12 PM', 13: '1 PM', 14: '2 PM', 15: '3 PM', 16: '4 PM', 17: '5 PM', 18: '6 PM', 19: '7 PM', 20: '8 PM', 21: '9 PM', 22: '10 PM', 23: '11 PM'
    }

    # Map the military hours to standard time
    mega_df['hour_of_day_standard'] = mega_df['hour_of_day'].map(hour_mapping)

    # Create a list with the desired order of hours in standard time
    hour_order = [
        '12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM', '11 AM',
        '12 PM', '1 PM', '2 PM', '3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM'
    ]

    intersec_lookup = {
        3287: "Stirling Road and N 68th Avenue",
        3248: "Stirling Road and N 66th Avenue",
        3032: "Stirling Road and SR-7",
        3265: "Stirling Road and University Drive",
        3334: "Stirling Road and Carriage Hills Drive/SW 61st Avenue",
    }

    mph = {
        3287: {
            'NS': 45,
            'EW': 45,
        },
        3248: {
            'NS': 45,
            'EW': 45,
        },
        3032: {
            'NS': 45,
            'EW': 45,
        },
        3265: {
            'NS': 45,
            'EW': 45,
        },
        3334: {
            'NS': 30,
            'EW': 45,
        },
    }

    ttc_df['unique_ID1'] =  ttc_df['unique_ID1'].astype(str)
    ttc_df['unique_ID2'] =  ttc_df['unique_ID2'].astype(str)
    mega_df['track_id'] = mega_df['track_id'].astype(str)
    mega_df['conflict'] = 0

    # Mark conflicts in mega_df based on ttc_df
    for _, row in ttc_df.iterrows():
        id1, id2 = row['unique_ID1'], row['unique_ID2']
        adj_id1, adj_id2 = "1" + id1, "1" + id2
        mega_df.loc[(mega_df['track_id'] == adj_id1) | (mega_df['track_id'] == adj_id2), 'conflict'] = 1

    # Check for speeders
    speed_limit = mph[iid]['NS']  # Assuming 'NS' for simplicity
    mega_df['speeder'] = mega_df['Max_speed'] > speed_limit

    # Group by hour of day and count the number of vehicles and speeders
    grouped = mega_df.groupby('hour_of_day').agg({'track_id': 'count', 'speeder': 'sum'}).reset_index()
    grouped.columns = ['hour_of_day', 'vehicle_count', 'speeder_count']

    # Calculate correlation between vehicle count and speeder count
    pearson_r, p_val = pearsonr(grouped['vehicle_count'], grouped['speeder_count'])

    print(f"INTERSECTION {iid} ANALYSIS")
    print(f"Pearson correlation coefficient: {pearson_r:.4f}")
    print(f"P-Value: {p_val:.4f}")

    # Plot the data
    fig, ax1 = plt.subplots()

    sns.lineplot(data=grouped, x='hour_of_day', y='vehicle_count', ax=ax1, label='Vehicle Count', color='b')
    ax2 = ax1.twinx()
    sns.lineplot(data=grouped, x='hour_of_day', y='speeder_count', ax=ax2, label='Speeder Count', color='r')

    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Vehicle Count', color='b')
    ax2.set_ylabel('Speeder Count', color='r')

    plt.title(f'Vehicle Count and Speeder Count by Hour of Day\nIntersection: {intersec_lookup[iid]}')
    plt.show()


for intersec in [3032, 3265, 3334]:
    speed_plot(intersec)