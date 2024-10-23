from chocolatechip.MySQLConnector import MySQLConnector
from yaspin import yaspin
from yaspin.spinners import Spinners
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

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

def fetch_or_cache_data(my, iid, start_time, end_time):
    cache_filename = f"cache_{iid}_{start_time.replace(':', '').replace('-', '').replace(' ', '_')}_{end_time.replace(':', '').replace('-', '').replace(' ', '_')}.csv"
    cache_filename = os.path.join('cache', cache_filename)

    if not os.path.isdir('cache'):
        os.mkdir('cache')

    if os.path.exists(cache_filename):
        # print(f"Loading data from cache file: {cache_filename}")
        df = pd.read_csv(cache_filename, parse_dates=['start_timestamp', 'end_timestamp'])
    else:
        # print(f"Fetching data from MySQL for {start_time} to {end_time}")
        df = my.query_tracksreal(iid, start_time, end_time)
        df.to_csv(cache_filename, index=False)
        print(f"Data cached to file: {cache_filename}")
    
    return df


def speed_plot(iid: int):
    my = MySQLConnector()
    mega_df = pd.DataFrame()
    times = get_times(iid)
    for i in range(0, len(times), 2):
        start_time = times[i]
        end_time = times[i+1]
        # print(f"start: {start_time}, end: {end_time}")
        with yaspin(Spinners.earth, text=f"Fetching data from MySQL starting at {start_time}") as sp:
            df = fetch_or_cache_data(my, iid, start_time, end_time)
            mega_df = pd.concat([mega_df, df])

    print(mega_df['Approach'].unique())

    # First, ensure 'start_timestamp' is a datetime type (if not already converted)
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

    # Add a column for the day of the week
    mega_df['day_of_week'] = mega_df['start_timestamp'].dt.day_name()

    # Reorder the days of the week to start from Monday
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

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
            'NS': 30,
            'EW': 45,
        },
        3248: {
            'NS': 30,
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

    # List of approaches to create separate graphs for
    approaches = ['WBT', 'EBT', 'NBT', 'SBT']

    for approach in approaches:
        # Filter the DataFrame for the current approach
        approach_df = mega_df[mega_df['Approach'] == approach]
        
        # Set the speed limit based on the approach
        if approach in ['NBT', 'SBT']:
            speed_limit = mph[iid]['NS']
        else:
            speed_limit = mph[iid]['EW']
        
        # Add a column to identify speeding vehicles based on the current speed limit
        approach_df['is_speeding'] = approach_df['average_speed'] > speed_limit
        
        # Create a pivot table to calculate the percentage of speeding vehicles
        pivot_table = approach_df.pivot_table(
            values='is_speeding', 
            index='day_of_week', 
            columns='hour_of_day_standard', 
            aggfunc=lambda x: 100 * x.sum() / x.count()
        )

        # Reorder the days of the week to start from Monday and hours to be in the desired order
        pivot_table = pivot_table.reindex(columns=hour_order, index=days_order)

        # Drop columns with all NaN values
        pivot_table = pivot_table.dropna(axis=1, how='all')

        # Fill missing values with zeros
        pivot_table = pivot_table.fillna(0)


        plt.figure(figsize=(7, 4))
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.1f', 
                    cbar_kws={'label': 'Percentage of Speeding Vehicles (%)'}, vmin=0, vmax=15,
                    annot_kws={"size": 10})  # Change the annotation size

        # plt.title(f'Percentage of Speeding Vehicles ({approach})\n{intersec_lookup[iid]} (Speed Limit: {speed_limit} mph)', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Day of Week', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.savefig(f'{iid}_average_speed_by_hour_{approach}.png', bbox_inches='tight', pad_inches=0.1)
        plt.savefig(f'{iid}_average_speed_by_hour_{approach}.pdf', bbox_inches='tight', pad_inches=0.1)

for intersec in [3032, 3265, 3334, 3248, 3287]:
    speed_plot(intersec)