from chocolatechip.MySQLConnector import MySQLConnector
from cloudmesh.common.util import path_expand
from yaspin import yaspin
from yaspin.spinners import Spinners
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import time
import os

def validate_inputs(df_tyope: str, p2v: bool, conflict_type: str):
    if df_type not in ['track', 'conflict']:
        raise ValueError('df_type must be "track" or "conflict"')
    
    if df_type == 'conflict' and p2v is None:
        raise ValueError('p2v must be True or False when df_type is "conflict"')
    
    if p2v is False and conflict_type in ['left turning', 'right turning', 'thru']:
        raise ValueError('try commenting the three lines and uncommenting the one, or make p2v true')
    
def fetch_and_process_data(times: list, params: dict, df_type: str):
    omega = pd.DataFrame()
    my = MySQLConnector()

    for i in range(0, len(times), 2):
        params['start_date'] = times[i]
        params['end_date'] = times[i + 1]
        params['start_date_datetime_object'] = pd.to_datetime(params['start_date'])
        params['end_date_datetime_object'] = pd.to_datetime(params['end_date'])

        with yaspin(Spinners.pong, text=f"Fetching {df_type} {"P2V" if params['p2v'] is 1 else "V2V"} data from MySQL for {params['intersec_id']} starting at {times[i]}") as sp:
            df = my.handleRequest(params, df_type)

        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=[
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)

        omega = pd.concat([omega, df])

    return omega

def filter_data(df_type: str, omega: pd.DataFrame, p2v: bool, conflict_type: str):
    if df_type == 'track':
        omega = omega[omega['class'] == 'pedestrian']           # possible issue source for problem 1 and 2
    else:
        if p2v:
            conflict_type_filter = {
                'left turning': [3, 4],
                'right turning': [1, 2],
                'thru': [5, 6],
                'all': []
            }
            if conflict_type != 'all':
                omega = omega[(omega['p2v'] == 1) & (omega['conflict_type'].isin(conflict_type_filter[conflict_type]))]
            else:
                omega = omega[(omega['p2v'] == 1)]

    return omega

def create_pivot_table(omega: pd.DataFrame, column_name: str):
    omega['weekday_short'] = omega['timestamp'].dt.day_name().str[:3]
    omega['date_weekday_short'] = omega['date'].astype(str) + ' (' + omega['weekday_short'] + ')'

    return omega.pivot_table(values=column_name, index='date_weekday_short', columns='hour_digit', aggfunc='count', fill_value=0)

def plot_heatmap(pivot_table: pd.DataFrame, df_type: str, mean: bool, params: dict, intersec_lookup: dict, p2v: bool, conflict_type: str):
    plt.figure(figsize=(10, 6))
    vmax = 20 if df_type == 'conflict' else 4200
    cmap = 'YlGnBu' if not p2v else 'inferno'
    if df_type == 'track':
        cmap = 'viridis'

    sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt='.0f', vmin=0, vmax=vmax, annot_kws={"size": 10},
                cbar_kws={'format': plt.FuncFormatter(lambda x, pos: f'{int(x)}')})

    title = f'Heatmap of {"average" if mean else "total"} track counts by date and hour\n{intersec_lookup[params["intersec_id"]]}' if df_type == 'track' else f'Heatmap of {"average" if mean else "total"} {"V2V" if not p2v else "P2V"} conflict counts by date and hour\n{intersec_lookup[params["intersec_id"]]}{" - " + conflict_type.title() + " Vehicles" if conflict_type else ""}'
    plt.title(title, fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Date (Weekday)')
    name = f'heatmap_{params["intersec_id"]}_{df_type}_{"mean" if mean else "sum"}_{"p2v" if p2v else "v2v"}_{conflict_type.replace(" ", "_") if conflict_type else "all"}'

    i_id = str(params.get('intersec_id'))
    os.makedirs(i_id, exist_ok=True)
    plt.savefig(f'{i_id}/{name}.pdf', bbox_inches='tight')
    plt.savefig(f'{i_id}/{name}.png', bbox_inches='tight')

def plot_lineplot(pivot_table: pd.DataFrame, df_type: str, mean: bool, params: dict, intersec_lookup: dict, p2v: bool, conflict_type: str):
    plt.figure(figsize=(10, 6))
    for day in pivot_table.index:
        plt.plot(pivot_table.columns, pivot_table.loc[day], label=day)

    title = f'Lineplot of {"average" if mean else "total"} {""}track counts by date and hour\n{intersec_lookup[params["intersec_id"]]}' if df_type == 'track' else f'Lineplot of {"average" if mean else "total"} {"V2V" if not p2v else "P2V"} conflict counts by date and hour\n{intersec_lookup[params["intersec_id"]]}{" - " + conflict_type.title() + " Vehicles" if conflict_type else ""}'
    plt.title(title, fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Track Count')
    plt.legend(title='Day of Week', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.xticks(pivot_table.columns)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    vmax = 20 if df_type == 'conflict' else 4200
    plt.ylim(0, vmax)
    name = f'lineplot_{params["intersec_id"]}_{df_type}_{"mean" if mean else "sum"}_{"p2v" if p2v else "v2v"}_{conflict_type.replace(" ", "_") if conflict_type else "all"}'

    i_id = str(params.get('intersec_id'))
    os.makedirs(i_id, exist_ok=True)
    plt.savefig(f'{i_id}/{name}.pdf', bbox_inches='tight')
    plt.savefig(f'{i_id}/{name}.png', bbox_inches='tight')

def create_visualizations(df_type: str, omega: pd.DataFrame, mean: bool, params: dict, intersec_lookup: dict, p2v: bool, conflict_type: str):
    column_name = 'track_id' if df_type == 'track' else 'unique_ID1'
    omega['date'] = omega['timestamp'].dt.date
    omega['track_count'] = omega.groupby(['day_of_week', 'hour_digit'])[column_name].transform('nunique')

    if not mean:
        pivot_table = create_pivot_table(omega, column_name)
        plot_heatmap(pivot_table, df_type, mean, params, intersec_lookup, p2v, conflict_type)
        plot_lineplot(pivot_table, df_type, mean, params, intersec_lookup, p2v, conflict_type)
    else:
        calculate_average_track_count(omega, df_type, params, intersec_lookup, p2v, conflict_type)

def calculate_average_track_count(omega: pd.DataFrame, df_type: str, params: dict, intersec_lookup: dict, p2v: bool, conflict_type: str):
    omega['hour'] = omega['timestamp'].dt.hour
    non_zero_tracks = omega.groupby(['hour', 'day_of_week', 'date']).size().reset_index(name='track_count')
    non_zero_tracks = non_zero_tracks[non_zero_tracks['track_count'] > 0]

    unique_days_per_hour_and_weekday = non_zero_tracks.groupby(['hour', 'day_of_week'])['date'].nunique().reset_index(name='unique_day_count')
    tracks_per_hour_and_weekday = omega.groupby(['hour', 'day_of_week']).size().reset_index(name='track_count')

    merged_df = pd.merge(tracks_per_hour_and_weekday, unique_days_per_hour_and_weekday, on=['hour', 'day_of_week'], how='left')
    merged_df['unique_day_count'] = merged_df['unique_day_count'].fillna(1)
    merged_df['average_track_count'] = merged_df['track_count'] / merged_df['unique_day_count']

    pivot_table = merged_df.pivot_table(
        values='average_track_count', 
        index='day_of_week', 
        columns='hour', 
        fill_value=0
    )

    for hour in range(7, 19):
        if hour not in pivot_table.columns:
            pivot_table[hour] = 0

    pivot_table = pivot_table.sort_index(axis=1)

    plot_heatmap(pivot_table, df_type, True, params, intersec_lookup, p2v, conflict_type)
    plot_lineplot(pivot_table, df_type, True, params, intersec_lookup, p2v, conflict_type)


##################### main prg #######################

intersec_lookup = {
        3287: "Stirling Road and N 68th Avenue",
        3248: "Stirling Road and N 66th Avenue",
        3032: "Stirling Road and SR-7",
        3265: "Stirling Road and University Drive",
        3334: "Stirling Road and Carriage Hills Drive/SW 61st Avenue",
}

cam_lookup = {
    3287: 24,
    3248: 27,
    3032: 23,
    3265: 30,
    3334: 33,
}

'''
3334	Stirling Road and Carriage Hills Drive/SW 61st Avenue	31,32,33
3265	Stirling Road and University Drive	28,29,30
3248	Stirling Road and N 66th Avenue	25,26,27
3287	Stirling Road and N 68th Avenue	24
3032	Stirling Road and SR-7	21,22,23
'''
# intersection ids
inter = [3334, 3265, 3248, 3287, 3032]

# p2v or v2v
p2v = [True, False]

# conflict types
conflict_types = ['left turning', 'right turning', 'thru', 'all']

# conflict or track
df_types = ["track", "conflict"]

# mean or sum?
mean = [True, False]

for intersec_id in inter:
    for vehicle_type in p2v:
        for df_type in df_types:
            params = {
                'start_date': '2024-02-26 07:00:00',
                'end_date': '2024-02-27 00:00:00',
                'intersec_id': intersec_id,
                'cam_id': cam_lookup[intersec_id],
                'p2v': 0 if vehicle_type is False else 1
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
            else:
                raise ValueError('Invalid intersection ID')

            omega = fetch_and_process_data(times, params, df_type)
            if df_type == 'track':
                conflict_type = None
                for stat in mean:
                    omega = filter_data(df_type, omega, vehicle_type, conflict_type)
                    create_visualizations(df_type, omega, stat, params, intersec_lookup, vehicle_type, conflict_type)
            else:
                for conflict_type in conflict_types:
                    for stat in mean:
                        omega = filter_data(df_type, omega, vehicle_type, conflict_type)
                        create_visualizations(df_type, omega, stat, params, intersec_lookup, vehicle_type, conflict_type)
