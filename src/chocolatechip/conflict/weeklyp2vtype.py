from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
from yaspin import yaspin
from yaspin.spinners import Spinners
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_times(iid: int, period: str = 'before') -> list:
    try:
        return times_dict[iid][period]
    except KeyError:
        raise ValueError(f"Invalid intersection ID ({iid}) or period ({period}) in times config.")

def fetch_or_cache_data(my, iid, start_time, end_time, p2v, df_type='track'):
    cache_filename = f"cache_{iid}"
    if df_type == 'track':
        cache_filename += "_track"
    elif df_type == 'trackthru':
        cache_filename += "_trackthru"
    else:
        cache_filename += "_conflict"
    cache_filename += f"_{start_time.replace(':', '').replace('-', '').replace(' ', '_')}_{end_time.replace(':', '').replace('-', '')}.csv"
    cache_filename = os.path.join('cache', cache_filename)

    if not os.path.isdir('cache'):
        os.mkdir('cache')

    if os.path.exists(cache_filename):
        if df_type in ['track', 'trackthru']:
            df = pd.read_csv(cache_filename, parse_dates=['start_timestamp', 'end_timestamp'])
        else:
            df = pd.read_csv(cache_filename)
    else:
        if df_type == 'track':
            df = my.query_tracksreal(iid, start_time, end_time)
        elif df_type == 'trackthru':
            df = my.query_tracksreal(iid, start_time, end_time, True)
        elif df_type == 'conflict':
            params = {
                'intersec_id': iid,
                'start_date': start_time,
                'end_date': end_time,
                'p2v': p2v,
            }
            df = my.handleRequest(params, 'conflict')
                
        df.to_csv(cache_filename, index=False)
        print(f"\n\tData cached to file: {cache_filename}")

    return df

def get_intersection_data(iid, p2v, period='before'):
    my = MySQLConnector()
    ttc_df = pd.DataFrame()
    times = get_times(iid, period=period)
    for i in range(0, len(times), 2):
        start_time = times[i]
        end_time = times[i+1]
        with yaspin(Spinners.earth, text=f"Fetching data from MySQL starting at {start_time}") as sp:
            ttc_df = pd.concat([ttc_df, fetch_or_cache_data(my, iid, start_time, end_time, p2v, 'conflict')], ignore_index=True)
    ttc_df['unique_ID1'] = ttc_df['unique_ID1'].astype(str)
    ttc_df['unique_ID2'] = ttc_df['unique_ID2'].astype(str)
    return ttc_df

def count_conflicts_by_type(df, conflict_type):
    if conflict_type == 'P2V':
        types = ['Left Turning Vehs', 'Right Turning Vehs', 'Through Vehs']
    elif conflict_type == 'V2V':
        types = ['LOT', 'ROL', 'RMT']
    
    counts = {t: 0 for t in types}
    
    def extract_direction(cluster):
        return ''.join([c for c in cluster if c.isupper()])
    
    for index, row in df.iterrows():
        cluster1 = str(row['cluster1']).lower()
        cluster2 = str(row['cluster2']).lower()
        
        if conflict_type == 'P2V':
            if 'ped' in cluster1:
                direction = extract_direction(str(row['cluster2']))
            else:
                direction = extract_direction(str(row['cluster1']))
            if direction.endswith('R'):
                counts['Right Turning Vehs'] += 1
            elif direction.endswith('L'):
                counts['Left Turning Vehs'] += 1
            elif direction.endswith('T'):
                counts['Through Vehs'] += 1
        elif conflict_type == 'V2V':
            direction1 = extract_direction(str(row['cluster1']))
            direction2 = extract_direction(str(row['cluster2']))
            if (direction1.endswith('L') and direction2.endswith('T')) or (direction1.endswith('T') and direction2.endswith('L')):
                counts['LOT'] += 1
            elif (direction1.endswith('R') and direction2.endswith('T')) or (direction1.endswith('T') and direction2.endswith('R')):
                counts['RMT'] += 1
            elif (direction1.endswith('R') and direction2.endswith('L')) or (direction1.endswith('L') and direction2.endswith('R')):
                counts['ROL'] += 1

    return pd.Series(counts)

def analyze_and_plot(iid, p2v, period):
    print(f"Running analysis for intersection {iid} ({period})")
    ttc_df = get_intersection_data(iid, p2v, period=period)
    
    # Convert the timestamp column to datetime and extract the week number
    ttc_df['start_time'] = pd.to_datetime(ttc_df['timestamp'], errors='coerce')
    ttc_df['week_number'] = ttc_df['start_time'].dt.isocalendar().week

    weekly_counts_list = []
    weeks = ttc_df['week_number'].unique()

    for week in weeks:
        df_week = ttc_df[ttc_df['week_number'] == week]
        counts = count_conflicts_by_type(df_week, 'P2V')
        counts['week_number'] = week
        weekly_counts_list.append(counts)

    weekly_counts = pd.DataFrame(weekly_counts_list)
    average_counts = weekly_counts[['Left Turning Vehs', 'Right Turning Vehs', 'Through Vehs']].mean().reset_index()
    average_counts.columns = ['Movement Type', 'Average Count']

    print("Average weekly conflicts by movement type:")
    print(average_counts)

    # Plot the average counts with a fixed y-limit
    plt.figure(figsize=(7, 5))
    plt.bar(average_counts['Movement Type'], average_counts['Average Count'], color=['blue', 'green', 'red'])
    plt.xlabel('Movement Type')
    plt.ylabel('Average Number of Conflicts per Week')
    plt.ylim(0, 100)
    plt.title(f'Average Weekly P2V Conflicts by Movement Type at Intersection {iid} ({period.capitalize()})')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    filename_png = f'average_p2v_conflicts_by_movement_type_intersection_{iid}_{period}.png'
    filename_pdf = f'average_p2v_conflicts_by_movement_type_intersection_{iid}_{period}.pdf'
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    plt.savefig(filename_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plots as {filename_png} and {filename_pdf}")

# Main Loop: Run analysis for both 'before' and 'after'
iid = 3287
p2v = '1'
for period in ['before', 'after']:
    analyze_and_plot(iid, p2v, period)
