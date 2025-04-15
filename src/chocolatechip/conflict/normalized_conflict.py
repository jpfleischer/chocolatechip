from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
from datetime import datetime
from yaspin import yaspin
from yaspin.spinners import Spinners
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import numpy as np

def heatmap_generator(df_type: str,
                      mean: bool,
                      intersec_id: int,
                      times: list,
                      p2v: bool = None,
                      conflict_type: str = None,
                      pedestrian_counting: bool = False,
                      ):
    if df_type not in ['track', 'conflict']:
        raise ValueError('df_type must be "track" or "conflict"')
    
    if df_type == 'conflict' and p2v is None:
        raise ValueError('p2v must be True or False when df_type is "conflict"')
    
    if p2v is False and conflict_type in ['left turning', 'right turning', 'thru']:
        raise ValueError('Try commenting the three lines and uncommenting the one, or make p2v true')

    # Intersection and camera lookup dictionaries
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
        5060: 7
    }

    params = {
        'start_date': '2024-02-26 07:00:00',
        'end_date': '2024-02-27 00:00:00',
        'intersec_id': intersec_id,
        'cam_id': cam_lookup[intersec_id],
        'p2v': 0 if p2v is False else 1
    }

    omega = pd.DataFrame()

    for i in range(0, len(times), 2):
        params['start_date'] = times[i]
        params['end_date'] = times[i+1]
        params['start_date_datetime_object'] = pd.to_datetime(params['start_date'])
        params['end_date_datetime_object'] = pd.to_datetime(params['end_date'])

        my = MySQLConnector()

        with yaspin(Spinners.pong, text=f"Fetching data from MySQL starting at {times[i]}") as sp:
            df = my.handleRequest(params, df_type)
            sp.ok("✔")  # Mark the spinner as done

        print(f"Fetched {len(df)} rows for time range {params['start_date']} to {params['end_date']}")


        if df.empty:
            continue  # Skip if the DataFrame is empty

        df['day_of_week'] = df['timestamp'].dt.day_name()

        # Convert 'day_of_week' to categorical to maintain order in the heatmap
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=[
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            ordered=True)

        omega = pd.concat([omega, df], ignore_index=True)

    if omega.empty:
        # Return empty counts_df if no data is fetched
        counts_df = pd.DataFrame(columns=['day_of_week', 'count', 'num_days', 'average_count'])
        total_hours = 0
        return counts_df, total_hours

    if df_type == 'track':
        if pedestrian_counting:
            omega = omega[omega['class'] == 'pedestrian']
        else:
            omega = omega[omega['class'] != 'pedestrian']
    else:
        if p2v:
            if conflict_type == 'left turning':
                omega = omega[((omega['p2v'] == 1) & ((omega['conflict_type'] == 3) | (omega['conflict_type'] == 4)))]
            elif conflict_type == 'right turning':
                omega = omega[((omega['p2v'] == 1) & ((omega['conflict_type'] == 1) | (omega['conflict_type'] == 2)))]
            elif conflict_type == 'thru':
                omega = omega[((omega['p2v'] == 1) & ((omega['conflict_type'] == 5) | (omega['conflict_type'] == 6)))]
            elif conflict_type == 'all':
                omega = omega[(omega['p2v'] == 1)]

    column_name = 'track_id' if df_type == 'track' else 'unique_ID1'

    # Convert 'timestamp' to date only
    omega['date'] = omega['timestamp'].dt.date

    # Group by 'day_of_week' and 'date', count unique entries
    counts_per_date = omega.groupby(['day_of_week', 'date'])[column_name].nunique().reset_index(name='count')

    # Sum counts per day of week
    counts_per_dayofweek = counts_per_date.groupby('day_of_week')['count'].sum().reset_index()

    # Get number of days per day of week
    days_per_dayofweek = counts_per_date.groupby('day_of_week')['date'].nunique().reset_index(name='num_days')

    # Merge counts and days
    counts_df = pd.merge(counts_per_dayofweek, days_per_dayofweek, on='day_of_week')

    # Compute average counts per day
    counts_df['average_count'] = counts_df['count'] / counts_df['num_days']

    total_seconds = 0
    for i in range(0, len(times), 2):
        start = datetime.strptime(times[i], "%Y-%m-%d %H:%M:%S.%f")
        end = datetime.strptime(times[i+1], "%Y-%m-%d %H:%M:%S.%f")
        total_seconds += (end - start).total_seconds()
    total_hours = total_seconds / 3600.0

    return counts_df, total_hours


def calculate_conflict_rates(conflict_counts, volume_counts, volume_type):
    # Rename columns for clarity
    conflict_counts = conflict_counts.rename(columns={'count': 'conflict_count', 'average_count': 'average_conflict_count'})
    volume_counts = volume_counts.rename(columns={'count': f'{volume_type}_count', 'average_count': f'average_{volume_type}_count'})

    print("Conflict Counts:")
    print(conflict_counts[['day_of_week', 'num_days']].dtypes)
    print(conflict_counts[['day_of_week', 'num_days']].drop_duplicates())

    print("\nVolume Counts:")
    print(volume_counts[['day_of_week', 'num_days']].dtypes)
    print(volume_counts[['day_of_week', 'num_days']].drop_duplicates())


    # Merge conflict counts with volume counts
    merged = pd.merge(
        conflict_counts[['day_of_week', 'conflict_count', 'num_days', 'average_conflict_count']],
        volume_counts[['day_of_week', f'{volume_type}_count', 'num_days', f'average_{volume_type}_count']],
        on='day_of_week'
    )

    # Calculate conflicts per 1,000 units, handling division by zero
    merged[f'conflicts_per_1000_{volume_type}'] = np.where(
        merged[f'{volume_type}_count'] > 0,
        (merged['conflict_count'] / merged[f'{volume_type}_count']) * 1000,
        0
    )
    merged[f'average_conflicts_per_1000_{volume_type}'] = np.where(
        merged[f'average_{volume_type}_count'] > 0,
        (merged['average_conflict_count'] / merged[f'average_{volume_type}_count']) * 1000,
        0
    )

    return merged


##################### main program #######################

# Main program
iid = 3287  # Intersection ID

times_before = times_dict[iid]['before']
times_after = times_dict[iid]['after']

# Fetch data for 'before' period
df_type = "track"
mean = False  # We want total counts, not averages

vehicle_counts_before, total_hours_before = heatmap_generator(df_type, mean, iid, times_before, p2v=False, pedestrian_counting=False)

pedestrian_counts_before, _ = heatmap_generator(df_type, mean, iid, times_before, p2v=False, pedestrian_counting=True)


df_type = "conflict"
p2v = True
conflict_type = 'all'

conflict_counts_before, _ = heatmap_generator(df_type, mean, iid, times_before, p2v=p2v, conflict_type=conflict_type)


print("Conflict Counts (Before):")
print(conflict_counts_before)
print("Vehicle Counts (Before):")
print(vehicle_counts_before)
print("Pedestrian Counts (Before):")
print(pedestrian_counts_before)

total_vehicle_count_before = vehicle_counts_before['count'].sum()
total_conflict_count_before = conflict_counts_before['count'].sum()
total_pedestrian_count_before = pedestrian_counts_before['count'].sum()

print(f"Total Conflict Count (Before): {total_conflict_count_before}")
print(f"Total Vehicle Count (Before): {total_vehicle_count_before}")
print(f"Total Pedestrian Count (Before): {total_pedestrian_count_before}")


# Fetch data for 'after' period
df_type = "track"
mean = False

vehicle_counts_after, total_hours_after = heatmap_generator(df_type, mean, iid, times_after, p2v=False, pedestrian_counting=False)
pedestrian_counts_after, _ = heatmap_generator(df_type, mean, iid, times_after, p2v=False, pedestrian_counting=True)

total_vehicle_count_after = vehicle_counts_after['count'].sum()
total_pedestrian_count_after = pedestrian_counts_after['count'].sum()

print("Vehicle Counts (After):", total_vehicle_count_after)
print("Pedestrian Counts (After):", total_pedestrian_count_after)


df_type = "conflict"
p2v = True

conflict_counts_after, _ = heatmap_generator(df_type, mean, iid, times_after, p2v=p2v, conflict_type=conflict_type)

# Calculate conflict rates
merged_before_vehicle = calculate_conflict_rates(conflict_counts_before, vehicle_counts_before, 'vehicle')
merged_after_vehicle = calculate_conflict_rates(conflict_counts_after, vehicle_counts_after, 'vehicle')
merged_before_pedestrian = calculate_conflict_rates(conflict_counts_before, pedestrian_counts_before, 'pedestrian')
merged_after_pedestrian = calculate_conflict_rates(conflict_counts_after, pedestrian_counts_after, 'pedestrian')

print("Merged Before Vehicle:")
print(merged_before_vehicle)
print("Merged Before Pedestrian:")
print(merged_before_pedestrian)


# Plotting
categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for df in [merged_before_vehicle, merged_after_vehicle, merged_before_pedestrian, merged_after_pedestrian]:
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=categories, ordered=True)
    df.sort_values('day_of_week', inplace=True)

print(merged_before_pedestrian)

# Plot conflicts per 1,000 vehicles
plt.figure(figsize=(10, 6))
plt.plot(merged_before_vehicle['day_of_week'], merged_before_vehicle['average_conflicts_per_1000_vehicle'], marker='o', label='Before')
plt.plot(merged_after_vehicle['day_of_week'], merged_after_vehicle['average_conflicts_per_1000_vehicle'], marker='o', label='After')
plt.title('Average P2V Conflicts per 1,000 Vehicles by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Conflicts per 1,000 Vehicles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'p2v_conflicts_per_1000_vehicles_{iid}.png', dpi=300)
plt.close()

# Plot conflicts per 1,000 pedestrians
plt.figure(figsize=(10, 6))
plt.plot(merged_before_pedestrian['day_of_week'], merged_before_pedestrian['average_conflicts_per_1000_pedestrian'], marker='o', label='Before')
plt.plot(merged_after_pedestrian['day_of_week'], merged_after_pedestrian['average_conflicts_per_1000_pedestrian'], marker='o', label='After')
plt.title('Average P2V Conflicts per 1,000 Pedestrians by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Conflicts per 1,000 Pedestrians')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'p2v_conflicts_per_1000_pedestrians_{iid}.png', dpi=300)
plt.close()