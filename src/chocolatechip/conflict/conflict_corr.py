from chocolatechip.MySQLConnector import MySQLConnector
from yaspin import yaspin
from yaspin.spinners import Spinners
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency

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
        times = [
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
        times = [
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


def extract_direction(cluster):
    return cluster.split('_')[0]

def count_conflicts_by_direction(df):
    df['direction1'] = df['cluster1'].apply(extract_direction)
    df['direction2'] = df['cluster2'].apply(extract_direction)
    direction_counts = df['direction1'].value_counts() + df['direction2'].value_counts()
    return direction_counts

def fetch_or_cache_data(my, iid, start_time, end_time, df_type='track'):
    cache_filename = f"cache_{iid}"
    if df_type == 'track':
        cache_filename += "_track"
    elif df_type == 'trackthru':
        cache_filename += "_trackthru"
    else:
        cache_filename += "_conflict"
    cache_filename += f"_{start_time.replace(':', '').replace('-', '').replace(' ', '_')}_{end_time.replace(':', '').replace('-', '').replace(' ', '_')}.csv"
    cache_filename = os.path.join('cache', cache_filename)

    if not os.path.isdir('cache'):
        os.mkdir('cache')

    if os.path.exists(cache_filename):
        if df_type == 'track' or df_type == 'trackthru':
            df = pd.read_csv(cache_filename, parse_dates=['start_timestamp', 'end_timestamp'])
        else :
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
                'end_date': end_time
            }
            df = my.handleRequest(params, 'speedcorr')
            
        df.to_csv(cache_filename, index=False)
        print(f"\n\tData cached to file: {cache_filename}")
    
    return df

def get_intersection_data(iid):
    my = MySQLConnector()
    ttc_df = pd.DataFrame()

    times = get_times(iid)

    for i in range(0, len(times), 2):
        start_time = times[i]
        end_time = times[i+1]
        with yaspin(Spinners.earth, text=f"Fetching data from MySQL starting at {start_time}") as sp:
            ttc_df = pd.concat([ttc_df, fetch_or_cache_data(my, iid, start_time, end_time, 'conflict')])

    ttc_df['unique_ID1'] = ttc_df['unique_ID1'].astype(str)
    ttc_df['unique_ID2'] = ttc_df['unique_ID2'].astype(str)

    return ttc_df

# Visualization: Stacked Bar Chart for p2v == 0
intersection_lookup = {
    3287: "Stirling Rd. & N 68th Ave.",
    3248: "Stirling Rd. & N 66th Ave.",
    3032: "Stirling Rd. & SR-7",
    3265: "Stirling Rd. & University Dr.",
    3334: "Stirling Rd. & Carriage Hills Drive/SW 61st Ave.",
}

# List of intersection IDs to analyze
intersection_ids = [3032, 3248, 3287, 3265, 3334]

# Colors for each intersection using tab10 colormap
colors = plt.get_cmap('tab10').colors
intersection_colors = dict(zip(intersection_ids, colors))

# DataFrame to hold all data
total_data_v2v = pd.DataFrame()
total_data_p2v = pd.DataFrame()

for iid in intersection_ids:
    ttc_df = get_intersection_data(iid)
    
    # Filter data for V2V (p2v == 0) and P2V (p2v == 1)
    ttc_df_v2v = ttc_df[ttc_df['p2v'] == 0]
    ttc_df_p2v = ttc_df[ttc_df['p2v'] == 1]
    
    # Count conflicts by direction for V2V and P2V
    direction_counts_v2v = count_conflicts_by_direction(ttc_df_v2v)
    direction_counts_p2v = count_conflicts_by_direction(ttc_df_p2v)
    
    # Add intersection ID column for coloring
    direction_counts_v2v.name = intersection_lookup[iid]
    direction_counts_p2v.name = intersection_lookup[iid]
    
    total_data_v2v = pd.concat([total_data_v2v, direction_counts_v2v], axis=1, sort=False)
    total_data_p2v = pd.concat([total_data_p2v, direction_counts_p2v], axis=1, sort=False)

# Remove rows with 'ped' from total_data_p2v
total_data_p2v = total_data_p2v[~total_data_p2v.index.str.contains('ped')]

# Sort the data by total number of conflicts
total_data_v2v['total'] = total_data_v2v.sum(axis=1)
total_data_v2v = total_data_v2v.sort_values(by='total').drop(columns='total')

total_data_p2v['total'] = total_data_p2v.sum(axis=1)
total_data_p2v = total_data_p2v.sort_values(by='total').drop(columns='total')

# Adjust these font sizes as needed
title_fontsize = 16
label_fontsize = 14
tick_labelsize = 14
legend_fontsize = 10

# Plot the stacked bar chart for V2V
fig_v2v, ax_v2v = plt.subplots(figsize=(6, 5))
total_data_v2v.plot(kind='bar', stacked=True, color=colors, ax=ax_v2v)
ax_v2v.set_title('Number of V2V Conflicts by Directionality\nAcross All Intersections', fontsize=title_fontsize)
ax_v2v.set_xlabel('Direction', fontsize=label_fontsize)
ax_v2v.set_ylabel('Number of Conflicts', fontsize=label_fontsize)
ax_v2v.tick_params(axis='x', labelsize=tick_labelsize)
ax_v2v.tick_params(axis='y', labelsize=tick_labelsize)
ax_v2v.set_xticklabels([label.get_text()[:-1] if label.get_text().endswith('1') else label.get_text() for label in ax_v2v.get_xticklabels()], fontsize=tick_labelsize)

legend = ax_v2v.legend(title='Intersection', labels=[intersection_lookup[iid] for iid in intersection_ids])
plt.setp(legend.get_texts(), fontsize=legend_fontsize)
plt.setp(legend.get_title(), fontsize=legend_fontsize)
plt.ylim(0, 1200)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(alpha=0.3)
plt.savefig('v2v_approach_all_intersections.pdf', dpi=300, bbox_inches='tight')

# Plot the stacked bar chart for P2V
fig_p2v, ax_p2v = plt.subplots(figsize=(6, 5))
total_data_p2v.plot(kind='bar', stacked=True, color=colors, ax=ax_p2v)
ax_p2v.set_title('Number of P2V Conflicts by Directionality\nAcross All Intersections', fontsize=title_fontsize)
ax_p2v.set_xlabel('Direction', fontsize=label_fontsize)
ax_p2v.set_ylabel('Number of Conflicts', fontsize=label_fontsize)
ax_p2v.tick_params(axis='x', labelsize=tick_labelsize)
ax_p2v.tick_params(axis='y', labelsize=tick_labelsize)
ax_p2v.set_xticklabels([label.get_text()[:-1] if label.get_text().endswith('1') else label.get_text() for label in ax_p2v.get_xticklabels()], fontsize=tick_labelsize)

legend = ax_p2v.legend(title='Intersection', labels=[intersection_lookup[iid] for iid in intersection_ids])
plt.setp(legend.get_texts(), fontsize=legend_fontsize)
plt.setp(legend.get_title(), fontsize=legend_fontsize)
plt.ylim(0, 1200)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(alpha=0.3)
plt.savefig('p2v_approach_all_intersections.pdf', dpi=300, bbox_inches='tight')


# Combine V2V and P2V data into a single DataFrame for chi-square test
combined_data = pd.concat([total_data_v2v.sum(axis=1), total_data_p2v.sum(axis=1)], axis=1)
combined_data.columns = ['V2V', 'P2V']

# Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(combined_data.T)

print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print("Expected frequencies:")
print(expected)
