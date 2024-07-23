from chocolatechip.MySQLConnector import MySQLConnector
from yaspin import yaspin
from yaspin.spinners import Spinners
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

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

def get_intersection_data(iid, df_type='track'):
    my = MySQLConnector()
    mega_df = pd.DataFrame()
    ttc_df = pd.DataFrame()

    times = get_times(iid)

    for i in range(0, len(times), 2):
        start_time = times[i]
        end_time = times[i+1]
        with yaspin(Spinners.earth, text=f"Fetching data from MySQL starting at {start_time}") as sp:
            df = fetch_or_cache_data(my, iid, start_time, end_time, df_type)
            ttc_df = pd.concat([ttc_df, fetch_or_cache_data(my, iid, start_time, end_time, 'conflict')])
            mega_df = pd.concat([mega_df, df])

    mega_df['start_timestamp'] = pd.to_datetime(mega_df['start_timestamp'])
    mega_df['hour_of_day'] = mega_df['start_timestamp'].dt.hour
    mega_df['day_of_week'] = mega_df['start_timestamp'].dt.day_name()
    mega_df['track_id'] = mega_df['track_id'].astype(str)
    mega_df['conflict'] = 0

    ttc_df['unique_ID1'] =  ttc_df['unique_ID1'].astype(str)
    ttc_df['unique_ID2'] =  ttc_df['unique_ID2'].astype(str)

    approach_mapping = {'NBT': 1, 'NBL': 2, 'NBR': 3, 'NBU': 4 , 
                        'SBT': 1, 'SBL': 2, 'SBR': 3, 'SBU': 4, 
                        'EBT': 1, 'EBL': 2, 'EBR': 3, 'EBU': 4, 
                        'WBT': 1, 'WBL': 2, 'WBR': 3, 'WBU': 4
                        }
    
    print('11111111')

    mega_df = mega_df[mega_df['Approach'] != '0']  # filter out bad data
    mega_df['approach_numeric'] = mega_df['Approach'].map(approach_mapping)   

    print('22222222')

    for _, row in ttc_df.iterrows():
        id1, id2 = row['unique_ID1'], row['unique_ID2']
        if iid != 3287:
            adj_id1, adj_id2 = "1" + id1, "1" + id2
        else:
            adj_id1, adj_id2 = id1, id2
        mega_df.loc[(mega_df['track_id'] == adj_id1) | (mega_df['track_id'] == adj_id2), 'conflict'] = 1


    print('3333333')

    return mega_df

def analyze_multiple_intersections(intersection_ids):
    all_data = pd.DataFrame()

    for iid in intersection_ids:
        intersection_data = get_intersection_data(iid)
        intersection_data['intersection_id'] = iid
        all_data = pd.concat([all_data, intersection_data])

    print('AAAAA')

    # Calculate conflict rates for each approach and intersection
    conflict_counts = all_data.groupby(['intersection_id', 'Approach'])['conflict'].sum()
    total_counts = all_data.groupby(['intersection_id', 'Approach']).size()
    conflict_rates = (conflict_counts / total_counts).fillna(0).unstack()

    print('BBBBB')

    print("Conflict Rates by Approach Across All Intersections:")
    print(conflict_rates)

    # Chi-Square Test for Independence
    contingency_table = pd.crosstab(all_data['Approach'], all_data['conflict'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test: chi2 = {chi2}, p-value = {p:.50e}")
    print('other p ', p)


    print('CCCCC')

    # Polynomial Regression Analysis
    if 'average_speed' in all_data.columns and 'Max_speed' in all_data.columns and 'Min_speed' in all_data.columns:
        X = all_data[['average_speed', 'Max_speed', 'Min_speed']].values
        y = all_data['conflict'].values

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        r2 = r2_score(y, y_pred)
        print(f'R^2 Score for polynomial regression: {r2}')
    else:
        print("Necessary columns for polynomial regression are missing.")


    print('DDDDD')

    return all_data, conflict_rates, chi2, p


# List of intersection IDs to analyze
intersection_ids = [3032, 3248, 3287, 3265, 3334]

# Analyze all intersections
all_data, conflict_rates, chi2, p = analyze_multiple_intersections(intersection_ids)

# Visualization: Stacked Bar Chart
intersection_lookup = {
    3287: "Stirling Road and N 68th Avenue",
    3248: "Stirling Road and N 66th Avenue",
    3032: "Stirling Road and SR-7",
    3265: "Stirling Road and University Drive",
    3334: "Stirling Road and Carriage Hills Drive/SW 61st Avenue",
}

# Transpose the conflict_rates DataFrame to switch x-axis and legend
conflict_rates = conflict_rates.T

# Apply intersection lookup to index
# Ensure the columns (originally intersections) are correctly named
conflict_rates.columns = conflict_rates.columns.map(intersection_lookup)

# Fill NaN values with 0.0
conflict_rates = conflict_rates.fillna(0.0)

# Sort the DataFrame based on the sum of conflict rates for each intersection
conflict_rates = conflict_rates.loc[conflict_rates.sum(axis=1).sort_values().index]

# Plotting the data
conflict_rates.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Conflict Rates by Approach and Intersection')
plt.xlabel('Approach')
plt.ylabel('Conflict Rate')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Intersections')
plt.tight_layout()
if not os.path.isdir('exp_results'):
    os.mkdir('exp_results')
plt.savefig(f'exp_results/speed_correlation_conflict_rates_stacked_bar.png', bbox_inches='tight')
# plt.show()