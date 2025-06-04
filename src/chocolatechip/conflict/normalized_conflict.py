from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
from datetime import datetime
from yaspin import yaspin
from yaspin.spinners import Spinners
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def heatmap_generator(df_type: str,
                      mean: bool,
                      intersec_id: int,
                      times: list,
                      p2v: bool = None,
                      conflict_type: str = None,
                      pedestrian_counting: bool = False,
                      ):
    """
    Fetch track or conflict data over the given time windows,
    return (counts_df, total_hours).
    counts_df has columns ['day_of_week','count','num_days','average_count'].
    """
    if df_type not in ['track', 'conflict']:
        raise ValueError('df_type must be "track" or "conflict"')
    if df_type == 'conflict' and p2v is None:
        raise ValueError('p2v must be True or False for conflict')
    if p2v is False and conflict_type in ['left turning','right turning','thru']:
        raise ValueError('that conflict_type requires p2v=True')

    cam_lookup = {3287:24, 3248:27, 3032:23, 3265:30, 3334:33, 3252:36, 5060:7}
    params = {'intersec_id': intersec_id,
              'cam_id': cam_lookup[intersec_id],
              'p2v': 0 if p2v is False else 1,
              'start_date': None, 'end_date': None}

    omega = pd.DataFrame()
    for i in range(0, len(times), 2):
        params['start_date'], params['end_date'] = times[i], times[i+1]
        my = MySQLConnector()
        with yaspin(Spinners.pong, text=f"Fetching {df_type}@{times[i]}") as sp:
            df = my.handleRequest(params, df_type)
            sp.ok("✔")
        if df.empty:
            continue
        df['day_of_week'] = pd.Categorical(
            df['timestamp'].dt.day_name(),
            categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
            ordered=True
        )
        omega = pd.concat([omega, df], ignore_index=True)

    if omega.empty:
        return pd.DataFrame(columns=['day_of_week','count','num_days','average_count']), 0

    # filter classes
    if df_type == 'track':
        if pedestrian_counting:
            omega = omega[omega['class']=='pedestrian']
        else:
            omega = omega[omega['class']!='pedestrian']
    else:
        if conflict_type == 'all':
            omega = omega[omega['p2v']==1]
        elif conflict_type == 'left turning':
            omega = omega[(omega['p2v']==1)&omega['conflict_type'].isin([3,4])]
        elif conflict_type == 'right turning':
            omega = omega[(omega['p2v']==1)&omega['conflict_type'].isin([1,2])]
        elif conflict_type == 'thru':
            omega = omega[(omega['p2v']==1)&omega['conflict_type'].isin([5,6])]

    col = 'track_id' if df_type=='track' else 'unique_ID1'
    omega['date'] = omega['timestamp'].dt.date

    per_date = omega.groupby(['day_of_week','date'])[col]\
                    .nunique().reset_index(name='count')
    sum_by_day = per_date.groupby('day_of_week')['count']\
                         .sum().reset_index()
    days_by_day = per_date.groupby('day_of_week')['date']\
                          .nunique().reset_index(name='num_days')
    counts_df = pd.merge(sum_by_day, days_by_day, on='day_of_week')
    counts_df['average_count'] = counts_df['count'] / counts_df['num_days']

    total_seconds = sum(
        (datetime.strptime(times[j+1], "%Y-%m-%d %H:%M:%S.%f") -
         datetime.strptime(times[j],   "%Y-%m-%d %H:%M:%S.%f")
        ).total_seconds()
        for j in range(0, len(times), 2)
    )
    return counts_df, total_seconds/3600.0


def calculate_conflict_rates(conflict_counts, volume_counts, volume_type):
    cc = conflict_counts.rename(
        columns={'count':'conflict_count','average_count':'avg_conflict_count'}
    )
    vc = volume_counts.rename(
        columns={'count':f'{volume_type}_count',
                 'average_count':f'avg_{volume_type}_count'}
    )
    merged = pd.merge(
        cc[['day_of_week','conflict_count','num_days','avg_conflict_count']],
        vc[['day_of_week',f'{volume_type}_count','num_days',f'avg_{volume_type}_count']],
        on='day_of_week'
    )
    merged[f'avg_conflicts_per_1000_{volume_type}'] = np.where(
        merged[f'{volume_type}_count']>0,
        merged['avg_conflict_count']/merged[f'avg_{volume_type}_count']*1000, 0
    )
    return merged


def summarize_intersection(iid):
    before = times_dict[iid]['before']
    after  = times_dict[iid]['after']

    # totals
    vc_b,_ = heatmap_generator("track",    False, iid, before, p2v=False, pedestrian_counting=False)
    vc_a,_ = heatmap_generator("track",    False, iid, after,  p2v=False, pedestrian_counting=False)
    pc_b,_ = heatmap_generator("track",    False, iid, before, p2v=False, pedestrian_counting=True)
    pc_a,_ = heatmap_generator("track",    False, iid, after,  p2v=False, pedestrian_counting=True)
    cc_b,_ = heatmap_generator("conflict", False, iid, before, p2v=True, conflict_type='all')
    cc_a,_ = heatmap_generator("conflict", False, iid, after,  p2v=True, conflict_type='all')

    Cb, Ca = cc_b['count'].sum(), cc_a['count'].sum()
    Vb, Va = vc_b['count'].sum(), vc_a['count'].sum()
    Pb, Pa = pc_b['count'].sum(), pc_a['count'].sum()

    r1_v, r2_v = Cb/Vb, Ca/Va
    r1_p, r2_p = Cb/Pb, Ca/Pa

    se_v = np.sqrt(r1_v/Vb + r2_v/Va)
    p_v  = 2*(1 - stats.norm.cdf(abs((r2_v-r1_v)/se_v)))

    se_p = np.sqrt(r1_p/Pb + r2_p/Pa)
    p_p  = 2*(1 - stats.norm.cdf(abs((r2_p-r1_p)/se_p)))

    return pd.DataFrame({
        'Intersection':[iid,iid],
        'Metric':[
          'Conflicts per 1 000 vehicles',
          'Conflicts per 1 000 pedestrians'
        ],
        'Before':    [r1_v*1000, r1_p*1000],
        'After':     [r2_v*1000, r2_p*1000],
        'Change (%)':[(r2_v-r1_v)/r1_v*100,(r2_p-r1_p)/r1_p*100],
        'p‑value':   [p_v,p_p]
    })


if __name__=="__main__":
    intersections = [3248, 3287]

    for iid in intersections:
        # prepare merged DataFrames for normalized‐conflict plots
        before = times_dict[iid]['before']
        after  = times_dict[iid]['after']

        vc_b,_ = heatmap_generator("track", False, iid, before, p2v=False, pedestrian_counting=False)
        vc_a,_ = heatmap_generator("track", False, iid, after,  p2v=False, pedestrian_counting=False)
        pc_b,_ = heatmap_generator("track", False, iid, before, p2v=False, pedestrian_counting=True)
        pc_a,_ = heatmap_generator("track", False, iid, after,  p2v=False, pedestrian_counting=True)
        cc_b,_ = heatmap_generator("conflict", False, iid, before, p2v=True, conflict_type='all')
        cc_a,_ = heatmap_generator("conflict", False, iid, after,  p2v=True, conflict_type='all')

        mv_b = calculate_conflict_rates(cc_b, vc_b, 'vehicle')
        mv_a = calculate_conflict_rates(cc_a, vc_a, 'vehicle')
        mp_b = calculate_conflict_rates(cc_b, pc_b, 'pedestrian')
        mp_a = calculate_conflict_rates(cc_a, pc_a, 'pedestrian')

        # ensure ordering
        for df in (mv_b, mv_a, mp_b, mp_a):
            df['day_of_week'] = pd.Categorical(
                df['day_of_week'],
                categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                ordered=True
            )
            df.sort_values('day_of_week', inplace=True)

        # plot per‐1 000 vehicles
        plt.figure(figsize=(8,5))
        plt.plot(mv_b['day_of_week'], mv_b['avg_conflicts_per_1000_vehicle'], marker='o', label='Before')
        plt.plot(mv_a['day_of_week'], mv_a['avg_conflicts_per_1000_vehicle'], marker='o', label='After')
        # plt.title(f'Average P2V Conflicts / 1,000 Vehicles @ Intersection {iid}')
        plt.xlabel('Day of Week')
        plt.ylabel('Conflicts / 1,000 Vehicles')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'normalized_conflicts_vehicles_{iid}.pdf', dpi=300)
        plt.close()

        # plot per‐1 000 pedestrians
        plt.figure(figsize=(8,5))
        plt.plot(mp_b['day_of_week'], mp_b['avg_conflicts_per_1000_pedestrian'], marker='o', label='Before')
        plt.plot(mp_a['day_of_week'], mp_a['avg_conflicts_per_1000_pedestrian'], marker='o', label='After')
        # plt.title(f'Average P2V Conflicts / 1,000 Pedestrians @ Intersection {iid}')
        plt.xlabel('Day of Week')
        plt.ylabel('Conflicts / 1,000 Pedestrians')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'normalized_conflicts_pedestrians_{iid}.pdf', dpi=300)
        plt.close()

    # finally, print the combined summary table
    all_summaries = pd.concat(
      [summarize_intersection(i) for i in intersections],
      ignore_index=True
    )
    colspec = "p{2cm}p{4cm}XXXXL"
    print(r"\begin{table}[htbp]")
    print(r"  \begin{flushleft}")
    print(
      r"    \caption{Average number of P2V conflicts per 1,000 vehicles and per 1,000 pedestrians, "
      r"before vs after, for the 66th and 68th Avenue intersections.}"
    )
    print(r"    \label{tab:p2v_compare_two}")
    print(r"    \begin{tabularx}{\textwidth}{" + colspec + r"}")
    print(r"      \hline")
    headers = ["Intersection","Metric","Before","After","Change (\\%)","p‑value"]
    print("      " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\")
    print(r"      \hline")
    for _, row in all_summaries.iterrows():
        print(
            f"      {int(row['Intersection'])} & {row['Metric']} & "
            f"{row['Before']:.2f} & {row['After']:.2f} & "
            f"{row['Change (%)']:.2f} & {row['p‑value']:.8f} \\\\"
        )
    print(r"      \hline")
    print(r"    \end{tabularx}")
    print(r"  \end{flushleft}")
    print(r"\end{table}")
