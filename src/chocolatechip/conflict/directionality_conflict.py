from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
from yaspin import yaspin
from yaspin.spinners import Spinners
import pandas as pd
import matplotlib.pyplot as plt
import os



def extract_direction(cluster):
    return cluster.split('_')[0]


def count_conflicts_by_direction(df, p2v: bool = False):
    """
    If p2v==False: count both cluster1 & cluster2 legs.
    If p2v==True : pick only the non-pedestrian cluster per row.
    """
    if not p2v:
        # V2V: sum both sides
        dirs = df['cluster1'].apply(extract_direction)._append(
               df['cluster2'].apply(extract_direction),
               ignore_index=True)
        return dirs.value_counts()

    # P2V: pick the vehicle side
    def pick_vehicle(row):
        return row['cluster2'] if row['cluster1'].startswith('ped_') else row['cluster1']

    veh = df.apply(pick_vehicle, axis=1)
    return veh.apply(extract_direction).value_counts()


def fetch_or_cache_data(my, iid, start_time, end_time, p2v, df_type='track'):
    os.makedirs('cache', exist_ok=True)

    # build a cache name that includes the p2v flag
    p2v_flag = int(p2v)
    def clean(ts):
        return ts.replace(':','').replace('-','').replace(' ','_')
    fname = (
        f"{iid}_{df_type}_p2v{p2v_flag}_"
        f"{clean(start_time)}_{clean(end_time)}.csv"
    )
    cache_path = os.path.join('cache', fname)

    # if it’s cached, load it
    if os.path.exists(cache_path):
        if df_type in ('track','trackthru'):
            return pd.read_csv(cache_path, parse_dates=['start_timestamp','end_timestamp'])
        else:
            return pd.read_csv(cache_path)

    # otherwise fetch fresh
    if df_type == 'track':
        df = my.query_tracksreal(iid, start_time, end_time)
    elif df_type == 'trackthru':
        df = my.query_tracksreal(iid, start_time, end_time, True)
    else:  # conflict
        print(f"→ [DEBUG] fetching conflicts via fetchConflictRecords(p2v={p2v_flag})")
        df = my.fetchConflictRecords(iid, p2v_flag, start_time, end_time)
        df['p2v'] = p2v_flag 
        print(f"   got {len(df)} rows")

    # cache and return
    df.to_csv(cache_path, index=False)
    print(f"→ Data cached to file: {cache_path}")
    return df



def get_intersection_data(iid, p2v, period):
    start_ts, end_ts = times_dict[iid][period][0], times_dict[iid][period][-1]
    my = MySQLConnector()
    df = fetch_or_cache_data(my, iid, start_ts, end_ts, p2v, 'conflict')
    if df.empty:
        return df
    return df.dropna(axis=1, how='all')



def plot_before_after_counts(counts_b, counts_a, iid: int, p2v_label: str):
    """
    Plot side-by-side bar chart of before vs after counts,
    sorted low→high by their combined totals.
    """
    # 1) build DataFrame and fill missing directions with 0
    df = pd.DataFrame({
        'Before': counts_b,
        'After':  counts_a
    }).fillna(0)

    # 2) sort by total count
    df['total'] = df.sum(axis=1)
    df = df.sort_values('total').drop(columns='total')

    # 3) plot
    plt.figure(figsize=(8,5))
    df.plot(
        kind='bar',
        figsize=(8,5),
        color=['#1f77b4', '#ff7f0e'],   # or whatever two distinct colors
        width=0.8
    )
    ax = plt.gca()
    # ax.set_title(f"Intersection {iid} — {p2v_label} Conflicts\nBefore vs After", fontsize=16)
    ax.set_xlabel("Direction", fontsize=14)
    ax.set_ylabel("Number of Conflicts", fontsize=14)
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 4) save
    outdir = "directionality"
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"{iid}_{p2v_label.lower()}_before_after.png")
    plt.savefig(fn, dpi=300)
    plt.close()
    print(f"→ Saved {fn}")


def plot_conflicts(direction_counts: pd.Series, iid: int, period: str, p2v_label: str):
    # ensure output directory exists
    outdir = "directionality"
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,5))
    direction_counts.plot(kind='bar', ax=ax)
    # ax.set_title(f"Intersection {iid} — {p2v_label} Conflicts ({period})", fontsize=16)
    ax.set_xlabel("Direction", fontsize=14)
    ax.set_ylabel("Number of Conflicts", fontsize=14)
    ax.tick_params(labelsize=12, rotation=45)
    plt.tight_layout()

    # save into the folder
    fn = os.path.join(outdir, f"{iid}_{period}_{p2v_label.lower()}.png")
    fig.savefig(fn, dpi=300)
    plt.close(fig)
    print(f"→ Saved {fn}")


# Visualization: Stacked Bar Chart for p2v == 0
intersection_lookup = {
    3287: "Stirling Rd. & N 68th Ave.",
    3248: "Stirling Rd. & N 66th Ave.",
    3032: "Stirling Rd. & SR-7",
    3265: "Stirling Rd. & University Dr.",
    3334: "Stirling Rd. & Carriage Hills Drive/SW 61st Ave.",
}

# intersection_ids    = [3032,3248,3287,3265,3334]
intersection_ids = [3248, 3287]
p2v_labels          = {'0':'V2V','1':'P2V'}
colors = plt.get_cmap('tab10').colors

# reset your totals
total_data_v2v = pd.DataFrame()
total_data_p2v = pd.DataFrame()

for iid in intersection_ids:
    # 1) BEFORE vs AFTER plots, one per conflict type
    for p2v_flag, label in p2v_labels.items():
        # fetch both periods
        df_b = get_intersection_data(iid, p2v_flag, 'before')
        df_b = df_b[df_b['p2v'] == int(p2v_flag)]
        df_a = get_intersection_data(iid, p2v_flag, 'after')
        df_a = df_a[df_a['p2v'] == int(p2v_flag)]

        # count directions
        cnt_b = count_conflicts_by_direction(df_b, p2v=(p2v_flag=='1'))
        cnt_a = count_conflicts_by_direction(df_a, p2v=(p2v_flag=='1'))
        # drop pedestrians in P2V
        if p2v_flag == '1':
            cnt_b = cnt_b[~cnt_b.index.str.contains('ped')]
            cnt_a = cnt_a[~cnt_a.index.str.contains('ped')]

        # skip if absolutely no data
        if cnt_b.sum()==0 and cnt_a.sum()==0:
            print(f"→ no {label} data for {iid}, skipping before/after plot")
        else:
            plot_before_after_counts(cnt_b, cnt_a, iid, label)

    # 2) now aggregate both periods for your all-intersection stacked bars
    all_v2v = pd.concat([
        get_intersection_data(iid,'0','before'),
        get_intersection_data(iid,'0','after')
    ], ignore_index=True)
    all_p2v = pd.concat([
        get_intersection_data(iid,'1','before'),
        get_intersection_data(iid,'1','after')
    ], ignore_index=True)

    vc = all_v2v[all_v2v['p2v']==0]
    pc = all_p2v[all_p2v['p2v']==1]
    cnt_v2v = count_conflicts_by_direction(vc);  cnt_v2v.name = intersection_lookup[iid]
    cnt_p2v = count_conflicts_by_direction(pc);  cnt_p2v.name = intersection_lookup[iid]

    total_data_v2v = pd.concat([total_data_v2v, cnt_v2v], axis=1, sort=False)
    total_data_p2v = pd.concat([total_data_p2v, cnt_p2v], axis=1, sort=False)

# drop the peds and sort as before
total_data_p2v = total_data_p2v[~total_data_p2v.index.str.contains('ped')]
total_data_v2v['total'] = total_data_v2v.sum(axis=1)
total_data_p2v['total'] = total_data_p2v.sum(axis=1)
total_data_v2v = total_data_v2v.sort_values('total').drop(columns='total')
total_data_p2v = total_data_p2v.sort_values('total').drop(columns='total')

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

# ensure our output folder exists
outdir = "directionality"
os.makedirs(outdir, exist_ok=True)

# save the V2V stacked‐bar
v2v_fn = os.path.join(outdir, 'v2v_approach_all_intersections.png')
fig_v2v.savefig(v2v_fn, dpi=300, bbox_inches='tight')
print(f"→ Saved {v2v_fn}")



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
# save the P2V stacked‐bar
p2v_fn = os.path.join(outdir, 'p2v_approach_all_intersections.png')
fig_p2v.savefig(p2v_fn, dpi=300, bbox_inches='tight')
print(f"→ Saved {p2v_fn}")



