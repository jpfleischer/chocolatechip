from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
from yaspin import yaspin
from yaspin.spinners import Spinners
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime



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


def get_recorded_days(iid, period):
    """
    Sum up the total recording time (in days) for an intersection & period,
    given that times_dict[iid][period] is a flat list of alternating
    start/end ISO-strings.
    """
    ts_list = times_dict[iid][period]
    total_seconds = 0.0
    # take (start, end) in pairs
    for start_str, end_str in zip(ts_list[0::2], ts_list[1::2]):
        start = datetime.fromisoformat(start_str)
        end   = datetime.fromisoformat(end_str)
        total_seconds += (end - start).total_seconds()
    return total_seconds / 86400.0  # seconds → days


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
    plt.figure(figsize=(5,5))
    df.plot(
        kind='bar',
        figsize=(5,5),
        color=['#1f77b4', '#ff7f0e'],   # or whatever two distinct colors
        width=0.8
    )
    ax = plt.gca()
    # ax.set_title(f"Intersection {iid} — {p2v_label} Conflicts\nBefore vs After", fontsize=16)
    ax.set_xlabel("Direction", fontsize=14)
    ax.set_ylabel("Conflicts per Day", fontsize=14)

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
    3287: "Stirling Road and N 68th Avenue",
    3248: "Stirling Road and N 66th Avenue",
    3032: "Stirling Road and SR-7",
    3265: "Stirling Road and University Drive",
    3334: "Stirling Road and SW 61st Avenue",
    3252: "Stirling Road and Davie Road Extension"
}


if __name__ == "__main__":
    intersection_ids = [3032, 3248, 3287, 3265, 3334, 3252]  # adjust as needed
    p2v_labels       = {'0': 'V2V', '1': 'P2V'}
    colors           = plt.get_cmap('tab10').colors
    outdir           = "directionality"
    os.makedirs(outdir, exist_ok=True)

    # 1) BEFORE/AFTER only for the special pair
    if intersection_ids == [3248, 3287]:
        for iid in intersection_ids:
            for p2v_flag, label in p2v_labels.items():
                # fetch before
                df_b = get_intersection_data(iid, p2v_flag, 'before')
                df_b = df_b[df_b['p2v'] == int(p2v_flag)]
                # fetch after
                df_a = get_intersection_data(iid, p2v_flag, 'after')
                df_a = df_a[df_a['p2v'] == int(p2v_flag)]

                days_b = get_recorded_days(iid, 'before')
                days_a = get_recorded_days(iid, 'after')

                cnt_b = count_conflicts_by_direction(df_b, p2v=(p2v_flag=='1'))
                cnt_a = count_conflicts_by_direction(df_a, p2v=(p2v_flag=='1'))

                rate_b = cnt_b / days_b
                rate_a = cnt_a / days_a

                plot_before_after_counts(rate_b, rate_a, iid, label)

    # 2) Build the comprehensive BEFORE-only P2V table
    total_data_p2v = pd.DataFrame()
    for iid in intersection_ids:
        df_b = get_intersection_data(iid, '1', 'before')
        df_b = df_b[df_b['p2v'] == 1]

        cnt       = count_conflicts_by_direction(df_b, p2v=True)
        cnt.name  = intersection_lookup.get(iid, str(iid))
        total_data_p2v = pd.concat([total_data_p2v, cnt], axis=1)

    total_data_p2v = total_data_p2v[~total_data_p2v.index.str.contains('ped', case=False)]
    total_data_p2v = total_data_p2v.fillna(0)

    # 3) Build the comprehensive BEFORE-only V2V table
    total_data_v2v = pd.DataFrame()
    for iid in intersection_ids:
        df_v = get_intersection_data(iid, '0', 'before')
        df_v = df_v[df_v['p2v'] == 0]

        cnt_v     = count_conflicts_by_direction(df_v, p2v=False)
        cnt_v.name = intersection_lookup.get(iid, str(iid))
        total_data_v2v = pd.concat([total_data_v2v, cnt_v], axis=1)

    total_data_v2v = total_data_v2v[~total_data_v2v.index.str.contains('ped', case=False)]
    total_data_v2v = total_data_v2v.fillna(0)

    # 4) If NOT the special pair, plot the comprehensive V2V stacked bar (before only)
    if intersection_ids != [3248, 3287]:
        # clean direction labels
        clean_labels_v2v = [
            lbl[:-1] if lbl.endswith('1') else lbl
            for lbl in total_data_v2v.index
        ]

        fig_v2v, ax_v2v = plt.subplots(figsize=(10, 5))
        total_data_v2v.plot(
            kind='bar',
            stacked=True,
            color=colors[:len(intersection_ids)],
            width=0.7,
            ax=ax_v2v
        )
        ax_v2v.set_xticklabels(clean_labels_v2v, rotation=45)
        ax_v2v.grid(axis='y', linestyle='--', alpha=0.6)
        ax_v2v.set_xlabel("Movement of Involved Vehicle")
        ax_v2v.set_ylabel("Number of Conflicts")
        legend_v2v = ax_v2v.legend(title="Intersection")
        plt.tight_layout()

        fn_v2v_png = os.path.join(outdir, 'v2v_all_intersections_directionality_before.png')
        fig_v2v.savefig(fn_v2v_png, dpi=300, bbox_inches='tight')
        fn_v2v_pdf = os.path.join(outdir, 'v2v_all_intersections_directionality_before.pdf')
        fig_v2v.savefig(fn_v2v_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig_v2v)
        print(f"→ Saved comprehensive V2V (before-only) plot: {fn_v2v_pdf}")


    # 5) If NOT the special pair, plot the comprehensive P2V stacked bar (before only),
    #    sorted by total conflicts descending and with cleaned labels.
    if intersection_ids != [3248, 3287]:
        # 1) Sort movements by total conflicts (sum across intersections)
        movement_order = (
            total_data_p2v
              .sum(axis=1)
              .sort_values(ascending=False)
              .index
        )
        total_data_p2v = total_data_p2v.loc[movement_order]

        # 2) Clean labels (drop trailing '1' if present)
        clean_labels = [
            lbl[:-1] if lbl.endswith('1') else lbl
            for lbl in total_data_p2v.index
        ]

        # 3) Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        total_data_p2v.plot(
            kind='bar',
            stacked=True,
            color=colors[:len(intersection_ids)],
            width=0.7,
            ax=ax
        )
        ax.set_xticklabels(clean_labels, rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlabel("Movement of Involved Vehicle")
        ax.set_ylabel("Number of P2V Conflicts")
        ax.legend(title="Intersection", fontsize=9)

        # 4) Save outputs
        fn_png = os.path.join(outdir, 'p2v_all_intersections_directionality_before_stacked_sorted.png')
        fig.savefig(fn_png, dpi=300, bbox_inches='tight')
        fn_pdf = fn_png.replace('.png', '.pdf')
        fig.savefig(fn_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"→ Saved sorted stacked P2V plot: {fn_pdf}")
