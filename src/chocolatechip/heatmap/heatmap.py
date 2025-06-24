from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
import datetime
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
import base64


def _add_weekday_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame whose index is dates (or timestamps), add a 'day_of_week'
    column as a pandas.Categorical (Monday→Sunday).
    """
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekdays = pd.to_datetime(df.index).day_name()
    df['day_of_week'] = pd.Categorical(
        weekdays,
        categories=day_order,
        ordered=True
    )
    return df


def aggregate_period_counts(times: list,
                            df_type: str,
                            params: dict,
                            pedestrian_counting: bool = False,
                            p2v: bool = None,
                            conflict_type: str = None) -> pd.DataFrame:
    """
    1) For track data: fetch per-minute distinct counts via SQL,
       reindex onto a 7 AM-7 PM minute grid, then:
         - if vehicles → interpolate gaps ≤ 120 minutes “inside” each day
         - if pedestrians → scale partial hours to a full 60-minute estimate,
           but drop any hours with fewer than a threshold of camera-on minutes.
    2) For conflict data: fetch the raw 15-minute distinct counts (no interpolation).
    """
    conn = MySQLConnector()
    column_name = 'track_id' if df_type == 'track' else 'unique_ID1'

    # 1) Determine absolute start/end (for logging only)
    start, end = times[0], times[-1]
    params['start_date'], params['end_date'] = start, end
    print(f"Fetching {df_type} data from {start} to {end}")

    if df_type == 'track':
        # ─── TRACK PATH ───

        # (a) Pull per-minute counts from the DB
        df_counts = conn.fetchPeriodicTrackCountsPerMinute(
            intersec_id         = params['intersec_id'],
            cam_id              = params['cam_id'],
            start               = start,
            end                 = end,
            pedestrian_counting = pedestrian_counting
        )
        # Expect df_counts columns: ['period_start', 'count']
        df_counts.set_index('period_start', inplace=True)
        minute_counts = df_counts['count']

        # ─── BUILD active_idx from the fine-grained times list ───
        intervals = []
        for i in range(0, len(times), 2):
            seg_start = pd.to_datetime(times[i]).floor('min')
            seg_end   = pd.to_datetime(times[i+1]).floor('min')
            intervals.append((seg_start, seg_end))

        all_minutes = []
        for seg_start, seg_end in intervals:
            minutes_in_segment = pd.date_range(start=seg_start, end=seg_end, freq='T')
            # Keep only 07:00-19:00
            minutes_in_segment = minutes_in_segment[
                (minutes_in_segment.hour >= 7) & (minutes_in_segment.hour <= 19)
            ]
            all_minutes.append(minutes_in_segment)

        if all_minutes:
            active_idx = pd.DatetimeIndex(
                pd.concat([pd.Series(idx) for idx in all_minutes]).unique()
            ).sort_values()
        else:
            active_idx = pd.DatetimeIndex([])

        # ─── 2) HOURLY “PRE” PIVOT (raw or zeros) ───────────────────────────────────
        # Fill missing-SQL minutes with 0 just to see raw zeros in hours with no data
        hourly_pre = (
            minute_counts
              .reindex(active_idx, fill_value=0)  # missing SQL rows → 0
              .to_frame(name='count')
              .assign(
                 date=lambda df: df.index.date,
                 hour=lambda df: df.index.hour
              )
              .groupby(['date','hour'])['count']
              .sum()
              .reset_index()
              .pivot(index='date', columns='hour', values='count')
              .fillna(0)
        )
        hourly_pre = _add_weekday_category(hourly_pre)
        hourly_pre = (
            hourly_pre
              .loc[(hourly_pre.drop(columns='day_of_week') > 0).any(axis=1)]
              .sort_values('day_of_week')
        )
        print("\n=== HOURLY RAW COUNTS (7 AM-7 PM; zeros if missing) ===")
        print(hourly_pre.to_string())

        # ─── 3) VEHICLE vs PEDESTRIAN BRANCH ────────────────────────────────────────
        if not pedestrian_counting:
            # ───── VEHICLES: interpolate gaps ≤ 120 minutes per day ─────
            df_min = minute_counts.reindex(active_idx).to_frame(name='count')

            df_min['count'] = (
                df_min['count']
                  .groupby(df_min.index.date)
                  .transform(lambda g: g.interpolate(
                      method='time',
                      limit=120,
                      limit_area='inside'
                  ))
            )
            df_min = df_min[df_min['count'].notna()]

            hourly_post = (
                df_min
                  .assign(
                     date=lambda df: df.index.date,
                     hour=lambda df: df.index.hour
                  )
                  .groupby(['date','hour'])['count']
                  .sum()
                  .reset_index(name='hourly_count')
                  .pivot(index='date', columns='hour', values='hourly_count')
                  .fillna(0)
            )
            hourly_post = _add_weekday_category(hourly_post)
            hourly_post = (
                hourly_post
                  .loc[(hourly_post.drop(columns='day_of_week') > 0).any(axis=1)]
                  .sort_values('day_of_week')
            )
            print("\n=== HOURLY POST-INTERPOLATION (VEHICLES) ===")
            print(hourly_post.to_string())

            df_pc = df_min

        else:
            # ───── PEDESTRIANS: scale partial hours, drop low-coverage ─────

            # Step 1: Reindex onto active_idx (all camera-on minutes), leaving missing-SQL as NaN
            df_pc = minute_counts.reindex(active_idx).to_frame(name='count')

            # Step 2: Every row is “camera-on this minute,” so has_data = True
            df_pc = df_pc.assign(
                date=lambda df: df.index.date,
                hour=lambda df: df.index.hour,
                has_data=True
            )

            # Step 3: Count how many minutes the camera was on per (date, hour)
            observed = (
                df_pc
                .groupby(['date','hour'])['has_data']
                .sum()
                .reset_index(name='observed_minutes')
            )
            # (A) Remember the old index by turning it into a column named "timestamp"
            df_pc = df_pc.reset_index().rename(columns={'index': 'timestamp'})

            # (B) Merge on ['date','hour'] exactly as before
            df_pc = df_pc.merge(observed, on=['date','hour'], how='left')

            # (C) Now put "timestamp" back as the index
            df_pc = df_pc.set_index('timestamp')


            # Step 4: Drop any hour with fewer than X minutes of camera-on time
            min_minutes_threshold = 20
            df_pc = df_pc[df_pc['observed_minutes'] >= min_minutes_threshold].copy()

            # Step 5: Compute scale_factor = 60 / observed_minutes
            df_pc['scale_factor'] = 60.0 / df_pc['observed_minutes']

            # Step 6: Replace NaN→0 for “camera-on but no ped,” then multiply
            df_pc['count'] = df_pc['count'].fillna(0) * df_pc['scale_factor']

            # ─ Step 7: Print a “post-scaling” pivot for sanity
            hourly_post = (
                df_pc
                .groupby(['date','hour'])['count']
                .sum()
                .reset_index(name='hourly_scaled_count')
                .pivot(index='date', columns='hour', values='hourly_scaled_count')
                .fillna(0)
            )
            hourly_post = _add_weekday_category(hourly_post)
            hourly_post = (
                hourly_post
                .loc[(hourly_post.drop(columns='day_of_week') > 0).any(axis=1)]
                .sort_values('day_of_week')
            )
            print(
                "\n=== HOURLY POST-SCALING (PEDESTRIANS; scaled to 60 "
                f"minutes if ≥ {min_minutes_threshold} camera-minutes) ==="
            )
            print(hourly_post.to_string())

            # Build a “human-friendly” index of the form “YYYY-MM-DD (Mon)”
            # and then print the same hourly table with that index.
            ped_by_date = hourly_post.copy()
            # ped_by_date.index is currently a DatetimeIndex of dates; convert each to "YYYY-MM-DD (DayAbbrev)"
            ped_by_date.index = [
                f"{d.strftime('%Y-%m-%d')} ({d.strftime('%a')})"
                for d in ped_by_date.index
            ]
            print("\n--- Pedestrian counts by date (with weekday) and hour ---")
            print(ped_by_date.to_string())
            # ─────────────────────────────────────────────────────────────

            # temp patch not a good fix
            df_pc = df_pc[df_pc.index.date != datetime.date(2025, 1, 6)]
            return df_pc

    else:
        # --- conflict path: first fetch EVERYTHING in [start_date, end_date] ---
        full_df = conn.handleRequest(params, df_type)

        # apply include_flag (p2v vs v2v) filters
        if p2v:
            full_df = full_df[full_df['p2v'] == 1]
            if conflict_type in ('left turning','right turning','thru'):
                code_map = {
                    'left turning': (3,4),
                    'right turning': (1,2),
                    'thru': (5,6)
                }
                full_df = full_df[full_df['conflict_type']\
                                 .isin(code_map[conflict_type])]
        else:
            full_df = full_df[full_df['p2v'] == 0]

        # ── FILTER full_df TO ONLY THOSE “camera-on” INTERVALS ──
        # times is a flat list: [start1, end1, start2, end2, …] (strings)
        # build a list of (datetime,start, datetime,end) tuples:
        intervals = []
        for i in range(0, len(times), 2):
            start_dt = pd.to_datetime(times[i])
            end_dt   = pd.to_datetime(times[i+1])
            intervals.append((start_dt, end_dt))

        # now keep only rows whose timestamp falls in ANY of these intervals
        full_df['ts'] = pd.to_datetime(full_df['timestamp'])
        mask = pd.Series(False, index=full_df.index)
        for (st, et) in intervals:
            mask |= (full_df['ts'] >= st) & (full_df['ts'] <= et)
        filtered = full_df.loc[mask].copy()

        # 1) Create a “pair” column on the camera-on subset:
        filtered['pair'] = filtered['unique_ID1'].astype(str) + '_' + filtered['unique_ID2'].astype(str)

        # 2) Extract date and hour from that filtered set
        filtered['date'] = filtered['timestamp'].dt.date
        filtered['hour'] = filtered['timestamp'].dt.hour

        # 3) Keep only hours between 7 and 19
        hour_mask = (filtered['hour'] >= 7) & (filtered['hour'] <= 19)
        conf = filtered.loc[hour_mask].copy()


        # 4) Group by date & hour and count distinct “pair”
        hourly = (
            conf
            .groupby(['date', 'hour'])['pair']
            .nunique()
            .reset_index(name='distinct_pair_count')
        )

        # 5) Add weekday for debugging or plotting
        hourly['weekday'] = pd.to_datetime(hourly['date']).dt.day_name()

        # 6) If you need a DataFrame indexed by a Timestamp index (date+hour), do:
        hourly['timestamp'] = pd.to_datetime(hourly['date']) + pd.to_timedelta(hourly['hour'], unit='h')
        df_pc = hourly.set_index('timestamp')[['distinct_pair_count']]
        df_pc.rename(columns={'distinct_pair_count': 'count'}, inplace=True)

        # Now df_pc.index is full hours (2024-02-26 07:00, 2024-02-26 08:00, …), 
        # and df_pc['count'] holds “number of distinct (ID1,ID2) pairs in that hour.”
        return df_pc


    # ─── print the *interpolated* hourly pivot for sanity ───
    interp_hourly = (
        df_pc
        .assign(
            date = df_pc.index.date,
            hour = df_pc.index.hour
        )
        .groupby(['date','hour'])['count']
        .sum()
        .reset_index(name='interp_hourly_count')
    )
    interp_pivot = interp_hourly.pivot(
        index='date', columns='hour', values='interp_hourly_count'
    ).fillna(0)
    interp_pivot = _add_weekday_category(interp_pivot)
    print("\nInterpolated hourly counts:")
    print(interp_pivot.to_string())
    # ───────────────────────────────────────────────────────

    return df_pc


def heatmap_generator(df_type: str,
                      mean: bool,
                      intersec_id: int,
                      period: str = 'before',
                      p2v: bool = None,
                      conflict_type: str = None,
                      pedestrian_counting: bool = False,
                      return_agg: bool = False):
    """
    Generate a heatmap (and lineplot) for 'track' or 'conflict' data,
    with missing 15-min clips interpolated and pedestrian hours scaled.
    """
    # 1) Validate
    if df_type not in ['track', 'conflict']:
        raise ValueError('df_type must be "track" or "conflict"')
    if df_type == 'conflict' and p2v is None:
        raise ValueError('p2v must be True or False for conflict data')
    if p2v is False and conflict_type in ['left turning','right turning','thru']:
        raise ValueError('use p2v=True or conflict_type=None')

    # 2) Intersection lookups
    intersec_lookup = {
        3287: "Stirling Road and N 68th Avenue", 3248: "Stirling Road and N 66th Avenue",
        3032: "Stirling Road and SR-7", 3265: "Stirling Road and University Drive",
        3334: "Stirling Road and Carriage Hills Drive/SW 61st Avenue",
        3252: "Stirling Road and Davie Road Extension", 5060: "SW 13th St and University Ave"
    }
    cam_lookup = {3287:24,3248:27,3032:23,3265:30,3334:33,3252:36,5060:7}

    # 3) Base params
    params = {
        'intersec_id': intersec_id,
        'cam_id': cam_lookup[intersec_id],
        'p2v': 0 if p2v is False else 1,
        'start_date': None,
        'end_date': None
    }

    # 4) Time windows
    try:
        times = times_dict[intersec_id][period]
    except KeyError:
        raise ValueError(f"Missing period '{period}' for intersection {intersec_id}")

    # 5) Build interpolated / scaled counts via single query
    df_pc = aggregate_period_counts(
        times, df_type, params,
        pedestrian_counting=pedestrian_counting,
        p2v=p2v,
        conflict_type=conflict_type
    )

    # 6) Enrich for plotting
    df_pc['day_of_week'] = df_pc.index.day_name()
    df_pc['hour']       = df_pc.index.hour
    df_pc['date']       = df_pc.index.date

    # 7) If requested, return aggregated summary
    if return_agg:
        agg = df_pc.groupby(['day_of_week','date'])['count']\
                   .sum().reset_index(name='count')
        total = agg.groupby('day_of_week')['count']\
                   .sum().reset_index(name='total_count')
        days  = agg.groupby('day_of_week')['date']\
                   .nunique().reset_index(name='num_days')
        merged = pd.merge(total, days, on='day_of_week')
        merged['average_count'] = merged['total_count'] / merged['num_days']
        return merged

    # 8) Titles & filenames
    if df_type == 'track':
        if pedestrian_counting:
            title = (f"Heatmap of {'average' if mean else 'total'} pedestrian counts "
                     f"by day of week and hour\n{intersec_lookup[intersec_id]}")
            label = 'peds'
        else:
            title = (f"Heatmap of {'average' if mean else 'total'} vehicle counts "
                     f"by day of week and hour\n{intersec_lookup[intersec_id]}")
            label = 'vehs'
    else:
        if p2v:
            title = (f"Heatmap of {'average' if mean else 'total'} pedestrian-vehicle conflicts "
                     f"by day of week and hour\n{intersec_lookup[intersec_id]} - {conflict_type.title()} Vehicles")
        else:
            title = (f"Heatmap of {'average' if mean else 'total'} vehicle-vehicle conflicts "
                     f"by day of week and hour\n{intersec_lookup[intersec_id]}")
        label = 'p2v' if p2v else 'v2v'
    name = (
        f"heatmap_{intersec_id}_{period}_"
        f"{label}_{df_type}_"
        f"{'mean' if mean else 'sum'}_"
        f"{conflict_type.replace(' ', '_') if conflict_type else 'all'}"
    )

    # 9) Build pivot
    if mean:
        hourly = (
            df_pc
            .groupby([df_pc.index.date, df_pc.index.hour])['count']
            .sum()
            .reset_index(name='hourly_count')
        )

        hourly.columns = ['date', 'hour', 'hourly_count']
        hourly['date'] = pd.to_datetime(hourly['date'])
        hourly = _add_weekday_category(hourly.set_index('date')).reset_index()

        # build kwargs so we only add fill_value=0 for conflicts
        pivot_kwargs = {}
        if df_type == "conflict":
            pivot_kwargs['fill_value'] = 0

        pivot_table = hourly.pivot_table(
            index='day_of_week',
            columns='hour',
            values='hourly_count',
            aggfunc='mean',
            **pivot_kwargs
        )

        full_hours = list(range(7, 20))                    # 7,8,9,…,19
        pivot_table = pivot_table.reindex(columns=full_hours, fill_value=0)


    else:
        df_pc['weekday_short'] = df_pc['day_of_week'].str[:3]
        df_pc['date_weekday_short'] = df_pc['date'].astype(str) + ' (' + df_pc['weekday_short'] + ')'
        pivot_table = df_pc.pivot_table(
            values='count', index='date_weekday_short', columns='hour',
            aggfunc='sum', fill_value=0
        )

    # Restrict pivot to 7-19
    pivot_table = pivot_table.loc[
        :,
        (pivot_table.columns >= 7) & (pivot_table.columns <= 19)
    ]


    # 10) Heatmap
    plt.figure(figsize=(10,6) if not mean else (8,5))
    cmap, vmax = ('viridis', 5000) if df_type=='track' else ('YlGnBu', 4200)
    if df_type=='conflict' and p2v: cmap, vmax = ('inferno', 30)
    sns.heatmap(
        pivot_table, cmap=cmap, annot=True, fmt='.0f',
        vmin=0, vmax=vmax,
        annot_kws={'size':10},
        cbar_kws={'format': plt.FuncFormatter(lambda x,pos: f'{int(x)}')}
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Date (Weekday)' if not mean else 'Day of Week')

    # decide folder name and create it if needed
    if df_type == "conflict":
        os.makedirs("conflict_plots", exist_ok=True)
        folder = "conflict_plots"
    else:  # df_type == "track"
        os.makedirs("track_plots", exist_ok=True)
        folder = "track_plots"

    # save the heatmap PDF + PNG into that folder
    pdf_path = os.path.join(folder, f"{name}.pdf")
    png_path = os.path.join(folder, f"{name}.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")


    # 11) Lineplot
    plt.figure(figsize=(10,6) if not mean else (8,5))
    for idx in pivot_table.index:
        plt.plot(pivot_table.columns, pivot_table.loc[idx], label=idx)
    plt.xlabel('Hour of Day')

    if df_type == "conflict":
        plt.ylabel('Average Conflict Count')
    elif pedestrian_counting:
        plt.ylabel('Average Pedestrian Count')
    else:
        plt.ylabel('Average Vehicle Count')


    plt.grid(True)
    # dynamic y-limit
    if df_type == 'track':
        if pedestrian_counting:
            plt.ylim(0, 90)
        else:
            plt.ylim(0, 5000)
    else:  # conflict data
        # if p2v:
        if intersec_id == 3287:
            plt.ylim(0, 15)
        else:
            plt.ylim(0, 12)
        # else:
            # plt.ylim(0, 30)

    hours = list(pivot_table.columns)
    labels = [(datetime.datetime(2025,1,1,h).strftime("%I %p").lstrip("0")) for h in hours]
    plt.xticks(hours, labels, rotation=30, ha='right', rotation_mode='anchor')

    # ─── sort legend in weekday order ───
    handles, labels = plt.gca().get_legend_handles_labels()
    if mean:
        # labels are full weekday names (Monday, Tuesday, …)
        day_order_full = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        sorted_pairs = sorted(
            zip(labels, handles),
            key=lambda x: day_order_full.index(x[0])
        )
    else:
        # labels are e.g. "2024-02-26 (Mon)", so extract the "Mon" part
        abbrev_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        def key_fn(pair):
            label_text = pair[0]                # e.g. "2024-02-26 (Mon)"
            # split out the three-letter weekday inside parentheses
            try:
                abbrev = label_text.split('(')[1].split(')')[0]
            except IndexError:
                # if format is unexpected, put it at the end
                return len(abbrev_order)
            return abbrev_order.index(abbrev)
        sorted_pairs = sorted(zip(labels, handles), key=key_fn)

    sorted_labels, sorted_handles = zip(*sorted_pairs)
    plt.legend(
        sorted_handles,
        sorted_labels,
        title='Day of Week',
        loc='upper left',
        bbox_to_anchor=(1,1)
    )


    plt.tight_layout(); plt.subplots_adjust(right=0.8)

    line_name = name.replace("heatmap", "lineplot") + f"_lineplot_{period}"

    # (folder variable is already set above; no need to recreate it)
    pdf_lp = os.path.join(folder, f"{line_name}.pdf")
    png_lp = os.path.join(folder, f"{line_name}.png")
    plt.savefig(pdf_lp, bbox_inches="tight")
    plt.savefig(png_lp, bbox_inches="tight")

    # return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def calculate_conflict_rates(conflict_counts_df, volume_counts_df, volume_type='vehicle'):
    """
    Takes two aggregated DataFrames that each have:
      - 'day_of_week'
      - 'total_count' (or some numeric count)
      - 'num_days'
      - 'average_count' (if you want)
    and calculates conflicts per 1,000 <volume_type>.

    conflict_counts_df: aggregated DataFrame of conflict data
      columns: [day_of_week, total_count, num_days, average_count, ...]
    volume_counts_df: aggregated DataFrame of volume data (vehicles/peds)
      columns: [day_of_week, total_count, num_days, average_count, ...]
    volume_type: str, e.g. 'vehicle' or 'pedestrian'

    Returns a merged DataFrame with new columns:
      - conflicts_per_1000_<volume_type>
      - average_conflicts_per_1000_<volume_type>
    """
    # Rename to avoid collisions
    conflict_df = conflict_counts_df.rename(
        columns={'total_count': 'conflict_count', 'average_count': 'avg_conflict_count'}
    )
    volume_df = volume_counts_df.rename(
        columns={'total_count': f'{volume_type}_count', 'average_count': f'avg_{volume_type}_count'}
    )

    # Merge on day_of_week
    merged = pd.merge(
        conflict_df[['day_of_week', 'conflict_count', 'avg_conflict_count']],
        volume_df[['day_of_week', f'{volume_type}_count', f'avg_{volume_type}_count']],
        on='day_of_week',
        how='outer'  # or 'inner' if you only want matching days
    )

    # Calculate conflicts per 1,000 <volume_type>
    merged[f'conflicts_per_1000_{volume_type}'] = np.where(
        merged[f'{volume_type}_count'] > 0,
        (merged['conflict_count'] / merged[f'{volume_type}_count']) * 1000,
        0
    )

    merged[f'avg_conflicts_per_1000_{volume_type}'] = np.where(
        merged[f'avg_{volume_type}_count'] > 0,
        (merged['avg_conflict_count'] / merged[f'avg_{volume_type}_count']) * 1000,
        0
    )

    return merged


def plot_normalized_conflicts(
    df: pd.DataFrame,
    day_col: str,
    rate_col: str,
    intersection_id: int,
    plot_title: str,
    output_filename: str
):
    """
    Plot the specified 'rate_col' of 'df' against 'day_col', 
    saving the figure to 'output_filename'.

    :param df: DataFrame that includes day_of_week and a 
               conflicts-per-1,000 column (or average conflicts).
    :param day_col: Name of the column representing the day of week (e.g., "day_of_week").
    :param rate_col: Name of the column in 'df' that has the conflict rate 
                     (e.g., "avg_conflicts_per_1000_pedestrian").
    :param intersection_id: Intersection ID, used for labeling.
    :param plot_title: Title for the plot.
    :param output_filename: File path/name to save the figure.
    """
    if df.empty:
        print(f"[WARNING] No data to plot for intersection {intersection_id}: '{plot_title}'.")
        return

    # Ensure the day_col is a categorical in a sensible order
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df[day_col] = pd.Categorical(df[day_col], categories=day_order, ordered=True)

    # Sort DataFrame by that day_col
    df.sort_values(day_col, inplace=True)

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        df[day_col],
        df[rate_col],
        marker='o',
        label=plot_title
    )
    plt.title(plot_title, fontsize=14)

    plt.ylim(0, 10)

    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel(rate_col.replace('_', ' ').title(), fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Optionally add a legend
    plt.legend()

    # Save the plot

    #change here 3

    plt.savefig(output_filename, dpi=300)
    print(f"Saved normalized plot: {output_filename}")
    # plt.close()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
        
    base64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return base64_img


##################### main prg #######################
##
## first, p2v
##
# conflict or track
# df_type = "track"
df_type = "conflict"

# mean or sum?
mean = True

p2v = False


# conflict_types = ['left turning', 'right turning', 'thru', 'all']
# for conflict_type in conflict_types:
    # heatmap_generator(df_type, mean, inter, p2v, conflict_type)

##
## v2v
##

# p2v = False

# heatmap_generator(df_type, mean, inter, p2v)


# df_type = "track"

# for inter in [3032, 3265, 3334, 3248, 3287]:
for inter in [3248, 3287]:
    for period in ['before', 'after']:
        print(f"\n=== Processing intersection: {inter} ({period}) ===")

        # # (1) P2V conflict + Ped volume (aggregated)
        # p2v_conflicts_agg = heatmap_generator(
        #     df_type="conflict", mean=True,
        #     intersec_id=inter, period=period,
        #     p2v=True, conflict_type='all',
        #     pedestrian_counting=False,
        #     return_agg=True
        # )
        # ped_volume_agg = heatmap_generator(
        #     df_type="track", mean=True,
        #     intersec_id=inter, period=period,
        #     p2v=False, conflict_type=None,
        #     pedestrian_counting=True,
        #     return_agg=True
        # )

        # # (2) Draw the two heatmaps & lineplots for peds & vehs
        heatmap_generator(
            df_type="track", mean=True,
            intersec_id=inter, period=period,
            p2v=False, conflict_type=None,
            pedestrian_counting=True,
            return_agg=False
        )
        heatmap_generator(
            df_type="track", mean=True,
            intersec_id=inter, period=period,
            p2v=False, conflict_type=None,
            pedestrian_counting=False,
            return_agg=False
        )

        # # (3) Compute P2V per 1k peds
        # p2v_per_1000_peds = calculate_conflict_rates(
        #     conflict_counts_df=p2v_conflicts_agg,
        #     volume_counts_df=ped_volume_agg,
        #     volume_type='pedestrian'
        # )
        # print(f"--- P2V per 1k peds ({period}):")
        # print(p2v_per_1000_peds)
        # plot_normalized_conflicts(
        #     df=p2v_per_1000_peds, day_col="day_of_week",
        #     rate_col="avg_conflicts_per_1000_pedestrian",
        #     intersection_id=inter,
        #     plot_title=f"Intersection {inter}: P2V per 1k peds ({period})",
        #     output_filename=f"intersection_{inter}_p2v_per_1000_peds_{period}.png"
        # )

        # # (4) V2V conflict + Veh volume
        # v2v_conflicts_agg = heatmap_generator(
        #     df_type="conflict", mean=True,
        #     intersec_id=inter, period=period,
        #     p2v=False, conflict_type='all',
        #     pedestrian_counting=False,
        #     return_agg=True
        # )
        # vehicle_volume_agg = heatmap_generator(
        #     df_type="track", mean=True,
        #     intersec_id=inter, period=period,
        #     p2v=False, conflict_type=None,
        #     pedestrian_counting=False,
        #     return_agg=True
        # )

        # # (5) Compute V2V per 1k vehicles
        # v2v_per_1000_vehicles = calculate_conflict_rates(
        #     conflict_counts_df=v2v_conflicts_agg,
        #     volume_counts_df=vehicle_volume_agg,
        #     volume_type='vehicle'
        # )
        # print(f"--- V2V per 1k vehicles ({period}):")
        # print(v2v_per_1000_vehicles)
        # plot_normalized_conflicts(
        #     df=v2v_per_1000_vehicles, day_col="day_of_week",
        #     rate_col="avg_conflicts_per_1000_vehicle",
        #     intersection_id=inter,
        #     plot_title=f"Intersection {inter}: V2V per 1k vehicles ({period})",
        #     output_filename=f"intersection_{inter}_v2v_per_1000_vehicle_{period}.png"
        # )

        heatmap_generator(
            df_type="conflict", mean=True,
            intersec_id=inter, period=period,
            p2v=True, conflict_type='all',
            pedestrian_counting=False,
            return_agg=False
        )

        heatmap_generator(
            df_type="conflict", mean=True,
            intersec_id=inter, period=period,
            p2v=False, conflict_type='all',
            pedestrian_counting=False,
            return_agg=False
        )

#if the heatmap() does not have return_agg = True, then binary image data should be returned
#to find changes if you ever want to revert, ctrl f and then look up change. 
#Remove lines relating to base64 and uncomment other lines