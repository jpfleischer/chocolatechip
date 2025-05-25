from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
import datetime
import numpy as np
from yaspin import yaspin
from yaspin.spinners import Spinners
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import time
import io
import base64


def aggregate_period_counts(times: list,
                            df_type: str,
                            params: dict,
                            pedestrian_counting: bool = False,
                            p2v: bool = None,
                            conflict_type: str = None) -> pd.DataFrame:
    """
    1) Fetch everything from the very first start to the very last end
    2) Apply your class / p2v filters
    3) Floor timestamps to 15-minute bins
    4) Count uniques per bin
    5) Reindex on a true 15-min grid globally, but only interpolate *within* each day's recorded window,
       then drop any bins outside the actual video intervals so you never extrapolate past the final clip.
    """
    conn = MySQLConnector()
    column_name = 'track_id' if df_type == 'track' else 'unique_ID1'

    # 1) Big fetch
    params['start_date'], params['end_date'] = times[0], times[-1]
    print(f"Fetching {df_type} data from {params['start_date']} to {params['end_date']}")
    full_df = conn.handleRequest(params, df_type)
    print(f"Fetched {len(full_df)} rows")
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

    # 2) Filters
    if df_type == 'track':
        if pedestrian_counting:
            full_df = full_df[full_df['class'] == 'pedestrian']
        else:
            full_df = full_df[full_df['class'] != 'pedestrian']
    else:
        if p2v:
            full_df = full_df[full_df['p2v'] == 1]
            if conflict_type in ('left turning', 'right turning', 'thru'):
                code_map = {
                    'left turning': (3,4),
                    'right turning': (1,2),
                    'thru': (5,6)
                }
                full_df = full_df[full_df['conflict_type'].isin(code_map[conflict_type])]
        else:
            full_df = full_df[full_df['p2v'] == 0]

    # 3) Bin into 15-minute
    full_df['period_start'] = full_df['timestamp'].dt.floor('15min')

    # 4) Unique-count per bin
    counts = (
        full_df
        .groupby('period_start')[column_name]
        .nunique()
        .rename('count')
    )

    # 5a) Build a global 15-min grid
    start = pd.to_datetime(times[0]).floor('15min')
    end   = pd.to_datetime(times[-1]).floor('15min')
    grid  = pd.date_range(start, end, freq='15min')

    df_pc = counts.reindex(grid).to_frame()

    # 5b) Interpolate only *within* each calendar day
    df_pc['count'] = (
        df_pc['count']
          .groupby(df_pc.index.date)
          .transform(lambda grp: grp.interpolate(method='time'))
    )

    # 5c) Drop any bins that lie outside the actual video intervals
    #     (those will still be NaN after the interpolation above)
    df_pc = df_pc[df_pc['count'].notna()]

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
    with missing 15-min clips interpolated.
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

    # 5) Build interpolated counts via single query
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
        # 1) sum the four 15-min bins into each hour, per day
        hourly = (
            df_pc
            .groupby([df_pc.index.date, df_pc.index.hour])['count']
            .sum()                        # total vehicles per day-per-hour
            .reset_index(name='hourly_count')
        )

        # 2) clean up the column names
        hourly.rename(columns={'level_0':'date','level_1':'hour'}, inplace=True)
        hourly['date'] = pd.to_datetime(hourly['date'])

        # 3) extract the weekday name
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        hourly['day_of_week'] = pd.Categorical(
            hourly['date'].dt.day_name(),
            categories=day_order,
            ordered=True
        )

        
        # 1) find which hours each weekday actually appears in:
        hours_by_day = (
            hourly
            .groupby(['day_of_week','hour'])
            .size()
            .unstack(fill_value=0)
            # this is a DataFrame: rows=day_of_week, cols=hour, values=#Mondays with data at that hour
        )

        # 2) pick only hours where *all* weekdays have at least one sample:
        common_hours = [
            h for h in hours_by_day.columns
            if (hours_by_day[h] > 0).all()
        ]
        print(hours_by_day.loc['Monday'])


        # 3) when you pivot for the mean, drop all other hours:
        pivot_table = (
            hourly
            .pivot_table(
            index='day_of_week',
            columns='hour',
            values='hourly_count',
            aggfunc='mean'
            # no fill_value
            )
        )

        pivot_table = pivot_table.loc[:, common_hours]
    else:
        df_pc['weekday_short'] = df_pc['day_of_week'].str[:3]
        df_pc['date_weekday_short'] = df_pc['date'].astype(str) + ' (' + df_pc['weekday_short'] + ')'
        pivot_table = df_pc.pivot_table(
            values='count', index='date_weekday_short', columns='hour',
            aggfunc='sum', fill_value=0
        )

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
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.savefig(f'{name}.png', bbox_inches='tight')

    # 11) Lineplot
    plt.figure(figsize=(10,6) if not mean else (8,5))
    for idx in pivot_table.index:
        plt.plot(pivot_table.columns, pivot_table.loc[idx], label=idx)
    plt.title(title.replace('Heatmap','Lineplot'))
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.grid(True)
    hours = list(pivot_table.columns)
    labels = [(datetime.datetime(2025,1,1,h).strftime("%I %p").lstrip("0")) for h in hours]
    plt.xticks(hours, labels, rotation=30, ha='right', rotation_mode='anchor')
     # get current handles & labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # define the correct chronological order
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    # sort by the weekday’s position in day_order
    sorted_pairs = sorted(
        zip(labels, handles),
        key=lambda x: day_order.index(x[0])
    )
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    # reapply legend in that order
    plt.legend(
        sorted_handles,
        sorted_labels,
        title='Day of Week',
        loc='upper left',
        bbox_to_anchor=(1,1)
    )

    plt.tight_layout(); plt.subplots_adjust(right=0.8)
    plt.savefig(f'{name.replace("heatmap","lineplot")}_lineplot_{period}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.replace("heatmap","lineplot")}_lineplot_{period}.png', bbox_inches='tight')

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

        # (1) P2V conflict + Ped volume (aggregated)
        p2v_conflicts_agg = heatmap_generator(
            df_type="conflict", mean=True,
            intersec_id=inter, period=period,
            p2v=True, conflict_type='all',
            pedestrian_counting=False,
            return_agg=True
        )
        ped_volume_agg = heatmap_generator(
            df_type="track", mean=True,
            intersec_id=inter, period=period,
            p2v=False, conflict_type=None,
            pedestrian_counting=True,
            return_agg=True
        )

        # (2) Draw the two heatmaps & lineplots for peds & vehs
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

        # (3) Compute P2V per 1k peds
        p2v_per_1000_peds = calculate_conflict_rates(
            conflict_counts_df=p2v_conflicts_agg,
            volume_counts_df=ped_volume_agg,
            volume_type='pedestrian'
        )
        print(f"--- P2V per 1k peds ({period}):")
        print(p2v_per_1000_peds)
        plot_normalized_conflicts(
            df=p2v_per_1000_peds, day_col="day_of_week",
            rate_col="avg_conflicts_per_1000_pedestrian",
            intersection_id=inter,
            plot_title=f"Intersection {inter}: P2V per 1k peds ({period})",
            output_filename=f"intersection_{inter}_p2v_per_1000_peds_{period}.png"
        )

        # (4) V2V conflict + Veh volume
        v2v_conflicts_agg = heatmap_generator(
            df_type="conflict", mean=True,
            intersec_id=inter, period=period,
            p2v=False, conflict_type='all',
            pedestrian_counting=False,
            return_agg=True
        )
        vehicle_volume_agg = heatmap_generator(
            df_type="track", mean=True,
            intersec_id=inter, period=period,
            p2v=False, conflict_type=None,
            pedestrian_counting=False,
            return_agg=True
        )

        # (5) Compute V2V per 1k vehicles
        v2v_per_1000_vehicles = calculate_conflict_rates(
            conflict_counts_df=v2v_conflicts_agg,
            volume_counts_df=vehicle_volume_agg,
            volume_type='vehicle'
        )
        print(f"--- V2V per 1k vehicles ({period}):")
        print(v2v_per_1000_vehicles)
        plot_normalized_conflicts(
            df=v2v_per_1000_vehicles, day_col="day_of_week",
            rate_col="avg_conflicts_per_1000_vehicle",
            intersection_id=inter,
            plot_title=f"Intersection {inter}: V2V per 1k vehicles ({period})",
            output_filename=f"intersection_{inter}_v2v_per_1000_vehicle_{period}.png"
        )

#if the heatmap() does not have return_agg = True, then binary image data should be returned
#to find changes if you ever want to revert, ctrl f and then look up change. 
#Remove lines relating to base64 and uncomment other lines