# conflict_analysis.py

from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ────────────────────────────────────────────────────────────────────────────────
# 1) TIME WINDOW UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def get_time_windows(intersection_id: int, period: str = 'before') -> list[str]:
    """
    Return the list of [start, end, start, end, …] timestamp strings for
    a given intersection_id and period ('before' or 'after').
    """
    try:
        return times_dict[intersection_id][period]
    except KeyError:
        raise ValueError(f"Invalid intersection ID ({intersection_id}) or period ({period})")

def compute_unique_dates_and_weeks(time_windows: list[str]) -> tuple[set[pd.Timestamp], float]:
    """
    Given a list of timestamps [start1, end1, start2, end2, …], extract each
    start_i's date (YYYY-MM-DD) and build a set of unique dates.
    Returns (unique_dates_set, weeks_covered = len(unique_dates)/7.0).
    """
    dates = set()
    for i in range(0, len(time_windows), 2):
        date_only = pd.to_datetime(time_windows[i]).date()
        dates.add(date_only)
    num_days = len(dates)
    return dates, (num_days / 7.0 if num_days > 0 else 0.0)

# ────────────────────────────────────────────────────────────────────────────────
# 2) DATA FETCHING & CACHING
# ────────────────────────────────────────────────────────────────────────────────

def fetch_or_cache_conflicts(
    connector: MySQLConnector,
    intersection_id: int,
    p2v_flag: int,
    start_timestamp: str,
    end_timestamp: str,
    cache_dir: str = 'cache'
) -> pd.DataFrame:
    """
    Fetch conflict rows (timestamp, cluster1, cluster2, unique_ID1, unique_ID2)
    from TTCTable for the given intersection_id, p2v_flag, and time range.
    Uses a CSV cache file in <cache_dir>.
    """
    base = f"{intersection_id}_{p2v_flag}_conflict"
    clean = lambda s: s.replace(':', '').replace('-', '').replace(' ', '_')
    cache_name = f"{base}_{clean(start_timestamp)}_{clean(end_timestamp)}.csv"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        df = connector.fetchConflictRecords(intersection_id, p2v_flag, start_timestamp, end_timestamp)
        df.to_csv(cache_path, index=False)
        print(f"  → Cached conflicts to {cache_path}")
    return df

def get_all_conflicts_for_period(
    intersection_id: int,
    p2v_flag: int,
    period: str = 'before'
) -> pd.DataFrame:
    """
    Fetch and concatenate all conflict rows for the given intersection_id/p2v_flag
    across every [start, end] pair in times_dict[intersection_id][period].
    Returns a single DataFrame.
    """
    connector = MySQLConnector()
    all_slices = []
    windows = get_time_windows(intersection_id, period)

    for i in range(0, len(windows), 2):
        start_ts = windows[i]
        end_ts   = windows[i + 1]
        slice_df = fetch_or_cache_conflicts(connector, intersection_id, p2v_flag, start_ts, end_ts)
        if not slice_df.empty:
            all_slices.append(slice_df)

    if not all_slices:
        return pd.DataFrame(columns=['timestamp','cluster1','cluster2','unique_ID1','unique_ID2'])

    combined = pd.concat(all_slices, ignore_index=True)

    # Ensure IDs are strings
    for col in ['unique_ID1', 'unique_ID2']:
        if col in combined.columns:
            combined[col] = combined[col].astype(str)

    # Convert timestamp to datetime
    if 'timestamp' in combined.columns:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')

    return combined

# ────────────────────────────────────────────────────────────────────────────────
# 3) MOVEMENT-TYPE COUNTERS
# ────────────────────────────────────────────────────────────────────────────────

def count_p2v_conflicts(df: pd.DataFrame) -> pd.Series:
    """
    Count P2V conflict movements in the given DataFrame of conflict rows.
    Returns a Series:
      ['Left Turning Vehs', 'Right Turning Vehs', 'Through Vehs'] → integer counts.
    """
    if df.empty:
        return pd.Series({'Left Turning Vehs': 0,
                          'Right Turning Vehs': 0,
                          'Through Vehs':       0})

    # Extract uppercase-only direction codes
    d1 = df['cluster1'].str.findall(r"[A-Z]").str.join("")
    d2 = df['cluster2'].str.findall(r"[A-Z]").str.join("")

    # If cluster1 contains "ped", use d2; else use d1
    ped_in_1 = df['cluster1'].str.lower().str.contains("ped", na=False)
    active_dir = pd.Series(np.where(ped_in_1, d2, d1), index=df.index)

    left_mask  = active_dir.str.endswith("L")
    right_mask = active_dir.str.endswith("R")
    thru_mask  = active_dir.str.endswith("T")

    return pd.Series({
        'Left Turning Vehs':  int(left_mask.sum()),
        'Right Turning Vehs': int(right_mask.sum()),
        'Through Vehs':       int(thru_mask.sum())
    })

def count_v2v_conflicts(df: pd.DataFrame) -> pd.Series:
    """
    Count V2V conflict movements (LOT, RMT, ROL) in the given DataFrame.
    Returns a Series: ['LOT','RMT','ROL'] → integer counts.
    """
    if df.empty:
        return pd.Series({'LOT': 0, 'RMT': 0, 'ROL': 0})

    # Extract uppercase-only direction codes
    d1 = df['cluster1'].str.findall(r"[A-Z]").str.join("")
    d2 = df['cluster2'].str.findall(r"[A-Z]").str.join("")
    end1 = d1.str[-1]
    end2 = d2.str[-1]

    lot_mask = ((end1 == "L") & (end2 == "T")) | ((end1 == "T") & (end2 == "L"))
    rmt_mask = ((end1 == "R") & (end2 == "T")) | ((end1 == "T") & (end2 == "R"))
    rol_mask = ((end1 == "R") & (end2 == "L")) | ((end1 == "L") & (end2 == "R"))

    return pd.Series({
        'LOT': int(lot_mask.sum()),
        'RMT': int(rmt_mask.sum()),
        'ROL': int(rol_mask.sum())
    })

# ────────────────────────────────────────────────────────────────────────────────
# 4) P2V WEEKLY AVERAGE & PLOTTING
# ────────────────────────────────────────────────────────────────────────────────

def compute_p2v_weekly_averages(
    conflict_df: pd.DataFrame,
    weeks_covered: float
) -> pd.Series:
    """
    Given a DataFrame of all P2V conflict rows and the number of weeks covered,
    returns a Series of average conflicts per week by movement type:
      ['Left Turning Vehs','Right Turning Vehs','Through Vehs'] → float (rounded).
    """
    if weeks_covered <= 0:
        raise ValueError("weeks_covered must be > 0 to compute P2V averages.")

    totals = count_p2v_conflicts(conflict_df)
    return (totals / weeks_covered).round(2)


def plot_p2v_bar_chart(
    iid: int,
    period: str,
    avg_by_type: pd.Series,
    output_dir: str = '.'
) -> None:
    """
    Draw and save a bar chart for P2V averages per week:
      <output_dir>/p2v_avg_intersection_<iid>_<period>.png
    """

    # 1) Extract movement labels and their average values
    movement_types = avg_by_type.index.tolist()   # e.g. ['Left Turning Vehs','Right Turning Vehs','Through Vehs']
    values = avg_by_type.values.tolist()          # e.g. [4.5, 3.2, 2.8]

    # 2) Grab the first three colors from the "tab10" colormap
    tab10 = plt.get_cmap('tab10').colors  # returns a tuple of 10 RGBA colors
    # Use indices 0, 1, 2 for the P2V bars:
    bar_colors = tab10[:3]

    # 3) Create the bar chart
    plt.figure(figsize=(7, 5))
    bars = plt.bar(movement_types, values, color=bar_colors)

    plt.xlabel('P2V Movement Type')
    plt.ylabel('Avg Conflicts per Week')
    # plt.title(f'Intersection {iid} - P2V Avg/Week ({period.capitalize()})')

    # 4) Auto-scale the y-axis so that small values still fill most of the plot
    top = max(values) * 1.2 if values else 1.0
    plt.ylim(0, 200)

    # 5) Annotate each bar with its numeric value
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.1f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    # 6) Add horizontal grid lines and finalize
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()

    # 7) Save to file
    filename = f"p2v_avg_intersection_{iid}_{period}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved P2V bar chart: {filename}")



# ────────────────────────────────────────────────────────────────────────────────
# 5) V2V WEEKLY AVERAGE & PLOTTING
# ────────────────────────────────────────────────────────────────────────────────

def compute_v2v_weekly_averages(
    conflict_df: pd.DataFrame,
    weeks_covered: float
) -> pd.Series:
    """
    Given a DataFrame of all conflict rows and the number of weeks covered,
    returns a Series of average V2V conflicts per week:
      ['LOT','RMT','ROL'] → float (rounded).
    """
    if weeks_covered <= 0:
        raise ValueError("weeks_covered must be > 0 to compute V2V averages.")

    totals = count_v2v_conflicts(conflict_df)
    return (totals / weeks_covered).round(2)


def plot_v2v_weekly_bar_chart(
    iid: int,
    period: str,
    avg_by_type: pd.Series,
    output_dir: str = '.'
) -> None:
    """
    Draw and save a bar chart for V2V weekly averages:
      <output_dir>/v2v_weekly_<iid>_<period>.png
    """

    # 1) Extract movement labels and their average values
    movement_types = avg_by_type.index.tolist()   # e.g. ['LOT','RMT','ROL']
    values = avg_by_type.values.tolist()          # e.g. [1.8, 2.4, 0.6]

    # 2) Grab “tab10” and pick indices 3, 4, 5
    tab10 = plt.get_cmap('tab10').colors
    bar_colors = tab10[3:6]  # colors #3, #4, #5 from tab10

    # 3) Create the bar chart
    plt.figure(figsize=(7, 5))
    bars = plt.bar(movement_types, values, color=bar_colors)

    plt.xlabel('V2V Movement Type')
    plt.ylabel('Avg Conflicts per Week')
    # plt.title(f'Intersection {iid} - V2V Avg/Week ({period.capitalize()})')

    # 4) Auto-scale y-axis
    top = max(values) * 1.2 if values else 1.0
    plt.ylim(0, 200)

    # 5) Annotate each bar with its numeric value
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.1f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    # 6) Add grid and finalize
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()

    # 7) Save to file
    filename = f"v2v_weekly_intersection_{iid}_{period}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved V2V weekly bar chart: {filename}")


# ────────────────────────────────────────────────────────────────────────────────
# 6) V2V HOURLY AVERAGE & PLOTTING
# ────────────────────────────────────────────────────────────────────────────────

def compute_v2v_hourly_averages(
    conflict_df: pd.DataFrame,
    v2v_code: str,
    unique_dates: set[pd.Timestamp]
) -> pd.Series:
    """
    Given a DataFrame of all conflict rows, a v2v_code in ['LOT','RMT','ROL'],
    and the set of unique dates (days observed),
    return a Series indexed by hour [0..23] of average conflicts per day:
      (hourly_count) / (# unique_dates).
    """
    direction_map = {'LOT': ('L','T'), 'RMT': ('R','T'), 'ROL': ('R','L')}
    if v2v_code not in direction_map:
        raise ValueError(f"Unsupported v2v_code: {v2v_code}")

    dirA, dirB = direction_map[v2v_code]

    d1 = conflict_df['cluster1'].str.findall(r"[A-Z]").str.join("")
    d2 = conflict_df['cluster2'].str.findall(r"[A-Z]").str.join("")
    end1 = d1.str[-1]
    end2 = d2.str[-1]

    mask = ((end1 == dirA) & (end2 == dirB)) | ((end1 == dirB) & (end2 == dirA))
    filtered = conflict_df[mask].copy()

    if filtered.empty:
        return pd.Series(0, index=range(24), dtype=float)

    filtered['timestamp'] = pd.to_datetime(filtered['timestamp'], errors='coerce')
    filtered['hour'] = filtered['timestamp'].dt.hour

    hourly_counts = filtered.groupby('hour').size()
    n_days = len(unique_dates)

    avg_per_hour = (hourly_counts / n_days).reindex(range(24), fill_value=0)
    return avg_per_hour

def plot_v2v_hourly_line(
    iid: int,
    period: str,
    v2v_code: str,
    avg_per_hour: pd.Series,
    output_dir: str = '.'
) -> None:
    """
    Draw and save a line plot for V2V hourly averages:
      <output_dir>/v2v_hourly_<v2v_code>_intersection_<iid>_<period>.png
    """
    plt.figure(figsize=(8, 4))
    plt.plot(avg_per_hour.index, avg_per_hour.values, marker='o')
    hours = list(range(24))
    labels = [f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in hours]
    plt.xticks(hours, labels, rotation=45)

    plt.xlabel('Hour of Day')
    plt.ylabel(f'Avg {v2v_code} Conflicts per Day')
    # plt.title(f'Intersection {iid} - {v2v_code} Hourly Avg ({period.capitalize()})')
    plt.grid(axis='y', alpha=0.6)
    plt.tight_layout()

    filename = f"v2v_hourly_{v2v_code}_intersection_{iid}_{period}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved V2V hourly plot: {filename}")

# ────────────────────────────────────────────────────────────────────────────────
# 7) MAIN EXECUTION FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

def analyze_p2v_and_plot(iid: int, p2v_flag: int, period: str):
    """
    Full pipeline for P2V:
      1. Fetch all conflicts
      2. Compute unique dates & weeks
      3. Compute avg per week by movement type
      4. Plot P2V bar chart
    """
    print(f"\n--- P2V Analysis: Intersection {iid} ({period}) ---")
    conflicts = get_all_conflicts_for_period(iid, p2v_flag, period)
    print(f"  Raw conflicts fetched: {len(conflicts)}")

    windows = get_time_windows(iid, period)
    unique_dates, weeks = compute_unique_dates_and_weeks(windows)
    print(f"  Unique dates: {sorted(unique_dates)}")
    print(f"  Weeks covered: {weeks:.2f}")

    if weeks <= 0:
        print("  No days covered; skipping P2V average.")
        return

    avg_by_type = compute_p2v_weekly_averages(conflicts, weeks)
    print("  Avg conflicts/week by P2V type:")
    print(avg_by_type.to_frame(name="Avg/week"))
    plot_p2v_bar_chart(iid, period, avg_by_type)

def analyze_v2v_weekly_and_plot(iid: int, p2v_flag: int, period: str):
    """
    Pipeline for V2V weekly averages:
      1. Fetch all conflicts
      2. Compute unique dates & weeks
      3. Compute avg per week for LOT, RMT, ROL
      4. Plot V2V weekly bar chart
    """
    print(f"\n--- V2V Weekly Analysis: Intersection {iid} ({period}) ---")
    conflicts = get_all_conflicts_for_period(iid, p2v_flag, period)
    print(f"  Raw conflicts fetched: {len(conflicts)}")

    windows = get_time_windows(iid, period)
    unique_dates, weeks = compute_unique_dates_and_weeks(windows)
    print(f"  Unique dates: {sorted(unique_dates)}")
    print(f"  Weeks covered: {weeks:.2f}")

    if weeks <= 0:
        print("  No days covered; skipping V2V weekly average.")
        return

    totals = count_v2v_conflicts(conflicts)
    avg_weekly = compute_v2v_weekly_averages(conflicts, weeks)
    print("  Avg conflicts/week by V2V type:")
    print(avg_weekly.to_frame(name="Avg/week"))
    plot_v2v_weekly_bar_chart(iid, period, avg_weekly)

def analyze_v2v_hourly_and_plot(iid: int, p2v_flag: int, period: str):
    """
    Pipeline for V2V hourly averages:
      1. Fetch all conflicts
      2. Compute unique dates
      3. For each code in ['LOT','RMT','ROL'], compute & plot hourly averages
    """
    print(f"\n--- V2V Hourly Analysis: Intersection {iid} ({period}) ---")
    conflicts = get_all_conflicts_for_period(iid, p2v_flag, period)

    windows = get_time_windows(iid, period)
    unique_dates, _ = compute_unique_dates_and_weeks(windows)

    for code in ['LOT', 'RMT', 'ROL']:
        print(f"  V2V code: {code}")
        avg_hourly = compute_v2v_hourly_averages(conflicts, code, unique_dates)
        plot_v2v_hourly_line(iid, period, code, avg_hourly)

# ────────────────────────────────────────────────────────────────────────────────
# 8) RUN FOR SELECTED INTERSECTIONS
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    intersections = [3287, 3248]
    periods      = ['before', 'after']

    for iid in intersections:
        for period in periods:
            # ─── P2V analysis (use p2v_flag = 1) ───
            analyze_p2v_and_plot(iid, p2v_flag=1, period=period)

            # ─── V2V weekly bar chart (use p2v_flag = 0) ───
            analyze_v2v_weekly_and_plot(iid, p2v_flag=0, period=period)

            # ─── V2V hourly line charts (use p2v_flag = 0) ───
            analyze_v2v_hourly_and_plot(iid, p2v_flag=0, period=period)
