import pandas as pd
import warnings
# suppress pandas SQLAlchemy DBAPI2 connection warning
warnings.filterwarnings("ignore", message=".*supports SQLAlchemy connectable.*")
from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.times_config import times_dict
from chocolatechip.intersections import intersection_lookup, cam_lookup
from chocolatechip.heatmap.heatmap import aggregate_period_counts
from chocolatechip.conflict.weeklyp2vtype import get_time_windows, get_all_conflicts_for_period


def compute_daily_conflicts(
    iid: int,
    p2v_flag: int = 1,
    period: str = 'before'
) -> pd.Series:
    """
    Returns a Series indexed by each calendar date on which recordings exist,
    with the number of conflicts observed that day (including zeros for days
    with no conflicts).
    """
    # 1) fetch raw conflict rows
    df = get_all_conflicts_for_period(iid, p2v_flag, period)

    # 2) parse your window timestamp strings into datetimes
    windows = get_time_windows(iid, period)
    windows_dt = [pd.to_datetime(w) for w in windows]

    # 3) determine which calendar days were recorded (use every start timestamp)
    recorded_days = sorted({ts.date() for ts in windows_dt[0::2]})

    # if no recording windows -> empty
    if not recorded_days:
        return pd.Series(dtype=int)

    # 4) if no conflicts at all, return zeros for each recorded date
    if df.empty:
        return pd.Series(
            0,
            index=recorded_days,
            name='conflicts'
        )

    # 5) count raw conflicts per date
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size()

    # 6) reindex to include days with zero conflicts
    daily_full = daily_counts.reindex(recorded_days, fill_value=0)
    daily_full.name = 'conflicts'

    return daily_full


def compute_daily_track_counts_via_heatmap(intersection_id: int, pedestrian: bool, period: str = 'before') -> pd.Series:
    """
    Returns a Series indexed by date with total pedestrian or vehicle counts per day.
    """
    conn = MySQLConnector()
    params = {
        'intersec_id': intersection_id,
        'cam_id':      cam_lookup[intersection_id],
        'p2v':         0,
        'start_date':  None,
        'end_date':    None
    }
    times = get_time_windows(intersection_id, period)
    df_min = aggregate_period_counts(
        times, df_type='track', params=params, pedestrian_counting=pedestrian
    )
    df_min['date'] = df_min.index.date
    daily = df_min.groupby('date')['count'].sum()
    daily.name = 'peds' if pedestrian else 'vehs'
    return daily


def compute_pedestrian_risk(intersection_id: int, period: str = 'before') -> tuple[int,int,float]:
    conn    = MySQLConnector()
    windows = get_time_windows(intersection_id, period)

    # 1) collect full 16‑digit pedestrian IDs
    ped_ids = set()
    for start, end in zip(windows[0::2], windows[1::2]):
        df = conn.fetchPedestrianTracks(intersection_id,
                                        cam_lookup[intersection_id],
                                        start, end)
        ped_ids.update(df['unique_ID'].astype(str).tolist())

    # 2) collect raw 15‑digit conflict IDs
    raw_conflict_ids = set()
    for start, end in zip(windows[0::2], windows[1::2]):
        df = conn.fetchConflictRecords(intersection_id, p2v=1, start=start, end=end)
        raw_conflict_ids.update(df['unique_ID1'].astype(str).tolist())
        raw_conflict_ids.update(df['unique_ID2'].astype(str).tolist())

    # 3) build the overlap set without double‑counting
    overlap = set()
    for cid in raw_conflict_ids:
        has_raw  = cid    in ped_ids
        has_pref = ('1'+cid) in ped_ids

        if has_raw and has_pref:
            # both forms exist—keep only one
            overlap.add(cid)           # or add '1'+cid if you prefer the full form
        elif has_raw:
            overlap.add(cid)
        elif has_pref:
            overlap.add('1'+cid)

    total      = len(ped_ids)
    conflicted = len(overlap)
    prob       = conflicted / total if total > 0 else 0.0
    return total, conflicted, prob



if __name__ == '__main__':
    intersections = [
        3252, 3334, 3265, 
        3248, 3287, 
        3032
    ]
    period = 'before'

    rows = []
    for iid in intersections:
        name      = intersection_lookup[iid]
        avg_conf = compute_daily_conflicts(iid=iid,
                                           p2v_flag=1,
                                           period=period).mean()
        avg_peds  = compute_daily_track_counts_via_heatmap(iid, True, period).mean()
        avg_vehs  = compute_daily_track_counts_via_heatmap(iid, False, period).mean()
        # no f-strings here, keep them numeric
        rows.append({
            'Intersection':  name,
            'Conflicts/day': avg_conf,
            'Peds/day':      avg_peds,
            'Vehs/day':      avg_vehs,
        })

    df = pd.DataFrame(rows)

    # 1) Get full LaTeX (pandas will emit table+tabular)
    full = df.to_latex(
        index=False,
        column_format="lrrr",
        formatters={
            'Conflicts/day': '{:,.2f}'.format,
            'Peds/day':      '{:,.0f}'.format,
            'Vehs/day':      '{:,.0f}'.format,
        }
    )

    # 2) Extract only the tabular block
    lines = full.splitlines()
    start = next(i for i, L in enumerate(lines) if L.startswith(r'\begin{tabular}'))
    end   = next(i for i, L in enumerate(lines) if L.startswith(r'\end{tabular}'))
    tabular = "\n".join(lines[start:end+1])

    # 3) Wrap with your own table float + centering
    latex = f"""\\begin{{table}}[htbp]
\\centering
{tabular}
\\caption{{P2V conflict and volume summary for each surveyed intersection.}}
\\label{{tab:conflict_summary}}
\\end{{table}}
"""
    print(latex)