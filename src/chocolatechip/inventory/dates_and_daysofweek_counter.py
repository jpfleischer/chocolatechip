#!/usr/bin/env python3
from chocolatechip.times_config import times_dict
from chocolatechip.intersections import intersection_lookup
from datetime import datetime, timedelta
from collections import Counter
import sys
import pandas as pd
from collections import defaultdict

# Full weekday names and abbreviations
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ABBR = {
    'Monday':'Mon', 'Tuesday':'Tue', 'Wednesday':'Wed', 'Thursday':'Thu',
    'Friday':'Fri', 'Saturday':'Sat', 'Sunday':'Sun'
}

# Abbreviation map for intersection name components
NAME_ABBR = {
    'Road': 'Rd.', 'Avenue':'Ave.', 'Drive':'Dr.', 'Extension':'Ext.',
}

# String to append two LaTeX backslashes
NEWLINE = "\\\\"

def abbreviate_name(name):
    for long, short in NAME_ABBR.items():
        name = name.replace(long, short)
    return name

def analyze_times(times_dict):
    """
    For each intersection and each period ("before","after"):
      • Sum all recording windows into seconds per date
      • Aggregate those seconds by weekday
      • Convert to hours and print hours per weekday

    Additionally:
      • Build a naive weekday count table: for each intersection, count
        each recorded date once per weekday based *only* on the "before"
        period, regardless of duration, and print as a LaTeX table with
        abbreviated intersection names and weekdays.
    """
    naive_counts_all   = {}
    hours_records_all  = {}

    for intersection_id, periods in times_dict.items():
        # skip unwanted
        if intersection_id == 5060:
            continue

        raw_name = intersection_lookup.get(intersection_id, str(intersection_id))
        name     = abbreviate_name(raw_name)
        print(f"Intersection {name}:")

        naive_dates = set()

        for period_name in ("before", "after"):
            ts_list = periods.get(period_name, [])
            if not ts_list:
                print(f"  {period_name.capitalize()}: (no entries)\n")
                continue

            # accumulate total seconds per date for this period
            secs_per_date = Counter()
            for start_s, end_s in zip(ts_list[0::2], ts_list[1::2]):
                start = datetime.fromisoformat(start_s)
                end   = datetime.fromisoformat(end_s)

                # for naïve counts, record any date touched during "before"
                if period_name == "before":
                    current = start.date()
                    while current <= end.date():
                        naive_dates.add(current)
                        current += timedelta(days=1)

                # split if crossing midnight
                if start.date() == end.date():
                    secs_per_date[start.date()] += (end - start).total_seconds()
                else:
                    midnight = datetime.combine(start.date(), datetime.max.time())
                    secs_per_date[start.date()] += (midnight - start).total_seconds()
                    next_day = datetime.combine(end.date(), datetime.min.time())
                    secs_per_date[end.date()]   += (end - next_day).total_seconds()

            # aggregate seconds into hours by weekday
            secs_by_weekday = Counter()
            for d, secs in secs_per_date.items():
                secs_by_weekday[d.strftime("%A")] += secs
            hours_by_weekday = {wd: secs/3600.0 for wd, secs in secs_by_weekday.items()}

            # if this is "before", stash hours for the second table
            if period_name == "before":
                hours_records_all[name] = hours_by_weekday

            # print your existing detailed breakdown
            print(f"  {period_name.capitalize()} (hours by weekday):")
            for wd in WEEKDAYS:
                hrs = hours_by_weekday.get(wd, 0.0)
                if hrs > 0:
                    print(f"    {DAY_ABBR[wd]:<3} {hrs:5.2f} h")
            print()

        # stash naïve counts for the first table
        naive_counts_all[name] = Counter(d.strftime("%A") for d in naive_dates)
        print()

    # ——— First LaTeX table: naïve weekday counts ———
    print("Naive weekday counts per intersection (before only):")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Naive weekday counts per intersection for the before period.}")
    print(r"\label{tab:weekday_counts}")
    print(r"\begin{tabular}{lccccccc}")
    print(r"Intersection & Mon & Tue & Wed & Thu & Fri & Sat & Sun \\")
    print(r"\hline")
    for name, counts in naive_counts_all.items():
        vals = " & ".join(str(counts.get(wd, 0)) for wd in WEEKDAYS)
        print(f"{name} & {vals}{NEWLINE}")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # ——— Second LaTeX table: recorded hours by weekday ———
    print("Recorded hours per intersection (before only):")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    # pad columns and rows
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\renewcommand{\arraystretch}{1.1}")
    print(r"\caption{Total recorded hours by weekday per intersection for the before period.}")
    print(r"\label{tab:hours_by_weekday}")
    print(r"\begin{tabular}{lccccccc}")
    print(r"Intersection & Mon & Tue & Wed & Thu & Fri & Sat & Sun \\")
    print(r"\hline")
    for name, hours in hours_records_all.items():
        vals = " & ".join(f"{hours.get(wd, 0.0):.1f}" for wd in WEEKDAYS)
        print(f"{name} & {vals}{NEWLINE}")
    print(r"\end{tabular}")
    print(r"\end{table}")



def _get_date_set(ts_list):
    dates = set()
    for start_s, end_s in zip(ts_list[0::2], ts_list[1::2]):
        start = datetime.fromisoformat(start_s).date()
        end   = datetime.fromisoformat(end_s).date()
        cur = start
        while cur <= end:
            dates.add(cur)
            cur += timedelta(days=1)
    return dates

def _compress_days(days):
    if not days:
        return ""
    days = sorted(days)
    ranges = []
    start = prev = days[0]
    for d in days[1:]:
        if d == prev + 1:
            prev = d
        else:
            ranges.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = prev = d
    ranges.append(f"{start}" if start == prev else f"{start}–{prev}")
    return ", ".join(ranges)

def monthly_dates_df(iids, times_dict, intersection_lookup):
    rows = []
    for iid in iids:
        raw_name = intersection_lookup.get(iid, str(iid))
        name     = abbreviate_name(raw_name)
        before = _get_date_set(times_dict[iid].get("before", []))
        after  = _get_date_set(times_dict[iid].get("after", []))

        mb = defaultdict(list)
        ma = defaultdict(list)
        for d in before:
            mb[d.strftime("%b %Y")].append(d.day)
        for d in after:
            ma[d.strftime("%b %Y")].append(d.day)

        months = sorted(set(mb) | set(ma),
                        key=lambda m: datetime.strptime(m, "%b %Y"))

        for mon in months:
            rows.append({
                "Intersection": name,
                "Month":        mon,
                "Before (dates)": _compress_days(mb[mon]) or "—",
                "After (dates)":  _compress_days(ma[mon]) or "—",
            })

    df = pd.DataFrame(rows, columns=["Intersection","Month","Before (dates)","After (dates)"])
    return df


if __name__ == "__main__":
    if not times_dict:
        print("No data found in times_dict!", file=sys.stderr)
        sys.exit(1)
    analyze_times(times_dict)
    print('\n\n')
    df = monthly_dates_df([3248, 3287], times_dict, intersection_lookup)
    print(df.to_latex(index=False,
                      longtable=True,
                      caption="Data collection dates by month for each intersection.",
                      label="tab:monthly_dates",
                      na_rep="—"))
    