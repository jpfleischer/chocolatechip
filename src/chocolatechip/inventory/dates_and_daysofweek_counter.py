#!/usr/bin/env python3
from chocolatechip.times_config import times_dict
from chocolatechip.intersections import intersection_lookup
from datetime import datetime, timedelta
from collections import Counter
import sys

# Full weekday names and abbreviations
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ABBR = {
    'Monday':'Mon', 'Tuesday':'Tue', 'Wednesday':'Wed', 'Thursday':'Thu',
    'Friday':'Fri', 'Saturday':'Sat', 'Sunday':'Sun'
}

# Abbreviation map for intersection name components
NAME_ABBR = {
    'Road': 'Rd.', 'Avenue':'Ave.', 'Drive':'Dr.', 'Extension':'Ext.',
    # add more if needed
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
    naive_counts_all = {}

    for intersection_id, periods in times_dict.items():
        # skip intersection 5060
        if intersection_id == 5060:
            continue

        # Lookup and abbreviate name
        raw_name = intersection_lookup.get(intersection_id, str(intersection_id))
        name = abbreviate_name(raw_name)
        print(f"Intersection {name}:")

        # Collect naive dates only for "before"
        naive_dates = set()

        for period_name in ("before", "after"):
            ts_list = periods.get(period_name, [])
            if not ts_list:
                print(f"  {period_name.capitalize()}: (no entries)\n")
                continue

            # Accumulate total seconds per date for both periods
            secs_per_date = Counter()
            for start_s, end_s in zip(ts_list[0::2], ts_list[1::2]):
                start = datetime.fromisoformat(start_s)
                end   = datetime.fromisoformat(end_s)

                # Only for naive: record dates for "before"
                if period_name == "before":
                    current = start.date()
                    while current <= end.date():
                        naive_dates.add(current)
                        current += timedelta(days=1)

                # Split durations across midnight if needed
                if start.date() == end.date():
                    secs_per_date[start.date()] += (end - start).total_seconds()
                else:
                    midnight = datetime.combine(start.date(), datetime.max.time())
                    secs_per_date[start.date()] += (midnight - start).total_seconds()
                    next_day = datetime.combine(end.date(), datetime.min.time())
                    secs_per_date[end.date()]   += (end - next_day).total_seconds()

            # Aggregate and convert to hours
            secs_by_weekday = Counter()
            for d, secs in secs_per_date.items():
                secs_by_weekday[d.strftime("%A")] += secs
            hours_by_weekday = {wd: secs/3600.0 for wd, secs in secs_by_weekday.items()}

            # Print detailed results
            print(f"  {period_name.capitalize()} (hours by weekday):")
            for wd in WEEKDAYS:
                hrs = hours_by_weekday.get(wd, 0.0)
                if hrs > 0:
                    print(f"    {DAY_ABBR[wd]:<3} {hrs:5.2f} h")
            print()
        print()

        # Store only "before" naive counts
        naive_counts_all[name] = Counter(d.strftime("%A") for d in naive_dates)

    # LaTeX table with abbreviated weekdays wrapped in table environment
    print("Naive weekday counts per intersection (before only):")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Naive weekday counts per intersection for the 'before' period.}")
    print(r"\label{tab:weekday_counts}")
    print(r"\begin{tabular}{lccccccc}")
    print(r"Intersection & Mon & Tue & Wed & Thu & Fri & Sat & Sun \\")
    print(r"\hline")
    for name, counts in naive_counts_all.items():
        row = [counts.get(wd, 0) for wd in WEEKDAYS]
        vals = " & ".join(str(x) for x in row)
        print(f"{name} & {vals}{NEWLINE}")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    if not times_dict:
        print("No data found in times_dict!", file=sys.stderr)
        sys.exit(1)
    analyze_times(times_dict)
