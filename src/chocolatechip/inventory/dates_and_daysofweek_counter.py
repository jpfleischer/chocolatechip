#!/usr/bin/env python3
from chocolatechip.times_config import times_dict
from datetime import datetime
from collections import Counter
import sys

def analyze_times(times_dict):
    for intersection_id, periods in times_dict.items():
        print(f"Intersection {intersection_id}:")
        for period_name in ("before", "after"):
            timestamps = periods.get(period_name, [])
            # parse out dates and dedupe
            dates = {
                datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").date()
                for ts in timestamps
            }
            if not dates:
                print(f"  {period_name.capitalize()}: (no entries)\n")
                continue

            # sort dates and get weekday names
            sorted_dates = sorted(dates)
            weekdays = [d.strftime("%A") for d in sorted_dates]

            # count weekdays
            weekday_counts = Counter(weekdays)

            # print results
            print(f"  {period_name.capitalize()}:")
            print(f"    Unique dates ({len(sorted_dates)}):")
            for d, wd in zip(sorted_dates, weekdays):
                print(f"      {d} — {wd}")
            print(f"    Weekday counts ({sum(weekday_counts.values())} days):")
            for day in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]:
                count = weekday_counts.get(day, 0)
                if count:
                    print(f"      {day}: {count}")
            print()  # blank line between periods
        print()  # blank line between intersections

if __name__ == "__main__":
    if not times_dict:
        print("No data found in times_dict!", file=sys.stderr)
        sys.exit(1)
    analyze_times(times_dict)
