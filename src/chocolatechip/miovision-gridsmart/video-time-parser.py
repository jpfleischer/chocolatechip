#!/usr/bin/env python3
import os, re, json, subprocess, glob
from datetime import datetime, date, timedelta
from tqdm import tqdm

# === CONFIG ===
# If you want to test a single camera, you can comment out CAMERA_IDS and uncomment:
CAMERA_IDS  = [24]           # ← put your camera number here (the "24_…" prefix)
# CAMERA_IDS = [25, 26]      # ← your camera prefixes
TIMEFRAME  = "before"      # "before" or "after"
TRACKING_DIR = "/mnt/hdd/data/video_pipeline/tracking"
SSH_ALIAS    = "maltlab"

# cutoff = Oct 1 2024
CUTOFF_DATE = date(2024,10,1)

OUTPUT_TIMES_PY     = "video_times_output.py"

# build a regex that matches any of your camera prefixes
camera_group = "|".join(str(c) for c in CAMERA_IDS)
# match e.g. "25_2024-03-16_08-45-05.000.mp4" or without millis
FNAME_RE = re.compile(
    rf"^({camera_group})_"         # camera prefix
    r"(\d{4})-(\d{2})-(\d{2})_"     # YYYY-MM-DD_
    r"(\d{2})-(\d{2})-(\d{2})"      # HH-MM-SS
    r"(?:\.\d{3})?\.mp4$"           # optional .mmm + .mp4
)

# 1) Discover all .mp4 files (local first, then remote)
video_files = []

if os.path.isdir(TRACKING_DIR):
    # Local discovery
    for cam in CAMERA_IDS:
        pattern = os.path.join(TRACKING_DIR, f"{cam}_*.mp4")
        for p in glob.glob(pattern):
            if FNAME_RE.match(os.path.basename(p)):
                video_files.append((False, None, p))
else:
    # Remote discovery via ssh find, per-camera
    for cam in CAMERA_IDS:
        remote_pattern = f"{TRACKING_DIR}/{cam}_*.mp4"
        ssh_find = [
            "ssh", SSH_ALIAS,
            "find", TRACKING_DIR,
            "-type", "f",
            "-name", f"{cam}_*.mp4",
            "-print"
        ]
        try:
            out = subprocess.check_output(ssh_find, stderr=subprocess.DEVNULL, text=True)
            for line in out.splitlines():
                path = line.strip()
                if FNAME_RE.match(os.path.basename(path)):
                    video_files.append((True, SSH_ALIAS, path))
        except subprocess.CalledProcessError:
            # no files for this camera
            continue

# sort by filename (so you process in chronological order)
video_files.sort(key=lambda x: os.path.basename(x[2]))


# 2) Filter by year and before/after cutoff
filtered = []
for is_remote, host, path in video_files:
    fn = os.path.basename(path)
    m = FNAME_RE.match(fn)
        # pull out only groups 2,3,4
    yr = int(m.group(2))
    mo = int(m.group(3))
    dy = int(m.group(4))

    if yr == 2023:
        continue
    file_date = date(yr,mo,dy)
    if TIMEFRAME == "before" and file_date >= CUTOFF_DATE:
        continue
    if TIMEFRAME == "after"  and file_date <  CUTOFF_DATE:
        continue
    filtered.append((is_remote, host, path))


# 3) Probe durations and collect start/end
video_times = []  # tuples (start_dt, end_dt, start_str, end_str)
total_secs = 0.0

for is_remote, host, path in tqdm(filtered, desc="Probing"):
    fn = os.path.basename(path)
    yr,mo,dy, hh,mm,ss = map(int, FNAME_RE.match(fn).groups()[1:7])
    start_dt = datetime(yr,mo,dy,hh,mm,ss)
    cmd = (["ssh", host] if is_remote else []) + [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(json.loads(res.stdout)["format"]["duration"])
        if dur <= 0:
            continue
    except Exception:
        continue

    total_secs += dur
    end_dt = start_dt + timedelta(seconds=dur)
    s_str = start_dt.strftime("%Y-%m-%d %H:%M:%S") + ".000"
    e_str = end_dt.strftime("%Y-%m-%d %H:%M:%S") + ".000"
    video_times.append((start_dt, end_dt, s_str, e_str))


# 4) Dedupe overlapping starts: keep the longest clip for each start_dt
video_times.sort(key=lambda x: x[0])
best = {}
for s_dt, e_dt, s_str, e_str in video_times:
    if s_dt not in best or e_dt > best[s_dt][0]:
        best[s_dt] = (e_dt, s_str, e_str)

unique = [(s_dt, v[1], v[2]) for s_dt, v in sorted(best.items())]


# 5) Write to your times file
with open(OUTPUT_TIMES_PY, "w") as f:
    for _, s_str, e_str in unique:
        f.write(f"'{s_str}','{e_str}',\n")

print(f"Saved {len(unique)} ranges; total hours={total_secs/3600:.2f}")


# 6) Plot coverage timeline
import matplotlib.pyplot as plt

# rebuild datetime ranges
segs = [
    (
        datetime.strptime(s_str, "%Y-%m-%d %H:%M:%S.%f"),
        datetime.strptime(e_str, "%Y-%m-%d %H:%M:%S.%f")
    )
    for _, s_str, e_str in unique
]

by_day = {}
for sd, ed in segs:
    day = sd.date()
    start_sec = sd.hour*3600 + sd.minute*60 + sd.second
    dur = (ed - sd).total_seconds()
    by_day.setdefault(day, []).append((start_sec, dur))


days = sorted(by_day)

# prepare a clean underscore-separated camera label
cam_label = "_".join(str(c) for c in CAMERA_IDS)

# now build the coverage-plot filename
OUTPUT_COVERAGE_PNG = f"video_coverage_{cam_label}_{TIMEFRAME}.png"

fig, ax = plt.subplots(figsize=(10, len(days)*0.5 + 1))
for i, day in enumerate(days):
    ax.broken_barh([(0, 86400)], (i, 0.8), facecolors='lightgray')
    ax.broken_barh(by_day[day],     (i, 0.8), facecolors='blue')

ax.set_yticks([i + 0.4 for i in range(len(days))])
ax.set_yticklabels([d.strftime("%b %d") for d in days])
ax.set_xlim(0, 86400)
ax.set_xlabel("Seconds from midnight")

ax.set_title(f"Cameras {cam_label} — {TIMEFRAME.capitalize()} since Oct 1 2024")

plt.tight_layout()
plt.savefig(OUTPUT_COVERAGE_PNG, dpi=300)
print(f"Saved plot to {OUTPUT_COVERAGE_PNG}")