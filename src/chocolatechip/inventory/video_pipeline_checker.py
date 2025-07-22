#!/usr/bin/env python3
import os
import sys
import re
import argparse
import getpass
import math
from datetime import datetime, timedelta

from dotenv import load_dotenv
from tqdm import tqdm
from fabric import Connection

from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.miovision_gridsmart.filemuncher import get_new_filename

# ── CONFIG ──────────────────────────────────────────────────────────────────────
SSH_HOST_VAR   = "host"    # key in login.env for SSH host
SSH_USER_VAR   = "SSH_USER"    # key in login.env for SSH user

SOURCE_DIR     = "/mnt/vast/BrowardVideosAll/2023"
# SOURCE_DIR     = "/mnt/vast/68AvBefore/fromvast"
PIPELINE_DIR   = "/mnt/hdd/data/video_pipeline"
INTERSECTION   = 3287
DATE_FILTER    = "2023-08-21"   # e.g. YYYY‑MM‑DD, or None for no date filter
CAM_PATTERN    = "68Av"
# CAM_PATTERN    = ""
INTERVAL_MINS  = 15                # segment length in minutes
MIN_SIZE_B     = 5 * 1024 * 1024   # 5 MB
WEEKDAY_FILTER = "Monday"       # set to specific day or None for all days
# ── end CONFIG ─────────────────────────────────────────────────────────────────


# parse datetime from original or processed filename
# matches either "YYYY-MM-DD_HH-MM-SS-rtsp_SITE_suffix.mp4" or "CAMID_YYYY-MM-DD_HH-MM-SS.000.mp4"
_TS_RE = re.compile(
    r"^(?:"
    r"\d+_ (?P<date1>\d{4}-\d{2}-\d{2})_(?P<time1>\d{2}-\d{2}-\d{2})\.\d{3}\.mp4"  # processed
    r"|"
    r"(?P<date2>\d{4}-\d{2}-\d{2})_(?P<time2>\d{2}-\d{2}-\d{2})-rtsp_.*\.mp4"  # raw
    r")$"
    .replace(" ", "")
)

# regex to extract full source filename parts for trimming
_TS_FULL_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<time>\d{2}-\d{2}-\d{2})-rtsp_"
    r"(?P<site>[^_]+)_(?P<suffix>\d)\.mp4$"
)


def parse_start(fname):
    m = _TS_RE.match(fname)
    if not m:
        return None
    # pick whichever matched
    date = m.group('date1') or m.group('date2')
    time = m.group('time1') or m.group('time2')
    ts = time.replace('-', ':')
    return datetime.strptime(f"{date} {ts}", "%Y-%m-%d %H:%M:%S")


def find_missing(conn, mc):
    """
    Return list of filenames in SOURCE_DIR matching DATE_FILTER and CAM_PATTERN
    that have no car tracks in the next INTERVAL_MINS.
    """
    missing = []

    # build the shell‑glob pattern to include date & cam if set
    parts = []
    if DATE_FILTER:
        parts.append(f"*{DATE_FILTER}*")
    if CAM_PATTERN:
        parts.append(f"*{CAM_PATTERN}*")
    parts.append("*.mp4")
    name_pat = "".join(parts)

    # find all candidate files
    find_cmd = (
        f"find {SOURCE_DIR} -maxdepth 1 -type f -name '{name_pat}'"
        " -printf '%f|%s\n'"
    )
    result = conn.run(find_cmd, hide=True)

    for line in tqdm(result.stdout.strip().splitlines(),
                     desc="Checking videos", unit="file"):
        try:
            fn, size_str = line.split("|", 1)
            size = int(size_str)
        except ValueError:
            continue
        # skip too-small files
        if size < MIN_SIZE_B:
            continue

        # parse timestamp from filename
        dt = parse_start(fn)
        if dt is None:
            print(f"⚠️ skipping unknown filename: {fn}")
            continue

        # optional weekday filter
        if WEEKDAY_FILTER and dt.strftime("%A") != WEEKDAY_FILTER:
            continue

        # check for car tracks in this interval
        start_s = dt.strftime("%Y-%m-%d %H:%M:%S")
        end_s   = (dt + timedelta(minutes=INTERVAL_MINS)) \
                    .strftime("%Y-%m-%d %H:%M:%S")
        cnt = mc.countTracks(
            intersection_id=INTERSECTION,
            start=start_s,
            end=end_s,
            class_name="car"
        )
        if cnt < 1:
            missing.append(fn)

    return missing



def copy_and_trim(conn, missing, sudo_passwd):
    copied = []
    for fn in missing:
        src = f"{SOURCE_DIR.rstrip('/')}/{fn}"
        m_full = _TS_FULL_RE.match(fn)
        if not m_full:
            print(f"⚠️ cannot parse original parts, skipping: {fn}")
            continue
        site   = m_full.group("site")
        suffix = m_full.group("suffix")
        probe = conn.run(
            f"ffprobe -v error -show_entries format=duration "
            f"-of default=noprint_wrappers=1:nokey=1 {src!r}", hide=True, warn=True
        )
        try:
            duration = float(probe.stdout.strip())
        except Exception:
            print(f"❌ could not parse duration for {fn}")
            continue
        chunk_secs = INTERVAL_MINS * 60
        total_chunks = math.ceil(duration / chunk_secs)
        orig_dt = parse_start(fn)
        for i in range(total_chunks):
            offset = i * chunk_secs
            seg_start = orig_dt + timedelta(seconds=offset)
            seg_dur = min(chunk_secs, duration - offset)
            # construct stub in original schema
            ds = seg_start.strftime("%Y-%m-%d")
            ts = seg_start.strftime("%H-%M-%S")
            stub = f"{ds}_{ts}-rtsp_{site}_{suffix}.mp4"
            new_name = get_new_filename(stub)
            if not new_name:
                print(f"⚠️ get_new_filename failed for {stub}")
                continue
            dst = f"{PIPELINE_DIR.rstrip('/')}/{new_name}"
            exists = conn.run(f"[ -f {dst!r} ] && echo yes || echo no", hide=True).stdout.strip()
            if exists == "yes":
                continue
            print(f"✂️ trimming chunk {i+1}/{total_chunks}: {fn} → {new_name}")
            ff_cmd = f"ffmpeg -y -ss {offset} -i {src!r} -t {seg_dur} -c copy {dst!r}"
            conn.sudo(ff_cmd, password=sudo_passwd)
            stat = conn.run(f"stat -c '%s' {dst!r}", hide=True, warn=True)
            if stat.ok:
                size = stat.stdout.strip().strip("'")
                print(f"✅ {new_name} ({size} bytes)")
                copied.append(new_name)
            else:
                print(f"❌ failed to stat {new_name}")
    print(f"\nDone. {len(copied)} trimmed segments copied.")
    return copied


def main():
    parser = argparse.ArgumentParser(description="Sync and trim missing videos into 15m segments")
    parser.add_argument("--dryrun", action="store_true", help="List missing only; no copying/trimming")
    args = parser.parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), "../login.env"))
    host = os.getenv(SSH_HOST_VAR)
    user = os.getenv(SSH_USER_VAR)
    if not (host and user):
        print("ERROR: SSH_HOST and SSH_USER required in login.env")
        sys.exit(1)

    conn = Connection(host=host, user=user)
    mc = MySQLConnector()

    missing = find_missing(conn, mc)
    if not missing:
        print("✅ all videos have trajectories - nothing to trim.")
        return
    print(f"🔍 {len(missing)} missing clips to trim:")
    for fn in missing:
        print(f" - {fn}")

    if args.dryrun:
        print("\nDry run: no trimming performed.")
        return
    sudo_passwd = getpass.getpass("Enter sudo password for remote user 'jpf': ")
    print("\nStarting trim & copy…")
    copy_and_trim(conn, missing, sudo_passwd)

if __name__ == "__main__":
    main()
