# File: chocolatechip/unflag.py

import os
import sys
import yaml
from pprint import pprint

from yaspin import yaspin
from chocolatechip.MySQLConnector import MySQLConnector


def load_harmless_ids(yaml_path):
    """
    Read the YAML and collect every "ID" (the second underscore-delimited token)
    whose label == 'harmless'.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    harmless_ids = set()
    for filename, label in data.items():
        if str(label).strip().lower() == 'harmless' and filename.startswith('output_'):
            parts = os.path.basename(filename).split('_')
            if len(parts) >= 2:
                # parts[1] is the numeric ID
                harmless_ids.add(parts[1])
    return harmless_ids


def unflag_in_db(connection, uniqueID1, uniqueID2, cameraID):
    """
    UPDATE TTCTable SET include_flag=0 WHERE unique_ID1=%s AND unique_ID2=%s AND camera_id=%s
    """
    with connection.cursor() as cursor:
        sql = (
            "UPDATE TTCTable "
            "SET include_flag = 0 "
            "WHERE unique_ID1 = %s AND unique_ID2 = %s AND camera_id = %s"
        )
        cursor.execute(sql, (uniqueID1, uniqueID2, cameraID))
        connection.commit()
        return cursor.rowcount


def main(yaml_file=None):
    """
    Called by `chip unflag [flags.yaml]`. If no yaml_file is passed, defaults to "flags.yaml".
    This version does NOT scan the local folder for .mp4 files; it simply unflags
    any ID that appears in the YAML with label == 'harmless'.
    """

    if yaml_file is None:
        yaml_file = "flags.yaml"

    if not yaml_file.lower().endswith('.yaml'):
        print("Error: must pass a .yaml file as argument.")
        sys.exit(1)
    if not os.path.isfile(yaml_file):
        print(f"Error: \"{yaml_file}\" not found in current directory.")
        sys.exit(1)

    # 1) Parse YAML → set of harmless IDs
    harmless_ids = load_harmless_ids(yaml_file)
    # print("Harmless IDs (to be unflagged):")
    # pprint(sorted(harmless_ids))
    # print()

    if not harmless_ids:
        print("No 'harmless' entries found in the YAML; nothing to unflag.")
        sys.exit(0)

    # 2) Open a DB connection
    connector = MySQLConnector()
    conn = connector._connect()

    total_unflagged = 0
    count_total = len(harmless_ids)

    # 3) We still need uniqueID1, uniqueID2, cameraID for each conflict.
    #    If you know how to derive them from just the ID, adapt this section.
    #    Commonly, ID=parts[1], so we still need to scan the YAML’s keynames again
    #    (instead of scanning the directory). We’ll pull out uniqueID1, uniqueID2, camid 
    #    directly from each YAML key that had 'harmless'.

    #    Example key: 
    #       "output_008_26_2025-01-06_11-00_262501061100477_262501061100476.mp4"
    #    parts = [ "output", "008", "26", "2025-01-06", "11-00", 
    #              "262501061100477", "262501061100476.mp4" ]
    #    so:
    #       ID        = parts[1]          → "008"
    #       camera_id = parts[2]          → "26"
    #       uniqueID1 = parts[-2]         → "262501061100477"
    #       uniqueID2 = parts[-1].rstrip(".mp4") → "262501061100476"

    # Build a list of (video_filename, camid, uniqueID1, uniqueID2) for each harmless ID:
    to_unflag = []
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    for filename, label in data.items():
        if str(label).strip().lower() != 'harmless':
            continue
        if not filename.startswith('output_'):
            continue

        parts = os.path.basename(filename).split('_')
        if len(parts) < 7:
            # If it doesn’t match “output_<ID>_<cam>_..._<UID1>_<UID2>.mp4”, skip.
            continue

        vid_id = parts[1]
        if vid_id in harmless_ids:
            camid     = parts[2]
            uniqueID1 = parts[-2]
            uniqueID2 = parts[-1].replace('.mp4', '')
            to_unflag.append((filename, camid, uniqueID1, uniqueID2))

    # 4) Run the spinner + unflag loop
    with yaspin(text="Starting to unflag…", color="yellow") as spinner:
        for idx, (fname, camid, u1, u2) in enumerate(to_unflag, start=1):
            spinner.text = f"Unflagging {idx}/{count_total} → {fname}"
            rows = unflag_in_db(conn, u1, u2, camid)
            total_unflagged += rows

        spinner.ok("✅ Done unflagging all harmless IDs")

    conn.close()
    print()
    print(f"Finished. Total rows unflagged: {total_unflagged}")


if __name__ == "__main__":
    main()
