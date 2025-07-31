#!/usr/bin/env python3
"""
histogram_conflict.py

Fetches P2V “conflict” events (timestamp + numeric x/y + cluster codes) from TTCTable,
downloads the map image if needed, and plots a **single** 2D heatmap overlay with three
separate color gradients for Left-turn, Through, and Right-turn P2V conflicts—
all normalized by days observed and flipped in Y to match the map.

The script also plots V2V conflicts

Usage:
    python histogram_conflict.py
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chocolatechip.times_config import times_dict
from chocolatechip.MySQLConnector import MySQLConnector
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches


# ────────────────────────────────────────────────────────────────────────────────
# 1) TIME WINDOW UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def get_time_windows(iid: int, period: str = 'before') -> list[str]:
    try:
        return times_dict[iid][period]
    except KeyError:
        raise ValueError(f"Invalid intersection ID {iid} or period '{period}'")

def compute_unique_dates_and_weeks(windows: list[str]) -> tuple[set[pd.Timestamp], float]:
    dates = {pd.to_datetime(windows[i]).date() for i in range(0, len(windows), 2)}
    nd = len(dates)
    return dates, (nd/7.0 if nd else 0.0)



def fetch_or_cache_conflicts(
    connector: MySQLConnector,
    intersection_id: int,
    p2v_flag: int,
    start_ts: str,
    end_ts: str,
    cache_dir: str = 'cache'
) -> pd.DataFrame:
    required = {'timestamp','cluster1','cluster2','conflict_x','conflict_y'}
    base = f"{intersection_id}_{p2v_flag}_conflict"
    clean = lambda s: s.replace(':','').replace('-','').replace(' ','_')
    cache_name = f"{base}_{clean(start_ts)}_{clean(end_ts)}.csv"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            print(f"  → Cache missing columns {missing}, removing {cache_path}")
            os.remove(cache_path)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df

    # fetch fresh
    df = connector.fetchConflictCoordinates(intersection_id, p2v_flag, start_ts, end_ts)
    df.to_csv(cache_path, index=False)
    print(f"  → Cached fresh data to {cache_path}")
    return df



def get_all_conflicts_for_period(iid, p2v, period):
    conn = MySQLConnector()
    slices = []
    windows = get_time_windows(iid, period)
    for i in range(0, len(windows), 2):
        df = fetch_or_cache_conflicts(conn, iid, p2v, windows[i], windows[i+1])
        if not df.empty:
            slices.append(df)
    if not slices:
        return pd.DataFrame(columns=['timestamp','cluster1','cluster2','conflict_x','conflict_y'])
    all_df = pd.concat(slices, ignore_index=True)
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], errors='coerce')
    return all_df


# ────────────────────────────────────────────────────────────────────────────────
# 3) P2V spatial heatmap with three gradients
# ────────────────────────────────────────────────────────────────────────────────

def plot_p2v_triple_heatmap(df, img_path, out_path,
                            img_extent=(0,1280,0,960), bin_size=30, global_max=None):
    # require columns
    for col in ['cluster1','cluster2','conflict_x','conflict_y','timestamp']:
        if col not in df.columns:
            print(f"Missing '{col}' → skipping")
            return

    # drop NA
    df = df.dropna(subset=['timestamp','conflict_x','conflict_y'])
    if df.empty:
        print("No data → skipping")
        return

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    ndays = df['date'].nunique()
    if ndays==0:
        print("Zero days → skipping")
        return

    # classification
    d1 = df['cluster1'].str.findall(r"[A-Z]+").str.join("")
    d2 = df['cluster2'].str.findall(r"[A-Z]+").str.join("")
    ped1 = df['cluster1'].str.contains('ped', case=False, na=False)
    # classification via pandas.Series.where → preserves .str
    active = d2.where(ped1, other=d1)
    left_mask  = active.str.endswith("L")
    thru_mask  = active.str.endswith("T")
    right_mask = active.str.endswith("R")

    # numeric coords
    x = pd.to_numeric(df['conflict_x'], errors='coerce')
    y = pd.to_numeric(df['conflict_y'], errors='coerce')
    valid = df.index[~(x.isna()|y.isna())]
    df = df.loc[valid]
    x = x.loc[valid]; y = y.loc[valid]

    # bin edges
    xmin,xmax,ymin,ymax = img_extent
    xbins = np.arange(xmin, xmax+bin_size, bin_size)
    ybins = np.arange(ymin, ymax+bin_size, bin_size)

    # flip Y
    y_flipped = ymax - y

    # compute histograms
    H_left, _, _ = np.histogram2d(x[left_mask.loc[valid]],
                                  y_flipped[left_mask.loc[valid]],
                                  bins=[xbins,ybins])
    H_thru, _, _ = np.histogram2d(x[thru_mask.loc[valid]],
                                  y_flipped[thru_mask.loc[valid]],
                                  bins=[xbins,ybins])
    H_right, _, _= np.histogram2d(x[right_mask.loc[valid]],
                                  y_flipped[right_mask.loc[valid]],
                                  bins=[xbins,ybins])

    # normalize
    H_left  /= ndays
    H_thru  /= ndays
    H_right /= ndays

    if global_max is None:
        global_max = max(H_left.max(), H_thru.max(), H_right.max(), 1e-3)

    # plot
    bg = plt.imread(img_path)
    plt.figure(figsize=(8,6))
    plt.imshow(bg, extent=img_extent, origin='upper', alpha=0.5)

    # overlay each
    plt.imshow(H_left.T,  extent=img_extent, origin='lower',
               cmap='Blues',   alpha=0.5, norm=LogNorm(vmin=1e-3, vmax=global_max))
    plt.imshow(H_thru.T,  extent=img_extent, origin='lower',
               cmap='Greens', alpha=0.5, norm=LogNorm(vmin=1e-3, vmax=global_max))
    plt.imshow(H_right.T, extent=img_extent, origin='lower',
               cmap='Oranges',  alpha=0.5, norm=LogNorm(vmin=1e-3, vmax=global_max))
    

    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.xlabel("X"); plt.ylabel("Y")
    # plt.title(f"P2V Spatial: L=Red, T=Green, R=Blue\n(Avg/day over {ndays} days)")
    plt.title(f"P2V Spatial: L=Blue, T=Green, R=Orange")

    # legend patches
    patch_L = mpatches.Patch(color='blue',   label='Left turns')
    patch_T = mpatches.Patch(color='green', label='Through')
    patch_R = mpatches.Patch(color='orange',  label='Right turns')
    plt.legend(handles=[patch_L,patch_T,patch_R], loc='upper right')

     # … after your three plt.imshow(...) overlays …

 

    # # grab figure & axes
    # fig = plt.gcf()
    # ax  = plt.gca()

    # # left-turn colorbar
    # sm1 = ScalarMappable(cmap="Reds",  norm=LogNorm(vmin=1e-3, vmax=global_max))
    # sm1.set_array([])
    # cbar1 = fig.colorbar(sm1, ax=ax, fraction=0.046, pad=0.01, label="Left turns (avg/day)")
    # cbar1.ax.yaxis.set_major_formatter(ScalarFormatter())
    # cbar1.ax.yaxis.get_offset_text().set_visible(False)  # hide the “×10^n” text


    # # through-move colorbar
    # sm2 = ScalarMappable(cmap="Greens",  norm=LogNorm(vmin=1e-3, vmax=global_max))
    # sm2.set_array([])
    # cbar2 = fig.colorbar(sm2, ax=ax, fraction=0.046, pad=0.07, label="Through (avg/day)")
    # cbar2.ax.yaxis.set_major_formatter(ScalarFormatter())
    # cbar2.ax.yaxis.get_offset_text().set_visible(False)  # hide the “×10^n” text

    # # right-turn colorbar
    # sm3 = ScalarMappable(cmap="Blues",  norm=LogNorm(vmin=1e-3, vmax=global_max))
    # sm3.set_array([])
    # cbar3 = fig.colorbar(sm3, ax=ax, fraction=0.046, pad=0.13, label="Right turns (avg/day)")
    # cbar3.ax.yaxis.set_major_formatter(ScalarFormatter())
    # cbar3.ax.yaxis.get_offset_text().set_visible(False)  # hide the “×10^n” text

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"→ Saved: {out_path}")


def plot_v2v_triple_heatmap(df, img_path, out_path,
                            img_extent=(0,1280,0,960), bin_size=30, global_max=None):
    # require same conflict_x, conflict_y, cluster1/2, timestamp columns
    for col in ['cluster1','cluster2','conflict_x','conflict_y','timestamp']:
        if col not in df.columns:
            print(f"Missing '{col}' → skipping V2V")
            return

    df = df.dropna(subset=['timestamp','conflict_x','conflict_y'])
    if df.empty:
        print("No V2V data → skipping")
        return

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    ndays = df['date'].nunique()
    if ndays == 0:
        print("Zero V2V days → skipping")
        return

    # build V2V masks
    d1 = df['cluster1'].str.findall(r"[A-Z]+").str.join("")
    d2 = df['cluster2'].str.findall(r"[A-Z]+").str.join("")
    end1 = d1.str[-1]
    end2 = d2.str[-1]

    lot_mask = ((end1=="L") & (end2=="T")) | ((end1=="T") & (end2=="L"))
    rmt_mask = ((end1=="R") & (end2=="T")) | ((end1=="T") & (end2=="R"))
    rol_mask = ((end1=="R") & (end2=="L")) | ((end1=="L") & (end2=="R"))

    # coords + flip
    x = pd.to_numeric(df['conflict_x'], errors='coerce')
    y = pd.to_numeric(df['conflict_y'], errors='coerce')
    valid = df.index[~(x.isna()|y.isna())]
    x = x.loc[valid]
    y = y.loc[valid]
    xmin,xmax,ymin,ymax = img_extent
    y_flipped = ymax - y

    # bins
    xbins = np.arange(xmin, xmax+bin_size, bin_size)
    ybins = np.arange(ymin, ymax+bin_size, bin_size)

    # histograms normalized
    H_lot = np.histogram2d(x[lot_mask.loc[valid]],
                           y_flipped[lot_mask.loc[valid]],
                           bins=[xbins,ybins])[0] / ndays
    H_rmt = np.histogram2d(x[rmt_mask.loc[valid]],
                           y_flipped[rmt_mask.loc[valid]],
                           bins=[xbins,ybins])[0] / ndays
    H_rol = np.histogram2d(x[rol_mask.loc[valid]],
                           y_flipped[rol_mask.loc[valid]],
                           bins=[xbins,ybins])[0] / ndays

    if global_max is None:
        global_max = max(H_lot.max(), H_rmt.max(), H_rol.max(), 1e-3)

    # start plot
    bg = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(bg, extent=img_extent, origin='upper', alpha=0.5)

    im1 = ax.imshow(H_lot.T, extent=img_extent, origin='lower',
                    cmap='Reds', alpha=0.5,
                    norm=LogNorm(vmin=1e-3, vmax=global_max))
    im2 = ax.imshow(H_rmt.T, extent=img_extent, origin='lower',
                    cmap='Purples', alpha=0.5,
                    norm=LogNorm(vmin=1e-3, vmax=global_max))
    im3 = ax.imshow(H_rol.T, extent=img_extent, origin='lower',
                    cmap='copper', alpha=0.5,
                    norm=LogNorm(vmin=1e-3, vmax=global_max))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("V2V Spatial: LOT=Red, RMT=Purple, ROL=Brown")

    # legend
    patch_LOT = mpatches.Patch(color='red', label='LOT')
    patch_RMT = mpatches.Patch(color='purple', label='RMT')
    patch_ROL = mpatches.Patch(color='brown',   label='ROL')
    ax.legend(handles=[patch_LOT, patch_RMT, patch_ROL], loc='upper right')

    # # three colorbars, no scientific notation
    # for im, label, pad in [
    #     (im1, "LOT (avg/day)", 0.01),
    #     (im2, "RMT (avg/day)", 0.07),
    #     (im3, "ROL (avg/day)", 0.13),
    # ]:
    #     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=pad, label=label)
    #     cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    #     cbar.ax.yaxis.get_offset_text().set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"→ Saved V2V: {out_path}")


if __name__ == "__main__":
    # list of intersections to process
    intersections = [3287, 3248, 3032, 3265, 3334, 3252]
    default_intersections = [3287, 3248]

    # camera lookup map
    cam_lookup = {
        3287: 24,
        3248: 27,
        3032: 21,
        3265: 30,
        3334: 33,
        3252: 36,
    }

    # Build the list of expected map filenames
    map_pics = [f"{cam}_Map.png" for cam in cam_lookup.values()]
    for picture in map_pics:
        if not os.path.isfile(picture):
            print(f"Downloading {picture} …")
            url = f"http://maltlab.cise.ufl.edu:30101/api/image/{picture}"
            resp = requests.get(url)
            # guard against HTML error pages
            if resp.status_code == 200 and 'png' in resp.headers.get('Content-Type',''):
                with open(picture, 'wb') as f:
                    f.write(resp.content)
            else:
                print(f"  → Failed to fetch a valid PNG for {picture} (status={resp.status_code})")
    # ─────────────────────────────────


    # helper: build P2V histograms exactly as before
    def _p2v_hists(df):
        x = pd.to_numeric(df['conflict_x'], errors='coerce')
        y = pd.to_numeric(df['conflict_y'], errors='coerce')
        valid = df.index[~(x.isna() | y.isna())]
        x = x.loc[valid]
        y = (960 - y.loc[valid])
        bins = [np.arange(0, 1280+30, 30), np.arange(0, 960+30, 30)]
        d1 = df['cluster1'].str.findall(r"[A-Z]+").str.join("")
        d2 = df['cluster2'].str.findall(r"[A-Z]+").str.join("")
        ped1 = df['cluster1'].str.contains('ped', case=False, na=False)
        active = d2.where(ped1, other=d1)
        lm = active.str.endswith("L")
        tm = active.str.endswith("T")
        rm = active.str.endswith("R")
        H_L = np.histogram2d(x[lm.loc[valid]], y[lm.loc[valid]], bins=bins)[0]
        H_T = np.histogram2d(x[tm.loc[valid]], y[tm.loc[valid]], bins=bins)[0]
        H_R = np.histogram2d(x[rm.loc[valid]], y[rm.loc[valid]], bins=bins)[0]
        return H_L, H_T, H_R

    # helper: build V2V histograms exactly as before
    def _v2v_hists(df):
        x = pd.to_numeric(df['conflict_x'], errors='coerce')
        y = pd.to_numeric(df['conflict_y'], errors='coerce')
        valid = df.index[~(x.isna() | y.isna())]
        x = x.loc[valid]
        y = (960 - y.loc[valid])
        bins = [np.arange(0, 1280+30, 30), np.arange(0, 960+30, 30)]
        d1 = df['cluster1'].str.findall(r"[A-Z]+").str.join("")
        d2 = df['cluster2'].str.findall(r"[A-Z]+").str.join("")
        e1, e2 = d1.str[-1], d2.str[-1]
        lot = ((e1 == "L") & (e2 == "T")) | ((e1 == "T") & (e2 == "L"))
        rmt = ((e1 == "R") & (e2 == "T")) | ((e1 == "T") & (e2 == "R"))
        rol = ((e1 == "R") & (e2 == "L")) | ((e1 == "L") & (e2 == "R"))
        H_LOT = np.histogram2d(x[lot.loc[valid]], y[lot.loc[valid]], bins=bins)[0]
        H_RMT = np.histogram2d(x[rmt.loc[valid]], y[rmt.loc[valid]], bins=bins)[0]
        H_ROL = np.histogram2d(x[rol.loc[valid]], y[rol.loc[valid]], bins=bins)[0]
        return H_LOT, H_RMT, H_ROL

    # main loop
    for iid in intersections:
        cam = cam_lookup.get(iid)
        if cam is None:
            continue

        # ← this is the one change: pick both periods only for the default list
        if intersections == default_intersections:
            periods = ['before', 'after']
        else:
            periods = ['before']

        # ensure map image is present
        pic = f"{cam}_Map.png"
        if not os.path.isfile(pic):
            print(f"Downloading {pic} …")
            r = requests.get(f"http://maltlab.cise.ufl.edu:30101/api/image/{pic}")
            with open(pic, 'wb') as f:
                f.write(r.content)

        # ─── P2V ───
        p2v_data = {period: get_all_conflicts_for_period(iid, 1, period)
                    for period in periods}
        all_p2v_H = []
        for df in p2v_data.values():
            if not df.empty:
                all_p2v_H.extend(_p2v_hists(df))
        shared_max_p2v = max([H.max() for H in all_p2v_H] + [1e-3])

        for period, df in p2v_data.items():
            if df.empty:
                print(f"No P2V data for {iid} {period}; skipping.")
                continue
            out = f"heatmaps/{iid}_{period}_p2v_triple.png"
            plot_p2v_triple_heatmap(
                df, pic, out,
                img_extent=(0,1280,0,960),
                bin_size=30,
                global_max=shared_max_p2v
            )

        # ─── V2V ───
        v2v_data = {period: get_all_conflicts_for_period(iid, 0, period)
                    for period in periods}
        all_v2v_H = []
        for df in v2v_data.values():
            if not df.empty:
                all_v2v_H.extend(_v2v_hists(df))
        shared_max_v2v = max([H.max() for H in all_v2v_H] + [1e-3])

        for period, df in v2v_data.items():
            if df.empty:
                print(f"No V2V data for {iid} {period}; skipping.")
                continue
            out = f"heatmaps/{iid}_{period}_v2v_triple.png"
            plot_v2v_triple_heatmap(
                df, pic, out,
                img_extent=(0,1280,0,960),
                bin_size=30,
                global_max=shared_max_v2v
            )
