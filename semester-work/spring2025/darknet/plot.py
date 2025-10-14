#!/usr/bin/env python3
# Grouped boxplots of last mAP@0.50 by validation ratio and YOLO template (from run CSVs)
import os
import re
import csv
import glob
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# e.g., benchmark__...__yolov7-tiny__val15__20251011_014001
FOLDER_RX = re.compile(r"__(?P<template>yolov[0-9a-zA-Z\-_.]+)__val(?P<val>\d{2})__", re.I)

# columns to try for the "last" mAP
MAP50_LAST_KEYS = [
    "mAP@0.50 (last %)",   # current writer
    "mAP@0.50 (last)",     # fallback
    "mAP@0.50",            # older fallback
]

def read_last_map50_pct(csv_path: str, debug: bool=False) -> Optional[float]:
    """Return last mAP@0.50 as PERCENT (float), or None if missing/unreadable."""
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                if debug:
                    print(f"[debug] CSV empty: {csv_path}")
                return None
            row = rows[-1]  # usually single-row
            for key in MAP50_LAST_KEYS:
                if key in row and row[key] not in ("", "nan", None):
                    try:
                        v = float(row[key])
                    except Exception:
                        if debug:
                            print(f"[debug] Could not parse {key}='{row[key]}' in {csv_path}")
                        continue
                    v = v * 100.0 if v <= 1.0 else v  # ensure percent
                    if debug:
                        print(f"[debug] Using {key}={v:.2f}% from {os.path.basename(csv_path)}")
                    return v
            if debug:
                print(f"[debug] No mAP key found in {os.path.basename(csv_path)}; headers={list(row.keys())}")
    except Exception as e:
        if debug:
            print(f"[debug] Failed reading CSV {csv_path}: {e}")
        return None
    return None

def collect_map(outputs_dir: str,
                include_templates: Optional[List[str]] = None,
                debug: bool=False
                ) -> Tuple[Dict[str, Dict[str, List[float]]], List[str], List[str]]:
    """
    Returns:
      data[template][val_tag] -> list of mAP%
      templates (sorted, filtered)
      val_tags (sorted ascending numerically: ["10","15","20"])
    """
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    templates_seen = set()
    vals_seen = set()
    if debug:
        print(f"[debug] Scanning: {outputs_dir}")
    for entry in os.scandir(outputs_dir):
        if not entry.is_dir():
            continue
        m = FOLDER_RX.search(entry.name)
        if not m:
            if debug:
                print(f"[debug] Skip (no __valNN__): {entry.name}")
            continue
        template = m.group("template")
        val_tag = m.group("val")  # "10", "15", "20"
        if include_templates and template not in include_templates:
            if debug:
                print(f"[debug] Skip (template filtered): {entry.name}")
            continue
        csvs = glob.glob(os.path.join(entry.path, "*.csv"))
        if not csvs:
            if debug:
                print(f"[debug] No CSVs in run dir: {entry.name}")
            continue
        any_added = False
        for csv_path in csvs:
            v = read_last_map50_pct(csv_path, debug=debug)
            if v is not None:
                data[template][val_tag].append(v)
                any_added = True
        if any_added:
            templates_seen.add(template)
            vals_seen.add(val_tag)
            if debug:
                print(f"[debug] + {entry.name} -> template={template} val={val_tag} count={len(data[template][val_tag])}")
        elif debug:
            print(f"[debug] No usable mAP in {entry.name} (CSV keys missing?)")
    templates_sorted = sorted(templates_seen, key=lambda s: s.lower())
    vals_sorted = sorted(vals_seen, key=lambda x: int(x))
    # ensure every (template,val) key exists
    for t in templates_sorted:
        for vt in vals_sorted:
            data[t].setdefault(vt, [])
    return data, templates_sorted, vals_sorted

def plot_grouped_boxplots(data, templates, val_tags, out_path, show_points=True):
    if not templates or not val_tags:
        print("[warn] nothing to plot.")
        return

    G, K = len(val_tags), len(templates)
    box_gap = 0.18
    box_width = 0.14
    total_width = (K - 1) * box_gap
    offsets = [(-total_width/2.0) + i*box_gap for i in range(K)]

    fig, ax = plt.subplots(figsize=(max(6, 1.6*G), 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])
    legend_handles, legend_labels = [], []

    # Collect geometry for smart placement
    box_info = []  # dict keys: pos_x, left_edge, right_edge, lower_y, upper_y, max_y, color, n, val_tag

    for j, template in enumerate(templates):
        c = colors[j % len(colors)]
        series, positions, vt_used = [], [], []
        for i, vt in enumerate(val_tags):
            arr = data.get(template, {}).get(vt, [])
            if arr:
                series.append(arr)
                positions.append(i + 1 + offsets[j])
                vt_used.append(vt)
        if not series:
            continue

        ax.boxplot(
            series,
            positions=positions,
            widths=box_width,
            showmeans=True,
            patch_artist=True,
            medianprops=dict(color=c, linewidth=2),
            boxprops=dict(edgecolor=c, facecolor='none', linewidth=1.8),
            whiskerprops=dict(color=c, linewidth=1.5),
            capprops=dict(color=c, linewidth=1.5),
            meanprops=dict(marker='^', markerfacecolor=c, markeredgecolor=c, markersize=6),
        )

        # Capture geometry and scatter points
        for pos, vt in zip(positions, vt_used):
            arr = data[template][vt]
            if show_points and arr:
                xs = [pos] * len(arr)
                ax.scatter(xs, arr, s=24, alpha=0.8, edgecolors='none', color=c, zorder=3)

            if not arr:
                continue
            arr_np = np.array(arr)
            q1, q3 = np.percentile(arr_np, [25, 75])
            iqr = q3 - q1
            upper_whisker = min(float(np.max(arr_np)), float(q3 + 1.5 * iqr))
            lower_whisker = max(float(np.min(arr_np)), float(q1 - 1.5 * iqr))

            box_info.append({
                'pos_x': pos,
                'left_edge': pos - box_width/2,
                'right_edge': pos + box_width/2,
                'lower_y': lower_whisker,
                'upper_y': upper_whisker,
                'max_y': float(np.max(arr_np)),
                'color': c,
                'n': len(arr),
                'val_tag': vt,
            })

        # legend proxy
        h, = ax.plot([], [], color=c, label=template, linewidth=2)
        legend_handles.append(h); legend_labels.append(template)

    # Add a small headroom above 100.0
    y_bottom, _y_top = ax.get_ylim()
    y_headroom = 1.5
    # If some data went above 100 (shouldn't, but just in case), respect that
    max_data = max((b['max_y'] for b in box_info), default=0.0)
    ax.set_ylim(bottom=y_bottom, top=max(100.0, max_data) + y_headroom)

    # Build neighbor spacing per validation tag to prefer open side
    vt_groups: Dict[str, List[float]] = defaultdict(list)
    for b in box_info:
        vt_groups[b['val_tag']].append(b['pos_x'])
    for vt in vt_groups:
        vt_groups[vt].sort()

    def side_preference(b):
        # Prefer side with more horizontal space to nearest neighbor within same val group
        positions = vt_groups[b['val_tag']]
        idx = positions.index(b['pos_x'])
        left_gap = positions[idx] - positions[idx-1] if idx > 0 else float('inf')
        right_gap = positions[idx+1] - positions[idx] if idx < len(positions)-1 else float('inf')
        return 'left' if left_gap >= right_gap else 'right'

    # Smart placement of n= labels (only left or right)
    xlim_initial = ax.get_xlim()
    x_margin = 0.04   # minimal margin from edges
    x_pad_small = 0.08
    x_pad_large = 0.16
    y_clear = 0.8     # vertical clearance to count as not intersecting
    x_clear = 0.01    # horizontal clearance expansion on box span
    extend_right_needed = 0.0
    placed_labels = []  # track chosen positions to avoid overlaps between labels (optional)

    def collides_with_box(cand_x, cand_y, ignore_b=None):
        for ob in box_info:
            if ob is ignore_b:
                continue
            # If text x lies horizontally over the box span extended by x_clear
            if (ob['left_edge'] - x_clear) <= cand_x <= (ob['right_edge'] + x_clear):
                # And vertically overlaps box-whisker range
                if (ob['lower_y'] - y_clear) <= cand_y <= (ob['upper_y'] + y_clear):
                    return True
        return False

    def collides_with_label(cand_x, cand_y):
        # simple proximity check to avoid labels overlapping each other
        for (lx, ly) in placed_labels:
            if abs(lx - cand_x) < 0.06 and abs(ly - cand_y) < 1.0:
                return True
        return False

    for b in box_info:
        pos_x = b['pos_x']
        left_edge = b['left_edge']
        right_edge = b['right_edge']
        label_y = b['max_y']
        c = b['color']
        n = b['n']

        preferred = side_preference(b)
        primary_left = (preferred == 'left')

        # candidate list: prefer open side small pad, then larger pad, then other side
        candidates = []
        if primary_left:
            candidates += [
                (left_edge - x_pad_small, label_y, 'right', 'center'),
                (left_edge - x_pad_large, label_y, 'right', 'center'),
                (right_edge + x_pad_small, label_y, 'left', 'center'),
                (right_edge + x_pad_large, label_y, 'left', 'center'),
            ]
        else:
            candidates += [
                (right_edge + x_pad_small, label_y, 'left', 'center'),
                (right_edge + x_pad_large, label_y, 'left', 'center'),
                (left_edge - x_pad_small, label_y, 'right', 'center'),
                (left_edge - x_pad_large, label_y, 'right', 'center'),
            ]

        chosen = None
        xlim_now = ax.get_xlim()  # may extend later; use current for edge checks
        for cand_x, cand_y, ha, va in candidates:
            # edge bounds: allow slightly outside right edge; we'll extend if needed
            hits_left_edge = cand_x < (xlim_now[0] + x_margin)
            # For right edge, allow slight overflow; record extension
            overflow_right = cand_x > (xlim_now[1] - x_margin)
            if hits_left_edge:
                continue
            if collides_with_box(cand_x, cand_y, ignore_b=b):
                continue
            if collides_with_label(cand_x, cand_y):
                continue
            chosen = (cand_x, cand_y, ha, va)
            if overflow_right:
                extend_right_needed = max(extend_right_needed, cand_x + x_margin - xlim_now[1])
            break

        # if no side candidate works, nudge vertically a bit and retry same order
        if chosen is None:
            for dy in (0.8, 1.2):
                for cand_x, base_y, ha, va in candidates:
                    cand_y = min(base_y + dy, ax.get_ylim()[1] - 0.8)  # keep within top headroom
                    hits_left_edge = cand_x < (xlim_now[0] + x_margin)
                    overflow_right = cand_x > (xlim_now[1] - x_margin)
                    if hits_left_edge:
                        continue
                    if collides_with_box(cand_x, cand_y, ignore_b=b):
                        continue
                    if collides_with_label(cand_x, cand_y):
                        continue
                    chosen = (cand_x, cand_y, ha, va)
                    if overflow_right:
                        extend_right_needed = max(extend_right_needed, cand_x + x_margin - xlim_now[1])
                    break
                if chosen is not None:
                    break

        # Final fallback: put on preferred side with large pad, clamp y slightly below top
        if chosen is None:
            if primary_left:
                cand_x = left_edge - x_pad_large
                ha = 'right'
            else:
                cand_x = right_edge + x_pad_large
                ha = 'left'
            cand_y = min(label_y, ax.get_ylim()[1] - 0.8)
            overflow_right = cand_x > (xlim_now[1] - x_margin)
            if overflow_right:
                extend_right_needed = max(extend_right_needed, cand_x + x_margin - xlim_now[1])
            chosen = (cand_x, cand_y, ha, 'center')

        ax.text(chosen[0], chosen[1], f"n={n}", ha=chosen[2], va=chosen[3], fontsize=8, color=c, zorder=10)
        placed_labels.append((chosen[0], chosen[1]))

    # If any right-side labels needed a bit more space, extend xlim slightly
    if extend_right_needed > 0:
        xl0, xl1 = xlim_initial
        ax.set_xlim(xl0, xl1 + min(extend_right_needed, 0.25))  # cap tiny extension

    ax.set_xticks(range(1, G+1))
    ax.set_xticklabels([f"{int(v)}%" for v in val_tags])
    ax.set_xlabel("Validation fraction")
    ax.set_ylabel("mAP@0.50 (%, last)")
    ax.set_title("mAP@0.50 (last) by validation fraction and YOLO template")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.legend(legend_handles, legend_labels, title="Template", loc="best")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[ok] saved plot -> {out_path}")
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser(
        description="Grouped boxplots of last mAP@0.50 by validation ratio and YOLO template (reads run CSVs)."
    )
    ap.add_argument("--outputs-dir", default="LegoGearsFiles/outputs",
                    help="Directory containing benchmark__* run folders.")
    ap.add_argument("--templates", nargs="*", default=None,
                    help="Optional list to include (e.g., yolov4-tiny yolov7-tiny). Defaults to all found.")
    ap.add_argument("--out", default="map_boxplot_by_template_and_val.png",
                    help="Output PNG path. Set empty to show interactively.")
    ap.add_argument("--debug", action="store_true", help="Print why runs/CSVs are included or skipped.")
    args = ap.parse_args()

    data, templates, vals = collect_map(args.outputs_dir, include_templates=args.templates, debug=args.debug)
    if not templates or not vals:
        print(f"[warn] no matching runs found under: {args.outputs_dir}")
        return

    # quick summary
    for vt in sorted(vals, key=lambda x: int(x)):
        print(f"\nval{vt}%")
        for t in templates:
            arr = data[t][vt]
            if arr:
                arr_sorted = sorted(arr)
                n = len(arr_sorted)
                median = arr_sorted[n//2] if n % 2 == 1 else 0.5*(arr_sorted[n//2-1] + arr_sorted[n//2])
                print(f"  {t:12s}: n={n:2d} min={min(arr):5.2f} med={median:5.2f} max={max(arr):5.2f}")
            else:
                print(f"  {t:12s}: n=0")

    plot_grouped_boxplots(data, templates, vals, args.out or None)

if __name__ == "__main__":
    main()