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
    total_width = (K - 1) * box_gap
    offsets = [(-total_width/2.0) + i*box_gap for i in range(K)]

    fig, ax = plt.subplots(figsize=(max(6, 1.6*G), 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])

    legend_handles, legend_labels = [], []

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

        bp = ax.boxplot(
            series,
            positions=positions,
            widths=0.14,
            showmeans=True,
            patch_artist=True,
            medianprops=dict(color=c, linewidth=2),
            boxprops=dict(edgecolor=c, facecolor='none', linewidth=1.8),
            whiskerprops=dict(color=c, linewidth=1.5),
            capprops=dict(color=c, linewidth=1.5),
            meanprops=dict(marker='^', markerfacecolor=c, markeredgecolor=c, markersize=6),
        )

        # scatter individual runs + n labels (helps when n=1)
        if show_points:
            for pos, vt in zip(positions, vt_used):
                arr = data[template][vt]
                xs = [pos] * len(arr)
                ax.scatter(xs, arr, s=24, alpha=0.8, edgecolors='none', color=c, zorder=3)
                ax.text(pos, max(arr) + 0.6, f"n={len(arr)}", ha="center", va="bottom", fontsize=8, color=c)

        # legend proxy
        h, = ax.plot([], [], color=c, label=template, linewidth=2)
        legend_handles.append(h); legend_labels.append(template)

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
