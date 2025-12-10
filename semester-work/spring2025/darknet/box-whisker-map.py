#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_common import (
    infer_dataset_name_from_csv,
    infer_input_resolution_from_csv,
    get_ordered_yolos,
    git_repo_root,
    iter_benchmark_csvs,
)

KNOWN_DATASETS = ["LegoGears", "Leather", "FisheyeTraffic", "Cubes"]


def infer_dataset(profile: str | None, path_hint: str | None) -> str:
    """
    Infer dataset name from a 'Profile' string if present, else from path hints.
    Falls back to 'unknown' if nothing matches.
    """
    text = " ".join([str(profile or ""), str(path_hint or "")])
    for ds in KNOWN_DATASETS:
        if ds.lower() in text.lower():
            return ds
    if profile:
        m = re.match(r"^([A-Za-z]+)", profile)
        if m:
            return m.group(1)
    # Try last few path segments for a dataset-like token
    if path_hint:
        parts = [p for p in re.split(r"[\/\\_\-]", path_hint) if p]
        for p in parts[::-1]:
            for ds in KNOWN_DATASETS:
                if ds.lower() == p.lower():
                    return ds
    return "unknown"


def collect_records(base_dirs: List[str]) -> pd.DataFrame:
    """
    Use plot_common.iter_benchmark_csvs to find benchmark__*.csv files
    anywhere under the given base dirs and collect COCO AP50–95 results.
    """
    records = []

    for csv_path in iter_benchmark_csvs(base_dirs):
        root = os.path.dirname(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.read_csv(csv_path, engine="python")

        # Prefer COCO AP50-95 (%), fallback to mAP50-95 (%)
        ap_col_candidates = [
            "COCO AP50-95 (%)",
            "mAP50-95 (%)",
            "mAP50-95 (%) ",
        ]
        present_cols = [c for c in ap_col_candidates if c in df.columns]
        if not present_cols:
            continue

        for _, row in df.iterrows():
            ap_val = pd.to_numeric(row[present_cols[0]], errors="coerce")
            if ap_val is None or pd.isna(ap_val):
                continue

            # YOLO version from CSV ("YOLO Template"), fallback to directory name
            if "YOLO Template" in df.columns:
                yolo_raw = str(row["YOLO Template"])
                yolo = yolo_raw.replace(".pt", "")
            else:
                yolo = os.path.basename(os.path.dirname(root))

            # Validation fraction from CSV, fallback to guessing from path
            val_frac = None
            if "Val Fraction" in df.columns:
                val_frac = pd.to_numeric(row["Val Fraction"], errors="coerce")
            if val_frac is None or pd.isna(val_frac):
                parts = root.split(os.sep)
                val_tokens = [p for p in parts if p.lower().startswith("val")]
                if val_tokens:
                    try:
                        val_frac = float(re.sub(r"[^\d.]", "", val_tokens[0])) / 100.0
                    except Exception:
                        val_frac = None
            if val_frac is None or pd.isna(val_frac):
                continue

            # Dataset inference, from Profile if available, else path hints
            profile = str(row["Profile"]) if "Profile" in df.columns else None
            dataset = infer_dataset(profile, csv_path)

            # --- Color preset (Leather special) ---
            color_preset = "unknown"
            if "Color Preset" in df.columns:
                color_preset = str(row["Color Preset"]).strip()
            else:
                # fallback: parse from profile or path
                src = " ".join([str(profile or ""), csv_path])
                m = re.search(r"color_(off|on|preserve|auto)", src, re.IGNORECASE)
                if m:
                    color_preset = m.group(1).lower()

            records.append(
                {
                    "dataset": dataset,
                    "yolo": str(yolo),
                    "val_frac": float(val_frac),
                    "ap5095": float(ap_val),
                    "source_csv": csv_path,
                    "profile": profile or "",
                    "color_preset": color_preset,
                }
            )

    return pd.DataFrame.from_records(records)


def balance_replicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Balance the dataset so that each (yolo, val_frac) group has the same count n.
    Extras are dropped deterministically via sort/head.
    """
    if df.empty:
        return df, 0
    counts = df.groupby(["yolo", "val_frac"]).size()
    positive_counts = counts[counts > 0]
    if positive_counts.empty:
        return df, 0
    n = int(positive_counts.min())
    if int(positive_counts.max()) == n:
        return df, n
    balanced_parts = []
    for (y, vf), group in df.groupby(["yolo", "val_frac"]):
        if len(group) > n:
            grp_bal = group.sort_values("source_csv").head(n)
        else:
            grp_bal = group
        balanced_parts.append(grp_bal)
    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    return balanced_df, n


def make_plot(
    df: pd.DataFrame,
    out_path: Path,
    dataset_name: str,
    input_res: str,
    n_per_group: int | None,
):
    df = df.copy()
    df["val_pct_num"] = (df["val_frac"] * 100).round().astype(int)
    df["val_pct"] = df["val_pct_num"].astype(str) + "%"
    order = sorted(df["val_pct"].unique(), key=lambda s: int(s.rstrip("%")))
    hue_order = get_ordered_yolos(df["yolo"].unique())

    # --- dynamic figure width based on number of validation fractions ---
    num_val_fracs = max(len(order), 1)
    fig_width = max(5.0, 2.5 * num_val_fracs)
    fig_height = 6.0

    plt.figure(figsize=(fig_width, fig_height))

    # ===== SPECIAL LEATHER LOGIC =====
    if dataset_name == "Leather":
        # Collapse actual presets into two groups: preserve vs non-preserve
        def _preset_group(cp: str) -> str:
            cp = (cp or "").lower()
            return "preserve" if cp == "preserve" else "non-preserve"

        df["preset_group"] = df["color_preset"].apply(_preset_group)

        data_by_group = {}
        for vp in order:
            for y in hue_order:
                for pg in ("non-preserve", "preserve"):
                    mask = (
                        (df["val_pct"] == vp)
                        & (df["yolo"] == y)
                        & (df["preset_group"] == pg)
                    )
                    data_by_group[(vp, y, pg)] = df.loc[mask, "ap5095"].tolist()

        x_positions = []
        plot_data = []
        tick_positions = []
        tick_labels = order

        group_width = 0.8
        n_yolo = max(len(hue_order), 1)
        # For spacing: for each YOLO, 3 “slots”: [non-preserve, preserve, gap]
        slots_per_val = n_yolo * 3

        slot_width = group_width / slots_per_val

        for j, vp in enumerate(order):
            center = j + 1.0
            tick_positions.append(center)
            start = center - group_width / 2.0 + slot_width / 2.0

            for yi, y in enumerate(hue_order):
                base_slot = yi * 3
                for pi, pg in enumerate(("non-preserve", "preserve")):
                    slot_idx = base_slot + pi  # 0 or 1 within the triple
                    x = start + slot_idx * slot_width
                    x_positions.append(x)
                    plot_data.append(data_by_group[(vp, y, pg)])

        bp = plt.boxplot(
            plot_data,
            positions=x_positions,
            widths=slot_width * 0.8,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1),
        )

        colors = plt.cm.tab10.colors
        # Color by YOLO index; fill style distinguishes non-preserve vs preserve
        for i, patch in enumerate(bp["boxes"]):
            yolo_idx = (i // 2) % len(hue_order)  # every 2 boxes = one YOLO (2 presets)
            preset_idx = i % 2                    # 0 = non-preserve, 1 = preserve
            color = colors[yolo_idx % len(colors)]

            if preset_idx == 0:
                # non-preserve: solid colored box
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                patch.set_linewidth(1.0)
            else:
                # preserve: hollow box with colored edge
                patch.set_facecolor("none")
                patch.set_edgecolor(color)
                patch.set_linewidth(1.8)


        # Mean markers per (val_pct, yolo, preset_group)
        means = (
            df.groupby(["val_pct", "yolo", "preset_group"], as_index=False)["ap5095"]
            .mean()
        )

        group_width = 0.8
        slot_width = group_width / slots_per_val
        for j, vp in enumerate(order):
            center = j + 1.0
            start = center - group_width / 2.0 + slot_width / 2.0
            for yi, y in enumerate(hue_order):
                base_slot = yi * 3
                for pi, pg in enumerate(("non-preserve", "preserve")):
                    slot_idx = base_slot + pi
                    x = start + slot_idx * slot_width
                    m = means[
                        (means["val_pct"] == vp)
                        & (means["yolo"] == y)
                        & (means["preset_group"] == pg)
                    ]["ap5095"]
                    if len(m):
                        marker = "o" if pg == "non-preserve" else "D"
                        plt.scatter(
                            x,
                            float(m.iloc[0]),
                            marker=marker,
                            color="black",
                            s=40,
                            zorder=3,
                        )


        plt.xticks(tick_positions, tick_labels)

        colors = plt.cm.tab10.colors

        # Legend entries for YOLO versions (color)
        yolo_handles = [
            Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=y)
            for i, y in enumerate(hue_order)
        ]

        # Legend entries for color preset (fill style)
        preset_handles = [
            Patch(facecolor="0.7", edgecolor="black", label="non-preserve"),
            Patch(facecolor="none", edgecolor="black", label="preserve"),
        ]

        handles = yolo_handles + preset_handles

        plt.legend(
            handles=handles,
            title="YOLO / color preset",
            loc="lower right",   # inside the axes
            frameon=True,
        )



    # ===== DEFAULT (non-Leather) LOGIC =====
    else:
        data_by_group = {}
        for vp in order:
            for y in hue_order:
                mask = (df["val_pct"] == vp) & (df["yolo"] == y)
                data_by_group[(vp, y)] = df.loc[mask, "ap5095"].tolist()

        x_positions = []
        plot_data = []
        tick_positions = []
        tick_labels = order
        group_width = 0.8
        n_hue = max(len(hue_order), 1)
        dodge = group_width / n_hue
        group_center = 1.0
        for vp in order:
            center = group_center
            tick_positions.append(center)
            start = center - group_width / 2.0 + dodge / 2.0
            for i, y in enumerate(hue_order):
                x = start + i * dodge
                x_positions.append(x)
                plot_data.append(data_by_group[(vp, y)])
            group_center += 1.0

        bp = plt.boxplot(
            plot_data,
            positions=x_positions,
            widths=dodge * 0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1),
        )

        colors = plt.cm.tab10.colors
        # Color by hue index, consistent across groups
        for i, patch in enumerate(bp["boxes"]):
            hue_idx = i % n_hue
            patch.set_facecolor(colors[hue_idx % len(colors)])

        # Mean markers
        means = df.groupby(["val_pct", "yolo"], as_index=False)["ap5095"].mean()
        group_center = 1.0
        for vp in order:
            center = group_center
            start = center - group_width / 2.0 + dodge / 2.0
            for i, y in enumerate(hue_order):
                m = means[(means["val_pct"] == vp) & (means["yolo"] == y)]["ap5095"]
                if len(m):
                    x = start + i * dodge
                    plt.scatter(
                        x,
                        float(m.iloc[0]),
                        marker="D",
                        color="black",
                        s=40,
                        zorder=3,
                    )
            group_center += 1.0

        plt.xticks(tick_positions, tick_labels)

        # Legend matching hue colors
        for i, y in enumerate(hue_order):
            plt.scatter([], [], marker="s", color=colors[i % len(colors)], label=y)
        plt.legend(
            title="YOLO version",
            loc="lower right",   # try 'upper left', 'lower left', etc. if you prefer
            frameon=True,
        )


    plt.xlabel("Validation ratio")
    plt.ylabel("COCO AP50-95 (%)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved box-and-whisker plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Box-and-whisker plot of COCO AP50-95 grouped by YOLO version "
            "and validation ratio."
        )
    )
    parser.add_argument(
        "base",
        nargs="*",
        help=(
            "Base directory/directories to search. "
            "If omitted, defaults to <git_root>/artifacts/outputs."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Optional dataset name filter (e.g., 'Leather', 'LegoGears'). "
            "If omitted, process all datasets found."
        ),
    )
    args = parser.parse_args()

    # Base-dir logic like the first script
    if args.base:
        base_dirs = [str(Path(b).resolve()) for b in args.base]
    else:
        repo_root = git_repo_root()
        base_dirs = [str(repo_root)]

    df = collect_records(base_dirs)
    if df.empty:
        raise SystemExit("No benchmark CSVs found with required AP columns.")

    # Optional filter to a single dataset; otherwise do all of them
    if args.dataset:
        df = df[df["dataset"].str.lower() == args.dataset.lower()]
        if df.empty:
            raise SystemExit(f"No records found for dataset: {args.dataset}")
        target_datasets = [args.dataset]
    else:
        target_datasets = sorted(df["dataset"].unique())

    output_dir = Path("heatmaps")
    output_dir.mkdir(exist_ok=True)

    for ds in target_datasets:
        sub = df[df["dataset"].str.lower() == ds.lower()]
        if sub.empty:
            continue

        print(f"\n=== Processing dataset: {ds} ===")

        # Infer dataset name and input resolution from one CSV in this dataset
        first_csv = sub["source_csv"].iloc[0]
        dataset_name = infer_dataset_name_from_csv(first_csv) or ds
        input_res = infer_input_resolution_from_csv(first_csv) or ""

        # Balance replicates per dataset
        df_balanced, n_per_group = balance_replicates(sub)
        if n_per_group > 0:
            print(f"Balanced to n={n_per_group} per (YOLO, val_frac) group (extras dropped).")
        else:
            print("Could not balance replicates; using full dataset.")
            df_balanced = sub

        summary = (
            df_balanced.groupby(["yolo", "val_frac"])
            .agg(n=("ap5095", "size"), mean_ap=("ap5095", "mean"))
            .reset_index()
        )
        print(f"Found results (after balancing) for dataset {dataset_name}:")
        for _, r in summary.iterrows():
            print(
                f"  YOLO={r['yolo']}, val={r['val_frac']:.2f}, "
                f"n={int(r['n'])}, mean AP50-95={r['mean_ap']:.3f}"
            )

        pdf_name = f"{dataset_name.lower()}_boxplot_coco_ap5095.pdf"
        out_path = output_dir / pdf_name
        make_plot(df_balanced, out_path, dataset_name, input_res, n_per_group)

        # -------- LaTeX figure snippet --------
        dataset_label = dataset_name.lower().replace(" ", "-")

        if n_per_group is not None and n_per_group > 0:
            n_text = f" (n={n_per_group} runs per YOLO$\\times$validation-fraction group)"
        else:
            n_text = ""

        res_text = f" at input resolution {input_res}" if input_res else ""

        if dataset_name == "Leather":
            extra = " For each YOLO, the left box is without color preserve and the right box uses color preserve."
        else:
            extra = ""

        caption = (
            f"COCO AP50--95 (\\%) by YOLO version and validation ratio "
            f"for the {dataset_name} dataset{res_text}{n_text}.{extra}"
        )

        latex = f"""
\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=\\columnwidth]{{images/{pdf_name}}}
    \\caption{{{caption}}}
    \\label{{fig:coco-ap5095-boxplot-{dataset_label}}}
\\end{{figure}}
        """



        print(latex)
        print()  # blank line between figures


if __name__ == "__main__":
    main()
