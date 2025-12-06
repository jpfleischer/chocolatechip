#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from plot_common import (
    infer_dataset_name_from_csv,
    infer_input_resolution_from_csv,
    get_ordered_yolos,
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

def collect_records(base_dirs: list[str]) -> pd.DataFrame:
    records = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if not f.endswith(".csv"):
                    continue
                if "benchmark__" not in f:
                    continue
                csv_path = os.path.join(root, f)
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

                    records.append(
                        {
                            "dataset": dataset,
                            "yolo": str(yolo),
                            "val_frac": float(val_frac),
                            "ap5095": float(ap_val),
                            "source_csv": csv_path,
                            "profile": profile or "",
                        }
                    )
    return pd.DataFrame.from_records(records)

def balance_replicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
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
    out_path: str,
    dataset_name: str,
    input_res: str,
    n_per_group: int | None,
):
    df = df.copy()
    df["val_pct_num"] = (df["val_frac"] * 100).round().astype(int)
    df["val_pct"] = df["val_pct_num"].astype(str) + "%"
    order = sorted(df["val_pct"].unique(), key=lambda s: int(s.rstrip("%")))
    hue_order = get_ordered_yolos(df["yolo"].unique())

    plt.figure(figsize=(10, 6))
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
                plt.scatter(x, float(m.iloc[0]), marker="D", color="black", s=40, zorder=3)
        group_center += 1.0

    plt.xticks(tick_positions, tick_labels)

    # Legend matching hue colors
    for i, y in enumerate(hue_order):
        plt.scatter([], [], marker="s", color=colors[i % len(colors)], label=y)
    plt.legend(title="YOLO version")

    if n_per_group is not None and n_per_group > 0:
        n_text = f" (n={n_per_group} per YOLO×val_frac group)"
    else:
        n_text = ""
    title_ds = dataset_name if dataset_name and dataset_name != "unknown" else "Dataset"
    title_res = input_res if input_res else ""
    title_prefix = f"{title_ds} {title_res}".strip()
    plt.title(f"{title_prefix}: COCO AP50-95 (%)\nby YOLO version and validation ratio{n_text}")
    plt.xlabel("Validation ratio")
    plt.ylabel("COCO AP50-95 (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved box-and-whisker plot to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Box-and-whisker plot of COCO AP50-95 grouped by YOLO version "
            "and validation ratio."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more base output directories to scan for benchmark CSVs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name filter (e.g., 'Leather', 'LegoGears'). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the plot image. If omitted, uses 'coco_ap5095_boxplot_<dataset>.pdf'.",
    )
    args = parser.parse_args()

    for d in args.base_dir:
        if not os.path.isdir(d):
            raise SystemExit(f"Base directory not found: {d}")

    df = collect_records(args.base_dir)
    if df.empty:
        raise SystemExit("No benchmark CSVs found with required AP columns.")

    # Dataset selection
    dataset = args.dataset
    if dataset:
        df = df[df["dataset"].str.lower() == dataset.lower()]
        if df.empty:
            raise SystemExit(f"No records found for dataset: {dataset}")
    else:
        # Auto-pick the most common dataset in the scan
        counts = df["dataset"].value_counts()
        dataset = counts.idxmax()
        df = df[df["dataset"] == dataset]
        print(f"Auto-selected dataset: {dataset}")

    # Infer dataset name and input resolution from one CSV in this dataset
    first_csv = df["source_csv"].iloc[0]
    dataset_name = infer_dataset_name_from_csv(first_csv) or dataset
    input_res = infer_input_resolution_from_csv(first_csv) or ""

    # Balance replicates
    df_balanced, n_per_group = balance_replicates(df)
    if n_per_group > 0:
        print(f"Balanced to n={n_per_group} per (YOLO, val_frac) group (extras dropped).")
    else:
        print("Could not balance replicates; using full dataset.")
        df_balanced = df

    summary = (
        df_balanced.groupby(["yolo", "val_frac"])
        .agg(n=("ap5095", "size"), mean_ap=("ap5095", "mean"))
        .reset_index()
    )
    print(f"Found results (after balancing) for dataset {dataset}:")
    for _, r in summary.iterrows():
        print(
            f"  YOLO={r['yolo']}, val={r['val_frac']:.2f}, "
            f"n={int(r['n'])}, mean AP50-95={r['mean_ap']:.3f}"
        )

    out_default = f"coco_ap5095_boxplot_{dataset}.pdf"
    out_path = args.out or out_default
    make_plot(df_balanced, out_path, dataset_name, input_res, n_per_group)

if __name__ == "__main__":
    main()