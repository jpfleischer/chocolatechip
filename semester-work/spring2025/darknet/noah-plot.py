#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from plot_common import (
    infer_dataset_name_from_csv,
    infer_input_resolution_from_csv,
    get_ordered_yolos,
)


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
                    # Some CSVs can have odd formatting; fallback parser
                    df = pd.read_csv(csv_path, engine="python")
                # Expect single-row benchmark CSVs; but handle multi-row just in case
                for _, row in df.iterrows():
                    # Prefer COCO AP50-95 (%), fallback to mAP50-95 (%)
                    ap_col_candidates = [
                        "COCO AP50-95 (%)",
                        "mAP50-95 (%)",
                        "mAP50-95 (%) ",  # sometimes trailing space happens
                    ]
                    ap_val = None
                    for col in ap_col_candidates:
                        if col in df.columns:
                            ap_val = pd.to_numeric(row[col], errors="coerce")
                            break
                    if ap_val is None or pd.isna(ap_val):
                        continue

                    # YOLO version from the CSV ("YOLO Template"), fallback to directory name
                    if "YOLO Template" in df.columns:
                        yolo_raw = str(row["YOLO Template"])
                        # strip .pt so legend shows "yolo11n" not "yolo11n.pt"
                        yolo = yolo_raw.replace(".pt", "")
                    else:
                        yolo = os.path.basename(os.path.dirname(root))

                    # Validation fraction from CSV, fallback to guessing from path
                    val_frac = None
                    if "Val Fraction" in df.columns:
                        val_frac = pd.to_numeric(row["Val Fraction"], errors="coerce")
                    if val_frac is None or pd.isna(val_frac):
                        # Try to infer from directory segments like "val10", "val15", etc.
                        parts = root.split(os.sep)
                        val_tokens = [p for p in parts if p.startswith("val")]
                        if val_tokens:
                            try:
                                val_frac = (
                                    float(val_tokens[0].replace("val", "")) / 100.0
                                )
                            except Exception:
                                val_frac = None

                    if val_frac is None or pd.isna(val_frac):
                        # If still missing, skip this file
                        continue

                    records.append(
                        {
                            "yolo": str(yolo),
                            "val_frac": float(val_frac),
                            "ap5095": float(ap_val),
                            "source_csv": csv_path,
                        }
                    )
    return pd.DataFrame.from_records(records)


def balance_replicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Balance the dataset so that each (yolo, val_frac) group has the same count n.
    Extras are dropped (kept deterministically via sort/head).

    Returns:
        balanced_df, n_per_group
    """
    if df.empty:
        return df, 0

    counts = df.groupby(["yolo", "val_frac"]).size()

    # Only consider groups that actually exist (count > 0)
    positive_counts = counts[counts > 0]
    if positive_counts.empty:
        return df, 0

    n = int(positive_counts.min())  # target per-group count

    # If already balanced, just return as-is
    if int(positive_counts.max()) == n:
        return df, n

    balanced_parts = []
    for (y, vf), group in df.groupby(["yolo", "val_frac"]):
        if len(group) > n:
            # Deterministic trim: sort then take first n
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
    # Copy to avoid mutating caller data
    df = df.copy()

    # Numeric and string versions of validation percentage
    df["val_pct_num"] = (df["val_frac"] * 100).round().astype(int)
    df["val_pct"] = df["val_pct_num"].astype(str) + "%"

    # Ordering of x-axis by numeric val percentage, using the label strings
    order = sorted(df["val_pct"].unique(), key=lambda s: int(s.rstrip("%")))

    # Consistent hue order by YOLO template name, with fixed preferred order
    hue_order = get_ordered_yolos(df["yolo"].unique())

    plt.figure(figsize=(10, 6))

    # --- Grouped boxplots using pure matplotlib ---
    # Build data per group
    data_by_group = {}
    for vp in order:
        for y in hue_order:
            mask = (df["val_pct"] == vp) & (df["yolo"] == y)
            data_by_group[(vp, y)] = df.loc[mask, "ap5095"].tolist()

    # Compute positions, properly centered under ticks
    x_positions = []
    plot_data = []
    tick_positions = []
    tick_labels = order

    group_width = 0.8              # total width allotted per val_pct group
    n_hue = max(len(hue_order), 1) # number of YOLO variants
    dodge = group_width / n_hue    # horizontal separation between boxes

    group_center = 1.0             # x position of first group's center
    for vp in order:
        center = group_center
        tick_positions.append(center)

        # Leftmost box center so that the group is centered at `center`
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

    # Color boxes by hue
    colors = plt.cm.tab10.colors
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(hue_order)])

    # Add mean markers
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

    # Simple legend
    for i, y in enumerate(hue_order):
        plt.scatter([], [], marker="s", color=colors[i % len(hue_order)], label=y)
    plt.legend(title="YOLO version")

    if n_per_group is not None and n_per_group > 0:
        n_text = f" (n={n_per_group} per YOLO×val_frac group)"
    else:
        n_text = ""

    plt.title(
        f"{dataset_name} {input_res}: COCO AP50-95 (%)\n"
        f"by YOLO version and validation ratio{n_text}"
    )

    plt.xlabel("Validation ratio")
    plt.ylabel("COCO AP50-95 (%)")

    # No ylim – let matplotlib autoscale

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved box-and-whisker plot to: {out_path}")


def main():
    default_bases = [
        "/home/jpf/deleteme/chocolatechip/semester-work/spring2025/darknet/artifacts/outputs/LegoGearsDarknet"
        # you’ll pass the ultralytics base dir on the CLI
    ]

    parser = argparse.ArgumentParser(
        description=(
            "Box-and-whisker plot of COCO AP50-95 grouped by YOLO version "
            "and validation ratio."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        nargs="+",  # allow one or more dirs
        default=default_bases,
        help="One or more base output directories to scan for benchmark CSVs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="coco_ap5095_boxplot_by_yolo_val.pdf",
        help="Path to save the plot image.",
    )
    args = parser.parse_args()

    # Validate dirs
    for d in args.base_dir:
        if not os.path.isdir(d):
            raise SystemExit(f"Base directory not found: {d}")

    df = collect_records(args.base_dir)
    if df.empty:
        raise SystemExit("No benchmark CSVs found with the required columns.")

    # Infer dataset name and resolution from the first CSV we saw
    first_csv = df["source_csv"].iloc[0]
    dataset_name = infer_dataset_name_from_csv(first_csv)
    input_res = infer_input_resolution_from_csv(first_csv)

    # Balance replicates across (yolo, val_frac)
    df_balanced, n_per_group = balance_replicates(df)
    if n_per_group > 0:
        print(f"Balanced to n={n_per_group} per (YOLO, val_frac) group (extras dropped).")
    else:
        print("Could not balance replicates; using full dataset.")
        df_balanced = df  # fallback

    # Basic sanity check printout (on the balanced data)
    summary = (
        df_balanced.groupby(["yolo", "val_frac"])
        .agg(n=("ap5095", "size"), mean_ap=("ap5095", "mean"))
        .reset_index()
    )
    print("Found results (after balancing):")
    for _, r in summary.iterrows():
        print(
            f"  YOLO={r['yolo']}, val={r['val_frac']:.2f}, "
            f"n={int(r['n'])}, mean AP50-95={r['mean_ap']:.3f}"
        )

    make_plot(df_balanced, args.out, dataset_name, input_res, n_per_group)


if __name__ == "__main__":
    main()
