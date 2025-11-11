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


def collect_confusion_records(base_dirs: list[str]) -> pd.DataFrame:
    """
    Walk one or more base_dirs, read benchmark__*.csv, and collect confusion totals
    plus yolo version and validation fraction.
    """
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

                # Skip if required columns are missing
                required = ["CM_TotalTP", "CM_TotalFP", "CM_TotalFN"]
                if not all(col in df.columns for col in required):
                    continue

                for _, row in df.iterrows():
                    # YOLO version from CSV ("YOLO Template"), fallback directory name
                    if "YOLO Template" in df.columns:
                        yolo_raw = str(row["YOLO Template"])
                        # strip .pt so legend shows "yolo11n" not "yolo11n.pt"
                        yolo = yolo_raw.replace(".pt", "")
                    else:
                        yolo = os.path.basename(os.path.dirname(root))

                    # Validation fraction
                    val_frac = None
                    if "Val Fraction" in df.columns:
                        val_frac = pd.to_numeric(row["Val Fraction"], errors="coerce")
                    if val_frac is None or pd.isna(val_frac):
                        # Infer from directory segment like "val10", "val15", etc.
                        parts = root.split(os.sep)
                        val_tokens = [p for p in parts if p.startswith("val")]
                        if val_tokens:
                            try:
                                val_frac = float(val_tokens[0].replace("val", "")) / 100.0
                            except Exception:
                                val_frac = None

                    if val_frac is None or pd.isna(val_frac):
                        continue  # can't place on val_frac axis

                    tp = pd.to_numeric(row["CM_TotalTP"], errors="coerce")
                    fp = pd.to_numeric(row["CM_TotalFP"], errors="coerce")
                    fn = pd.to_numeric(row["CM_TotalFN"], errors="coerce")
                    if any(pd.isna(x) for x in (tp, fp, fn)):
                        continue

                    records.append(
                        {
                            "yolo": str(yolo),
                            "val_frac": float(val_frac),
                            "tp": float(tp),
                            "fp": float(fp),
                            "fn": float(fn),
                            "source_csv": csv_path,
                        }
                    )

    return pd.DataFrame.from_records(records)


def make_fp_fn_plot(summary_df: pd.DataFrame,
                    out_path: str,
                    dataset_name: str,
                    input_res: str):
    """
    Plot mean FP vs mean FN for each (yolo, val_frac) combination.
    Color = YOLO version, marker = validation ratio.
    """
    if summary_df.empty:
        raise SystemExit("No summary rows to plot.")

    # Pretty val labels
    summary_df = summary_df.copy()
    summary_df["val_pct_num"] = (summary_df["val_frac"] * 100).round().astype(int)
    summary_df["val_pct"] = summary_df["val_pct_num"].astype(str) + "%"

    # YOLO order: v4-tiny, v7-tiny, v11n, v11s, then any others
    present_yolos = summary_df["yolo"].unique()
    yolos = get_ordered_yolos(present_yolos)

    vals = sorted(summary_df["val_pct"].unique(),
                  key=lambda s: int(s.rstrip("%")))

    # Color per YOLO
    colors = plt.cm.tab10.colors
    yolo_to_color = {
        y: colors[i % len(colors)]
        for i, y in enumerate(yolos)
    }

    # Marker per val ratio
    marker_cycle = ["o", "s", "^", "D", "v", "*", "X", "P"]
    val_to_marker = {
        v: marker_cycle[i % len(marker_cycle)]
        for i, v in enumerate(vals)
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each group point
    for _, r in summary_df.iterrows():
        yolo = r["yolo"]
        val_label = r["val_pct"]
        mean_fp = r["mean_fp"]
        mean_fn = r["mean_fn"]

        ax.scatter(
            mean_fp,
            mean_fn,
            marker=val_to_marker[val_label],   # shape = val ratio
            color=yolo_to_color[yolo],         # color = YOLO version
            s=80,
            edgecolor="black",
            linewidth=0.5,
            clip_on=False,   # keep markers visible even on axes
            zorder=3,        # draw above axes lines
        )

    # Legend for YOLO versions (color)
    yolo_handles = []
    for y in yolos:
        h = plt.Line2D(
            [], [],
            marker="o",
            linestyle="None",
            color=yolo_to_color[y],
            label=y,
            markersize=8,
        )
        yolo_handles.append(h)

    # Legend for validation ratios (marker shape)
    val_handles = []
    for v in vals:
        h = plt.Line2D(
            [], [],
            marker=val_to_marker[v],
            linestyle="None",
            color="black",
            label=v,
            markersize=8,
        )
        val_handles.append(h)

    legend1 = ax.legend(handles=yolo_handles, title="YOLO version (color)",
                        loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=val_handles, title="Validation ratio (shape)",
              loc="lower right")

    # Labels / title
    n_min = int(summary_df["n"].min())
    n_max = int(summary_df["n"].max())
    n_text = f"(runs per group: n_min={n_min}, n_max={n_max})"

    ax.set_title(
        f"{dataset_name} {input_res}: mean FP vs FN by YOLO version and "
        f"validation ratio\n{n_text}"
    )
    ax.set_xlabel("Mean CM_TotalFP (false positives)")
    ax.set_ylabel("Mean CM_TotalFN (false negatives)")

    ax.grid(True, alpha=0.3)

    plt.ylim(0, 5)
    plt.xlim(0, 6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved FP/FN scatter plot to: {out_path}")


def main():
    default_bases = [
        "/home/noah/chocolatechip/semester-work/spring2025/"
        "darknet/artifacts/outputs/LegoGearsDarknet"
        # you'll add the ultralytics base dir on the CLI
    ]
    parser = argparse.ArgumentParser(
        description="Compare confusion totals (FP/FN) across YOLO versions "
                    "and validation ratios."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        nargs="+",  # allow one or more dirs
        default=default_bases,
        help="One or more base output directories to scan for benchmark CSVs."
    )
    parser.add_argument(
        "--out", type=str,
        default="coco_confusion_fp_fn_by_yolo_val.png",
        help="Path to save the plot image."
    )
    args = parser.parse_args()

    # Validate dirs
    for d in args.base_dir:
        if not os.path.isdir(d):
            raise SystemExit(f"Base directory not found: {d}")

    df = collect_confusion_records(args.base_dir)
    if df.empty:
        raise SystemExit("No benchmark CSVs found with CM_Total* columns.")

    # Dataset / resolution from first CSV
    first_csv = df["source_csv"].iloc[0]
    dataset_name = infer_dataset_name_from_csv(first_csv)
    input_res = infer_input_resolution_from_csv(first_csv)

    # Aggregate per (yolo, val_frac)
    summary = (
        df.groupby(["yolo", "val_frac"])
          .agg(
              n=("tp", "size"),
              mean_tp=("tp", "mean"),
              mean_fp=("fp", "mean"),
              mean_fn=("fn", "mean"),
          )
          .reset_index()
    )

    print("Summary by YOLO + val_frac:")
    for _, r in summary.iterrows():
        print(
            f"  YOLO={r['yolo']}, val={r['val_frac']:.2f}, "
            f"n={int(r['n'])}, mean TP={r['mean_tp']:.1f}, "
            f"mean FP={r['mean_fp']:.1f}, mean FN={r['mean_fn']:.1f}"
        )

    make_fp_fn_plot(summary, args.out, dataset_name, input_res)


if __name__ == "__main__":
    main()
