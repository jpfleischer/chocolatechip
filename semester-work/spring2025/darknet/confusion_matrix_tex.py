#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from typing import List
import re

import subprocess
from pathlib import Path


def git_repo_root() -> Path:
    """
    Return git toplevel (repo root). If not in a git repo, fall back to CWD.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return Path(out)
    except Exception:
        return Path.cwd()

def collect_confusion_records(base_dirs: List[str]) -> pd.DataFrame:
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
                    
                    # Determine framework (darknet vs ultralytics)
                    framework = "ultralytics"  # default
                    if "darknet" in csv_path.lower() or "darknet" in root.lower():
                        framework = "darknet"

                    # Get Profile if available
                    profile = "unknown"
                    if "Profile" in df.columns:
                        profile = str(row["Profile"])

                    # Extract dataset identifier from profile (e.g., "LegoGears" from "LegoGears_color_off")
                    dataset = "unknown"
                    if profile != "unknown":
                        # Look for common dataset names in the profile
                        if "LegoGears" in profile:
                            dataset = "LegoGears"
                        elif "FisheyeTraffic" in profile:
                            dataset = "FisheyeTraffic"
                        elif "Leather" in profile:
                            dataset = "Leather"
                        else:
                            # Try to extract the first part before underscore or other delimiter
                            match = re.match(r'^([A-Za-z]+)', profile)
                            if match:
                                dataset = match.group(1)

                    # Get Val Fraction if available
                    val_fraction = "unknown"
                    if "Val Fraction" in df.columns:
                        val_frac_raw = pd.to_numeric(row["Val Fraction"], errors="coerce")
                        if not pd.isna(val_frac_raw):
                            val_fraction = f"{val_frac_raw:.2f}"

                    tp = pd.to_numeric(row["CM_TotalTP"], errors="coerce")
                    fp = pd.to_numeric(row["CM_TotalFP"], errors="coerce")
                    fn = pd.to_numeric(row["CM_TotalFN"], errors="coerce")
                    if any(pd.isna(x) for x in (tp, fp, fn)):
                        continue

                    total = tp + fn  # Only sum TP and FN
                    
                    records.append(
                        {
                            "framework": str(framework),
                            "yolo_type": str(yolo),
                            "dataset": str(dataset),
                            "profile": str(profile),
                            "val_fraction": str(val_fraction),
                            "tp": float(tp),
                            "fp": float(fp),
                            "fn": float(fn),
                            "total": float(total),
                            "source_csv": csv_path,
                            "source_dir": root,
                        }
                    )

    return pd.DataFrame.from_records(records)


def escape_latex(text):
    """Escape special LaTeX characters"""
    text = str(text)
    text = text.replace('_', '\\_')
    text = text.replace('#', '\\#')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    return text


def main():
    repo_root = git_repo_root()

    default_bases = [
        repo_root / "semester-work" / "spring2025" / "darknet",
        repo_root / "semester-work" / "fall2025" / "ultralytics",
    ]

    parser = argparse.ArgumentParser(
        description="Compare confusion totals across YOLO versions and frameworks."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        nargs="+",
        default=[str(p) for p in default_bases],
        help="One or more base output directories to scan for benchmark CSVs."
    )
    args = parser.parse_args()

    for d in args.base_dir:
        if not os.path.isdir(d):
            raise SystemExit(f"Base directory not found: {d}")

    df = collect_confusion_records(args.base_dir)
    if df.empty:
        raise SystemExit("No benchmark CSVs found with CM_Total* columns.")

    # Find the most common total for each dataset + profile + val_fraction combination
    combo_cols = ['dataset', 'profile', 'val_fraction']
    profile_val_combos = df[combo_cols].drop_duplicates()
    
    # CONSOLE OUTPUT
    print("Total samples distribution by dataset, profile and validation fraction (TP + FN):")
    all_outliers = []
    
    for _, combo in profile_val_combos.iterrows():
        dataset = combo['dataset']
        profile = combo['profile']
        val_fraction = combo['val_fraction']
        
        combo_df = df[(df['dataset'] == dataset) & (df['profile'] == profile) & (df['val_fraction'] == val_fraction)]
        total_counts = combo_df['total'].value_counts().sort_index()
        most_common_total = total_counts.idxmax()  # Total with highest count for this combo
        
        print(f"\n  Dataset: {dataset}, Profile: {profile}, Val Fraction: {val_fraction}")
        for total_val, count in total_counts.items():
            marker = "←" if total_val == most_common_total else " "
            print(f"    Total {total_val:.0f}: {count} records {marker}")
        
        # Find outliers for this combination
        combo_outliers = combo_df[combo_df['total'] != most_common_total]
        if not combo_outliers.empty:
            all_outliers.append((dataset, profile, val_fraction, combo_outliers, most_common_total))
    
    # Show all outliers grouped by dataset, profile and val fraction
    if all_outliers:
        print(f"\nTraining runs with non-standard totals:")
        for dataset, profile, val_fraction, outliers, expected_total in all_outliers:
            print(f"\n  Dataset: {dataset}, Profile: {profile}, Val Fraction: {val_fraction} (expected total: {expected_total:.0f})")
            outlier_dirs = outliers.groupby(['source_dir', 'total']).size().reset_index(name='count')
            for _, row in outlier_dirs.iterrows():
                print(f"    {row['source_dir']} (total: {row['total']:.0f})")
    else:
        print(f"\n✓ All training runs have consistent totals within each dataset/profile/val_fraction combination")
    
    print("-" * 140)

    # Group by dataset, val_fraction, framework, then other dimensions
    summary = (
        df.groupby(["dataset", "val_fraction", "framework", "yolo_type", "profile"])
          .agg(
              count=("tp", "size"),
              avg_tp=("tp", "mean"),
              avg_fp=("fp", "mean"),
              avg_fn=("fn", "mean"),
          )
          .reset_index()
    )

    # Sort by dataset, validation fraction, then framework (darknet first)
    summary = summary.sort_values(['dataset', 'val_fraction', 'framework', 'yolo_type', 'profile'])

    print("Dataset      | Val Frac | Framework | YOLO Type | Profile        | Count | Avg TP | Avg FP | Avg FN")
    print("-" * 140)

    for _, row in summary.iterrows():
        print(f"{row['dataset']:<12} | {row['val_fraction']:<8} | {row['framework']:<9} | {row['yolo_type']:<9} | {row['profile']:<14} | {int(row['count']):<5} | "
              f"{row['avg_tp']:<6.1f} | {row['avg_fp']:<6.1f} | {row['avg_fn']:<6.1f}")

    # LATEX OUTPUT - Separate tables by dataset
    print("\n" + "="*50)
    print("LATEX TABLES")
    print("="*50)
    
    # Get unique datasets
    datasets = sorted(df['dataset'].unique())
    
    for dataset in datasets:
        dataset_summary = summary[summary['dataset'] == dataset]
        
        if dataset_summary.empty:
            continue
            
        print(f"\n% LaTeX Table: {dataset} Dataset Summary Statistics")
        print("\\begin{table}[htbp]")
        print("\\centering")
        print("\\begin{tabular}{|l|l|l|c|c|c|c|c|}")
        print("\\hline")
        print("Framework & YOLO Type & Profile & Val Frac & Count & Avg TP & Avg FP & Avg FN \\\\")
        print("\\hline")

        for _, row in dataset_summary.iterrows():
            print(f"{escape_latex(row['framework'])} & {escape_latex(row['yolo_type'])} & {escape_latex(row['profile'])} & "
                  f"{escape_latex(row['val_fraction'])} & {int(row['count'])} & "
                  f"{row['avg_tp']:.1f} & {row['avg_fp']:.1f} & {row['avg_fn']:.1f} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print(f"\\caption{{Average confusion matrix values for {dataset} dataset by framework, YOLO type, profile, and validation fraction}}")
        print("\\label{tab:" + dataset.lower() + "_summary}")
        print("\\end{table}")


if __name__ == "__main__":
    main()