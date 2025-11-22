#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from typing import List
import re

import subprocess
from pathlib import Path
import hashlib



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

                    # Color preset (needed for Leather)
                    color_preset = "unknown"
                    if "Color Preset" in df.columns:
                        color_preset = str(row["Color Preset"]).strip()
                    else:
                        # fallback: try to parse from profile or path
                        m = re.search(r'color_(off|on|preserve|auto)', profile, re.IGNORECASE)
                        if not m:
                            m = re.search(r'color_(off|on|preserve|auto)', csv_path, re.IGNORECASE)
                        if m:
                            color_preset = m.group(1).lower()


                    tp = pd.to_numeric(row["CM_TotalTP"], errors="coerce")
                    fp = pd.to_numeric(row["CM_TotalFP"], errors="coerce")
                    fn = pd.to_numeric(row["CM_TotalFN"], errors="coerce")
                    if any(pd.isna(x) for x in (tp, fp, fn)):
                        continue

                    total = tp + fn  # Only sum TP and FN

                    # Jaccard / IoU-like score from TP, FP, FN
                    denom_j = tp + fp + fn
                    jaccard = float(tp / denom_j) if denom_j > 0 else 0.0


                    # --- Find valid split file and hash it ---
                    valid_path = find_valid_file(root, max_up=5)
                    valid_hash = file_sha1(valid_path) if valid_path else "missing"

                    
                    records.append(
                        {
                            "framework": str(framework),
                            "yolo_type": str(yolo),
                            "dataset": str(dataset),
                            "profile": str(profile),
                            "val_fraction": str(val_fraction),
                            "color_preset": str(color_preset),
                            "tp": float(tp),
                            "fp": float(fp),
                            "fn": float(fn),
                            "total": float(total),
                            "jaccard": jaccard,
                            "valid_path": valid_path or "missing",
                            "valid_hash": valid_hash,
                            "source_csv": csv_path,
                            "source_dir": root,
                        }
                    )

    return pd.DataFrame.from_records(records)


def find_valid_file(run_dir: str, max_up=5):
    run_path = Path(run_dir).resolve()
    parents = run_path.parents
    for up in range(max_up + 1):
        if up == 0:
            p = run_path
        else:
            if up - 1 >= len(parents):
                break
            p = parents[up - 1]
        candidate = p / "valid.txt"
        if candidate.is_file():
            return str(candidate)
    return None



def file_sha1(path: str) -> str:
    """SHA1 of file contents (small & stable for equality checks)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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

    # --- CONSISTENCY CHECKS PER DATASET + VAL_FRACTION ---
    combo_cols = ["dataset", "val_fraction"]
    combos = df[combo_cols].drop_duplicates()

    print("Consistency checks by dataset + validation fraction:")
    valid_outliers = []
    total_outliers = []

    for _, combo in combos.iterrows():
        dataset = combo["dataset"]
        val_fraction = combo["val_fraction"]
        combo_df = df[(df["dataset"] == dataset) & (df["val_fraction"] == val_fraction)]

        print(f"\n  Dataset: {dataset}, Val Fraction: {val_fraction}")

        # ---- TP+FN totals must match ----
        total_counts = combo_df["total"].value_counts().sort_index()
        expected_total = total_counts.idxmax()
        for total_val, count in total_counts.items():
            marker = "←" if total_val == expected_total else " "
            print(f"    Total TP+FN {total_val:.0f}: {count} runs {marker}")

        bad_total_df = combo_df[combo_df["total"] != expected_total]
        if not bad_total_df.empty:
            total_outliers.append((dataset, val_fraction, expected_total, bad_total_df))

        # ---- valid.txt must match ----
        valid_hashes = combo_df["valid_hash"].unique().tolist()

        if len(valid_hashes) == 1 and valid_hashes[0] != "missing":
            print("    ✓ valid.txt is identical across runs")
        else:
            print("    ✗ valid.txt mismatch detected")
            valid_outliers.append((dataset, val_fraction, combo_df))

            valid_groups = combo_df.groupby(["valid_hash"]).size().reset_index(name="count")
            for _, r in valid_groups.iterrows():
                print(f"      valid_hash={r['valid_hash']}  ({r['count']} runs)")

    if total_outliers:
        print("\nRuns with non-standard TP+FN totals:")
        for dataset, val_fraction, expected_total, bad_df in total_outliers:
            print(f"\n  Dataset: {dataset}, Val Fraction: {val_fraction} (expected {expected_total:.0f})")
            for _, r in bad_df[["source_dir", "total"]].drop_duplicates().iterrows():
                print(f"    {r['source_dir']}  TP+FN={r['total']:.0f}")
    else:
        print("\n✓ All TP+FN totals consistent per dataset+val_fraction")

    if valid_outliers:
        print("\nRuns with valid.txt mismatches:")
        for dataset, val_fraction, bad_df in valid_outliers:
            print(f"\n  Dataset: {dataset}, Val Fraction: {val_fraction}")
            # for _, r in bad_df[["source_dir", "valid_path", "valid_hash"]].drop_duplicates().iterrows():
            #     print(f"    {r['source_dir']}")
            #     print(f"      valid: {r['valid_path']}  hash={r['valid_hash']}")
    else:
        print("\n✓ All valid.txt splits consistent per dataset+val_fraction")

    
    print("-" * 140)

    # Group by dataset, val_fraction, framework, then other dimensions
    summary = (
        df.groupby(["dataset", "val_fraction", "framework", "yolo_type", "profile", "color_preset"])
          .agg(
              count=("tp", "size"),
              avg_tp=("tp", "mean"),
              avg_fp=("fp", "mean"),
              avg_fn=("fn", "mean"),
              avg_jaccard=("jaccard", "mean"),
          )
          .reset_index()
    )

    summary = summary.sort_values(
        ['dataset', 'val_fraction', 'framework', 'yolo_type', 'profile', 'color_preset']
    )

    # Which (dataset, val_fraction) have inconsistent TP+FN totals across runs?
    unequal_totals = set()
    for (ds, vf), g in df.groupby(["dataset", "val_fraction"]):
        totals = g["total"].round(6).unique()  # total = TP+FN per run
        if len(totals) > 1:
            unequal_totals.add((ds, vf))



    print("Dataset      | Val Frac | Framework | YOLO Type | Profile        | Count | Avg TP | Avg FP | Avg FN | Avg Jaccard")
    print("-" * 160)

    for _, row in summary.iterrows():
        print(
            f"{row['dataset']:<12} | {row['val_fraction']:<8} | {row['framework']:<9} | "
            f"{row['yolo_type']:<9} | {row['profile']:<14} | {int(row['count']):<5} | "
            f"{row['avg_tp']:<6.1f} | {row['avg_fp']:<6.1f} | {row['avg_fn']:<6.1f} | "
            f"{row['avg_jaccard']:<10.3f}"
        )


    # LATEX OUTPUT - Separate tables by dataset
    print("\n" + "="*50)
    print("LATEX TABLES")
    print("="*50)

    datasets = sorted(df['dataset'].unique())

    def wavg(g, col):
        return (g[col] * g["count"]).sum() / g["count"].sum()

    for dataset in datasets:
        dataset_summary = summary[summary['dataset'] == dataset].copy()
        if dataset_summary.empty:
            continue

        print(f"\n% LaTeX Table: {dataset} Dataset Summary Statistics")

        if dataset == "Leather":
            # --- Leather: include Color column, NO Profile column ---
            leather_latex = (
                dataset_summary
                .groupby(["framework", "yolo_type", "color_preset", "val_fraction"], as_index=False)
                .apply(lambda g: pd.Series({
                    "count": int(g["count"].sum()),
                    "avg_tp": wavg(g, "avg_tp"),
                    "avg_fp": wavg(g, "avg_fp"),
                    "avg_fn": wavg(g, "avg_fn"),
                    "avg_jaccard": wavg(g, "avg_jaccard"),
                }))
                .reset_index(drop=True)
            )

            print("\\begin{table}[htbp]")
            print("\\centering")
            print("\\begin{tabular}{|l|l|l|c|c|c|c|c|c|}")
            print("\\hline")
            print("Framework & YOLO Type & Color & Val Frac & Count & Avg TP & Avg FP & Avg FN & Avg Jaccard \\\\")
            print("\\hline")

            for _, row in leather_latex.iterrows():
                j_str = f"{row['avg_jaccard']:.3f}"
                if (dataset, row["val_fraction"]) in unequal_totals:
                    j_str += "*"

                print(
                    f"{escape_latex(row['framework'])} & "
                    f"{escape_latex(row['yolo_type'])} & "
                    f"{escape_latex(row['color_preset'])} & "
                    f"{escape_latex(row['val_fraction'])} & "
                    f"{int(row['count'])} & "
                    f"{row['avg_tp']:.1f} & {row['avg_fp']:.1f} & {row['avg_fn']:.1f} & "
                    f"{j_str} \\\\"
                )


            print("\\hline")
            print("\\end{tabular}")
            bad_vfs = sorted({vf for (ds, vf) in unequal_totals if ds == dataset})
            cap = f"Average confusion matrix values for {dataset} by framework, YOLO type, color preset, and validation fraction"
            if bad_vfs:
                cap += f". TP+FN totals were unequal across runs for val frac(s): {', '.join(bad_vfs)}; * marks affected rows."

            print(f"\\caption{{{cap}}}")

            print("\\label{tab:" + dataset.lower() + "_summary}")
            print("\\end{table}")

        else:
            # --- Non-Leather: grouped by val_fraction ---
            ds_df = df[df["dataset"] == dataset].copy()
            if ds_df.empty:
                continue

            other_latex = (
                ds_df
                .groupby(["val_fraction", "framework", "yolo_type"], as_index=False)
                .agg(
                    count=("tp", "size"),
                    avg_tp=("tp", "mean"),
                    avg_fp=("fp", "mean"),
                    avg_fn=("fn", "mean"),
                    avg_jaccard=("jaccard", "mean"),
                )
                .sort_values(["val_fraction", "framework", "yolo_type"])
            )

            print("\\begin{table}[htbp]")
            print("\\centering")
            # NOTE: Val Frac column removed, since it becomes a block header
            print("\\begin{tabular}{|l|l|c|c|c|c|c|}")
            print("\\hline")
            print("Framework & YOLO Type & Count & Avg TP & Avg FP & Avg FN & Avg Jaccard \\\\")
            print("\\hline")

            for vf, g in other_latex.groupby("val_fraction", sort=False):
                # block header row spanning all 7 cols
                print(f"\\multicolumn{{7}}{{|c|}}{{\\textbf{{Val Frac = {escape_latex(vf)}}}}} \\\\")
                print("\\hline")

                for _, row in g.iterrows():
                    j_str = f"{row['avg_jaccard']:.3f}"
                    if (dataset, vf) in unequal_totals:
                        j_str += "*"

                    print(
                        f"{escape_latex(row['framework'])} & "
                        f"{escape_latex(row['yolo_type'])} & "
                        f"{int(row['count'])} & "
                        f"{row['avg_tp']:.1f} & {row['avg_fp']:.1f} & {row['avg_fn']:.1f} & "
                        f"{j_str} \\\\"
                    )
                print("\\hline")

            print("\\end{tabular}")

            bad_vfs = sorted({vf for (ds, vf) in unequal_totals if ds == dataset})
            cap = f"Average confusion matrix values for {dataset} grouped by validation fraction, framework, and YOLO type"
            if bad_vfs:
                cap += f". TP+FN totals were unequal across runs for val frac(s): {', '.join(bad_vfs)}; * marks affected rows."
            print(f"\\caption{{{cap}}}")
            print("\\label{tab:" + dataset.lower() + "_summary}")
            print("\\end{table}")

if __name__ == "__main__":
    main()