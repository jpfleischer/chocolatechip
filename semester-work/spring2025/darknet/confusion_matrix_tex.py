#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import re
from typing import List

import pandas as pd
import numpy as np


from plot_common import (
    git_repo_root,
    find_valid_file,
    valid_basename_signature,
    normalize_dataset_name,
    iter_benchmark_csvs,
)


try:
    from scipy import stats
except ImportError:
    stats = None


def collect_confusion_records(base_dirs: List[str]) -> pd.DataFrame:
    """
    Walk one or more base_dirs, read benchmark__*.csv, and collect confusion totals
    plus yolo version, dataset, validation fraction, color preset, etc.
    """
    records = []

    for csv_path in iter_benchmark_csvs(base_dirs):
        root = os.path.dirname(csv_path)

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

            # Extract dataset identifier from profile/path
            dataset = normalize_dataset_name(profile, csv_path)

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
            valid_sig = valid_basename_signature(valid_path) if valid_path else None

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
                    "valid_sig": valid_sig,
                    "source_csv": csv_path,
                    "source_dir": root,
                }
            )

    return pd.DataFrame.from_records(records)


def escape_latex(text):
    """Escape special LaTeX characters"""
    text = str(text)
    text = text.replace("_", "\\_")
    text = text.replace("#", "\\#")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    return text


def _find_significant_best(g: pd.DataFrame, id_cols, alpha: float = 0.05):
    """
    Given an aggregated table g with columns:
      - 'framework'
      - 'count' (n)
      - 'avg_jaccard'
      - 'std_jaccard'
    and identifier columns id_cols (e.g. ["framework","yolo_type"] or
    ["framework","yolo_type","color_preset"]),

    Return (best_key_dict, True) if there is a **unique** best model whose
    avg_jaccard is significantly higher than all models in the *other framework*
    (darknet vs ultralytics) using Welch t-tests with Holm–Bonferroni
    correction at level alpha.

    Otherwise return (None, False).
    """
    if stats is None:
        # SciPy not available; skip significance
        return None, False

    g2 = g.copy()

    # Need usable stats
    g2 = g2[g2["std_jaccard"].notna() & (g2["count"] > 1)]
    if len(g2) < 2:
        return None, False

    # Find unique best by avg_jaccard across all rows (any framework)
    max_val = g2["avg_jaccard"].max()
    best_rows = g2[np.isclose(g2["avg_jaccard"], max_val, rtol=1e-9, atol=1e-12)]
    if len(best_rows) != 1:
        # tie for best -> no unique best, so no star
        return None, False

    best = best_rows.iloc[0]
    best_framework = str(best["framework"])

    # Only compare against models from the *other* framework(s)
    pvals = []
    for idx, other in g2.iterrows():
        if idx == best.name:
            continue

        other_framework = str(other["framework"])
        if other_framework == best_framework:
            # same framework -> skip this comparison
            continue

        n1 = float(best["count"])
        n2 = float(other["count"])
        if n1 < 2 or n2 < 2:
            continue

        mean1 = float(best["avg_jaccard"])
        mean2 = float(other["avg_jaccard"])
        std1 = float(best["std_jaccard"])
        std2 = float(other["std_jaccard"])

        var1 = std1**2
        var2 = std2**2

        denom = np.sqrt(var1 / n1 + var2 / n2)
        if denom <= 0.0:
            continue

        t = (mean1 - mean2) / denom

        # Welch–Satterthwaite df
        num = (var1 / n1 + var2 / n2) ** 2
        denom_df = (var1**2 / (n1**2 * (n1 - 1))) + (var2**2 / (n2**2 * (n2 - 1)))
        if denom_df <= 0.0:
            continue

        df = num / denom_df
        p = 2.0 * stats.t.sf(abs(t), df)
        pvals.append(p)

    m = len(pvals)
    if m == 0:
        # No cross-framework competitors -> no way to declare cross-framework "best"
        return None, False

    # Holm–Bonferroni: require **all** cross-framework comparisons (best vs each other) to pass
    p_sorted = sorted(pvals)
    for i, p in enumerate(p_sorted):
        critical = alpha / (m - i)
        if p > critical:
            # at least one comparison is not significant
            return None, False

    # Survived all → unique best significantly better than all models in *other* framework(s)
    best_key = {col: best[col] for col in id_cols}
    return best_key, True


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
        help="One or more base output directories to scan for benchmark CSVs.",
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
        valid_sigs = combo_df["valid_sig"].dropna().unique().tolist()

        if len(valid_sigs) == 1:
            print("    ✓ valid.txt basenames identical across runs")
        else:
            print("    ✗ valid.txt basename mismatch detected")
            valid_outliers.append((dataset, val_fraction, combo_df))

            valid_groups = combo_df.groupby("valid_sig").size().reset_index(name="count")
            for i, r in valid_groups.iterrows():
                sig = r["valid_sig"]
                preview = ", ".join(sig[:3]) + ("..." if len(sig) > 3 else "")
                print(f"      group {i}: ({r['count']} runs) e.g. {preview}")

    if total_outliers:
        print("\nRuns with non-standard TP+FN totals:")
        for dataset, val_fraction, expected_total, bad_df in total_outliers:
            print(f"\n  Dataset: {dataset}, Val Fraction: {val_fraction} (expected {expected_total:.0f})")
            for _, r in bad_df[["source_dir", "total"]].drop_duplicates().iterrows():
                print(f"    {r['source_dir']}  TP+FN={r['total']:.0f}")
    else:
        print("\n✓ All TP+FN totals consistent per dataset+val_fraction")

    if valid_outliers:
        print("\nRuns with valid.txt mismatches (one example per basename-group, up to 2 groups):")
        for dataset, val_fraction, bad_df in valid_outliers:
            print(f"\n  Dataset: {dataset}, Val Fraction: {val_fraction}")

            tmp = bad_df.dropna(subset=["valid_sig"]).copy()

            # Find the two most common distinct basename-signatures
            counts = tmp["valid_sig"].value_counts()
            top_sigs = counts.index.tolist()[:2]

            if len(top_sigs) < 2:
                print("    Only one basename-group found.")
                continue

            for idx, sig in enumerate(top_sigs, start=1):
                example = (
                    tmp[tmp["valid_sig"] == sig][["source_dir", "valid_path"]]
                    .drop_duplicates()
                    .sort_values("source_dir")
                    .head(1)
                )
                if example.empty:
                    continue

                r = example.iloc[0]
                preview = ", ".join(sig[:5]) + ("..." if len(sig) > 5 else "")
                print(f"    Example group {idx} ({counts[sig]} runs):")
                print(f"      {r['source_dir']}")
                print(f"        valid: {r['valid_path']}")
                print(f"        basenames start: {preview}")
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
        ["dataset", "val_fraction", "framework", "yolo_type", "profile", "color_preset"]
    )

    # Which (dataset, val_fraction) have inconsistent TP+FN totals across runs?
    unequal_totals = set()
    for (ds, vf), g in df.groupby(["dataset", "val_fraction"]):
        totals = g["total"].round(6).unique()  # total = TP+FN per run
        if len(totals) > 1:
            unequal_totals.add((ds, vf))

    # Datasets that have ANY inconsistent TP+FN totals; we'll skip their LaTeX.
    bad_datasets = {ds for (ds, _vf) in unequal_totals}

    print(
        "Dataset      | Val Frac | Framework | YOLO Type | Profile        | "
        "Count | Avg TP | Avg FP | Avg FN | Avg Jaccard"
    )
    print("-" * 160)

    for _, row in summary.iterrows():
        print(
            f"{row['dataset']:<12} | {row['val_fraction']:<8} | {row['framework']:<9} | "
            f"{row['yolo_type']:<9} | {row['profile']:<14} | {int(row['count']):<5} | "
            f"{row['avg_tp']:<6.1f} | {row['avg_fp']:<6.1f} | {row['avg_fn']:<6.1f} | "
            f"{row['avg_jaccard']:<10.3f}"
        )

    # LATEX OUTPUT - Separate tables by dataset
    print("\n" + "=" * 50)
    print("LATEX TABLES")
    print("=" * 50)

    datasets = sorted(df["dataset"].unique())

    def wavg(g, col):
        return (g[col] * g["count"]).sum() / g["count"].sum()

    for dataset in datasets:
        dataset_summary = summary[summary["dataset"] == dataset].copy()
        if dataset_summary.empty:
            continue

        # If this dataset has any inconsistent TP+FN totals, warn and skip its LaTeX table.
        bad_vfs = sorted({vf for (ds, vf) in unequal_totals if ds == dataset})
        if bad_vfs:
            print(
                f"% WARNING: Skipping dataset {dataset} in LaTeX output because "
                f"TP+FN totals were unequal for val frac(s): {', '.join(bad_vfs)}"
            )
            continue

        print(f"\n% LaTeX Table: {dataset} Dataset Summary Statistics")

        if dataset == "Leather":
            leather_df = df[df["dataset"] == dataset].copy()

            leather_latex = (
                leather_df.groupby(
                    ["framework", "yolo_type", "color_preset", "val_fraction"], as_index=False
                ).agg(
                    count=("tp", "size"),
                    avg_tp=("tp", "mean"),
                    avg_fp=("fp", "mean"),
                    avg_fn=("fn", "mean"),
                    avg_jaccard=("jaccard", "mean"),
                    median_jaccard=("jaccard", "median"),
                    std_jaccard=("jaccard", "std"),
                )
            )

            # Ensure ultralytics rows are ordered: yolo11n, yolo11s, yolo11m
            yolo_order = {
                "yolo11n": 1,
                "yolo11s": 2,
                "yolo11m": 3,
            }
            leather_latex["yolo_sort"] = leather_latex["yolo_type"].map(yolo_order).fillna(0)

            leather_latex = leather_latex.sort_values(
                ["val_fraction", "framework", "yolo_sort", "yolo_type", "color_preset"]
            )

            print(f"\n% LaTeX Table: {dataset} Dataset Summary Statistics")

            cap = (
                f"Average confusion matrix values for {dataset} by framework, YOLO type, "
                f"color preset, and validation fraction."
            )

            # Table* with caption + label ABOVE the tabular
            print("\\begin{table*}[htbp]")
            print(f"\\caption{{{cap}}}")
            print("\\label{tab:" + dataset.lower() + "_summary}")
            print("\\centering")

            # Note: Val Frac is shown as a block header, not a separate column
            print("\\begin{tabular}{|l|l|l|c|c|c|c|c|c|}")
            print("\\hline")
            print(
                "\\rowcolor{darkgray!20}"
                "\\textbf{Framework} & \\textbf{YOLO Type} & \\textbf{Color} & "
                "\\textbf{Count} & \\textbf{Avg TP} & \\textbf{Avg FP} & "
                "\\textbf{Avg FN} & \\textbf{Med Jaccard} & \\textbf{Avg Jaccard} \\\\"
            )
            print("\\hline")

            for vf, g in leather_latex.groupby("val_fraction", sort=False):
                # Block header row for this Val Frac (gray band)
                print(
                    f"\\rowcolor{{gray!20}}"
                    f"\\multicolumn{{9}}{{|c|}}{{\\textbf{{Val Frac = {escape_latex(vf)}}}}} \\\\"
                )
                print("\\hline")

                # Best Avg Jaccard in this block, by thousandths
                max_j_3 = g["avg_jaccard"].round(3).max()

                # Decide if there's a unique significantly-best row (Welch + Holm)
                best_key = None
                sig_best = False
                if len(g) >= 2:
                    best_key, sig_best = _find_significant_best(
                        g,
                        id_cols=["framework", "yolo_type", "color_preset"],
                        alpha=0.05,
                    )

                for _, row in g.iterrows():
                    # Thousandths-based "best" logic
                    row_j_3 = round(row["avg_jaccard"], 3)
                    is_best = (row_j_3 == max_j_3)

                    # is this row the significant-best one?
                    is_sig = False
                    if sig_best and best_key is not None:
                        is_sig = (
                            (row["framework"] == best_key["framework"]) and
                            (row["yolo_type"] == best_key["yolo_type"]) and
                            (row["color_preset"] == best_key["color_preset"])
                        )

                    j_str = f"{row['avg_jaccard']:.3f}"
                    med_j_str = f"{row['median_jaccard']:.3f}"
                    if (dataset, row["val_fraction"]) in unequal_totals:
                        j_str += "*"

                    if is_best:
                        j_str = f"\\textbf{{{j_str}}}"

                    if is_sig:
                        j_str += "$^{*}$"

                    print(
                        f"{escape_latex(row['framework'])} & "
                        f"{escape_latex(row['yolo_type'])} & "
                        f"{escape_latex(row['color_preset'])} & "
                        f"{int(row['count'])} & "
                        f"{row['avg_tp']:.1f} & {row['avg_fp']:.1f} & {row['avg_fn']:.1f} & "
                        f"{med_j_str} & {j_str} \\\\"
                    )

                print("\\hline")

            print("\\end{tabular}")
            print("\\end{table*}")

        else:
            # --- Non-Leather: grouped by val_fraction ---
            ds_df = df[df["dataset"] == dataset].copy()
            if ds_df.empty:
                continue

            other_latex = (
                ds_df.groupby(["val_fraction", "framework", "yolo_type"], as_index=False)
                .agg(
                    count=("tp", "size"),
                    avg_tp=("tp", "mean"),
                    avg_fp=("fp", "mean"),
                    avg_fn=("fn", "mean"),
                    avg_jaccard=("jaccard", "mean"),
                    median_jaccard=("jaccard", "median"),
                    std_jaccard=("jaccard", "std"),
                )
            )

            # Ensure ultralytics rows are ordered: yolo11n, yolo11s, yolo11m
            yolo_order = {
                "yolo11n": 1,
                "yolo11s": 2,
                "yolo11m": 3,
            }
            other_latex["yolo_sort"] = other_latex["yolo_type"].map(yolo_order).fillna(0)

            other_latex = other_latex.sort_values(
                ["val_fraction", "framework", "yolo_sort", "yolo_type"]
            )

            # Caption text (used above the table)
            cap = (
                f"Average confusion matrix values for {dataset} grouped by validation "
                f"fraction, framework, and YOLO type."
            )

            # Table environment + caption/label ABOVE the tabular
            print("\\begin{table*}[htbp]")
            print(f"\\caption{{{cap}}}")
            print("\\label{tab:" + dataset.lower() + "_summary}")
            print("\\centering")

            print("\\begin{tabular}{|l|l|c|c|c|c|c|c|}")
            print("\\hline")
            print(
                "\\rowcolor{darkgray!20}"
                "\\textbf{Framework} & \\textbf{YOLO Type} & \\textbf{Count} & "
                "\\textbf{Avg TP} & \\textbf{Avg FP} & \\textbf{Avg FN} & "
                "\\textbf{Med Jaccard} & \\textbf{Avg Jaccard} \\\\"
            )
            print("\\hline")

            for vf, g in other_latex.groupby("val_fraction", sort=False):
                # best avg_jaccard in this block, rounded to 3 decimals (thousandths)
                max_j_3 = g["avg_jaccard"].round(3).max()

                # block header row spanning all cols, with gray background
                print(
                    f"\\rowcolor{{gray!20}}"
                    f"\\multicolumn{{8}}{{|c|}}{{\\textbf{{Val Frac = {escape_latex(vf)}}}}} \\\\"
                )
                print("\\hline")

                # Decide if there's a unique significantly-best row (Welch + Holm)
                best_key = None
                sig_best = False
                if len(g) >= 2:
                    best_key, sig_best = _find_significant_best(
                        g,
                        id_cols=["framework", "yolo_type"],
                        alpha=0.05,
                    )

                for _, row in g.iterrows():
                    # Thousandths-based "best" logic
                    row_j_3 = round(row["avg_jaccard"], 3)
                    is_best = (row_j_3 == max_j_3)

                    # does this row correspond to the significant-best one?
                    is_sig = False
                    if sig_best and best_key is not None:
                        is_sig = (
                            (row["framework"] == best_key["framework"]) and
                            (row["yolo_type"] == best_key["yolo_type"])
                        )

                    j_str = f"{row['avg_jaccard']:.3f}"
                    med_j_str = f"{row['median_jaccard']:.3f}"
                    if (dataset, vf) in unequal_totals:
                        j_str += "*"

                    if is_best:
                        j_str = f"\\textbf{{{j_str}}}"

                    if is_sig:
                        j_str += "$^{*}$"

                    print(
                        f"{escape_latex(row['framework'])} & "
                        f"{escape_latex(row['yolo_type'])} & "
                        f"{int(row['count'])} & "
                        f"{row['avg_tp']:.1f} & {row['avg_fp']:.1f} & "
                        f"{row['avg_fn']:.1f} & {med_j_str} & {j_str} \\\\"
                    )
                print("\\hline")

            print("\\end{tabular}")
            print("\\end{table*}")


if __name__ == "__main__":
    main()
