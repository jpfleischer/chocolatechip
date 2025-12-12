#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plot_common import (
    get_ordered_yolos,
    git_repo_root,
    iter_benchmark_csvs,
    normalize_dataset_name,
)

# Map between short labels used on plots and full YOLO names used in plot_common
SHORT_TO_FULL: Dict[str, str] = {
    "v4": "yolov4",
    "v4-tiny": "yolov4-tiny",
    "v7": "yolov7",
    "v7-tiny": "yolov7-tiny",
    "11n": "yolo11n",
    "11s": "yolo11s",
    "11m": "yolo11m",
}

FULL_TO_SHORT: Dict[str, str] = {v: k for k, v in SHORT_TO_FULL.items()}

# Preferred validation fractions ordering
VAL_ORDER: List[str] = ["0.1", "0.15", "0.2", "0.8"]


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def avg_jaccard(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def clean_yolo_name(yolo_template: str) -> str:
    """
    Normalize YOLO template to short labels like 'v4-tiny', '11n', etc.
    Example: 'yolov4-tiny' -> 'v4-tiny', 'yolo11n' -> '11n'.
    """
    cleaned = yolo_template.lower()
    if cleaned.startswith("yolo"):
        cleaned = cleaned[len("yolo"):]
    if "." in cleaned:
        cleaned = cleaned.split(".")[0]
    return cleaned


def infer_dataset_name(cm_path: Path) -> str:
    """
    Infer a dataset name using plot_common.normalize_dataset_name so that:
      - FisheyeTraffic*JPG → FisheyeTrafficJPG
      - other FisheyeTraffic variants → FisheyeTraffic
      - LegoGears*, Leather*, Cubes* → their canonical names
    """
    if len(cm_path.parents) >= 3:
        dataset_dir = cm_path.parents[2].name
    else:
        dataset_dir = cm_path.parent.name

    # Use the dir name as "profile" and the full path as csv_path-ish
    return normalize_dataset_name(dataset_dir, str(cm_path))


def iter_runs(base_dirs: List[str]):
    """
    Yield (csv_path, confusion_matrix_path) pairs.

    Use plot_common.iter_benchmark_csvs to find benchmark__*.csv files
    anywhere under the given base dirs, and assume confusion_matrix.json
    lives in the same directory.
    """
    for csv_path in iter_benchmark_csvs(base_dirs):
        csv_p = Path(csv_path)
        cm_path = csv_p.parent / "confusion_matrix.json"
        if cm_path.is_file():
            yield csv_p, cm_path


def extract_jaccard(rec: dict) -> float:
    if "jaccard" in rec and rec["jaccard"] is not None:
        try:
            return float(rec["jaccard"])
        except (TypeError, ValueError):
            pass
    else:
        raise NotImplementedError("No jaccard field in record")
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base",
        nargs="*",
        help=(
            "Base directory/directories to search. "
            "If omitted, defaults to <git_root>/artifacts/outputs."
        ),
    )
    args = parser.parse_args()

    if args.base:
        # user-specified roots
        base_dirs = [str(Path(b).resolve()) for b in args.base]
    else:
        # auto: search from git repo root using plot_common logic
        repo_root = git_repo_root()
        base_dirs = [str(repo_root)]

    # key: (dataset, class_name, "short_yolo / val_frac") -> list of jaccard values
    scores: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    for csv_path, cm_path in iter_runs(base_dirs):
        df_csv = pd.read_csv(csv_path, dtype=str)

        # Require at least these columns to avoid legacy runs with different schema
        if "YOLO Template" not in df_csv.columns or "Backend" not in df_csv.columns:
            continue

        yolo_template = df_csv["YOLO Template"].dropna().iloc[0]
        val_fraction = (
            df_csv["Val Fraction"].dropna().iloc[0]
            if "Val Fraction" in df_csv.columns
            else "unknown"
        )

        short_yolo = clean_yolo_name(yolo_template)
        combined_key = f"{short_yolo} / {val_fraction}"

        dataset_name = infer_dataset_name(cm_path)

        data = load_json(cm_path)
        for rec in data.get("per_class", []):
            cls = rec["class"]
            j = extract_jaccard(rec)
            key = (dataset_name, cls, combined_key)
            scores[key].append(j)

    # Build a DataFrame with dataset info included
    rows = [
        {
            "dataset": k[0],
            "class": k[1],
            "YOLO_Val": k[2],
            "avg_jaccard": avg_jaccard(v),
            "count": len(v),
        }
        for k, v in scores.items()
    ]
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("per_class_yolo_jaccard.csv", index=False)

    output_dir = Path("heatmaps")
    output_dir.mkdir(exist_ok=True)

    sns.set(style="whitegrid", font_scale=1.15)

    # One figure per dataset, with one subplot per YOLO version
    for dataset_name in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset_name]
        if sub.empty:
            continue

        # Order classes by mean Jaccard (descending) for this dataset
        class_means = (
            sub.groupby("class")["avg_jaccard"]
            .mean()
            .sort_values(ascending=False)
        )
        classes = list(class_means.index)

        # Find which YOLO short names are present
        present_pairs = sub["YOLO_Val"].unique()
        present_full_names = set()
        short_to_full_local: Dict[str, str] = {}

        # derive the set of val fractions that actually exist for this dataset
        val_fracs_present = {
            part.split(" / ")[1]
            for part in present_pairs
            if " / " in part
        }

        # sort them using VAL_ORDER as a preference, then anything extra at the end
        def _val_sort_key(v: str) -> tuple[int, str]:
            if v in VAL_ORDER:
                return (VAL_ORDER.index(v), v)
            return (len(VAL_ORDER), v)

        val_fracs = sorted(val_fracs_present, key=_val_sort_key)

        for yv in present_pairs:
            parts = yv.split(" / ")
            if len(parts) != 2:
                continue
            short_y, _val = parts
            full_name = SHORT_TO_FULL.get(short_y, short_y)
            present_full_names.add(full_name)
            short_to_full_local[short_y] = full_name

        if not present_full_names:
            continue

        ordered_full = get_ordered_yolos(list(present_full_names))
        ordered_short = [
            FULL_TO_SHORT.get(full_name, full_name) for full_name in ordered_full
        ]

        num_yolos = len(ordered_short)

        # --- figure size based on grid shape ---
        cell_size = 0.6  # inches per cell side; tweak 0.5–0.8 if you like
        fig_width = num_yolos * len(val_fracs) * cell_size + 2
        fig_height = len(classes) * cell_size + 2

        fig, axes = plt.subplots(
            1,
            num_yolos,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )
        axes = axes[0]

        fig.subplots_adjust(left=0.05, right=0.98, wspace=0.05)

        # Build one heatmap per YOLO version
        for ax, short_y in zip(axes, ordered_short):
            # Build array: rows = classes, cols = val_fracs that actually exist
            data_matrix = []
            for cls in classes:
                row_vals = []
                for v in val_fracs:
                    key_val = f"{short_y} / {v}"
                    mask = (sub["class"] == cls) & (sub["YOLO_Val"] == key_val)
                    vals = sub.loc[mask, "avg_jaccard"]
                    if not vals.empty:
                        row_vals.append(float(vals.iloc[0]))
                    else:
                        row_vals.append(float("nan"))
                data_matrix.append(row_vals)

            matrix_df = pd.DataFrame(data_matrix, index=classes, columns=val_fracs)

            sns.heatmap(
                matrix_df,
                ax=ax,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                cbar=False,
                vmin=0,
                vmax=1,
                square=True,
                annot_kws={"size": 10},
            )

            ax.set_title(short_y)
            ax.set_xticklabels(val_fracs, rotation=45, ha="right")
            ax.tick_params(axis="x", bottom=True, labelbottom=True)

        # One common x-label for the whole figure
        try:
            fig.supxlabel("Val Fraction")  # matplotlib ≥ 3.4
        except AttributeError:
            mid = len(axes) // 2
            axes[mid].set_xlabel("Val Fraction")

        # left-most subplot: keep class names, rotate nicely
        axes[0].set_ylabel("Class")
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

        # all other subplots: hide y tick labels
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.15)

        # Lowercase filename
        pdf_name = f"{dataset_name.lower()}_heatmap_jaccard.pdf"
        out_path = output_dir / pdf_name
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # -------- LaTeX figure snippet --------
        if len(classes) <= 5 and num_yolos <= 3:
            width = r"0.4\textwidth"
        else:
            width = r"\textwidth"

        caption_dataset = dataset_name
        label_name = dataset_name.lower()

        latex = f"""
\\begin{{figure*}}[t]
    \\centering
    \\includegraphics[width={width}]{{images/{pdf_name}}}
    \\caption{{Average Jaccard index (IoU) by YOLO version and validation fraction for {caption_dataset} dataset.}}
    \\label{{fig:class-heatmap-jaccard-{label_name}}}
\\end{{figure*}}
        """

        print(latex)
        print()  # blank line between figures


if __name__ == "__main__":
    main()
