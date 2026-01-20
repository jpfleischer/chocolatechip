#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from plot_common import (
    git_repo_root,
    iter_benchmark_csvs,
    infer_dataset_name_from_csv,
    get_ordered_yolos,
    DEFAULT_FAIR_KEYS,
    make_fair_key,
)

# -----------------------------------------------------------------------------
# Fair apples-to-apples comparison keys.
# We ONLY average timing/VRAM over runs that match on all of these.
# If a key is missing in a CSV row, that row is skipped.
# -----------------------------------------------------------------------------

EXCLUDE_MODELS_FOR_DATASET: dict[str, set[str]] = {
    "LegoGears": {"yolov3", "yolov3-tiny", "yolov3-tiny-3l", "yolov4-tiny-3l", "yolov4"},
}

# Static caption (no CLI override)
DEFAULT_CAPTION = "Dataset training summary: Rows per Model, apples-to-apples by CPU/GPU/config"


# ----------------------------
# Helpers: parsing / formatting
# ----------------------------

def fair_key_to_map(key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(DEFAULT_FAIR_KEYS, key))

def parse_gpu_indices_from_value(raw: str) -> Optional[List[int]]:
    """
    Parse CSV field like:
      "0", "3", "0,1", "0 1", "[0,1]"
    into [0], [3], [0,1], ...
    Returns None if cannot parse.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    s = s.strip("[]()")
    parts = re.split(r"[,\s]+", s)

    out: List[int] = []
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"\d+", p):
            out.append(int(p))

    return out or None


def parse_gpu_indices_from_csv_row(row: pd.Series) -> Optional[List[int]]:
    """
    Try to infer which GPU indices were used for the run.

    Priority:
      1) Explicit columns if you ever add them (best)
      2) Otherwise, only treat 'GPUs Used' as indices if it *looks like a list* ("0,1", "[0 1]", etc.)
         because many logs store it as a COUNT ("1", "4") not an index.
    """
    for col in ("GPU Indices", "GPU Index", "CUDA_VISIBLE_DEVICES"):
        v = row.get(col)
        idxs = parse_gpu_indices_from_value(v) if v is not None else None
        if idxs:
            return idxs

    v = row.get("GPUs Used")
    if v is None:
        return None

    s = str(v).strip()
    if not s:
        return None

    if re.search(r"[,\s\[\]\(\)]", s):
        return parse_gpu_indices_from_value(s)

    return None


def parse_max_vram_from_log(log_path: Path, *, gpu_indices: Optional[List[int]] = None) -> Optional[float]:
    """
    Parse mylogfile.log (CSV format) to find the maximum vram_mem_used value in MiB.

    If gpu_indices is provided (e.g., [3]), ONLY consider columns like:
      "3 vram_mem_used MiB"
    If None, falls back to considering all "vram_mem_used MiB" columns.
    """
    if not log_path.exists():
        return None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None

            if gpu_indices:
                wanted = [f"{i} vram_mem_used MiB" for i in gpu_indices]
                vram_columns = [c for c in wanted if c in reader.fieldnames]
            else:
                vram_columns = [c for c in reader.fieldnames if "vram_mem_used MiB" in c]

            if not vram_columns:
                return None

            max_vram = 0.0
            for row in reader:
                for col in vram_columns:
                    vram_str = (row.get(col, "") or "").strip()
                    if not vram_str:
                        continue
                    m = re.match(r"(\d+(?:\.\d+)?)", vram_str)
                    if m:
                        max_vram = max(max_vram, float(m.group(1)))

            return max_vram if max_vram > 0 else None
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None


def infer_gpu_indices_from_log_by_peak_vram(log_path: Path) -> Optional[List[int]]:
    """
    Fallback inference: choose the GPU index with the highest peak VRAM usage.
    Useful when the benchmark CSV does not reliably store GPU indices.
    """
    if not log_path.exists():
        return None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None

        cols: list[tuple[int, str]] = []
        for name in reader.fieldnames:
            m = re.match(r"^(\d+)\s+vram_mem_used\s+MiB$", name)
            if m:
                cols.append((int(m.group(1)), name))

        if not cols:
            return None

        max_by_idx = {idx: 0.0 for idx, _ in cols}
        for row in reader:
            for idx, col in cols:
                s = (row.get(col, "") or "").strip()
                m = re.match(r"(\d+(?:\.\d+)?)", s)
                if m:
                    max_by_idx[idx] = max(max_by_idx[idx], float(m.group(1)))

        best_idx, best_val = max(max_by_idx.items(), key=lambda kv: kv[1])
        return [best_idx] if best_val > 0 else None


def iqr_outlier_indices(xs: List[float], *, k: float = 1.5) -> List[int]:
    """
    Return indices of points outside [Q1 - k*IQR, Q3 + k*IQR].
    If too few points or IQR=0, returns [].
    """
    if xs is None or len(xs) < 4:
        return []
    arr = np.array(xs, dtype=float)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return []
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return [i for i, v in enumerate(arr) if v < lo or v > hi]


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or pd.isna(seconds):
        return "N/A"
    s = float(seconds)
    if s >= 3600:
        return f"{s / 3600.0:.2f} hr"
    if s >= 60:
        return f"{s / 60.0:.2f} min"
    return f"{s:.2f} s"


def mib_to_gb(mib: Optional[float]) -> str:
    # MiB -> GiB (binary) but labeled as "GB" for readability.
    if mib is None or pd.isna(mib):
        return "N/A"
    return f"{float(mib) / 1024.0:.2f} GB"


def strip_cpu_frequency(cpu: str) -> str:
    if not cpu:
        return cpu
    s = str(cpu)

    # Remove trailing frequency like "@ 3.20GHz"
    s = re.sub(r"\s*@\s*\d+(?:\.\d+)?\s*GHz\b", "", s, flags=re.IGNORECASE)

    # Remove common suffixes
    s = re.sub(r"\s+Processor\s*$", "", s, flags=re.IGNORECASE)

    # Remove standalone "CPU" anywhere
    s = re.sub(r"\bCPU\b", "", s, flags=re.IGNORECASE)

    # Normalize spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def normalize_cpu_name(cpu: str) -> str:
    if cpu is None:
        return "N/A"
    s = str(cpu).strip()
    if not s:
        return "N/A"

    s = strip_cpu_frequency(s)

    # If it already contains lowercase letters, assume it's human-formatted enough
    if re.search(r"[a-z]", s):
        return s

    tokens = s.split()

    # NOTE: intentionally NOT preserving "PLATINUM" in all-caps; will become "Platinum"
    preserve_upper = {"AMD", "EPYC", "APU", "GPU", "INTEL", "XEON", "GOLD", "SILVER"}

    out: list[str] = []
    for t in tokens:
        if re.fullmatch(r"v\d+", t, flags=re.IGNORECASE):
            out.append(t.lower())
            continue
        if re.fullmatch(r"\d+([A-Za-z]+)?", t):
            out.append(t)
            continue

        bare = re.sub(r"[^\w()/-]", "", t)
        if bare.upper() in preserve_upper:
            out.append(bare.upper())
            continue

        m = re.match(r"^([A-Z]+)(\([^)]+\))?$", t)
        if m:
            base = m.group(1).capitalize()
            suffix = m.group(2) or ""
            out.append(base + suffix)
        else:
            out.append(t.capitalize())

    return " ".join(out)


def infer_backend(df0: pd.DataFrame, csv_path: Path) -> Optional[str]:
    if "Backend" in df0.columns:
        v = str(df0.iloc[0].get("Backend", "")).strip().lower()
        if "darknet" in v:
            return "darknet"
        if "ultra" in v or "ultralytics" in v:
            return "ultralytics"

    p = str(csv_path).lower()
    if "darknet" in p:
        return "darknet"
    if "ultra" in p or "ultralytics" in p:
        return "ultralytics"

    return None


def get_row_value(row: pd.Series, col: str) -> Optional[str]:
    v = row.get(col)
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s != "" else None


def escape_latex(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def infer_model_id(row: pd.Series, csv_path: Path) -> str:
    """
    Model identifier for grouping rows.
    Primary source: CSV column 'YOLO Template' (your ground truth).
    Fallback: infer from path tokens if missing.
    """
    v = row.get("YOLO Template")
    if v is not None and not (isinstance(v, float) and pd.isna(v)):
        s = str(v).strip()
        if s:
            # Remove the '.pt' suffix from the model name if present
            s = s.replace(".pt", "")
            return s

    # Fallback: infer from path tokens if YOLO Template is missing
    p = str(csv_path).lower()
    m = re.search(r"(yolov4(?:-tiny)?|yolov7(?:-tiny)?|yolo11[nsmp])", p)
    if m:
        return m.group(1)

    return "UnknownModel"


# ----------------------------
# LaTeX table rendering
# ----------------------------

def df_to_latex_table(df: pd.DataFrame, *, caption: str, label: str) -> str:
    cols = [
        ("dataset", "Dataset"),
        ("model", "Model"),
        ("runs", "Runs"),
        ("cpu", "CPU"),
        ("cpu_threads_used", r"\shortstack{\rule{0pt}{2.6ex}CPU\\Threads\\Used}"),
        ("gpu", "GPU"),
        ("avg_time", "Avg Time"),
        ("std_time", "Std Time"),
        ("avg_vram", "Avg VRAM"),
        ("std_vram", "Std VRAM"),
    ]
    colspec = "|l|l|r|l|r|l|l|l|r|r|"

    # Ensure strings for stable comparison (and avoid NaNs)
    df2 = df.copy()
    for c in ("dataset", "cpu", "cpu_threads_used", "gpu"):
        if c in df2.columns:
            df2[c] = df2[c].astype(str).fillna("")

    # For each dataset, determine if cpu/threads/gpu are constant
    ds_can_merge: dict[str, bool] = {}
    ds_counts: dict[str, int] = {}
    for ds, g in df2.groupby("dataset", sort=False):
        ds_counts[ds] = len(g)
        if len(g) <= 1:
            ds_can_merge[ds] = True
            continue
        # constant iff each of these has exactly 1 unique value
        const = (
            g["cpu"].nunique(dropna=False) == 1
            and g["cpu_threads_used"].nunique(dropna=False) == 1
            and g["gpu"].nunique(dropna=False) == 1
        )
        ds_can_merge[ds] = bool(const)

    lines: list[str] = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join(h for _, h in cols) + r" \\")
    lines.append(r"\hline")

    prev_dataset: Optional[str] = None
    rows_left_in_block = 0
    use_multirow = False

    for _, row in df2.iterrows():
        dataset = str(row.get("dataset", ""))

        # new dataset block?
        if dataset != prev_dataset:
            if prev_dataset is not None:
                lines.append(r"\hline")

            prev_dataset = dataset
            rows_left_in_block = int(ds_counts.get(dataset, 1))
            use_multirow = bool(ds_can_merge.get(dataset, False))

            # base values (escaped)
            cpu = escape_latex(str(row.get("cpu", "")))
            thr = escape_latex(str(row.get("cpu_threads_used", "")))
            gpu = escape_latex(str(row.get("gpu", "")))

            if use_multirow and rows_left_in_block > 1:
                ds_cell = rf"\multirow{{{rows_left_in_block}}}{{*}}{{\centering {escape_latex(dataset)}}}"
                cpu_cell = rf"\multirow{{{rows_left_in_block}}}{{*}}{{\centering {cpu}}}"
                thr_cell = rf"\multirow{{{rows_left_in_block}}}{{*}}{{\centering {thr}}}"
                gpu_cell = rf"\multirow{{{rows_left_in_block}}}{{*}}{{\centering {gpu}}}"
            else:
                # no merging for this dataset (or only 1 row)
                ds_cell = escape_latex(dataset)
                cpu_cell = cpu
                thr_cell = thr
                gpu_cell = gpu
        else:
            # continuing block
            if use_multirow and rows_left_in_block > 0:
                ds_cell = ""
                cpu_cell = ""
                thr_cell = ""
                gpu_cell = ""
            else:
                # not using multirow: print full row each time
                ds_cell = escape_latex(dataset)
                cpu_cell = escape_latex(str(row.get("cpu", "")))
                thr_cell = escape_latex(str(row.get("cpu_threads_used", "")))
                gpu_cell = escape_latex(str(row.get("gpu", "")))

        rows_left_in_block -= 1

        # row cells
        line_cells = [
            ds_cell,
            escape_latex(str(row.get("model", ""))),
            str(row.get("runs", "")),
            cpu_cell,
            thr_cell,
            gpu_cell,
            str(row.get("avg_time", "")),
            str(row.get("std_time", "")),
            str(row.get("avg_vram", "")),
            str(row.get("std_vram", "")),
        ]
        lines.append(" & ".join(line_cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + escape_latex(caption) + r"}")
    lines.append(r"\label{" + escape_latex(label) + r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base",
        nargs="*",
        help=(
            "Base directory/directories to search. "
            "If omitted, defaults to repo root (then iter_benchmark_csvs finds benchmark__*.csv)."
        ),
    )
    # Removed: --out-csv
    # Removed: --caption
    parser.add_argument("--label", default="tab:dataset-summary-by-model", help="LaTeX label.")
    args = parser.parse_args()

    if args.base:
        base_dirs = [str(Path(b).resolve()) for b in args.base]
    else:
        base_dirs = [str(git_repo_root())]

    # dataset -> backend -> model -> key_tuple -> list[(time_seconds, run_dir)]
    ds_backend_model_key_times: Dict[str, Dict[str, Dict[str, Dict[tuple, List[Tuple[float, str]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    # dataset -> backend -> model -> key_tuple -> list[(vram_mib, run_dir)]
    ds_backend_model_key_vram: Dict[str, Dict[str, Dict[str, Dict[tuple, List[Tuple[float, str]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    # ---------- ingest ----------
    for csv_path_str in iter_benchmark_csvs(base_dirs):
        csv_path = Path(csv_path_str)
        run_dir = str(csv_path.parent.resolve())

        try:
            df_csv = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        if df_csv.empty:
            continue
        if "GPU Name" not in df_csv.columns:
            continue

        dataset_name = infer_dataset_name_from_csv(str(csv_path))
        dataset = str(dataset_name).strip() if dataset_name else "UnknownDataset"

        if re.search(r"cards", dataset, flags=re.IGNORECASE):
            continue
        if dataset.strip().lower() == "artifacts":
            continue

        backend = infer_backend(df_csv, csv_path)
        if backend not in ("darknet", "ultralytics"):
            continue

        keys_in_this_csv: set[tuple] = set()
        key_to_gpu_idxs: Dict[tuple, Optional[Tuple[int, ...]]] = {}
        key_to_model_id: Dict[tuple, str] = {}
        key_counts_in_this_csv: Dict[tuple, int] = defaultdict(int)

        for _, row in df_csv.iterrows():
            # # Skip runs with missing mAP50-95 (%)
            # map_str = get_row_value(row, "mAP50-95 (%)")
            # if map_str is None or str(map_str).strip().lower() in {"na", "n/a", "nan"}:
            #     print(f"[SKIP missing mAP50-95] dir={run_dir} csv={csv_path}")
            #     continue

            t_str = get_row_value(row, "Benchmark Time (s)")
            if t_str is None:
                continue
            try:
                t = float(t_str)
            except Exception:
                continue

            key = make_fair_key(row, fair_keys=DEFAULT_FAIR_KEYS)
            if key is None:
                continue

            keys_in_this_csv.add(key)

            model_id = infer_model_id(row, csv_path)
            if key not in key_to_model_id:
                key_to_model_id[key] = model_id

            ds_backend_model_key_times[dataset][backend][model_id][key].append((t, run_dir))
            key_counts_in_this_csv[key] += 1

            if key not in key_to_gpu_idxs:
                idxs = parse_gpu_indices_from_csv_row(row)
                key_to_gpu_idxs[key] = tuple(idxs) if idxs else None

        log_path = csv_path.parent / "mylogfile.log"
        if keys_in_this_csv and log_path.exists():
            cache: Dict[Optional[Tuple[int, ...]], Optional[float]] = {}

            for key in keys_in_this_csv:
                model_id = key_to_model_id.get(key, "UnknownModel")
                idxs_tup = key_to_gpu_idxs.get(key)

                if idxs_tup is None:
                    inferred = infer_gpu_indices_from_log_by_peak_vram(log_path)
                    idxs_tup = tuple(inferred) if inferred else None

                if idxs_tup not in cache:
                    idxs_list = list(idxs_tup) if idxs_tup else None
                    cache[idxs_tup] = parse_max_vram_from_log(log_path, gpu_indices=idxs_list)

                max_vram = cache[idxs_tup]
                if max_vram is not None:
                    n = key_counts_in_this_csv.get(key, 1)
                    ds_backend_model_key_vram[dataset][backend][model_id][key].extend([(max_vram, run_dir)] * n)

    # ---------- summarize ----------
    rows: list[dict] = []
    any_outliers = False

    for dataset in sorted(ds_backend_model_key_times.keys()):
        if re.search(r"cards", dataset, flags=re.IGNORECASE):
            continue
        if dataset.strip().lower() == "artifacts":
            continue

        for backend in ("darknet", "ultralytics"):
            model_map = ds_backend_model_key_times.get(dataset, {}).get(backend, {})
            if not model_map:
                continue

            for model_id in get_ordered_yolos(model_map.keys()):
                # Exclude specific models for specific datasets (final tables)
                if model_id in EXCLUDE_MODELS_FOR_DATASET.get(dataset, set()):
                    continue

                key_to_time_samples = model_map.get(model_id, {})
                if not key_to_time_samples:
                    continue

                # Choose the largest matching subset (most runs) for apples-to-apples within this model
                best_key, time_samples = max(key_to_time_samples.items(), key=lambda kv: len(kv[1]))
                runs = len(time_samples)

                times = [v for (v, _d) in time_samples]
                avg_t = float(np.mean(times)) if times else None
                std_t = float(np.std(times)) if times else None

                vram_samples = (
                    ds_backend_model_key_vram.get(dataset, {})
                    .get(backend, {})
                    .get(model_id, {})
                    .get(best_key, [])
                )
                vrams = [v for (v, _d) in vram_samples]
                avg_v = float(np.mean(vrams)) if vrams else None
                std_v = float(np.std(vrams)) if vrams else None

                # Sanity check: outliers (IQR)
                time_out_idx = iqr_outlier_indices(times)
                vram_out_idx = iqr_outlier_indices(vrams)

                for i in time_out_idx:
                    any_outliers = True
                    v, d = time_samples[i]
                    print(
                        f"[OUTLIER runtime] dataset={dataset} backend={backend} model={model_id} "
                        f"value={v:.3f}s avg={avg_t:.3f}s dir={d}"
                    )

                for i in vram_out_idx:
                    any_outliers = True
                    v, d = vram_samples[i]
                    print(
                        f"[OUTLIER vram]    dataset={dataset} backend={backend} model={model_id} "
                        f"value={v:.1f}MiB avg={avg_v:.1f}MiB dir={d}"
                    )

                key_map = fair_key_to_map(best_key)
                cpu = normalize_cpu_name(key_map.get("CPU Name", "N/A") or "N/A")
                cpu_threads_used = key_map.get("CPU Threads Used", "N/A") or "N/A"
                gpu = key_map.get("GPU Name", "N/A") or "N/A"

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_id,
                        "runs": runs,
                        "cpu": cpu,
                        "cpu_threads_used": cpu_threads_used,
                        "gpu": gpu,
                        "avg_time": format_duration(avg_t),
                        "std_time": format_duration(std_t),
                        "avg_vram": mib_to_gb(avg_v),
                        "std_vram": mib_to_gb(std_v),
                        # keep backend for internal sorting/debug even though it's not output
                        "_backend": backend,
                    }
                )

    if not any_outliers:
        print("No outliers found.")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found!")
        return

    # Sort using plot_common's preferred YOLO order (otherwise lexicographic sort breaks 11n/11s/11m)
    all_models = list(df["model"].dropna().astype(str).unique())
    ordered_models = get_ordered_yolos(all_models)
    rank = {m: i for i, m in enumerate(ordered_models)}

    df["_model_rank"] = df["model"].map(lambda m: rank.get(str(m), 10_000))

    df = df.sort_values(["dataset", "_backend", "_model_rank", "model"]).reset_index(drop=True)
    df_out = df.drop(columns=["_backend", "_model_rank"], errors="ignore")

    print("\n" + "=" * 120)
    print(DEFAULT_CAPTION)
    print("Fair subsets enforced by matching on: " + ", ".join(DEFAULT_FAIR_KEYS))
    print("=" * 120)
    print(df_out.to_string(index=False))
    print()

    # Removed: df_out.to_csv(...)

    latex = df_to_latex_table(df_out, caption=DEFAULT_CAPTION, label=args.label)
    print(latex)


if __name__ == "__main__":
    main()
