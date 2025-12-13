#!/usr/bin/env python3
from __future__ import annotations
import json
import argparse
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_common import (
    get_ordered_yolos,
    git_repo_root,
    iter_benchmark_csvs,
    normalize_dataset_name,
    infer_dataset_name_from_csv,
)


def normalize_profile_name(profile: str) -> str:
    """
    Normalize profile name to group darknet and ultralytics variants together.
    Examples:
      - 'FisheyeTraffic-darknet' -> 'FisheyeTraffic'
      - 'FisheyeTraffic-ultralytics' -> 'FisheyeTraffic'
      - 'LegoGears-darknet' -> 'LegoGears'
    """
    profile_lower = profile.lower()
    # Remove common suffixes
    for suffix in ['-darknet', '-ultralytics', '_darknet', '_ultralytics']:
        if profile_lower.endswith(suffix):
            return profile[:-(len(suffix))]
    return profile


def parse_max_vram_from_log(log_path: Path) -> Optional[float]:
    """
    Parse mylogfile.log (CSV format) to find the maximum vram_mem_used value in MiB.
    Returns the highest VRAM usage found across all GPUs, or None if not found.
    """
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            # Find columns that contain "vram_mem_used MiB"
            vram_columns = [col for col in reader.fieldnames if 'vram_mem_used MiB' in col]
            
            if not vram_columns:
                return None
            
            max_vram = 0.0
            
            for row in reader:
                for col in vram_columns:
                    vram_str = row.get(col, '').strip()
                    if vram_str:
                        # Remove "MiB" suffix and convert to float
                        vram_match = re.match(r'(\d+(?:\.\d+)?)', vram_str)
                        if vram_match:
                            vram_val = float(vram_match.group(1))
                            max_vram = max(max_vram, vram_val)
            
            return max_vram if max_vram > 0 else None
            
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None


def iter_runs(base_dirs: List[str]):
    """
    Yield csv_path for each benchmark CSV found.
    Use plot_common.iter_benchmark_csvs to find benchmark__*.csv files
    anywhere under the given base dirs.
    """
    for csv_path in iter_benchmark_csvs(base_dirs):
        yield Path(csv_path)


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
    
    # Store data for each profile
    profile_gpus: Dict[str, List[str]] = defaultdict(list)
    profile_cpus: Dict[str, List[str]] = defaultdict(list)
    profile_run_counts: Dict[str, int] = defaultdict(int)
    # Store times indexed by GPU: profile -> gpu -> list of times
    profile_gpu_times: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    # Store VRAM values indexed by GPU: profile -> gpu -> list of max VRAM values
    profile_gpu_vram: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for csv_path in iter_runs(base_dirs):
        try:
            df_csv = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
        
        # Check if required columns exist
        if "GPU Name" not in df_csv.columns:
            print(f"Warning: 'GPU Name' column not found in {csv_path}")
            continue
        
        # Use plot_common's infer_dataset_name_from_csv function
        dataset_name = infer_dataset_name_from_csv(str(csv_path))
        
        # Normalize to group darknet/ultralytics together
        profile = normalize_profile_name(dataset_name)
        
        # Skip artifacts profile
        if profile.lower() == "artifacts":
            continue
        
        # Count total runs (rows in CSV)
        profile_run_counts[profile] += len(df_csv)
        
        # Get GPU name for this run
        gpu_name = df_csv["GPU Name"].dropna().iloc[0] if not df_csv["GPU Name"].dropna().empty else None
        
        # Collect GPU names
        gpu_values = df_csv["GPU Name"].dropna().tolist()
        profile_gpus[profile].extend(gpu_values)
        
        # Collect CPU names if available
        if "CPU Name" in df_csv.columns:
            cpu_values = df_csv["CPU Name"].dropna().tolist()
            profile_cpus[profile].extend(cpu_values)
        
        # Parse VRAM from mylogfile.log
        log_path = csv_path.parent / "mylogfile.log"
        max_vram = parse_max_vram_from_log(log_path)
        if max_vram is not None and gpu_name is not None:
            profile_gpu_vram[profile][gpu_name].append(max_vram)
        
        # Collect benchmark times paired with GPU
        if "Benchmark Time (s)" in df_csv.columns and "GPU Name" in df_csv.columns:
            for idx, row in df_csv.iterrows():
                gpu = row.get("GPU Name")
                time_str = row.get("Benchmark Time (s)")
                
                if pd.notna(gpu) and pd.notna(time_str):
                    try:
                        time_val = float(time_str)
                        profile_gpu_times[profile][gpu].append(time_val)
                    except (ValueError, TypeError):
                        pass
    
    # Build DataFrame
    rows = []
    for profile in sorted(profile_gpus.keys()):
        if not profile_gpus[profile]:
            continue
        
        # Total runs
        total_runs = profile_run_counts[profile]
        
        # Most common GPU
        gpu_counter = Counter(profile_gpus[profile])
        most_common_gpu, _ = gpu_counter.most_common(1)[0]
        
        # Most common CPU
        most_common_cpu = "N/A"
        if profile in profile_cpus and profile_cpus[profile]:
            cpu_counter = Counter(profile_cpus[profile])
            most_common_cpu, _ = cpu_counter.most_common(1)[0]
        
        # Average and std dev of benchmark time for the most common GPU only
        avg_time = "N/A"
        std_time = "N/A"
        if profile in profile_gpu_times and most_common_gpu in profile_gpu_times[profile]:
            times_for_gpu = profile_gpu_times[profile][most_common_gpu]
            if times_for_gpu:
                avg_time = f"{np.mean(times_for_gpu):.2f}"
                std_time = f"{np.std(times_for_gpu):.2f}"
        
        # Average and std dev of VRAM for the most common GPU only
        avg_vram = "N/A"
        std_vram = "N/A"
        if profile in profile_gpu_vram and most_common_gpu in profile_gpu_vram[profile]:
            vram_for_gpu = profile_gpu_vram[profile][most_common_gpu]
            if vram_for_gpu:
                avg_vram = f"{np.mean(vram_for_gpu):.2f}"
                std_vram = f"{np.std(vram_for_gpu):.2f}"
        
        rows.append({
            "dataset": profile,
            "total_runs": total_runs,
            "most_common_cpu": most_common_cpu,
            "most_common_gpu": most_common_gpu,
            "avg_time_s": avg_time,
            "std_time_s": std_time,
            "avg_vram_mib": avg_vram,
            "std_vram_mib": std_vram,
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No data found!")
        return
    
    print("\n" + "=" * 120)
    print("Dataset Summary: Runs, Hardware, Performance, and VRAM Statistics")
    print("=" * 120)
    print(df.to_string(index=False))
    print()
    
    # Save to CSV
    output_file = "dataset_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()