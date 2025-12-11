#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List
import hashlib
import re

import pandas as pd

# Preferred YOLO ordering for all plots
PREFERRED_YOLO_ORDER = [
    "yolov4",      # non-tiny
    "yolov4-tiny",
    "yolov7",      # non-tiny
    "yolov7-tiny",
    "yolo11n",
    "yolo11s",
    "yolo11m",
]


def get_ordered_yolos(present_yolos) -> list[str]:
    """
    Given an iterable of YOLO names, return them in the preferred order:
    yolov4-tiny, yolov7-tiny, yolo11n, yolo11s, yolo11m, then any others sorted.
    """
    present_set = set(present_yolos)

    ordered = [y for y in PREFERRED_YOLO_ORDER if y in present_set]
    # Append any remaining YOLO variants in sorted order
    extras = sorted(y for y in present_set if y not in ordered)

    return ordered + extras


def git_repo_root() -> Path:
    """
    Return git toplevel (repo root). If not in a git repo, fall back to CWD.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return Path(out)
    except Exception:
        return Path.cwd()


def find_valid_file(run_dir: str, max_up: int = 5) -> str | None:
    """
    Starting from run_dir, search up to max_up parents for valid.txt.
    Returns the first match or None.
    """
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


def valid_basename_signature(path: str) -> tuple[str, ...]:
    """
    Read valid.txt and return a tuple of basenames, in order.
    Blank lines / comments are ignored.
    Two valid.txt are considered identical if this tuple matches.
    """
    names: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(Path(line).name)
    return tuple(names)


def normalize_dataset_name(profile: str, csv_path: str = "") -> str:
    """
    Extract dataset identifier from a profile string and/or csv_path.
    E.g., "LegoGears_color_off" -> "LegoGears".
    """
    dataset = "unknown"
    profile = str(profile) if profile is not None else ""
    lower_profile = profile.lower()
    lower_path = csv_path.lower() if csv_path else ""

    # Explicit known datasets first
    if "legogears" in lower_profile or "legogears" in lower_path:
        return "LegoGears"
    if "fisheyetraffic" in lower_profile or "fisheyetraffic" in lower_path:
        # Split JPEG vs non-JPEG
        if "jpg" in lower_profile or "jpg" in lower_path:
            return "FisheyeTrafficJPG"
        return "FisheyeTraffic"
    if "leather" in lower_profile or "leather" in lower_path:
        return "Leather"
    if "cubes" in lower_profile or "cubes" in lower_path:
        return "Cubes"

    # Fallback: leading alpha run from profile
    if profile:
        m = re.match(r"^([A-Za-z]+)", profile)
        if m:
            return m.group(1)

    return dataset


def iter_benchmark_csvs(base_dirs: List[str]) -> Iterable[str]:
    """
    Yield paths to benchmark__*.csv files under one or more base directories.
    """
    for base_dir in base_dirs:
        for root, _dirs, files in os.walk(base_dir):
            for f in files:
                if not f.endswith(".csv"):
                    continue
                if "benchmark__" not in f:
                    continue
                if 'val80' in f:
                    # silly patch
                    continue
                yield os.path.join(root, f)


def infer_dataset_name_from_csv(csv_path: str) -> str:
    """Try to infer a dataset/profile name from a benchmark CSV."""
    try:
        df0 = pd.read_csv(csv_path, nrows=1)
    except Exception:
        df0 = pd.read_csv(csv_path, nrows=1, engine="python")

    for col in ["Profile", "Dataset", "Data Profile"]:
        if col in df0.columns:
            val = str(df0.iloc[0][col])
            if val and val.strip():
                # Prefer normalized dataset name if possible
                return normalize_dataset_name(val, csv_path)

    if "Backend" in df0.columns:
        val = str(df0.iloc[0]["Backend"])
        if val and val.strip():
            return val.strip()

    # Fallback from directory structure
    d1 = os.path.dirname(csv_path)  # benchmark__...
    d2 = os.path.dirname(d1)        # yolov4-tiny / yolo11n / ...
    d3 = os.path.dirname(d2)        # LegoGearsDarknet / LegoGearsUltra / ...
    guess = os.path.basename(d3)
    return guess or "UnknownDataset"


def infer_input_resolution_from_csv(csv_path: str) -> str:
    """Infer input resolution (e.g. '224x160') from a benchmark CSV."""
    try:
        df0 = pd.read_csv(csv_path, nrows=1)
    except Exception:
        df0 = pd.read_csv(csv_path, nrows=1, engine="python")

    if "Input Size" in df0.columns:
        val = str(df0.iloc[0]["Input Size"])
        if val and val.strip():
            return val.strip()

    w = None
    h = None
    if "Input Width" in df0.columns:
        w = pd.to_numeric(df0.iloc[0]["Input Width"], errors="coerce")
    if "Input Height" in df0.columns:
        h = pd.to_numeric(df0.iloc[0]["Input Height"], errors="coerce")

    if w is not None and not pd.isna(w) and h is not None and not pd.isna(h):
        return f"{int(w)}x{int(h)}"

    return "unknown_res"
