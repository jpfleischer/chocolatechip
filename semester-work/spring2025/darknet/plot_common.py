#!/usr/bin/env python3
import os
import pandas as pd

# Preferred YOLO ordering for all plots
PREFERRED_YOLO_ORDER = [
    "yolov4-tiny",
    "yolov7-tiny",
    "yolo11n",
    "yolo11s",
]


def get_ordered_yolos(present_yolos) -> list[str]:
    """
    Given an iterable of YOLO names, return them in the preferred order:
    yolov4-tiny, yolov7-tiny, yolo11n, yolo11s, then any others sorted.
    """
    present_set = set(present_yolos)

    ordered = [y for y in PREFERRED_YOLO_ORDER if y in present_set]
    # Append any remaining YOLO variants in sorted order
    extras = sorted(y for y in present_set if y not in ordered)

    return ordered + extras


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
                return val.strip()

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
