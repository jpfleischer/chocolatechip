# src/chocolatechip/model_training/evaluators_darknet.py
from __future__ import annotations
from pathlib import Path
import os, re, subprocess

# ---- paste from train.py (unchanged logic) ----

def _maybe_percent_to_percent(val_str: str) -> float:
    try:
        v = float(val_str)
    except Exception:
        return float("nan")
    if v <= 1.0:
        return v * 100.0
    return v

def parse_darknet_summary(log_path: str):
    out = dict(
        map_iou=None, last_map_pct=None, best_map_pct=None, best_iter=None,
        conf_thresh_eval=None, prec=None, rec=None, f1=None
    )
    if not os.path.isfile(log_path):
        return out

    rx_map_line = re.compile(
        r"mean average precision\s*\(mAP@([0-9]+(?:\.[0-9]+)?)\)\s*=\s*([0-9]+(?:\.[0-9]+)?)%?",
        re.I
    )
    rx_last_best = re.compile(
        r"Last accuracy mAP@([0-9]+(?:\.[0-9]+)?)\s*=\s*([0-9]+(?:\.[0-9]+)?)%?,\s*best\s*=\s*([0-9]+(?:\.[0-9]+)?)%?\s*at iteration\s*#\s*(\d+)",
        re.I
    )
    rx_prf = re.compile(
        r"for\s+conf_thresh\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*precision\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*recall\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*F1\s*score\s*=\s*([0-9]*\.?[0-9]+)",
        re.I
    )

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = rx_last_best.search(line)
            if m:
                out["map_iou"]      = float(m.group(1))
                out["last_map_pct"] = _maybe_percent_to_percent(m.group(2))
                out["best_map_pct"] = _maybe_percent_to_percent(m.group(3))
                out["best_iter"]    = int(m.group(4))
                continue

            m = rx_map_line.search(line)
            if m:
                out["map_iou"]      = float(m.group(1))
                out["last_map_pct"] = _maybe_percent_to_percent(m.group(2))
                continue

            m = rx_prf.search(line)
            if m:
                ct = float(m.group(1))
                out["conf_thresh_eval"] = ct if ct <= 1.0 else ct / 100.0
                out["prec"] = float(m.group(2))
                out["rec"]  = float(m.group(3))
                out["f1"]   = float(m.group(4))
    return out

# ---- small helper that uses your existing darknet_path() from train.py ----
def build_darknet_map_cmd(p, *, weights: str, iou: float = 0.50,
                          points: int = 101, thresh: float = 0.001,
                          letter_box: bool = True, darknet_path_fn=None) -> str:
    dk = darknet_path_fn() if darknet_path_fn else "darknet"
    extras = ["-points", str(points), "-iou_thresh", f"{iou:.2f}", "-thresh", f"{thresh:.3f}"]
    if letter_box:
        extras.append("-letter_box")
    return f"{dk} detector map {p.data_path} {p.cfg_out} {weights} " + " ".join(extras) + " -dont_show -nocolor"

def darknet_map50_95(*args, **kwargs):
    raise NotImplementedError(
        "darknet_map50_95() has been removed. Export detections once and use COCO eval (pycocotools) for AP50-95."
    )
