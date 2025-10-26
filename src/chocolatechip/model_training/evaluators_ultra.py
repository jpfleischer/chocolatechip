# src/chocolatechip/model_training/evaluators_ultra.py
from __future__ import annotations
from pathlib import Path
import csv, os, yaml

def find_ultra_results_csv(output_dir: str) -> str | None:
    d = Path(output_dir)
    for p in (d / "results.csv", d / "train" / "results.csv"):
        if p.is_file():
            return str(p)
    for sub in d.glob("*/results.csv"):
        return str(sub)
    return None

def parse_ultra_map(results_csv_path: str) -> tuple[float|None, float|None]:
    if not results_csv_path or not os.path.isfile(results_csv_path):
        return None, None
    last = None
    with open(results_csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            last = row
    if not last:
        return None, None

    def _pick(*keys):
        for k in keys:
            if k in last and last[k] not in (None, "", "nan"):
                try:
                    return float(last[k])
                except Exception:
                    pass
        return None

    m50 = _pick("metrics/mAP50", "metrics/mAP_50", "mAP50", "metrics/mAP50(B)")
    m95 = _pick("metrics/mAP50-95", "metrics/mAP_50-95", "mAP50-95", "metrics/mAP50-95(B)")
    m50 = (m50 * 100.0) if (m50 is not None and m50 <= 1.0) else m50
    m95 = (m95 * 100.0) if (m95 is not None and m95 <= 1.0) else m95
    return m50, m95

def parse_ultra_final_val(run_dir: str) -> tuple[int|None, float|None]:
    """
    Reads Ultralytics' final val table from the console log if you saved it, OR
    just return None for best_epoch here (optional hook if you later store it).
    """
    # minimal placeholder; keep your current behavior if not stored:
    return None, None

def count_from_data_yaml(data_yaml: str) -> tuple[int, int]:
    """
    Return (train_count, valid_count) by reading the dataset YAML and counting files.
    """
    p = Path(data_yaml)
    if not p.is_file():
        return 0, 0
    doc = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore"))
    def _as_list(x):
        if x is None: return []
        if isinstance(x, (list, tuple)): return list(x)
        return [x]
    def _count(src):
        sp = Path(src)
        if sp.is_dir():
            exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
            return sum(len(list(sp.rglob(e))) for e in exts)
        if sp.is_file():
            if sp.suffix.lower()==".txt":
                try:
                    return sum(1 for ln in sp.read_text(errors="ignore").splitlines()
                               if ln.strip() and not ln.strip().startswith("#"))
                except Exception:
                    return 0
            return 1
        try:
            from glob import glob
            return len(glob(src))
        except Exception:
            return 0

    base = p.parent
    root = doc.get("path")
    if root:
        root = (base / root) if not Path(root).is_absolute() else Path(root)
    else:
        root = base

    def _abs(x):
        xp = Path(x)
        if xp.is_absolute():
            return xp
        return root / xp

    train_sources = _as_list(doc.get("train"))
    val_sources   = _as_list(doc.get("val")) or _as_list(doc.get("val_images"))  # some YAMLs
    t = sum(_count(str(_abs(s))) for s in train_sources)
    v = sum(_count(str(_abs(s))) for s in val_sources)
    return t, v
