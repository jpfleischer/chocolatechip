#!/usr/bin/env python3
from __future__ import annotations
import os, re, csv, glob, shutil, getpass, subprocess, zipfile, unicodedata, argparse
from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from typing import Optional
from dataclasses import replace

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu

from chocolatechip.model_training.hw_info import (
    summarize_env, resolve_gpu_selection, fio_seq_rw, get_disk_info
)
from chocolatechip.model_training.cfg_maker import generate_cfg_file
from chocolatechip.model_training.profiles import TrainProfile, get_profile
from chocolatechip.model_training.darknet_ultralytics_translation import build_ultralytics_cmd

# ---------- small utils ----------
def slugify(text: str, allowed: str = "-_.") -> str:
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    s = s.replace(" ", "_")
    s = re.sub(fr"[^A-Za-z0-9{re.escape(allowed)}]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:180]

def is_wsl() -> bool:
    try:
        with open("/proc/version", "r") as f:
            v = f.read()
        return ("Microsoft" in v) or ("WSL" in v)
    except Exception:
        return False

def effective_username() -> str:
    for key in ("TRUE_USER", "SUDO_USER", "USER"):
        if os.environ.get(key):
            return os.environ[key]
    return getpass.getuser()

def darknet_path() -> str:
    if "APPTAINER_ENVIRONMENT" in os.environ:
        return "/host_workspace/darknet/build/src-cli/darknet"
    elif os.path.exists("/.dockerenv"):
        return "/workspace/darknet/build/src-cli/darknet"
    return "darknet"

# ---------- cfg generation (darknet) ----------
def generate_cfg(p: TrainProfile, template: str) -> None:
    print(f"[cfg] template={template} -> {p.cfg_out}")
    generate_cfg_file(
        template=template,
        data_path=p.data_path,
        out_path=p.cfg_out,
        width=p.width, height=p.height,
        batch_size=p.batch_size, subdivisions=p.subdivisions,
        iterations=p.iterations, learning_rate=p.learning_rate,
        anchor_clusters=None,  # template default (6 for v3/v4-tiny, 9 for v7/v4-3l)
    )

# ---------- command builders ----------
def build_darknet_cmd(p: TrainProfile, gpus_str: str) -> str:
    dk = darknet_path()
    return (
        f"{dk} detector -map -dont_show -nocolor "
        + (f"-gpus {gpus_str} " if gpus_str else "")
        + f"train {p.data_path} {p.cfg_out} 2>&1 | tee training_output.log"
    )

# --- dataset split builder for a given val fraction using profile.dataset ---
def build_split_for(vf: float, ds) -> tuple[str, str]:
    ratio_tag = f"v{int(round(vf*100)):02d}"          # e.g., v10, v15, v20
    prefix = f"{ds.prefix}_{ratio_tag}"

    sets_str = " ".join(ds.sets)
    neg = f"--neg-subdirs {' '.join(ds.neg_subdirs)}" if ds.neg_subdirs else ""
    exts = f"--exts {' '.join(ds.exts)}" if ds.exts else ""
    legos = "--legos" if ds.legos else ""

    cmd = (
        "python -m chocolatechip.model_training.dataset_setup "
        f"--root {ds.root} --sets {sets_str} --classes {ds.classes} "
        f"--names {ds.names} --prefix {prefix} --val-frac {vf} --seed {ds.seed} "
        f"{neg} {exts} {legos}"
    )
    subprocess.check_call(cmd, shell=True)
    return str(Path(ds.root) / f"{prefix}.data"), ratio_tag

# ---------- metrics parsing ----------
def _maybe_percent_to_percent(val_str: str) -> float:
    """Return value as percent; if <=1 treat as fraction and convert to percent."""
    try:
        v = float(val_str)
    except Exception:
        return float("nan")
    if v <= 1.0:
        return v * 100.0
    return v

def parse_darknet_summary(log_path: str):
    """
    Returns dict:
      last_map50_pct, best_map50_pct, best_iter, prec, rec, f1
    mAP as percent (0..100), PRF as decimals (0..1). Missing -> None.
    """
    out = dict(last_map50_pct=None, best_map50_pct=None, best_iter=None,
               prec=None, rec=None, f1=None)
    if not os.path.isfile(log_path):
        return out

    rx_last_best = re.compile(
        r"Last accuracy mAP@0\.50\s*=\s*([0-9]*\.?[0-9]+)%?,\s*best\s*=\s*([0-9]*\.?[0-9]+)%?\s*at iteration\s*#(\d+)",
        re.I)
    rx_map_line = re.compile(r"mean average precision \(mAP@0\.50\)\s*=\s*([0-9]*\.?[0-9]+)%?", re.I)
    rx_prf = re.compile(
        r"for conf_thresh=.*?precision\s*=\s*([0-9]*\.?[0-9]+),\s*recall\s*=\s*([0-9]*\.?[0-9]+),\s*F1 score\s*=\s*([0-9]*\.?[0-9]+)",
        re.I)

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = rx_last_best.search(line)
            if m:
                out["last_map50_pct"] = _maybe_percent_to_percent(m.group(1))
                out["best_map50_pct"] = _maybe_percent_to_percent(m.group(2))
                out["best_iter"] = int(m.group(3))
            m = rx_map_line.search(line)
            if m:
                out["last_map50_pct"] = _maybe_percent_to_percent(m.group(1))
            m = rx_prf.search(line)
            if m:
                out["prec"] = float(m.group(1))
                out["rec"]  = float(m.group(2))
                out["f1"]   = float(m.group(3))
    return out

def parse_ultra_map(results_csv_path: str) -> tuple[float|None, float|None]:
    """
    Read Ultralytics results.csv and return (mAP50_pct, mAP50-95_pct).
    """
    if not os.path.isfile(results_csv_path):
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

    m50 = _pick("metrics/mAP50", "metrics/mAP_50", "mAP50")
    m95 = _pick("metrics/mAP50-95", "metrics/mAP_50-95", "mAP50-95", "metrics/mAP50-95(B)")

    # convert to percent if present
    m50 = (m50 * 100.0) if (m50 is not None and m50 <= 1.0) else m50
    m95 = (m95 * 100.0) if (m95 is not None and m95 <= 1.0) else m95
    return m50, m95


def parse_darknet_data_file(data_path: str) -> dict:
    """
    Returns dict with keys 'train', 'valid', 'names', 'backup' if present.
    """
    out = {}
    if not data_path or not os.path.isfile(data_path):
        return out
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


# ---------- one run ----------
def run_once(*, p: TrainProfile, template: Optional[str], out_root: str) -> None:
    user = effective_username()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpu = Gpu()
    sel = resolve_gpu_selection(gpu)
    indices = sel["indices_abs"]
    gpus_str = sel.get("gpus_str_for_cli", ",".join(str(i) for i in indices))
    gpu_name = ", ".join(sel["selected_names"]) if sel["selected_names"] else "Unknown GPU"
    vram = ", ".join(sel["selected_vram"]) if sel["selected_vram"] else "N/A"
    gpu_name_safe = slugify(gpu_name.replace(",", "-"))

    # --- derive ratio from p.data_path and append to tag ---
    m = re.search(r"_v(\d{2})(?:\.data)?$", os.path.basename(p.data_path or ""))
    ratio_pct = m.group(1) if m else None                   # e.g. "10", "15", "20"
    ratio_float = (int(ratio_pct) / 100.0) if ratio_pct else None
    ratio_suffix = f"__val{ratio_pct}" if ratio_pct else ""

    base_tag = template if (p.backend == "darknet" and template) else "ultralytics"
    tag = base_tag + ratio_suffix

    output_dir = os.path.join(out_root, f"benchmark__{user}__{gpu_name_safe}__{tag}__{now}")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"[out] {output_dir}")

    if p.backend == "darknet":
        assert template, "template required for darknet"
        generate_cfg(p, template)

    # --- copy only the validation list into the output dir as valid.txt ---
    if p.backend == "darknet" and p.data_path:
        dd = parse_darknet_data_file(p.data_path)
        valid_src = dd.get("valid")
        if valid_src and os.path.isfile(valid_src):
            try:
                shutil.copy2(valid_src, os.path.join(output_dir, "valid.txt"))
                print(f"[valid] copied {valid_src} -> {os.path.join(output_dir, 'valid.txt')}")
            except Exception as e:
                print(f"[valid] copy failed: {e}")
        else:
            print("[valid] no valid file found in .data")

    # GPU watcher
    watch_log = os.path.join(output_dir, "mylogfile.log")
    stop_evt = Event()
    t = None
    try:
        if gpu.count > 0:
            t = Thread(target=gpu.watch, kwargs={
                "logfile": watch_log, "delay": 1.0, "dense": True,
                "gpu": indices if indices else None,
                "install_signal_handler": False, "stop_event": stop_evt,
            })
            t.daemon = True; t.start()
            print(f"[gpuwatch] -> {watch_log}")
        else:
            print("[gpuwatch] no GPUs visible")
    except Exception as e:
        print(f"[gpuwatch] skipped: {e}")

    # Train
    try:
        StopWatch.start("benchmark")
        if p.backend == "darknet":
            cmd = build_darknet_cmd(p, gpus_str)
        else:
            cmd = build_ultralytics_cmd(profile=p, device_indices=indices, run_dir=output_dir)
        print(f"[train] {cmd}")
        subprocess.call(cmd, shell=True)
        StopWatch.stop("benchmark")
    finally:
        if t is not None:
            try:
                stop_evt.set()
                gpu.running = False
                t.join(timeout=3)
                print(f"[gpuwatch] stopped")
            except Exception as e:
                print(f"[gpuwatch] stop err: {e}")

    # Post: metrics, env, bundle
    bench = StopWatch.get_benchmark()
    sysinfo = bench["sysinfo"]; b = bench["benchmark"]["benchmark"]
    cpu_name_safe = slugify(sysinfo["cpu"])

    print("[disk] probing…")
    disk_info = get_disk_info()
    print("[disk] fio…")
    dd_w, dd_r = fio_seq_rw()
    print(f"[disk] write={dd_w} read={dd_r}")

    env = summarize_env(indices=indices, training_log_path=os.path.join(output_dir, "training_output.log"))

    # --- derive evaluation metrics ---
    map50_last_pct = None
    map50_best_pct = None
    best_iter = None
    prf = dict(prec=None, rec=None, f1=None)
    map5095_pct = None

    if p.backend == "darknet":
        summary = parse_darknet_summary(os.path.join(output_dir, "training_output.log"))
        map50_last_pct = summary["last_map50_pct"]
        map50_best_pct = summary["best_map50_pct"]
        best_iter = summary["best_iter"]
        prf["prec"] = summary["prec"]
        prf["rec"] = summary["rec"]
        prf["f1"] = summary["f1"]
    else:
        # Ultralytics: results.csv in output_dir
        m50, m95 = parse_ultra_map(os.path.join(output_dir, "results.csv"))
        map50_last_pct = m50
        map5095_pct = m95

    row = {
        "Backend": p.backend,
        "Profile": p.name,
        "YOLO Template": template if (p.backend == "darknet" and template) else p.ultra_model,
        "Benchmark Time (s)": b["time"],
        "CPU Name": sysinfo["cpu"],
        "CPU Threads": sysinfo["cpu_threads"],
        "GPU Name": gpu_name,
        "GPU VRAM": vram,
        "Total Memory": sysinfo["mem.total"],
        "OS": "WSL" if is_wsl() else sysinfo["uname.system"],
        "Architecture": sysinfo["uname.machine"],
        "Python Version": sysinfo["python.version"],
        "Disk Capacity": disk_info["Disk Capacity"],
        "Disk Model": disk_info["Disk Model"],
        "Write Speed": dd_w,
        "Read Speed": dd_r,
        "Working Dir": os.getenv("ACTUAL_PWD", "N/A"),
        "CUDA Version": env["cuda_version"],
        "cuDNN Version": env["cudnn_version"],
        "GPUs Used": env["num_gpus_used"],
        "Compute Capability": env["compute_caps_str"],
        "Val Fraction": ratio_float,
        "mAP@0.50 (last %)": map50_last_pct,
        "mAP@0.50 (best %)": map50_best_pct,
        "mAP@0.50-0.95 (%)": map5095_pct,
        "Best Iteration": best_iter,
        "Precision@thr": prf["prec"],
        "Recall@thr": prf["rec"],
        "F1@thr": prf["f1"],
    }

    csv_name = f"benchmark__{user}__{gpu_name_safe}__{cpu_name_safe}__{tag}__{now}.csv"
    with open(os.path.join(output_dir, csv_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerow(row)
    print(f"[csv] {csv_name}")

    # Move any Darknet weights (if present)
    cfg_dir = os.path.dirname(p.cfg_out)
    for f in glob.glob(os.path.join(cfg_dir, "*weights")):
        try:
            shutil.move(f, output_dir)
        except Exception:
            pass

    # Zip (exclude .weights)
    bundle = f"benchmark_bundle__{user}__{gpu_name_safe}__{cpu_name_safe}__{tag}__{now}.zip"
    bundle_path = Path(output_dir) / bundle
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for q in Path(output_dir).rglob("*"):
            if not q.is_file(): continue
            if q.name == bundle: continue
            if q.suffix.lower() == ".weights": continue
            z.write(q, arcname=q.relative_to(output_dir))
    print(f"[zip] {bundle_path}")

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train with a named profile (profiles may contain multiple templates).")
    ap.add_argument("--profile", default="LegoGearsDarknet", help="Profile name in profiles.PROFILES")
    args = ap.parse_args()

    p = get_profile(args.profile)
    out_root = "/outputs" if p.backend == "darknet" else "/ultralytics/outputs"
    templates = p.templates or ((p.template,) if p.template else (None,))
    os.makedirs(out_root, exist_ok=True)

    if p.backend == "darknet" and getattr(p, "dataset", None):
        for t in templates:
            for vf in p.val_fracs:
                data_path, ratio_tag = build_split_for(vf, p.dataset)
                p_iter = replace(p, data_path=data_path)  # TrainProfile is frozen
                run_once(p=p_iter, template=t, out_root=out_root)
    else:
        run_once(p=p, template=None if p.backend != "darknet" else p.template, out_root=out_root)
