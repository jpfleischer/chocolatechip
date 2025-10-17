#!/usr/bin/env python3
from __future__ import annotations
import os, re, csv, glob, shutil, getpass, subprocess, zipfile, unicodedata, argparse
from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from typing import Optional, Tuple
from dataclasses import replace
import json

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu

from chocolatechip.model_training.hw_info import (
    summarize_env, resolve_gpu_selection, fio_seq_rw, get_disk_info
)
from chocolatechip.model_training.cfg_maker import generate_cfg_file
from chocolatechip.model_training.profiles import TrainProfile, get_profile, equalize_for_split
from chocolatechip.model_training.darknet_ultralytics_translation import build_ultralytics_cmd


from chocolatechip.model_training.datasets import ensure_download_once


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
        anchor_clusters=None,
        color_preset=p.color_preset,   # <- single, purposeful knob
    )


def build_darknet_cmd(p: TrainProfile, gpus_str: str, *,
                      map_thresh: float | None = None,
                      iou_thresh: float | None = None,
                      points: int | None = None) -> str:
    dk = darknet_path()
    # Prefer explicit kwargs if given; otherwise fall back to profile fields
    mt = map_thresh if map_thresh is not None else p.map_thresh
    it = iou_thresh if iou_thresh is not None else p.iou_thresh
    pts = points     if points     is not None else p.map_points

    extras = []
    if mt  is not None: extras += ["-thresh", f"{mt:.2f}"]
    if it  is not None: extras += ["-iou_thresh", f"{it:.2f}"]
    if pts is not None: extras += ["-points", str(pts)]
    extra_str = (" " + " ".join(extras)) if extras else ""
    return (
        f"{dk} detector -map{extra_str} -dont_show -nocolor "
        + (f"-gpus {gpus_str} " if gpus_str else "")
        + f"train {p.data_path} {p.cfg_out} 2>&1 | tee training_output.log"
    )



def build_split_for(vf: float, ds) -> tuple[str, str]:
    ratio_tag = f"v{int(round(vf*100)):02d}"          # e.g., v10, v15, v20
    prefix = f"{ds.prefix}_{ratio_tag}"

    exts = f"--exts {' '.join(ds.exts)}" if ds.exts else ""
    legos = "--legos" if getattr(ds, 'legos', False) else ""

    if getattr(ds, "flat_dir", None):  # <-- FLAT MODE
        cmd = (
            "python -m chocolatechip.model_training.dataset_setup "
            f"--root {ds.root} --flat-dir {ds.flat_dir} --classes {ds.classes} "
            f"--names {ds.names} --prefix {prefix} --val-frac {vf} --seed {ds.seed} "
            f"{exts}"
        )
    else:  # existing hierarchical mode
        sets_str = " ".join(ds.sets)
        neg = f"--neg-subdirs {' '.join(ds.neg_subdirs)}" if ds.neg_subdirs else ""
        cmd = (
            "python -m chocolatechip.model_training.dataset_setup "
            f"--root {ds.root} --sets {sets_str} --classes {ds.classes} "
            f"--names {ds.names} --prefix {prefix} --val-frac {vf} --seed {ds.seed} "
            f"{neg} {exts} {legos}"
        )

    subprocess.check_call(cmd, shell=True)
    return str(Path(ds.root) / f"{prefix}.data"), ratio_tag


def parse_darknet_data_file(data_path: str) -> dict:
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

def split_manifest_from_data_path(data_path: str) -> Path:
    """'/.../LegoGears_v15.data' -> '/.../LegoGears_v15_split.json'"""
    p = Path(data_path)
    return p.with_name(p.stem + "_split.json")

def read_split_counts_from_data(data_path: str) -> Tuple[int, int]:
    """Return (train_count, valid_count); (0,0) if manifest missing."""
    try:
        manifest = split_manifest_from_data_path(data_path)
        js = json.loads(manifest.read_text(encoding="utf-8"))
        c = js.get("counts", {})
        return int(c.get("train_total", 0)), int(c.get("valid_total", 0))
    except Exception:
        return 0, 0

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
    Returns:
      {
        map_iou: float | None,             # e.g., 0.60
        last_map_pct: float | None,        # mAP (last) in percent (0..100)
        best_map_pct: float | None,        # mAP (best) in percent (0..100)
        best_iter: int | None,
        conf_thresh_eval: float | None,    # printed conf_thresh used for PR/F1 (0..1)
        prec: float | None,                # at printed conf_thresh (0..1)
        rec:  float | None,                # at printed conf_thresh (0..1)
        f1:   float | None,                # at printed conf_thresh (0..1)
      }
    """
    out = dict(
        map_iou=None,
        last_map_pct=None,
        best_map_pct=None,
        best_iter=None,
        conf_thresh_eval=None,
        prec=None, rec=None, f1=None
    )
    if not os.path.isfile(log_path):
        return out

    # Examples matched:
    # "mean average precision (mAP@0.60)=97.46%"
    rx_map_line = re.compile(
        r"mean average precision\s*\(mAP@([0-9]+(?:\.[0-9]+)?)\)\s*=\s*([0-9]+(?:\.[0-9]+)?)%?",
        re.I
    )

    # Example matched:
    # "Last accuracy mAP@0.60=97.46%, best=99.18% at iteration #5900."
    rx_last_best = re.compile(
        r"Last accuracy mAP@([0-9]+(?:\.[0-9]+)?)\s*=\s*([0-9]+(?:\.[0-9]+)?)%?,\s*best\s*=\s*([0-9]+(?:\.[0-9]+)?)%?\s*at iteration\s*#\s*(\d+)",
        re.I
    )

    # Example matched:
    # "for conf_thresh=0.50, precision=0.94, recall=0.93, F1 score=0.93"
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
                # conf_thresh in Darknet logs is typically 0..1 or 0..100; keep it as fraction if <=1
                ct = float(m.group(1))
                out["conf_thresh_eval"] = ct if ct <= 1.0 else ct / 100.0
                out["prec"] = float(m.group(2))
                out["rec"]  = float(m.group(3))
                out["f1"]   = float(m.group(4))

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

        # --- stash a copy of the CFG for provenance ---
        try:
            src_cfg = Path(p.cfg_out).resolve()
            dst_cfg = Path(output_dir) / src_cfg.name
            if not (dst_cfg.exists() and os.path.samefile(src_cfg, dst_cfg)):
                shutil.copy2(src_cfg, dst_cfg)
            print(f"[cfg] copied {src_cfg} -> {dst_cfg}")
        except Exception as e:
            print(f"[cfg] copy failed: {e}")

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
    map_last_pct = None
    map_best_pct = None
    map_iou = None
    map_points = None
    prf = dict(prec=None, rec=None, f1=None)
    conf_thresh_eval = None
    map5095_pct = None

    if p.backend == "darknet":
        summary = parse_darknet_summary(os.path.join(output_dir, "training_output.log"))
        map_iou        = summary.get("map_iou")
        map_last_pct   = summary.get("last_map_pct")
        map_best_pct   = summary.get("best_map_pct")
        best_iter      = summary.get("best_iter")
        conf_thresh_eval = summary.get("conf_thresh_eval")
        prf["prec"]    = summary.get("prec")
        prf["rec"]     = summary.get("rec")
        prf["f1"]      = summary.get("f1")
        map_points     = getattr(p, "map_points", None)
    else:
        m50, m95 = parse_ultra_map(os.path.join(output_dir, "results.csv"))
        map_last_pct = m50
        map_best_pct = None
        map5095_pct  = m95
        map_iou      = 0.50
        map_points   = None
        best_iter    = None
        conf_thresh_eval = None

    # --- dataset sizing & effective epochs (for CSV) ---
    train_count = valid_count = 0
    approx_epochs = None
    if p.backend == "darknet" and p.data_path:
        train_count, valid_count = read_split_counts_from_data(p.data_path)
        if train_count > 0:
            approx_epochs = (p.iterations * p.batch_size) / float(train_count)

    color_preset_for_csv = p.color_preset if p.color_preset is not None else "off"

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

        # Training knobs + dataset sizing
        "Iterations": p.iterations,
        "Batch Size": p.batch_size,
        "Subdivisions": p.subdivisions,
        "Train Images": train_count,
        "Valid Images": valid_count,
        "Approx Epochs": approx_epochs,
        "Color Preset": color_preset_for_csv,

        # Evaluation knobs (fixed schema)
        "Val Fraction": ratio_float,
        "IoU (mAP)": map_iou,
        "mAP Points": map_points,
        "mAP (last %)": map_last_pct,
        "mAP (best %)": map_best_pct,
        "mAP50-95 (%)": map5095_pct,
        "Best Iteration": best_iter,

        # PRF at printed conf
        "Conf Thresh (PRF)": conf_thresh_eval,
        "Precision@Conf": prf["prec"],
        "Recall@Conf": prf["rec"],
        "F1@Conf": prf["f1"],

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

    # --- main ---
    p = get_profile(args.profile)
    out_root_base = "/outputs" if p.backend == "darknet" else "/ultralytics/outputs"
    out_root = os.path.join(out_root_base, p.name)   # e.g., /outputs/LeatherDarknet

    templates = p.templates or ((p.template,) if p.template else (None,))
    os.makedirs(out_root, exist_ok=True)

    # ensure the raw dataset exists ONCE (idempotent), BEFORE we call dataset_setup
    if getattr(p, "dataset", None):
        ensure_download_once(p.dataset)   # <--- download/extract once into p.dataset.root

    if p.backend == "darknet" and getattr(p, "dataset", None):
        for t in (p.templates or ((p.template,) if p.template else ())):
            for vf in p.val_fracs:
                data_path, ratio_tag = build_split_for(vf, p.dataset)
                # keep approx epochs constant across splits
                p_iter = equalize_for_split(p, data_path=data_path, mode="iterations")
                # sweep color presets (e.g., (None, "preserve"))
                for color_preset in (p.color_presets if getattr(p, "color_presets", None) else (p.color_preset,)):
                    p_variant = replace(p_iter, color_preset=color_preset)
                    run_once(p=p_variant, template=t, out_root=out_root)
    else:
        run_once(p=p, template=None if p.backend != "darknet" else p.template, out_root=out_root)

