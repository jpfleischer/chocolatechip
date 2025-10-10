#!/usr/bin/env python3
from __future__ import annotations
import os, re, csv, glob, shutil, getpass, subprocess, zipfile, unicodedata, argparse
from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from typing import Optional

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu

from chocolatechip.model_training.hw_info import (
    summarize_env, resolve_gpu_selection, fio_seq_rw, get_disk_info
)
from chocolatechip.model_training.cfg_maker import generate_cfg_file
from chocolatechip.model_training.profiles import TrainProfile, get_profile
from chocolatechip.model_training.darknet_ultralytics_translation import build_ultralytics_cmd

from dataclasses import replace 

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

# --- helper: (re)build split for a given val fraction using profile.dataset ---
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

    if p.backend == "darknet" and p.dataset:
        for t in templates:
            for vf in p.val_fracs:
                data_path, ratio_tag = build_split_for(vf, p.dataset)
                p_iter = replace(p, data_path=data_path)  # TrainProfile is frozen
                run_once(p=p_iter, template=t, out_root=out_root)  # unchanged
    else:
        run_once(p=p, template=None if p.backend != "darknet" else p.template, out_root=out_root)
