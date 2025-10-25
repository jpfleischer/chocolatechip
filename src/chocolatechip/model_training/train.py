#!/usr/bin/env python3
from __future__ import annotations
import os, re, csv, glob, shutil, getpass, subprocess, zipfile, unicodedata, argparse
from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from typing import Optional, Tuple, Any, Dict, List
import itertools
from dataclasses import replace
import json
import yaml

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

def _color_token(p) -> str:
    # Only emit if the profile opted in
    if not getattr(p, "tag_color_preset", False):
        return ""
    v = getattr(p, "color_preset", None)
    # encode "off" vs specific preset
    if v in (None, "", "off"):
        return "__color_off"
    return f"__color_{slugify(str(v))}"

def _find_ultra_results_csv(output_dir: str) -> str | None:
    d = Path(output_dir)
    for p in (d / "results.csv", d / "train" / "results.csv"):
        if p.is_file():
            return str(p)
    for sub in d.glob("*/results.csv"):
        return str(sub)
    return None

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
    # Prefer explicit kwargs; else profile; else default to 101
    mt = map_thresh if map_thresh is not None else getattr(p, "map_thresh", None)
    it = iou_thresh if iou_thresh is not None else getattr(p, "iou_thresh", None)
    pts = (
        points
        if points is not None
        else (getattr(p, "map_points", None) if hasattr(p, "map_points") else None)
    )
    if pts is None:
        pts = 101  # <-- default to COCO-style 101 recall points

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
    # Make sure the dataset exists on disk (download/extract once if needed)
    ensure_download_once(ds)
    
    ratio_tag = f"v{int(round(vf*100)):02d}"          # e.g., v10, v15, v20
    prefix = f"{ds.prefix}_{ratio_tag}"

    exts = f"--exts {' '.join(ds.exts)}" if ds.exts else ""
    legos = "--legos" if getattr(ds, 'legos', False) else ""

    if getattr(ds, "flat_dir", None):  # <-- FLAT MODE
        cmd = (
            "python -m chocolatechip.model_training.dataset_setup "
            f"--root {ds.root} --flat-dir {ds.flat_dir} --classes {ds.classes} "
            f"--names {ds.names} --prefix {prefix} --val-frac {vf} --seed {ds.split_seed} "
            f"{exts}"
        )
    else:  # existing hierarchical mode
        sets_str = " ".join(ds.sets)
        neg = f"--neg-subdirs {' '.join(ds.neg_subdirs)}" if ds.neg_subdirs else ""
        cmd = (
            "python -m chocolatechip.model_training.dataset_setup "
            f"--root {ds.root} --sets {sets_str} --classes {ds.classes} "
            f"--names {ds.names} --prefix {prefix} --val-frac {vf} --seed {ds.split_seed} "
            f"{neg} {exts} {legos}"
        )

    subprocess.check_call(cmd, shell=True)
    return str(Path(ds.root) / f"{prefix}.data"), ratio_tag


# helper to pull value-list for a given key
def _values_for_key(p: TrainProfile, key: str) -> Tuple[Any, ...]:
    # explicit values win
    if getattr(p, "sweep_values", None) and key in p.sweep_values and p.sweep_values[key]:
        return tuple(p.sweep_values[key])

    # conventional mappings from existing profile fields
    if key == "templates":
        # prefer templated list; else single template; allow None (ultra)
        if p.templates:
            return tuple(p.templates)
        return (p.template,) if p.template is not None else (None,)

    if key == "val_fracs":
        return tuple(p.val_fracs)

    if key == "color_presets":
        if getattr(p, "color_presets", None):
            return tuple(p.color_presets)
        return (p.color_preset,)

    # generic: if the profile has a tuple/list field with this name, use it
    if hasattr(p, key):
        v = getattr(p, key)
        if isinstance(v, (tuple, list)):
            return tuple(v)
        # allow sweeping scalar via explicit sweep_values only
    raise KeyError(f"sweep key '{key}' has no values (add to profile.sweep_values or provide a plural field)")

def _apply_one(p: TrainProfile, key: str, val: Any) -> TrainProfile:
    # templates: set the *active* template in the variant (don’t mutate templates tuple)
    if key == "templates":
        return replace(p, template=val)
    if key == "color_presets":
        return replace(p, color_preset=val)
    if hasattr(p, key):
        return replace(p, **{key: val})
    # color_presets/val_fracs handled by control flow (we don’t store val_fracs per-run)
    return p


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

    m50 = _pick("metrics/mAP50", "metrics/mAP_50", "mAP50", "metrics/mAP50(B)")
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
    tag = base_tag + ratio_suffix + _color_token(p)

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
            if getattr(p, "training_seed", None) is not None:
                print("[seed] training_seed set but Darknet ignores training RNG; proceeding without it.")
            # right before you call build_darknet_cmd(...) in run_once()
            if p.backend == "darknet" and getattr(p, "map_points", None) is None:
                p = replace(p, map_points=101)   # record the choice for provenance/CSV

            cmd = build_darknet_cmd(p, gpus_str)
        else:
            # ensure we have a dataset YAML; auto-generate from DatasetSpec if provided
            if getattr(p, "dataset", None) and (not p.ultra_data or not os.path.isfile(p.ultra_data)):
                default_vf = (p.val_fracs[0] if getattr(p, "val_fracs", None) else 0.20)
                data_path, _ = build_split_for(default_vf, p.dataset)
                yaml_path = str(Path(data_path).with_suffix(".yaml"))  # dataset_setup wrote this
                p = replace(p, ultra_data=yaml_path)
                print(f"[ultra] using dataset YAML: {yaml_path}")
            if getattr(p, "training_seed", None) is None:
                p = replace(p, training_seed=42)

            # --- stash Ultralytics dataset artifacts for provenance ---
            try:
                ypath = Path(p.ultra_data).resolve()
                # copy the YAML used
                dst_yaml = Path(output_dir) / ypath.name
                if not (dst_yaml.exists() and os.path.samefile(ypath, dst_yaml)):
                    shutil.copy2(ypath, dst_yaml)
                # read YAML to find train/val lists and copy them
                ydoc = yaml.safe_load(ypath.read_text(encoding="utf-8"))
                for key, outname in (("val", "valid.txt"), ("train", "train.txt")):
                    src = ydoc.get(key)
                    if isinstance(src, str) and os.path.isfile(src):
                        shutil.copy2(src, Path(output_dir) / outname)
                        print(f"[{key}] copied {src} -> {Path(output_dir) / outname}")
                    else:
                        # might be a list or glob; ignore silently
                        pass
            except Exception as e:
                print(f"[ultra] provenance copy failed: {e}")

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
        map_points     = getattr(p, "map_points", 101)
    else:
        csv_path = _find_ultra_results_csv(output_dir)
        if csv_path:
            m50, m95 = parse_ultra_map(csv_path)
        else:
            print("[ultra] results.csv not found; looked in run dir and train/ subdir")
            m50, m95 = (None, None)

        map_last_pct = m50
        map_best_pct = None
        map5095_pct  = m95
        map_iou      = 0.50
        map_points   = 101  # Ultralytics uses 101-point PR
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

        # Seeds (explicit provenance)
        "Split Seed": getattr(p.dataset, "split_seed", None),
        "Training Seed": getattr(p, "training_seed", None),

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
    if p.backend == "darknet":
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

        
    # new (no helpers, single inline check)
    inside_container = os.path.exists("/.dockerenv") or ("APPTAINER_ENVIRONMENT" in os.environ)
    out_root_base = "/outputs" if inside_container else "artifacts/outputs"

    out_root = os.path.join(out_root_base, p.name)
    os.makedirs(out_root, exist_ok=True)

    sweep_keys = tuple(getattr(p, "sweep_keys", ()) or ())
    if p.backend == "darknet" and sweep_keys:
        # build cartesian product of declared sweep variables
        grid_lists = [ _values_for_key(p, k) for k in sweep_keys ]
        for combo in itertools.product(*grid_lists):
            # make a concrete variant for this combo
            p_variant = p
            combo_map = dict(zip(sweep_keys, combo))

            # apply non-split fields to the profile (e.g., template, color_preset, iterations, etc.)
            for k, v in combo_map.items():
                if k not in ("val_fracs",):  # val_fracs handled via split below
                    p_variant = _apply_one(p_variant, k, v)

            # decide dataset split for this run
            if "val_fracs" in combo_map:
                vf = float(combo_map["val_fracs"])
                data_path, _ = build_split_for(vf, p.dataset) if getattr(p, "dataset", None) else (p.data_path, None)
            else:
                # Prefer building a fresh split if we have a DatasetSpec
                if getattr(p, "dataset", None):
                    default_vf = (p.val_fracs[0] if getattr(p, "val_fracs", None) else 0.20)
                    data_path, _ = build_split_for(default_vf, p.dataset)
                elif p.data_path and os.path.isfile(p.data_path):
                    data_path = p.data_path
                else:
                    data_path = p.data_path  # last resort


            # equalize per-template to keep epochs ~constant
            p_variant = equalize_for_split(p_variant, data_path=data_path, mode="iterations")

            # run it
            run_once(p=p_variant, template=p_variant.template, out_root=out_root)

    elif p.backend == "darknet":
        # no sweep declared: single run using (first) template / existing data
        run_once(p=p, template=p.template or (p.templates[0] if p.templates else None), out_root=out_root)

    else:
        # ultralytics: you can still declare sweep_keys for things like epochs/batch_size if you want
        if sweep_keys:
            grid_lists = [ _values_for_key(p, k) for k in sweep_keys ]
            for combo in itertools.product(*grid_lists):
                p_variant = p
                for k, v in dict(zip(sweep_keys, combo)).items():
                    p_variant = _apply_one(p_variant, k, v)
                run_once(p=p_variant, template=None, out_root=out_root)
        else:
            run_once(p=p, template=None, out_root=out_root)
