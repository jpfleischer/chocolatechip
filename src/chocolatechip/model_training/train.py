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
from chocolatechip.model_training.dataset_setup import make_split, IMG_EXTS

from chocolatechip.model_training.evaluators_darknet import (
    parse_darknet_summary
)
from chocolatechip.model_training.evaluators_ultra import (
    find_ultra_results_csv,
    parse_ultra_map,
    count_from_data_yaml,
    parse_ultra_final_val,
)

from chocolatechip.model_training.datasets import ensure_download_once

WRITABLE_BASE = Path(os.environ.get("WRITABLE_BASE", "/workspace/.cache/splits"))

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


def _count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def copy_darknet_weights_into_output(p, output_dir: str) -> None:
    """
    Copy any relevant Darknet weights into this run's output directory.
    We look in:
      - directory of p.cfg_out
      - WRITABLE_BASE (/workspace/.cache/splits) where Darknet actually saved
    """
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    cfg_dir = Path(p.cfg_out).parent if getattr(p, "cfg_out", None) else Path(".")
    splits_dir = Path(os.environ.get("WRITABLE_BASE", "/workspace/.cache/splits"))

    stem = Path(p.cfg_out).stem if getattr(p, "cfg_out", None) else "weights"

    candidates = [
        splits_dir / f"{stem}_last.weights",
        splits_dir / "last.weights",
        cfg_dir / f"{stem}_last.weights",
        cfg_dir / "last.weights",
    ]


    for src in candidates:
        if src.is_file():
            dst = outp / src.name
            try:
                shutil.copy2(src, dst)
                print(f"[weights] copied {src} -> {dst}")
            except Exception as e:
                print(f"[weights] copy failed for {src}: {e}")



def _find_darknet_weights_for_export(p) -> str | None:
    """
    Try to find the best/last weights in the usual places.
    """
    cfg_path = Path(p.cfg_out).resolve()
    cfg_dir = cfg_path.parent
    stem = cfg_path.stem  # e.g. "LegoGears"

    # 1) where Darknet actually saved in your log
    splits_dir = Path(os.environ.get("WRITABLE_BASE", "/workspace/.cache/splits"))

    candidates = [
        cfg_dir / f"{stem}_best.weights",
        cfg_dir / "best.weights",
        splits_dir / f"{stem}_best.weights",
        splits_dir / "best.weights",
        splits_dir / f"{stem}_last.weights",
        splits_dir / "last.weights",
        splits_dir / f"{stem}_final.weights",
        splits_dir / "final.weights",
    ]

    for c in candidates:
        if c.is_file():
            print(f"[coco] using weights: {c}")
            return str(c)

    print("[coco] no weights found in cfg_dir or splits_dir; skipping export")
    return None


def read_ultra_counts(output_dir: str, yaml_path: str | None = None) -> tuple[int, int]:
    """
    Prefer the copies you already make: output_dir/train.txt and output_dir/valid.txt.
    If missing, try to read paths from the Ultralytics dataset YAML.
    """
    train_txt = os.path.join(output_dir, "train.txt")
    valid_txt = os.path.join(output_dir, "valid.txt")
    if os.path.isfile(train_txt) or os.path.isfile(valid_txt):
        return _count_lines(train_txt), _count_lines(valid_txt)

    # Fallback: peek into YAML if provided and points to list files
    try:
        if yaml_path and os.path.isfile(yaml_path):
            ydoc = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
            tr = ydoc.get("train"); va = ydoc.get("val")
            if isinstance(tr, str) and os.path.isfile(tr) and tr.endswith(".txt"):
                t = _count_lines(tr)
            else:
                t = 0
            if isinstance(va, str) and os.path.isfile(va) and va.endswith(".txt"):
                v = _count_lines(va)
            else:
                v = 0
            return t, v
    except Exception:
        pass
    return 0, 0


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


def build_split_for(vf: float, ds, out_dir: str | Path | None = None) -> tuple[str, str]:
    ratio_tag = f"v{int(round(vf*100)):02d}"
    prefix = f"{ds.prefix}_{ratio_tag}"

    sets = None if getattr(ds, "flat_dir", None) else list(ds.sets)

    # Default to legacy behavior if not provided
    out_dir = Path(out_dir) if out_dir is not None else Path(ds.root)

    data_path, yaml_path = make_split(
        root=ds.root,
        sets=sets,
        classes=ds.classes,
        names=ds.names,
        prefix=prefix,
        val_frac=vf,
        seed=ds.split_seed,
        neg_subdirs=list(getattr(ds, "neg_subdirs", ())) or None,
        exts=list(getattr(ds, "exts", IMG_EXTS)),
        flat_dir=getattr(ds, "flat_dir", None),
        legos=bool(getattr(ds, "legos", False)),

        # NEW: write outputs under out_dir (i.e., /host_workspace)
        out_dir=out_dir,
    )
    return data_path, ratio_tag


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
    if key == "templates":
        return replace(p, template=val)
    if key == "color_presets":
        return replace(p, color_preset=val)
    if key == "val_fracs":
        # Never store a scalar; keep the invariant that val_fracs is a tuple
        if isinstance(val, (int, float)):
            return replace(p, val_fracs=(float(val),))
        if isinstance(val, (list, tuple)):
            return replace(p, val_fracs=tuple(val))
        # Unknown type → don’t change it
        return p
    if hasattr(p, key):
        return replace(p, **{key: val})
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


# ---------- one run ----------
def run_once(*, p: TrainProfile, template: Optional[str], out_root: str) -> None:
    user = effective_username()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpu = Gpu()
    sel = resolve_gpu_selection(gpu)

    indices_all = sel["indices_abs"]
    requested = getattr(p, "num_gpus", None)

    if isinstance(requested, int) and requested > 0:
        effective = min(requested, len(indices_all))
        if requested > len(indices_all):
            print(f"[gpu] requested {requested} GPUs but only {len(indices_all)} visible; using {effective}.")
        indices = indices_all[:effective]
    else:
        indices = indices_all

    gpus_str = ",".join(str(i) for i in indices)

    names_all = sel.get("selected_names", []) or []
    vram_all  = sel.get("selected_vram", []) or []

    gpu_name = ", ".join(names_all[:len(indices)]) if names_all else "Unknown GPU"
    vram     = ", ".join(vram_all[:len(indices)])  if vram_all  else "N/A"
    gpu_name_safe = slugify(gpu_name.replace(",", "-"))

    # --- derive ratio from p.data_path and append to tag ---
    m = re.search(r"_v(\d{2})(?:\.data)?$", os.path.basename(p.data_path or ""))
    ratio_pct = m.group(1) if m else None                   # e.g. "10", "15", "20"
    ratio_float = (int(ratio_pct) / 100.0) if ratio_pct else None
    ratio_suffix = f"__val{ratio_pct}" if ratio_pct else ""

    # Keep size in folder names
    size_token = ""
    if getattr(p, "width", None) and getattr(p, "height", None):
        try:
            size_token = f"__{int(p.width)}x{int(p.height)}"
        except Exception:
            size_token = f"__{p.width}x{p.height}"

    # One subdir per YOLO variant (template for Darknet, model name for Ultralytics)
    yolo_variant_raw = (template if (p.backend == "darknet" and template) else p.ultra_model) or "unknown-model"
    yolo_variant_safe = slugify(Path(yolo_variant_raw).stem)
    variant_dir = os.path.join(out_root, yolo_variant_safe)
    os.makedirs(variant_dir, exist_ok=True)

    base_tag = p.backend  # "darknet" or "ultralytics" (no template here)
    tag = base_tag + ratio_suffix + size_token + _color_token(p)

    output_dir = os.path.join(variant_dir, f"benchmark__{user}__{gpu_name_safe}__{tag}__{now}")
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

    # GPU watcher (host-indexed for nvidia-smi)
    watch_log = os.path.join(output_dir, "mylogfile.log")
    stop_evt = Event()
    t = None

    # Map selected CUDA logical indices -> host nvidia-smi indices for gpu.watch
    watch_indices = None
    rt_map = sel.get("runtime_smi_map", [])  # [{'logical','bus_id','name','smi_index'}]
    if rt_map:
        l2s = {row["logical"]: row.get("smi_index") for row in rt_map if row.get("smi_index") is not None}
        mapped = [l2s.get(li) for li in indices]
        if all(m is not None for m in mapped) and len(mapped) > 0:
            watch_indices = mapped  # host indices that align with your selected logical indices

    try:
        if gpu.count > 0:
            t = Thread(target=gpu.watch, kwargs={
                "logfile": watch_log, "delay": 1.0, "dense": True,
                "gpu": watch_indices,  # None => watch all; else host indices
                "install_signal_handler": False, "stop_event": stop_evt,
            })
            t.daemon = True
            t.start()
            if watch_indices:
                print(f"[gpuwatch] -> {watch_log} (host idx: {','.join(map(str, watch_indices))} for logical {','.join(map(str, indices))})")
            else:
                print(f"[gpuwatch] -> {watch_log} (watching all)")
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
            # --- ensure dataset exists before making the split/YAML ---
            if getattr(p, "dataset", None):
                ds = p.dataset
                root = Path(ds.root)
                # anchor relative roots under a writable base (inside the container this is /workspace)
                base = Path(os.environ.get("DATA_ROOT", "/workspace"))
                if not root.is_absolute():
                    ds = replace(ds, root=str((base / root).resolve()))
                # download/extract/normalize once
                ensure_download_once(ds)
                # keep this normalized DatasetSpec on the profile for subsequent uses
                p = replace(p, dataset=ds)

            # existing code that builds the split + YAML
            if getattr(p, "dataset", None) and (not p.ultra_data or not os.path.isfile(p.ultra_data)):
                vf_field = getattr(p, "val_fracs", None)
                if isinstance(vf_field, (list, tuple)) and vf_field:
                    default_vf = float(vf_field[0])
                elif isinstance(vf_field, (int, float)):
                    default_vf = float(vf_field)
                else:
                    default_vf = 0.20  # fallback
                data_path, _ = build_split_for(default_vf, p.dataset, out_dir=WRITABLE_BASE)
                yaml_path = str(Path(data_path).with_suffix(".yaml"))
                p = replace(p, ultra_data=yaml_path)
                print(f"[ultra] using dataset YAML: {yaml_path}")

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

            # after you possibly copy train/valid lists into output_dir
            ultra_train_count, ultra_valid_count = read_ultra_counts(output_dir, p.ultra_data)


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

        # ensure points shows up in CSV even if profile didn’t set it explicitly
        if map_points is None:
            map_points = 101

    else:
        # --- Ultralytics: prefer final 'Validating ... best.pt' block; fallback to results.csv ---
        ultra_log = os.path.join(output_dir, "training_output.log")
        m50_best, m95_best = parse_ultra_final_val(ultra_log)

        if m50_best is None or m95_best is None:
            csv_path = find_ultra_results_csv(output_dir)
            if csv_path:
                m50_best, m95_best = parse_ultra_map(csv_path)
            else:
                print("[ultra] results.csv not found; looked in run dir and train/ subdir")
                m50_best, m95_best = (None, None)

        map_last_pct = m50_best        # fill your "mAP (last %)" with the best.pt value
        map_best_pct = None            # Ultralytics doesn't output a separate "best %" line like Darknet
        map_iou      = 0.50
        map_points   = 101             # Ultralytics uses 101-point PR integration
        best_iter    = None
        conf_thresh_eval = None

        ultra_train_count, ultra_valid_count = (0, 0)
        if getattr(p, "ultra_data", None):
            ultra_train_count, ultra_valid_count = count_from_data_yaml(p.ultra_data)


    # --- dataset sizing & effective epochs (for CSV) ---
    train_count = valid_count = 0
    approx_epochs = None

    if p.backend == "darknet" and p.data_path:
        train_count, valid_count = read_split_counts_from_data(p.data_path)
        if train_count > 0:
            approx_epochs = (p.iterations * p.batch_size) / float(train_count)
    else:
        # Ultralytics
        train_count, valid_count = ultra_train_count, ultra_valid_count
        # (optional) approx_epochs could be p.epochs if you want:
        # approx_epochs = p.epochs


    color_preset_for_csv = p.color_preset if p.color_preset is not None else "off"

    # defaults so names exist even on failure
    coco_ap5095 = coco_ap50 = coco_ap75 = None
    per_iou_cols = {}
    gt_json = det_json = None
    cm_csv_cols = {}   # what we'll merge into row later


    # === External COCO evaluation (framework-agnostic, no env vars) ===
    try:
        from chocolatechip.model_training.export_coco_dets import (
            export_ultra_detections, export_darknet_detections
        )
        from chocolatechip.model_training.coco_eval import (
            coco_eval_bbox
        )

        # 1) make sure we have a val list
        val_list = os.path.join(output_dir, "valid.txt")
        if not os.path.isfile(val_list):
            print("[coco] No valid.txt found; skipping external COCO eval")
            raise RuntimeError("no_valid_list")

        # 2) build COCO GT from DarkMark per-image JSONs (once), using the profile dataset root
        gt_json = os.path.join(output_dir, "val.coco.gt.json")
        if not os.path.isfile(gt_json):
            from chocolatechip.model_training.coco_build_gt import build_coco_gt

            if not getattr(p, "dataset", None) or not getattr(p.dataset, "root", None):
                raise RuntimeError("Profile is missing dataset.root; cannot build COCO GT from DarkMark JSONs.")

            ann_root = p.dataset.root                                 # e.g. /workspace/LegoGears_v2
            # names.txt is relative to the dataset root in your profiles
            names_path = os.path.join(p.dataset.root, p.dataset.names) if getattr(p.dataset, "names", None) else None
            if names_path and not os.path.isfile(names_path):
                # also try alongside data_path (if names is just a basename)
                alt = os.path.join(os.path.dirname(p.data_path), p.dataset.names)
                names_path = alt if os.path.isfile(alt) else None

            build_coco_gt(
                ann_root=ann_root,
                out_json=gt_json,
                list_file=val_list,     # include every image in valid.txt (positives + negatives)
                names_path=names_path,  # lock category order to your names file if present
            )
            print(f"[coco] built GT from DarkMark JSONs: {gt_json}")



        # 3) export detections to COCO results JSON
        det_json = os.path.join(output_dir, f"dets_{p.backend}.coco.json")
        if p.backend == "darknet":
            weights = _find_darknet_weights_for_export(p)
            if not weights:
                print("[coco] external COCO eval skipped: no weights to export")
            else:
                export_darknet_detections(
                    darknet_bin=darknet_path(),
                    data_path=p.data_path,
                    cfg_path=p.cfg_out,
                    weights_path=weights,
                    ann_json=gt_json,
                    out_json=det_json,
                    images_txt=val_list,
                    thresh=0.001,
                    letter_box=True,
                )

        else:
            # best.pt under Ultralytics run dir
            best_pt = str((Path(output_dir) / "train" / "weights" / "best.pt"))
            export_ultra_detections(
                weights=best_pt,
                ann_json=gt_json,
                out_json=det_json,
                images_txt=val_list,
                conf=0.001,     # match Darknet export threshold for fairness
                iou=0.45,        # NMS IoU
                imgsz=p.width if hasattr(p, "width") else None,
                device=indices,
                batch=16,
            )

        coco_metrics = coco_eval_bbox(gt_json, det_json)
        coco_ap5095 = coco_metrics["AP"]
        coco_ap50   = coco_metrics["AP50"]
        coco_ap75   = coco_metrics["AP75"]
        ap_per_iou_pairs = coco_metrics["AP_per_IoU"]
        per_iou_cols = {f"COCO AP@{int(iou*100)} (%)": ap for (iou, ap) in ap_per_iou_pairs}
    except Exception as e:
        coco_ap5095 = None
        coco_ap50 = coco_ap75 = None
        per_iou_cols = {}
        print(f"[coco] external COCO eval skipped: {e}")


    # Prepare defaults
    cm = None
    cm_csv_cols = {}

    # --- Confusion matrix at deployment operating point (dataset-agnostic CSV) ---
    try:
        if gt_json and det_json:
            from chocolatechip.model_training.confusion_eval import compute_confusion_from_coco
            cm = compute_confusion_from_coco(
                gt_json,
                det_json,
                iou_thresh=0.50,
                conf_thresh=0.50,
                csv_style="generic",  # "per-class" or "none"
                write_json_path=os.path.join(output_dir, "confusion_matrix.json"),
                json_indent=2,
            )
            print(f"[confusion] IoU>={cm['params']['iou_thresh']}, conf>={cm['params']['conf_thresh']}")
            cm_csv_cols = cm["csv_cols"]
        else:
            print("[confusion] skipped: missing COCO JSONs (gt/det)")
    except Exception as e:
        print(f"[confusion] skipped: {e}")
        cm = None
        cm_csv_cols = {}

    # --- optional pretty print (console only) ---
    if cm is not None:
        try:
            per_cls = cm.get("per_class", [])
            names   = [c["class"] for c in per_cls]
            M       = cm.get("matrix", [])

            print("[confusion] per-class")
            for c in per_cls:
                print(f"  {c['class']:<16} TP={c['TP']:>4} FP={c['FP']:>4} FN={c['FN']:>4} "
                    f"P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}")

            if M and names:
                col_header = " ".join(f"{n[:8]:>9}" for n in names)
                print("[confusion] matrix (pred rows × gt cols)")
                print(f"           {col_header}")
                for i, row in enumerate(M):
                    row_str = " ".join(f"{v:>9d}" for v in row)
                    print(f"{names[i][:10]:>10} {row_str}")
        except Exception:
            # don't fail the run if printing breaks
            pass

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
        "Input Width":  p.width,
        "Input Height": p.height,
        "Input Size":   f"{p.width}x{p.height}",
        "Iterations":   p.iterations,
        "Batch Size":   p.batch_size,
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
        "Best Iteration": best_iter,

        # PRF at printed conf
        "Conf Thresh (PRF)": conf_thresh_eval,
        "Precision@Conf": prf["prec"],
        "Recall@Conf": prf["rec"],
        "F1@Conf": prf["f1"],

    }

    row.update(cm_csv_cols)


    # If we have per-IoU APs (COCO), add them as extra columns
    if per_iou_cols:
        row.update({
            "COCO AP50-95 (%)": coco_ap5095,
            "COCO AP50 (%)": coco_ap50,
            "COCO AP75 (%)": coco_ap75,
        })
        row.update(per_iou_cols)

    # Only record pycocotools value; if COCO eval failed, omit the column
    if coco_ap5095 is not None:
        row["mAP50-95 (%)"] = coco_ap5095

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

        copy_darknet_weights_into_output(p, output_dir)


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

    # --- make sure Darknet dataset exists at the expected path on first run ---
    if p.backend == "darknet" and getattr(p, "dataset", None):
        ds = p.dataset
        base = Path(os.environ.get("DATA_ROOT", "/workspace"))
        root = Path(ds.root)
        # Normalize relative roots to DATA_ROOT (inside container)
        if not root.is_absolute():
            ds = replace(ds, root=str((base / root).resolve()))
        # Download/extract/promote once so set_* dirs are present
        ensure_download_once(ds)
        # Keep normalized spec on the profile for subsequent uses
        p = replace(p, dataset=ds)


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
            else:
                # accept either a tuple/list or a single float on the profile
                vf = (p.val_fracs[0] if isinstance(p.val_fracs, (tuple, list)) else float(p.val_fracs))

            if getattr(p_variant, "dataset", None):
                data_path, _ = build_split_for(vf, p_variant.dataset, out_dir=WRITABLE_BASE)
            else:
                data_path = p_variant.data_path  # must already exist



            # equalize per-template to keep epochs ~constant
            p_variant = equalize_for_split(p_variant, data_path=data_path, mode="iterations")

            # run it
            run_once(p=p_variant, template=p_variant.template, out_root=out_root)

    elif p.backend == "darknet":
        # single Darknet run: still build/refresh split if using DatasetSpec
        if getattr(p, "dataset", None):
            vf = (p.val_fracs[0] if isinstance(p.val_fracs, (tuple, list)) else float(p.val_fracs))
            data_path, _ = build_split_for(vf, p.dataset, out_dir=WRITABLE_BASE)
            p = replace(p, data_path=data_path)
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
