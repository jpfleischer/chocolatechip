#!/usr/bin/env python3
import os, re, csv, glob, shutil, getpass, subprocess, zipfile, unicodedata
from pathlib import Path
from datetime import datetime
from threading import Thread, Event

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu

from chocolatechip.model_training.hw_info import (
    summarize_env, resolve_gpu_selection, fio_seq_rw, get_disk_info
)

from chocolatechip.model_training.cfg_maker import generate_cfg_file

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

# ---------- config ----------
ALLOWED_TEMPLATES = ["yolov3-tiny", "yolov4-tiny", "yolov7-tiny", "yolov4-tiny-3l"]

def parse_templates_from_env() -> list[str]:
    raw = os.environ.get("YOLO_TEMPLATES", "all").strip().lower()
    if raw in ("", "all"):
        return ALLOWED_TEMPLATES
    parts = re.split(r"[,\s]+", raw)
    chosen = [p for p in parts if p in ALLOWED_TEMPLATES]
    return chosen or ALLOWED_TEMPLATES

def backend() -> str:
    # BACKEND=darknet | ultralytics (default: darknet)
    return os.environ.get("BACKEND", "darknet").strip().lower()

def darknet_path() -> str:
    if "APPTAINER_ENVIRONMENT" in os.environ or os.path.exists("/.dockerenv"):
        return "/host_workspace/darknet/build/src-cli/darknet"
    return "darknet"

# ---------- cfg generation for Darknet ----------

def generate_cfg(template: str) -> None:
    # Let cfg_maker pick the right anchor count per template
    print(f"[cfg] template={template} -> /workspace/LegoGears_v2/LegoGears.cfg")
    generate_cfg_file(
        template=template,
        data_path="/workspace/LegoGears_v2/LegoGears.data",
        out_path="/workspace/LegoGears_v2/LegoGears.cfg",
        width=224, height=160,
        batch_size=64, subdivisions=8,
        iterations=3000, learning_rate=0.00261,
        anchor_clusters=None,  # default per template
    )

# ---------- command builders ----------
def build_darknet_cmd(gpus_str: str) -> str:
    dk = darknet_path()
    return (
        f"{dk} detector -map -dont_show -nocolor "
        + (f"-gpus {gpus_str} " if gpus_str else "")
        + "train /workspace/LegoGears_v2/LegoGears.data "
          "/workspace/LegoGears_v2/LegoGears.cfg "
        + "2>&1 | tee training_output.log"
    )

def build_ultralytics_cmd(device_indices: list[int]) -> str:
    # override via ULTRA_ARGS (e.g., "task=detect mode=train data=LG_v2.yaml model=yolo11n.pt epochs=200 imgsz=640 batch=16")
    ultra_args = os.environ.get(
        "ULTRA_ARGS",
        "task=detect mode=train data=LG_v2.yaml model=yolo11n.pt epochs=2134 batch=64"
    )
    device_str = ",".join(str(i) for i in device_indices) if device_indices else ""
    return f"yolo {ultra_args} " + (f"device={device_str} " if device_str else "") + "2>&1 | tee training_output.log"

# ---------- one run ----------
def run_once(*, template: str | None, out_root: str) -> None:
    user = effective_username()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpu = Gpu()
    sel = resolve_gpu_selection(gpu)
    indices = sel["indices_abs"]
    gpus_str = sel.get("gpus_str_for_cli", ",".join(str(i) for i in indices))
    gpu_name = ", ".join(sel["selected_names"]) if sel["selected_names"] else "Unknown GPU"
    vram = ", ".join(sel["selected_vram"]) if sel["selected_vram"] else "N/A"
    gpu_name_safe = slugify(gpu_name.replace(",", "-"))

    tag = template if template else "ultralytics"
    output_dir = os.path.join(out_root, f"benchmark__{user}__{gpu_name_safe}__{tag}__{now}")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"[out] {output_dir}")

    if backend() == "darknet":
        assert template, "template required for darknet backend"
        generate_cfg(template)

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
        if backend() == "darknet":
            cmd = build_darknet_cmd(gpus_str)
        else:
            cmd = build_ultralytics_cmd(indices)
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
        "Backend": backend(),
        "YOLO Template": template or "",
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
    }

    csv_name = f"benchmark__{user}__{gpu_name_safe}__{cpu_name_safe}__{tag}__{now}.csv"
    with open(os.path.join(output_dir, csv_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerow(row)
    print(f"[csv] {csv_name}")

    # Move any Darknet weights (if present)
    for f in glob.glob("/workspace/LegoGears_v2/*weights"):
        shutil.move(f, output_dir)

    # Zip (exclude .weights)
    bundle = f"benchmark_bundle__{user}__{gpu_name_safe}__{cpu_name_safe}__{tag}__{now}.zip"
    bundle_path = Path(output_dir) / bundle
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in Path(output_dir).rglob("*"):
            if not p.is_file(): continue
            if p.name == bundle: continue
            if p.suffix.lower() == ".weights": continue
            z.write(p, arcname=p.relative_to(output_dir))
    print(f"[zip] {bundle_path}")

# ---------- main ----------
if __name__ == "__main__":
    out_root = "/outputs" if backend() == "darknet" else "/ultralytics/outputs"
    os.makedirs(out_root, exist_ok=True)

    if backend() == "darknet":
        for t in parse_templates_from_env():
            print(f"\n=== {t} ===")
            run_once(template=t, out_root=out_root)
    else:
        run_once(template=None, out_root=out_root)
