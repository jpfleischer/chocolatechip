import csv
from datetime import datetime
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu
import subprocess
import train_setup  # links training data to its labels
import os
import getpass
import glob
import shutil
import re
from threading import Thread, Event
import unicodedata

from pathlib import Path
import zipfile
from hw_info import summarize_env, resolve_gpu_selection, fio_seq_rw, get_disk_info


uva_running = os.environ.get("UVA_VIRGINIA_RUNNING", "false").lower() == "true"


def is_wsl():
    """
    Detect if running in Windows Subsystem for Linux (WSL)
    """
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read()
        # In WSL, /proc/version usually contains "Microsoft" or "WSL"
        if "Microsoft" in version_info or "WSL" in version_info:
            return True
    except Exception:
        pass
    return False


def slugify(text: str, allowed: str = "-_.") -> str:
    """
    Make a filesystem-friendly string:
    - ASCII only (strip accents)
    - Replace spaces with underscores
    - Any char not alnum or in `allowed` -> underscore
    - Collapse multiple underscores and trim punctuation
    """
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    s = s.replace(" ", "_")
    s = re.sub(fr"[^A-Za-z0-9{re.escape(allowed)}]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:180]  # keep it reasonable

def effective_username() -> str:
    # Prefer an explicit pass-through from the launcher
    for key in ("TRUE_USER", "SUDO_USER", "USER"):
        v = os.environ.get(key)
        if v:
            return v
    return getpass.getuser()

if __name__ == "__main__":

    ALLOWED_TEMPLATES = ["yolov3-tiny", "yolov4-tiny", "yolov7-tiny"]

    def parse_templates_from_env() -> list[str]:
        raw = os.environ.get("YOLO_TEMPLATES", "all").strip().lower()
        if raw in ("", "all"):
            return ALLOWED_TEMPLATES
        parts = re.split(r"[,\s]+", raw)
        chosen = [p for p in parts if p in ALLOWED_TEMPLATES]
        if not chosen:
            print(f"[warn] YOLO_TEMPLATES='{raw}' not recognized; defaulting to both.")
            return ALLOWED_TEMPLATES
        return chosen

    def darknet_path() -> str:
        if "APPTAINER_ENVIRONMENT" in os.environ:
            print("Running in Apptainer environment")
            return "/host_workspace/darknet/build/src-cli/darknet"
        elif os.path.exists("/.dockerenv"):
            print("Running in Docker")
            return "/host_workspace/darknet/build/src-cli/darknet"
        else:
            print("Running non-apptainer")
            return "darknet"

    def generate_cfg(template: str) -> None:
        """Run cfg_maker inside the container environment, keeping cfg_maker defaults."""
        cmd = [
            "python3", "/workspace/cfg_maker.py",
            "--template", template,
            "--data", "/workspace/LegoGears_v2/LegoGears.data",
            "--out",  "/workspace/LegoGears_v2/LegoGears.cfg",
            "--width", "224",
            "--height", "160",
            "--batch-size", "64",
            "--subdivisions", "8",
            "--iterations", "3000",
            "--learning-rate", "0.00261",
            "--anchor-clusters", "9"
        ]
        print(f"[cfg] generating cfg via: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    def run_one(template: str):
        username = effective_username()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        # GPU selection & inventory (centralized)
        gpu = Gpu()
        sel = resolve_gpu_selection(gpu)
        indices = sel["indices_abs"]
        gpus_str = sel["gpus_str_for_cli"]
        gpu_name = ", ".join(sel["selected_names"]) if sel["selected_names"] else "Unknown GPU"
        vram = ", ".join(sel["selected_vram"]) if sel["selected_vram"] else "N/A"
        gpu_name_safe = slugify(gpu_name.replace(",", "-"))

        # Prepare per-template output dir
        output_dir = os.path.join(
            "/outputs", f"benchmark__{username}__{gpu_name_safe}__{template}__{now}"
        )
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        print(f"Changed directory to output folder: {output_dir}")

        # Build cfg for this template (anchors recomputed each time)
        generate_cfg(template)

        # GPU watcher (per run)
        gpu_watch_thread = None
        watch_log = os.path.join(output_dir, "mylogfile.log")
        watch_stop = Event()
        try:
            if gpu.count > 0:
                gpu_watch_thread = Thread(target=gpu.watch, kwargs={
                    "logfile": watch_log, "delay": 1.0, "dense": True,
                    "gpu": indices if indices else None,
                    "install_signal_handler": False, "stop_event": watch_stop,
                })
                gpu_watch_thread.daemon = True
                gpu_watch_thread.start()
                print(f"GPU watcher logging to {watch_log}")
            else:
                print("No GPUs detected; skipping GPU watch.")
        except Exception as e:
            print(f"Skipping GPU watch: {e}")

        # Train
        try:
            StopWatch.start("benchmark")
            dk = darknet_path()
            cmd = (
                f"{dk} detector -map -dont_show -nocolor "
                + (f"-gpus {gpus_str} " if gpus_str else "")
                + "train /workspace/LegoGears_v2/LegoGears.data "
                "/workspace/LegoGears_v2/LegoGears.cfg "
                "2>&1 | tee training_output.log"
            )
            print(f"[train] {cmd}")
            subprocess.call(cmd, shell=True)
            StopWatch.stop("benchmark")
        finally:
            # Stop watcher
            if gpu_watch_thread is not None:
                try:
                    watch_stop.set()
                    gpu.running = False
                    gpu_watch_thread.join(timeout=3)
                    print(f"GPU watcher stopped; log at {watch_log}")
                except Exception as e:
                    print(f"Error stopping GPU watcher: {e}")

        # Benchmark & environment metadata
        benchmark_result = StopWatch.get_benchmark()
        sysinfo = benchmark_result["sysinfo"]
        benchmark = benchmark_result["benchmark"]["benchmark"]
        cpu_name_safe = slugify(sysinfo["cpu"])

        print("Getting disk information")
        disk_info = get_disk_info()
        print("Running disk speed test")
        dd_write_speed, dd_read_speed = fio_seq_rw()

        env = summarize_env(indices=indices, training_log_path=os.path.join(output_dir, "training_output.log"))

        data = {
            "YOLO Template": template,
            "Benchmark Time (s)": benchmark["time"],
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
            "Write Speed": dd_write_speed,
            "Read Speed": dd_read_speed,
            "Working Dir": os.getenv("ACTUAL_PWD", "N/A"),
            "CUDA Version": env["cuda_version"],
            "cuDNN Version": env["cudnn_version"],
            "GPUs Used": env["num_gpus_used"],
            "Compute Capability": env["compute_caps_str"],
        }

        # Save CSV
        csv_name = f"benchmark__{username}__{gpu_name_safe}__{cpu_name_safe}__{template}__{now}.csv"
        with open(os.path.join(output_dir, csv_name), "w", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=data.keys())
            w.writeheader(); w.writerow(data)
        print(f"Benchmark results saved to {os.path.join(output_dir, csv_name)}")

        # Move weights from /workspace into this template's output folder
        for file in glob.glob("/workspace/LegoGears_v2/*weights"):
            shutil.move(file, output_dir)

        # Zip outputs (excluding *.weights)
        bundle_name = f"benchmark_bundle__{username}__{gpu_name_safe}__{cpu_name_safe}__{template}__{now}.zip"
        bundle_path = Path(output_dir) / bundle_name
        with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in Path(output_dir).rglob("*"):
                if not p.is_file(): continue
                if p.name == bundle_name: continue
                if p.name.lower().endswith(".weights"): continue
                z.write(p, arcname=p.relative_to(output_dir))
        print(f"Zipped outputs to: {bundle_path}")

    # ---- main flow: parse templates and run each sequentially ----
    templates = parse_templates_from_env()
    print(f"[run.py] YOLO templates selected: {templates}")
    for t in templates:
        print(f"\n===== Starting run for {t} =====")
        run_one(t)
        print(f"===== Finished run for {t} =====\n")

