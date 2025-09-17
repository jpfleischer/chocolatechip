# LegoGearsFiles/hw_info.py
import os
import json
import re
import ctypes
import subprocess
from pathlib import Path

def _sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()

def get_cuda_version():
    # Prefer nvcc
    try:
        out = _sh("nvcc --version")
        m = re.search(r"release\s+(\d+\.\d+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    # Fallback: nvidia-smi banner
    try:
        out = _sh("nvidia-smi")
        m = re.search(r"CUDA Version:\s*([\d\.]+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "Unknown"

def get_cudnn_version(training_log_path=None):
    # Try cudnnGetVersion() from libcudnn
    lib = None
    for name in ("libcudnn.so",
                 "libcudnn.so.9",
                 "libcudnn.so.8",
                 "libcudnn_cnn_infer.so.9",
                 "libcudnn_cnn_infer.so.8"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            lib = None
    if lib:
        try:
            lib.cudnnGetVersion.restype = ctypes.c_size_t
            v = int(lib.cudnnGetVersion())
            major, minor, patch = v // 1000, (v % 1000) // 100, v % 100
            return f"{major}.{minor}.{patch} ({v})"
        except Exception:
            pass

    # Try headers (if dev package installed)
    for hdr in ("/usr/include/cudnn_version.h", "/usr/local/cuda/include/cudnn_version.h"):
        try:
            txt = Path(hdr).read_text(errors="ignore")
            ma = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", txt)
            mi = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", txt)
            pa = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", txt)
            if ma and mi and pa:
                return f"{ma.group(1)}.{mi.group(1)}.{pa.group(1)}"
        except Exception:
            pass

    # Optional: parse your training log (Darknet prints cuDNN)
    if training_log_path:
        try:
            txt = Path(training_log_path).read_text(errors="ignore")
            m = re.search(r"cuDNN version\s+([0-9]+)\s*\(v([0-9\.]+)\)", txt, re.IGNORECASE)
            if m:
                # e.g., "12010 (v8.9.0)"
                return f"{m.group(2)} ({m.group(1)})"
            m = re.search(r"cuDNN\s+v?([0-9\.]+)", txt, re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception:
            pass

    return "Unknown"

def get_compute_caps(indices=None):
    """
    Returns a list of compute capability strings for given absolute GPU indices.
    If indices is None, returns for all GPUs.
    """
    # First try nvidia-smi (fast path)
    try:
        out = _sh("nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader")
        rows = [r.strip() for r in out.splitlines() if r.strip()]
        table = {}
        for r in rows:
            parts = [p.strip() for p in r.split(",")]
            if len(parts) >= 2:
                table[int(parts[0])] = parts[1]
        if indices is None:
            return [table[i] for i in sorted(table)]
        return [table.get(i, "N/A") for i in indices]
    except Exception:
        pass

    # Fallback: NVML via pynvml
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        count = nvml.nvmlDeviceGetCount()
        want = indices if indices is not None else list(range(count))
        caps = []
        for i in want:
            if 0 <= i < count:
                h = nvml.nvmlDeviceGetHandleByIndex(i)
                major, minor = nvml.nvmlDeviceGetCudaComputeCapability(h)
                caps.append(f"{major}.{minor}")
            else:
                caps.append("N/A")
        nvml.nvmlShutdown()
        return caps
    except Exception:
        return []


def summarize_env(indices=None, training_log_path=None):
    cuda_ver = get_cuda_version()
    cudnn_ver = get_cudnn_version(training_log_path=training_log_path)
    caps = get_compute_caps(indices=indices)
    # Use all visible GPUs when indices is None
    num_used = len(indices) if indices is not None else len(caps)
    return {
        "cuda_version": cuda_ver,
        "cudnn_version": cudnn_ver,
        "num_gpus_used": num_used,
        "compute_caps": caps,
        "compute_caps_str": ", ".join(caps) if caps else "",
    }


def parse_cuda_visible_devices():
    """
    Returns (indices_or_None, env_is_set, numeric_mode).
    - If CUDA_VISIBLE_DEVICES is unset: (None, False, False)
    - If set to comma-separated integers: ([abs_indices], True, True)
    - If set to UUIDs or other non-integers: (None, True, False)
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not env:
        return None, False, False
    parts = [p.strip() for p in env.split(",") if p.strip()]
    ints = []
    for p in parts:
        try:
            ints.append(int(p))
        except ValueError:
            return None, True, False  # set, but not numeric
    return ints, True, True

def list_all_gpu_indices():
    """
    Returns a list of absolute GPU indices as reported by nvidia-smi.
    """
    out = _sh("nvidia-smi --query-gpu=index --format=csv,noheader")
    inds = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        # index is an int on each line
        try:
            inds.append(int(s.split(",")[0]))
        except ValueError:
            pass
    return inds


def gpu_inventory(cm_gpu):
    """
    Return (names, vram) without mutating cm_gpu._smi.
    """
    try:
        data = dict(cm_gpu.smi(output="json"))
        gpus = data["nvidia_smi_log"]["gpu"]
        if not isinstance(gpus, list):
            gpus = [gpus]
    except Exception:
        return [], []

    names = []
    vram  = []
    for g in gpus:
        names.append(g.get("product_name", "Unknown"))
        fb = g.get("fb_memory_usage") or {}
        vram.append(fb.get("total", "N/A"))
    return names, vram


def resolve_gpu_selection(cm_gpu):
    """
    Decide which GPUs to use and how to pass them to Darknet.
    Returns a dict with:
      indices_abs        -> list[int] (absolute indices)
      gpus_str_for_cli   -> str to use with '-gpus' (may be '')
      selected_names     -> list[str] names of selected GPUs
      selected_vram      -> list[str] VRAM of selected GPUs
      env_set            -> bool (CUDA_VISIBLE_DEVICES set?)
      env_numeric        -> bool (set and numeric?)
    Rules:
      - If CUDA_VISIBLE_DEVICES is numeric (e.g., "2,3"): use those absolute indices,
        and for Darknet pass "0,1,..." (relative to the visible set).
      - If env is unset: use all indices from nvidia-smi and pass them as absolute.
      - If env is set to UUIDs: we can’t map cleanly; fall back to all indices and pass absolute.
    """
    env_indices, env_set, env_numeric = parse_cuda_visible_devices()

    names, vram = gpu_inventory(cm_gpu)
    all_abs = list_all_gpu_indices()
    if not all_abs:
        # No GPUs visible
        return {
            "indices_abs": [],
            "gpus_str_for_cli": "",
            "selected_names": [],
            "selected_vram": [],
            "env_set": env_set,
            "env_numeric": env_numeric,
        }

    if env_set and env_numeric and env_indices:
        indices_abs = env_indices
        # When env is numeric, frameworks typically treat indices as re-mapped 0..N-1.
        gpus_str_for_cli = ",".join(str(i) for i in range(len(indices_abs)))
    else:
        # env unset or UUID/mixed -> use all absolute indices, pass as absolute
        indices_abs = all_abs
        gpus_str_for_cli = ",".join(str(i) for i in indices_abs)

    selected_names = [names[i] for i in indices_abs if i < len(names)]
    selected_vram  = [vram[i]  for i in indices_abs if i < len(vram)]

    return {
        "indices_abs": indices_abs,
        "gpus_str_for_cli": gpus_str_for_cli,
        "selected_names": selected_names,
        "selected_vram": selected_vram,
        "env_set": env_set,
        "env_numeric": env_numeric,
    }



def fio_seq_rw(test_file=None, block_size="1M", runtime=20, size="1G"):
    """
    Sequential write/read using fio.
    Returns (write_speed, read_speed) as strings like '1234.56 MiB/s'.
    If fio isn't available or errors occur, returns 'N/A' for that leg.
    """
    if test_file is None:
        test_file = os.path.join("/tmp", f"fio_test_{os.getpid()}")

    # WRITE
    try:
        write_cmd = (
            f"fio --name=seqwrite --ioengine=libaio --direct=1 --rw=write "
            f"--bs={block_size} --runtime={runtime} --time_based --size={size} "
            f"--filename={test_file} --output-format=json"
        )
        r = subprocess.run(write_cmd, shell=True, capture_output=True, text=True, check=True)
        write_output = json.loads(r.stdout)
        write_bw_kib = write_output["jobs"][0]["write"]["bw"]  # KiB/s
        write_speed = f"{write_bw_kib / 1024:.2f} MiB/s"
    except Exception:
        write_speed = "N/A"

    # READ
    try:
        read_cmd = (
            f"fio --name=seqread --ioengine=libaio --direct=1 --rw=read "
            f"--bs={block_size} --runtime={runtime} --time_based --size={size} "
            f"--filename={test_file} --output-format=json"
        )
        r = subprocess.run(read_cmd, shell=True, capture_output=True, text=True, check=True)
        read_output = json.loads(r.stdout)
        read_bw_kib = read_output["jobs"][0]["read"]["bw"]  # KiB/s
        read_speed = f"{read_bw_kib / 1024:.2f} MiB/s"
    except Exception:
        read_speed = "N/A"

    # Cleanup
    try:
        os.remove(test_file)
    except Exception:
        pass

    return write_speed, read_speed


def disk_benchmark_summary(**kwargs):
    """
    Convenience wrapper returning a dict with fio results.
    Usage: disk_benchmark_summary(block_size="1M", runtime=20, size="1G")
    """
    w, r = fio_seq_rw(**kwargs)
    return {"fio_write_mib_s": w, "fio_read_mib_s": r}


def human_readable_size(num_bytes):
    """
    Convert a size in bytes to a human-readable string.
    """
    try:
        num_bytes = float(num_bytes)
    except (ValueError, TypeError):
        return "N/A"
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PiB"


def get_disk_info():
    """
    Returns disk info for the drive containing the current working directory.

    Keys:
      - Disk Capacity
      - Disk Model

    If WINDOWS_HARD_DRIVE is set (on Windows/Git Bash path), use that.
    Otherwise, determine disk info using Linux utilities (df/lsblk).
    """
    try:
        # Windows/Git Bash path via env (your Makefile sets these for Docker)
        windows_hd = os.environ.get("WINDOWS_HARD_DRIVE")
        if windows_hd:
            windows_cap = os.environ.get("WINDOWS_HARD_DRIVE_CAPACITY", "N/A")
            windows_cap_hr = human_readable_size(windows_cap) if windows_cap != "N/A" else "N/A"
            return {
                "Disk Capacity": windows_cap_hr,
                "Disk Model": windows_hd
            }

        # Non-Windows: use df to find the backing device for CWD
        cwd = os.getcwd()
        df_cmd = f"df --output=source {cwd}"
        df_output = subprocess.check_output(df_cmd, shell=True, text=True).strip().splitlines()
        if len(df_output) < 2:
            raise ValueError("Could not determine device from df output")
        current_device = df_output[1].strip()

        # lsblk physical disks (NAME, TYPE, SIZE, MODEL)
        lsblk_cmd = "lsblk -d -o NAME,TYPE,SIZE,MODEL -n"
        lsblk_output = subprocess.check_output(lsblk_cmd, shell=True, text=True).strip()
        disk_info_list = [[], []]  # [non-matches, matches]

        for line in lsblk_output.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0]
            dev_type = parts[1]
            size = parts[2]
            model = " ".join(parts[3:])
            if dev_type != "disk":
                continue
            matches = name in current_device
            disk_info_list[1 if matches else 0].append({"size": size, "model": model})

        if disk_info_list[1]:
            disk_capacities = ", ".join(d["size"] for d in disk_info_list[1])
            disk_models = ", ".join(d["model"] for d in disk_info_list[1])
        elif disk_info_list[0]:
            disk_capacities = ", ".join(d["size"] for d in disk_info_list[0])
            disk_models = ", ".join(d["model"] for d in disk_info_list[0])
        else:
            disk_capacities = disk_models = "N/A"

    except Exception as e:
        print("Error retrieving disk info:", e)
        disk_capacities = disk_models = "N/A"

    return {
        "Disk Capacity": disk_capacities,
        "Disk Model": disk_models,
    }
