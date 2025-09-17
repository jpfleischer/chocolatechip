# LegoGearsFiles/hw_info.py
import os
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
        return []

def summarize_env(indices=None, training_log_path=None):
    """
    indices: the *absolute* GPU indices you decided to use (or None).
    training_log_path: optional path to parse cuDNN from Darknet log.

    Returns a dict with:
      - cuda_version
      - cudnn_version
      - num_gpus_used
      - compute_caps       (list)
      - compute_caps_str   (comma-joined)
    """
    cuda_ver = get_cuda_version()
    cudnn_ver = get_cudnn_version(training_log_path=training_log_path)
    caps = get_compute_caps(indices=indices)
    num_used = len(indices) if indices else (0 if not caps else 1)
    return {
        "cuda_version": cuda_ver,
        "cudnn_version": cudnn_ver,
        "num_gpus_used": num_used,
        "compute_caps": caps,
        "compute_caps_str": ", ".join(caps) if caps else "",
    }
