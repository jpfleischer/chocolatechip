# LegoGearsFiles/hw_info.py
import os
import json
import re
import ctypes
import shutil
import subprocess
from pathlib import Path

_BUS_RE = re.compile(
    r'^(?:(?P<domain>[0-9a-fA-F]{0,8}):)?(?P<bus>[0-9a-fA-F]{2}):(?P<device>[0-9a-fA-F]{2})\.(?P<func>[0-7])$'
)

def _canon_bus_forms(s: str) -> tuple[str, str]:
    """
    Return two canonical forms of a PCI bus id:
      - 8-zero domain form: '00000000:BB:DD.F'
      - 4-zero domain form: '0000:BB:DD.F'
    Accepts inputs like '65:00.0', '0000:65:00.0', or '00000000:65:00.0'.
    """
    if not s:
        return ("(unknown)", "(unknown)")
    s = s.strip().lower()
    m = _BUS_RE.match(s) or _BUS_RE.match("00000000:" + s)  # allow missing domain
    if not m:
        return (s, s)
    dom = m.group("domain") or "00000000"
    if len(dom) < 8:
        dom = dom.rjust(8, "0")
    bus = m.group("bus").rjust(2, "0")
    dev = m.group("device").rjust(2, "0")
    fn  = m.group("func")
    eight = f"{dom}:{bus}:{dev}.{fn}"
    four  = f"{dom[-4:]}:{bus}:{dev}.{fn}"
    return (eight, four)


def _load_cudart():
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11", "libcudart.so.10.2", "libcudart.so.10.1", "libcudart.so.10.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None

_cudart = _load_cudart()

def _cuda_check(code):
    if code != 0:
        raise RuntimeError(f"CUDA runtime error {code}")


def runtime_visible_devices():
    """
    Returns the CUDA runtime-visible devices for *this process* in logical order:
      [{'logical': 0, 'name': 'NVIDIA ...', 'bus_id': '00000000:bb:dd.f'}, ...]
    Names are resolved by matching PCI bus IDs against nvidia-smi (host view).
    """
    cudart = _cudart or _load_cudart()
    if cudart is None:
        return []

    # Runtime API signatures
    cudart.cudaGetDeviceCount.restype  = ctypes.c_int
    cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaDeviceGetPCIBusId.restype  = ctypes.c_int
    cudart.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

    # Count runtime-visible devices
    n = ctypes.c_int(0)
    _cuda_check(cudart.cudaGetDeviceCount(ctypes.byref(n)))

    # Build host-view index by bus id (with both 8-zero & 4-zero forms)
    smi_by_bus = smi_inventory_by_bus()

    devices = []
    for logical_idx in range(n.value):
        # PCI bus id from runtime
        buf = ctypes.create_string_buffer(64)
        rc  = cudart.cudaDeviceGetPCIBusId(buf, 64, logical_idx)
        raw = buf.value.decode().lower() if rc == 0 else "(unknown)"

        eight, four = _canon_bus_forms(raw)
        meta = smi_by_bus.get(eight) or smi_by_bus.get(four) or {}
        name = meta.get("name", "Unknown")

        # Use the 8-zero canonical string as our stored bus_id
        devices.append({
            "logical": logical_idx,
            "name": name,
            "bus_id": eight,
        })

    return devices


# --- replace your smi_inventory_by_bus() with this version ---
def smi_inventory_by_bus():
    """
    Returns a dict mapping BOTH canonical forms to metadata:
      {
        '00000000:65:00.0': {'smi_index': 3, 'name': 'NVIDIA ...', 'uuid': 'GPU-...'},
        '0000:65:00.0':     {... same dict ...}
      }
    """
    try:
        out = _sh("nvidia-smi --query-gpu=index,pci.bus_id,name,uuid --format=csv,noheader")
        table = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            idx  = int(parts[0])
            bus  = parts[1].lower()
            name = parts[2]
            uuid = parts[3]
            eight, four = _canon_bus_forms(bus)
            meta = {"smi_index": idx, "name": name, "uuid": uuid}
            # Register both keys to maximize hit rate
            table[eight] = meta
            table[four]  = meta
        return table
    except Exception:
        return {}


def runtime_to_smi_map():
    """
    Returns list of rows mapping runtime logical -> smi index by bus ID.
    [{'logical': 0, 'bus_id': '0000:bb:dd.f', 'name': '...', 'smi_index': 3}, ...]
    """
    vis = runtime_visible_devices()
    smi = smi_inventory_by_bus()
    rows = []
    for r in vis:
        bus = r["bus_id"].lower()
        match = smi.get(bus, {})
        rows.append({**r, "smi_index": match.get("smi_index", None)})
    return rows


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

    # NEW: prefer runtime-visible view for compute caps & names
    rt = runtime_visible_devices()
    if rt:
        # compute caps via nvidia-smi, but aligned by bus id → runtime order
        smi = smi_inventory_by_bus()
        caps = []
        for d in rt:
            entry = smi.get(d["bus_id"].lower())
            if entry:
                # nvidia-smi compute cap is X.Y; if missing fall back later
                try:
                    out = _sh(f"nvidia-smi --id={entry['smi_index']} --query-gpu=compute_cap --format=csv,noheader")
                    caps.append(out.strip())
                except Exception:
                    caps.append("N/A")
            else:
                caps.append("N/A")
        num_used = len(rt) if indices is None else len(indices)
    else:
        # Fallback to your old path
        caps = get_compute_caps(indices=indices)
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

    Darknet's -gpus argument always expects *logical indices in the CUDA runtime view*,
    i.e., the 0..N-1 order returned by cudaGetDeviceCount / cudaGetDeviceProperties
    for this process (after any CUDA_VISIBLE_DEVICES filtering).

    We therefore:
      - Inspect the runtime-visible devices (names + bus IDs) to define the index set.
      - Build a mapping to nvidia-smi indices (for reporting only).
      - Always pass "0,1,...,K-1" to Darknet for K selected devices.
    """
    # Runtime view (authoritative for Darknet)
    rt = runtime_visible_devices()  # [{'logical': i, 'name':..., 'bus_id':...}, ...]
    if not rt:
        # Fallback to zero devices
        return {
            "indices_abs": [],            # deprecated; retained for compatibility
            "gpus_str_for_cli": "",
            "selected_names": [],
            "selected_vram": [],
            "env_set": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
            "env_numeric": parse_cuda_visible_devices()[2],
            "runtime_devices": [],        # NEW: full runtime list
            "runtime_smi_map": [],        # NEW: runtime <-> smi mapping
        }

    # Optional: look up VRAM from cloudmesh (if available)
    try:
        names, vram = gpu_inventory(cm_gpu)
    except Exception:
        names, vram = ([], [])

    # Build Darknet argument: always 0..K-1
    k = len(rt)
    gpus_str_for_cli = ",".join(str(i) for i in range(k))

    # For reporting, map runtime logical -> nvidia-smi index (by PCI bus)
    rt_smi = runtime_to_smi_map()  # includes 'smi_index' if resolvable

    # Selected names from runtime (authoritative)
    selected_names = [d["name"] for d in rt]

    return {
        # Historically your code called these 'absolute', but they are runtime logical.
        "indices_abs": list(range(k)),
        "gpus_str_for_cli": gpus_str_for_cli,
        "selected_names": selected_names,
        "selected_vram": vram[:k] if vram else [],
        "env_set": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "env_numeric": parse_cuda_visible_devices()[2],
        "runtime_devices": rt,     # [{'logical', 'name', 'bus_id'}]
        "runtime_smi_map": rt_smi, # [{'logical','bus_id','name','smi_index'}]
    }


def fio_seq_rw(test_file=None, block_size="1M", runtime=5, size="32M"):
    """
    Sequential write/read using fio.
    Returns (write_speed, read_speed) as strings like '1234.56 MiB/s'.
    If fio isn't available or errors occur, returns 'N/A' for that leg.
    """
    # opt-out
    if os.environ.get("DISABLE_FIO"):
        return "N/A", "N/A"

    # find fio
    fio_bin = os.environ.get("FIO_BIN") or shutil.which("fio")
    if not fio_bin:
        return "N/A", "N/A"

    # choose a writable dir
    base = os.environ.get("DATA_ROOT") or "/workspace"
    if not os.path.isdir(base) or not os.access(base, os.W_OK):
        base = "/tmp"

    if test_file is None:
        test_file = os.path.join(base, f".fio_test_{os.getpid()}")

    def _run(cmd: list[str]) -> tuple[bool, dict]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, json.loads(r.stdout)
        except Exception:
            return False, {}

    # Common args (portable: psync + direct=1 + json)
    common = [
        fio_bin,
        "--ioengine=psync", "--direct=1",
        f"--bs={block_size}",
        f"--runtime={runtime}", "--time_based",
        f"--size={size}",
        f"--filename={test_file}",
        "--output-format=json",
        "--group_reporting=1",
    ]

    # WRITE
    ok_w, out_w = _run(common + ["--name=seqwrite", "--rw=write"])
    if ok_w:
        try:
            bw_kib = out_w["jobs"][0]["write"]["bw"]  # KiB/s
            write_speed = f"{bw_kib / 1024:.2f} MiB/s"
        except Exception:
            write_speed = "N/A"
    else:
        write_speed = "N/A"

    # READ
    ok_r, out_r = _run(common + ["--name=seqread", "--rw=read"])
    if ok_r:
        try:
            bw_kib = out_r["jobs"][0]["read"]["bw"]  # KiB/s
            read_speed = f"{bw_kib / 1024:.2f} MiB/s"
        except Exception:
            read_speed = "N/A"
    else:
        read_speed = "N/A"

    # Cleanup
    try:
        Path(test_file).unlink(missing_ok=True)
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
