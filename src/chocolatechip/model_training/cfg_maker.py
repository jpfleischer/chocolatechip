#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cfg_maker.py — Generate a Darknet .cfg from the yolov7-tiny template (downloaded in-memory),
ALWAYS recalculating anchors from your dataset labels, using a .data file.

Example:
python cfg_maker.py \
--template yolov4-tiny \
--data /workspace/LegoGears_v2/LegoGears.data \
--out myproject.cfg \
--width 224 --height 160 \
--batch-size 64 --subdivisions 8 \
--iterations 3000 --learning-rate 0.00261
"""

from __future__ import annotations
import argparse, re, sys, math, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


TEMPLATE_URLS = {
    "yolov7-tiny": "https://raw.githubusercontent.com/hank-ai/darknet/master/cfg/yolov7-tiny.cfg",
    "yolov4-tiny": "https://raw.githubusercontent.com/hank-ai/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov3-tiny": "https://raw.githubusercontent.com/hank-ai/darknet/master/cfg/yolov3-tiny.cfg",
}


def pick_template_url(name: str) -> str:
    name = name.strip().lower()
    if name not in TEMPLATE_URLS:
        sys.exit(f"Unsupported template '{name}'. Choose one of: {', '.join(TEMPLATE_URLS)}")
    return TEMPLATE_URLS[name]

KV_RE  = re.compile(r'^\s*([^#;\s=]+)\s*=\s*([^\s#;]+)')
SEC_RE = re.compile(r'^\s*\[([^\]]+)\]\s*$')

# -------------------- .data parsing --------------------

def parse_data_file(data_path: Path) -> Dict[str, str]:
    if not data_path.exists():
        sys.exit(f".data file not found: {data_path}")
    cfg: Dict[str, str] = {}
    for ln in data_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = KV_RE.match(ln)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            cfg[key] = val
    if "train" not in cfg:
        sys.exit("Missing 'train=' entry in .data file.")
    if "classes" not in cfg:
        sys.exit("Missing 'classes=' entry in .data file.")
    return cfg

# -------------------- IO helpers --------------------

def fetch_template_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "cfg_maker/1.3"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
    except HTTPError as e:
        sys.exit(f"HTTP error fetching template: {e.code} {e.reason}")
    except URLError as e:
        sys.exit(f"Network error fetching template: {e.reason}")
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        sys.exit("Failed to decode template as UTF-8")

def cfg_text_to_lines(txt: str) -> List[str]:
    return txt.splitlines()

def write_lines(p: Path, lines: List[str]) -> None:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

def find_section_ranges(lines: List[str]) -> List[Tuple[str,int,int]]:
    ranges: List[Tuple[str,int,int]] = []
    current_start = None
    current_name  = None
    for i, ln in enumerate(lines):
        m = SEC_RE.match(ln)
        if m:
            if current_start is not None:
                ranges.append((current_name, current_start, i))
            current_name  = m.group(1).strip().lower()
            current_start = i
    if current_start is not None:
        ranges.append((current_name, current_start, len(lines)))
    return ranges

def section_key_set(lines: List[str], sec_start: int, sec_end: int, key: str, value: str, insert_if_missing=True):
    key_l = key.lower()
    for i in range(sec_start+1, sec_end):
        m = KV_RE.match(lines[i])
        if m and m.group(1).lower() == key_l:
            lines[i] = f"{key}={value}"
            return
    if insert_if_missing:
        lines.insert(sec_end, f"{key}={value}")

def section_key_get(lines: List[str], sec_start: int, sec_end: int, key: str) -> Optional[str]:
    key_l = key.lower()
    for i in range(sec_start+1, sec_end):
        m = KV_RE.match(lines[i])
        if m and m.group(1).lower() == key_l:
            return m.group(2)
    return None

def parse_csv_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]

# -------------------- Data & anchors --------------------

def read_train_list(train_list_path: Path) -> List[Path]:
    imgs: List[Path] = []
    for ln in train_list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            imgs.append(Path(ln))
    return imgs

def label_path_for_image(img_path: Path) -> Path:
    p = img_path
    stem = p.suffix and p.name[: -len(p.suffix)] or p.name
    return p.with_name(stem + ".txt")

def load_wh_from_labels(img_paths: List[Path], net_w: int, net_h: int, num_classes: int) -> Tuple[List[Tuple[float,float]], List[int], int]:
    wh_list: List[Tuple[float,float]] = []
    counters = [0] * num_classes
    num_boxes = 0

    for img in img_paths:
        lbl = label_path_for_image(img)
        if not lbl.exists():
            continue
        for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = re.split(r"[,\s]+", ln)
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                w_norm = float(parts[3]); h_norm = float(parts[4])
            except Exception:
                continue
            if w_norm <= 0 or h_norm <= 0:
                continue
            if cls >= num_classes and cls < 10000:
                need = cls + 1 - num_classes
                counters.extend([0] * need)
                num_classes = cls + 1
            if 0 <= cls < len(counters):
                counters[cls] += 1
            wh_list.append((w_norm * net_w, h_norm * net_h))
            num_boxes += 1

    if num_boxes == 0:
        sys.exit("No boxes found from labels. Check .data 'train' file and label files.")
    return wh_list, counters, num_boxes

def iou_distance(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    mw = a[0] if a[0] < b[0] else b[0]
    mh = a[1] if a[1] < b[1] else b[1]
    inter = mw * mh
    un = a[0]*a[1] + b[0]*b[1] - inter
    iou = inter / un if un > 0 else 0.0
    return 1.0 - iou

def kmeans_iou(wh: List[Tuple[float,float]], k: int, max_iter: int = 1000) -> List[Tuple[float,float]]:
    # deterministic seed as requested
    random.seed(9001)

    if k <= 1:
        sys.exit("anchor_clusters must be > 1")
    pts = [(max(1.0, w), max(1.0, h)) for (w, h) in wh]
    if len(pts) < k:
        sys.exit(f"Not enough boxes ({len(pts)}) for k={k}. Reduce --anchor-clusters or provide more data.")
    centers = random.sample(pts, k)
    assignments = [0] * len(pts)

    def closest_center(p):
        best_j, best_d = 0, iou_distance(p, centers[0])
        for j in range(1, k):
            d = iou_distance(p, centers[j])
            if d < best_d:
                best_d, best_j = d, j
        return best_j

    def expectation():
        converged = True
        for i, p in enumerate(pts):
            j = closest_center(p)
            if j != assignments[i]:
                converged = False
            assignments[i] = j
        return converged

    def maximization():
        old = centers[:]
        counts = [0] * k
        sums = [(0.0, 0.0) for _ in range(k)]
        for i, p in enumerate(pts):
            j = assignments[i]
            counts[j] += 1
            sums[j] = (sums[j][0] + p[0], sums[j][1] + p[1])
        new = []
        for j in range(k):
            if counts[j] > 0:
                new.append((sums[j][0]/counts[j], sums[j][1]/counts[j]))
            else:
                new.append(old[j])  # keep old center if empty
        return new

    for _ in range(max_iter):
        if expectation():
            break
        centers = maximization()

    centers.sort(key=lambda x: x[0]*x[1])                 # small -> large
    centers = [(round(w), round(h)) for (w, h) in centers] # round like DarkMark
    return centers

def avg_iou_against_anchors(wh: List[Tuple[float,float]], anchors: List[Tuple[float,float]]) -> float:
    total = 0.0
    count = 0
    for (bw, bh) in wh:
        best = 0.0
        for (aw, ah) in anchors:
            mw = bw if bw < aw else aw
            mh = bh if bh < ah else ah
            inter = mw * mh
            un = bw*bh + aw*ah - inter
            iou = inter / un if un > 0 else 0.0
            if iou > best:
                best = iou
        if 0.0 < best < 1.0:
            total += best
            count += 1
    return (100.0 * total / count) if count else 0.0

def group_masks(num_anchors: int, num_heads: int) -> List[List[int]]:
    if num_heads <= 0:
        sys.exit("internal error: num_heads <= 0")
    if num_anchors >= num_heads * 3:
        masks, idx = [], 0
        for _ in range(num_heads):
            masks.append([idx, idx+1, idx+2]); idx += 3
        return masks
    base = num_anchors // num_heads
    rem  = num_anchors %  num_heads
    masks, idx = [], 0
    for h in range(num_heads):
        take = base + (1 if h < rem else 0)
        masks.append(list(range(idx, idx+take)))
        idx += take
    return masks

# -------------------- CFG transform --------------------

def transform_cfg_from_text(template_text: str, *,
                            template_name: str,
                            classes: int,
                            width: int,
                            height: int,
                            batch_size: int,
                            subdivisions: int,
                            iterations: int,
                            learning_rate: float,
                            saturation: float,
                            exposure: float,
                            hue: float,
                            flip: int,
                            angle: int,
                            mosaic: int,
                            cutmix: int,
                            mixup: int,
                            write_counters_per_class: bool,
                            anchors_wh: List[Tuple[float,float]],
                            counters_per_class: List[int]) -> List[str]:

    lines = cfg_text_to_lines(template_text)
    secs  = find_section_ranges(lines)

    # ---- patch [net] ----
    net_idx = next((i for i, (nm, _, _) in enumerate(secs) if nm == "net"), None)
    if net_idx is None:
        sys.exit("No [net] section in template")
    _, ns, ne = secs[net_idx]

    def set_net(k, v): section_key_set(lines, ns, ne, k, str(v))

    set_net("width", width)
    set_net("height", height)
    set_net("batch", batch_size)
    set_net("subdivisions", subdivisions)
    set_net("learning_rate", f"{learning_rate:.6f}")
    set_net("max_batches", iterations)
    set_net("saturation", f"{saturation:.6f}")
    set_net("exposure",   f"{exposure:.6f}")
    set_net("hue",        f"{hue:.6f}")
    set_net("flip",       flip)
    set_net("angle",      angle)
    set_net("mosaic",     mosaic)
    set_net("cutmix",     cutmix)
    set_net("mixup",      mixup)

    # match DarkMark step scheduling: 80% and 90% of max_batches
    policy = section_key_get(lines, ns, ne, "policy")
    if policy and policy.lower() == "steps":
        s1 = int(math.floor(0.80 * iterations))
        s2 = int(math.floor(0.90 * iterations))
        section_key_set(lines, ns, ne, "steps", f"{s1},{s2}")
        section_key_set(lines, ns, ne, "scales", ".1,.1")

    if write_counters_per_class:
        section_key_set(lines, ns, ne, "counters_per_class", ", ".join(str(x) for x in counters_per_class))

    # Recompute sections after [net] edits so indexes are fresh
    secs = find_section_ranges(lines)

    # ---- compute anchors & masks ----
    yolo_idxs = [i for i, (nm, _, _) in enumerate(secs) if nm == "yolo"]
    if not yolo_idxs:
        sys.exit("Template has no [yolo] sections")
    num_heads = len(yolo_idxs)

    # Decide anchor count by template (no CLI)
    if template_name in ("yolov4-tiny", "yolov3-tiny"):
        k = 6   # 2 heads * 3 anchors
    elif template_name == "yolov7-tiny":
        k = 9   # 3 heads * 3 anchors
    else:
        # conservative fallback
        k = max(3 * num_heads, 6)

    anchors = kmeans_iou(anchors_wh, k)
    avg_iou = avg_iou_against_anchors(anchors_wh, anchors)

    # Sort small->large and group in triplets
    anchors.sort(key=lambda x: x[0] * x[1])
    groups = [list(range(3*g, min(3*g+3, len(anchors)))) for g in range((len(anchors)+2)//3)]
    # pad last group if needed (rare)
    if groups and len(groups[-1]) < 3:
        last = groups[-1]
        while len(last) < 3:
            last.append(last[-1])

    # Map groups to heads so the FIRST [yolo] gets the LARGEST anchors (DarkMark-style)
    if template_name in ("yolov4-tiny", "yolov3-tiny") and num_heads == 2:
        # groups: 0=small, 1=large
        group_order = [1, 0]  # head0->large, head1->small
    elif template_name == "yolov7-tiny" and num_heads == 3:
        # groups: 0=small,1=mid,2=large
        group_order = [2, 1, 0]  # head0->large, head1->mid, head2->small
    else:
        # generic: largest to first, then descending
        group_order = list(reversed(range(min(num_heads, len(groups)))))

    anchors_csv = ", ".join(f"{int(w)}, {int(h)}" for (w, h) in anchors)
    total_num = len(anchors)

    # Helper to read a key in a section block
    def get_key_in_block(sec_start: int, sec_end: int, key: str) -> Optional[str]:
        key_l = key.lower()
        for i in range(sec_start + 1, sec_end):
            m = KV_RE.match(lines[i])
            if m and m.group(1).lower() == key_l:
                return m.group(2)
        return None

    # Write heads in file order (top→bottom)
    for head_idx, yi in enumerate(yolo_idxs):
        _, ys, ye = secs[yi]

        # mask for this head from the chosen group ordering
        if head_idx >= len(group_order):
            mask_ids = groups[-1]  # fallback
        else:
            mask_ids = groups[group_order[head_idx]]

        section_key_set(lines, ys, ye, "classes", str(classes))
        section_key_set(lines, ys, ye, "mask", ",".join(str(i) for i in mask_ids))
        section_key_set(lines, ys, ye, "anchors", anchors_csv)
        section_key_set(lines, ys, ye, "num", str(total_num))

        expected_filters = len(mask_ids) * (classes + 5)

        # Find and update the preceding 1x1 conv
        target = None
        conv_idx = yi - 1
        while conv_idx >= 0:
            cnm, cs, ce = secs[conv_idx]
            if cnm == "convolutional":
                sz = get_key_in_block(cs, ce, "size")
                if sz and sz.isdigit() and int(sz) == 1:
                    target = (cs, ce)
                    break
                if target is None:
                    target = (cs, ce)
            conv_idx -= 1
        if target is None:
            sys.exit("No [convolutional] found before [yolo]")

        cs, ce = target
        replaced = False
        for i in range(cs + 1, ce):
            m = KV_RE.match(lines[i])
            if m and m.group(1).lower() == "filters":
                lines[i] = f"filters={expected_filters}"
                replaced = True
                break
        if not replaced:
            lines.insert(ce, f"filters={expected_filters}")

        # refresh section indices after edits
        secs = find_section_ranges(lines)

    print(f"[anchors] template={template_name}, k={k}, avg IoU ≈ {avg_iou:.2f}%")
    print(f"[class balance] counters_per_class = {counters_per_class}")
    return lines


def generate_cfg_file(
    *,
    template: str,
    data_path: str,
    out_path: str,
    width: int = 416,
    height: int = 416,
    batch_size: int = 64,
    subdivisions: int = 2,
    iterations: int = 20000,
    learning_rate: float = 0.001,
    saturation: float = 1.5,
    exposure: float = 1.5,
    hue: float = 0.1,
    flip: int = 0,
    angle: int = 0,
    mosaic: int = 0,
    cutmix: int = 0,
    mixup: int = 0,
    write_counters_per_class: bool = False,
    anchor_clusters: int | None = None,
) -> str:
    """Programmatic wrapper that mirrors the CLI behavior."""
    # Resolve template text
    template_text = fetch_template_text(pick_template_url(template))

    # Default k per template if not provided
    if anchor_clusters is None:
        if template == "yolov7-tiny":
            anchor_clusters = 9
        elif template in ("yolov4-tiny", "yolov3-tiny"):
            anchor_clusters = 6

    # Parse .data and gather boxes
    data_path_p = Path(data_path)
    data_cfg = parse_data_file(data_path_p)
    classes = int(data_cfg["classes"])

    train_list_path = Path(data_cfg["train"])
    if not train_list_path.exists():
        sys.exit(f"train list not found: {train_list_path}")
    img_paths = read_train_list(train_list_path)
    if not img_paths:
        sys.exit("No images found in train list from .data.")

    anchors_wh, counters_per_class, _ = load_wh_from_labels(
        img_paths, width, height, classes
    )

    # Transform and write
    cfg_lines = transform_cfg_from_text(
        template_text,
        template_name=template,
        classes=classes,
        width=width, height=height,
        batch_size=batch_size, subdivisions=subdivisions,
        iterations=iterations, learning_rate=learning_rate,
        saturation=saturation, exposure=exposure, hue=hue,
        flip=flip, angle=angle, mosaic=mosaic, cutmix=cutmix, mixup=mixup,
        write_counters_per_class=write_counters_per_class,
        anchors_wh=anchors_wh, counters_per_class=counters_per_class,
    )

    out_p = Path(out_path)
    write_lines(out_p, cfg_lines)
    print(f"Wrote: {out_p}")
    return str(out_p)


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate a Darknet .cfg from an online template (yolov7-tiny or yolov4-tiny), recalculating anchors from a .data file."
    )
    ap.add_argument("--data", required=True, help="path to .data file (must include 'classes=' and 'train=')")
    ap.add_argument("--out", required=True, help="output .cfg path")

    # [net] + training knobs
    ap.add_argument("--width", type=int, default=416)
    ap.add_argument("--height", type=int, default=416)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--subdivisions", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=20000, help="Darknet max_batches")
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--saturation", type=float, default=1.5)
    ap.add_argument("--exposure",  type=float, default=1.5)
    ap.add_argument("--hue",       type=float, default=0.1)
    ap.add_argument("--flip",      type=int,   default=0)
    ap.add_argument("--angle",     type=int,   default=0)
    ap.add_argument("--mosaic",    type=int,   default=0)
    ap.add_argument("--cutmix",    type=int,   default=0)
    ap.add_argument("--mixup",     type=int,   default=0)
    ap.add_argument("--write-counters-per-class", action="store_true",
                    help="also write counters_per_class=<csv> into [net]")

    ap.add_argument("--template", choices=["yolov7-tiny", "yolov4-tiny", "yolov3-tiny"], default="yolov7-tiny",
                    help="which Darknet template to start from")

    # anchors
    ap.add_argument("--anchor-clusters", type=int, default=None,
                    help="total anchors to compute (defaults: 9 for yolov7-tiny, 6 for yolov4-tiny)")

    args = ap.parse_args()

    # Validate img size
    if args.width < 32 or args.height < 32 or (args.width % 32) or (args.height % 32):
        sys.exit("width and height must be multiples of 32 and >= 32.")

    # Decide template + default anchor count
    template_name = args.template
    template_text = fetch_template_text(pick_template_url(template_name))
    anchor_clusters = 9 if (args.anchor_clusters is None and template_name == "yolov7-tiny") else \
                      6 if (args.anchor_clusters is None and template_name in ("yolov4-tiny", "yolov3-tiny")) else \
                      args.anchor_clusters

    if anchor_clusters is None or anchor_clusters <= 1:
        sys.exit("--anchor-clusters must be > 1 (or omit it to use the template-based default)")

    # Parse .data and gather boxes
    data_path = Path(args.data)
    data_cfg = parse_data_file(data_path)

    try:
        classes = int(data_cfg["classes"])
    except Exception:
        sys.exit("Invalid 'classes=' value in .data file (must be integer).")

    train_list_path = Path(data_cfg["train"])
    if not train_list_path.exists():
        sys.exit(f"train list not found: {train_list_path}")

    img_paths = read_train_list(train_list_path)
    if not img_paths:
        sys.exit("No images found in train list from .data.")

    anchors_wh, counters_per_class, _ = load_wh_from_labels(
        img_paths, args.width, args.height, classes
    )

    # Transform + write
    out_cfg = Path(args.out)
    cfg_lines = transform_cfg_from_text(
        template_text,
        template_name=template_name,   # NEW
        classes=classes,
        width=args.width, height=args.height,
        batch_size=args.batch_size, subdivisions=args.subdivisions,
        iterations=args.iterations, learning_rate=args.learning_rate,
        saturation=args.saturation, exposure=args.exposure, hue=args.hue,
        flip=args.flip, angle=args.angle, mosaic=args.mosaic, cutmix=args.cutmix, mixup=args.mixup,
        write_counters_per_class=args.write_counters_per_class,
        anchors_wh=anchors_wh, counters_per_class=counters_per_class,
    )

    write_lines(out_cfg, cfg_lines)
    print(f"Wrote: {out_cfg}")

if __name__ == "__main__":
    main()
