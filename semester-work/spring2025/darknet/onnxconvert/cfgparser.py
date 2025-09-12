#!/usr/bin/env python3
"""
Parse a Darknet YOLO .cfg and emit:
  • NUM_CLASSES
  • INPUT_SHAPE
  • LAYER_FACTORS
  • SCALES
  • ANCHORS
Usage:
  python cfgparser.py path/to/yolov4.cfg
"""
import sys
import re
from pathlib import Path

def parse_darknet_cfg(cfg_path: Path):
    # 1) Read and clean lines
    lines = []
    with open(cfg_path, 'r') as f:
        for raw in f:
            l = raw.strip()
            if not l or l.startswith('#'):
                continue
            lines.append(l)

    # 2) Parse the [net] block for width, height, channels
    if '[net]' not in lines:
        raise ValueError("No [net] section in cfg")
    idx_net = lines.index('[net]')
    width = height = channels = None
    for ln in lines[idx_net+1:]:
        if ln.startswith('['):
            break
        if ln.startswith('width='):
            width = int(ln.split('=',1)[1])
        elif ln.startswith('height='):
            height = int(ln.split('=',1)[1])
        elif ln.startswith('channels='):
            channels = int(ln.split('=',1)[1])
    if None in (width, height, channels):
        raise ValueError("Failed to find width, height or channels in [net]")
    input_shape = (channels, height, width)

    # 3) Parse each [yolo] block for mask, anchors, scale_x_y, classes
    anchors_per_head = []
    scales_per_head  = []
    classes_list     = []
    for i, ln in enumerate(lines):
        if ln == '[yolo]':
            mask = anc = scale = cls = None
            j = i + 1
            while j < len(lines) and not lines[j].startswith('['):
                cur = lines[j]
                if cur.startswith('mask'):
                    mask = cur.split('=',1)[1].strip()
                elif cur.startswith('anchors'):
                    anc = cur.split('=',1)[1].strip()
                elif cur.startswith('scale_x_y'):
                    scale = cur.split('=',1)[1].strip()
                elif cur.startswith('classes'):
                    cls = int(cur.split('=',1)[1].strip())
                j += 1
            if None in (mask, anc, scale, cls):
                raise ValueError(f"Missing mask/anchors/scale/classes in [yolo] at line {i}")
            masks = [int(x) for x in mask.split(',')]
            vals  = [int(x) for x in re.findall(r'\d+', anc)]
            group = []
            for m in masks:
                group.extend(vals[2*m:2*m+2])
            anchors_per_head.append(group)
            scales_per_head.append(float(scale))
            classes_list.append(cls)

    # assume all heads use same classes
    num_classes = classes_list[0]

    # 4) Compute LAYER_FACTORS (strides) for 3 heads at 1/8,1/16,1/32
    fm8  = height // 8
    fm16 = height // 16
    fm32 = height // 32
    factors = [height // fm8, height // fm16, height // fm32]

    return num_classes, input_shape, anchors_per_head, scales_per_head, factors

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/yolo.cfg", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    if not cfg_path.is_file():
        print(f"Error: {cfg_path} not found", file=sys.stderr)
        sys.exit(1)

    num_classes, inp_shape, anchors, scales, layer_factors = parse_darknet_cfg(cfg_path)

    print(f"NUM_CLASSES  = {num_classes}")
    print(f"INPUT_SHAPE  = {inp_shape}")
    print(f"LAYER_FACTORS= {layer_factors}")
    print(f"SCALES       = {scales}")
    print("ANCHORS      = [")
    for grp in anchors:
        print(f"    {grp},")
    print("]")

if __name__ == "__main__":
    main()
