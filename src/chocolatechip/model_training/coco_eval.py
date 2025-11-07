# src/chocolatechip/model_training/coco_eval.py
from __future__ import annotations
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json, os
import imagesize


def coco_eval_bbox(gt_json: str, det_json: str) -> Dict[str, object]:
    """
    Evaluate detections (COCO bbox) using pycocotools.
    Returns a dict with AP, AP50, AP75 (as percentages) and per-IoU AP curve.
    """
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(det_json)

    eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    # Headline metrics (COCOeval puts them in stats array)
    # stats indexes: [0]=AP .50:.95, [1]=AP50, [2]=AP75, [3]=AP_small, [4]=AP_medium, [5]=AP_large
    ap      = float(eval.stats[0]) * 100.0
    ap50    = float(eval.stats[1]) * 100.0
    ap75    = float(eval.stats[2]) * 100.0

    # Build AP per IoU (averaged over recall, classes, area=all, maxDets=100)
    # eval.eval['precision'] shape = [T, R, K, A, M]
    # T=len(thrs), R=len(recalls), K=classes, A=areas, M=maxDets configs
    P = eval.eval['precision']  # T x R x K x A x M
    thrs = eval.params.iouThrs   # array of IoU thresholds (0.50 -> 0.95 step 0.05)

    per_iou: List[Tuple[float, float]] = []
    if P is not None:
        for t_idx, thr in enumerate(thrs):
            pi = P[t_idx]  # R x K x A x M
            # mean over recall, classes, area, maxDets ignoring -1
            valid = pi[pi > -1]
            per_iou_ap = float(valid.mean()) * 100.0 if valid.size else float('nan')
            per_iou.append((float(thr), per_iou_ap))

    return {
        "AP": ap,
        "AP50": ap50,
        "AP75": ap75,
        "AP_per_IoU": per_iou,  # list of (iou, AP%) for the 10 thresholds
    }



def _image_size(path: str) -> Tuple[int, int]:
    w, h = imagesize.get(path)
    if not w or not h:
        raise RuntimeError(f"Unable to read image size for: {path}")
    return int(w), int(h)


def _guess_label_for_image(img_path: str) -> str | None:
    p = Path(img_path)
    # 1) same directory, same stem, .txt
    cand1 = str(p.with_suffix(".txt"))
    if os.path.isfile(cand1):
        return cand1
    # 2) replace .../images/... -> .../labels/... and swap extension
    parts = list(p.parts)
    if "images" in parts:
        i = parts.index("images")
        parts[i] = "labels"
        q = Path(*parts).with_suffix(".txt")
        if q.is_file():
            return str(q)
    return None

def build_coco_gt_from_yolo(val_list_path: str, class_names: List[str], out_json: str) -> str:
    """
    Create a COCO GT JSON from a YOLO-style validation list (one image path per line)
    and per-image YOLO label .txt files.
    Returns the path to the written JSON.
    """
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    categories: List[Dict[str, Any]] = [
        {"id": int(i), "name": str(n)} for i, n in enumerate(class_names)
    ]

    with open(val_list_path, "r", encoding="utf-8", errors="ignore") as f:
        img_paths = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    ann_id = 1
    for img_id, img_path in enumerate(img_paths, start=1):
        w, h = _image_size(img_path)
        images.append({
            "id": int(img_id),
            "file_name": str(Path(img_path).name),  # file_name compared by name in our exporters
            "width": int(w),
            "height": int(h),
        })

        lbl = _guess_label_for_image(img_path)
        if not lbl or not os.path.isfile(lbl):
            continue
        with open(lbl, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                # YOLO normalized cx,cy,w,h -> COCO x,y,w,h in pixels
                x = (cx - bw/2.0) * w
                y = (cy - bh/2.0) * h
                ww = bw * w
                hh = bh * h
                annotations.append({
                    "id": int(ann_id),
                    "image_id": int(img_id),
                    "category_id": int(cls),
                    "bbox": [float(x), float(y), float(ww), float(hh)],
                    "area": float(ww * hh),
                    "iscrowd": 0,
                })
                ann_id += 1

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    # COCO-compliant header + body
    coco = {
        "info": {
            "description": "YOLO TXT → COCO GT",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # Ensure each category has supercategory (not required, but many tools expect it)
    for c in coco["categories"]:
        c.setdefault("supercategory", "none")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return out_json

