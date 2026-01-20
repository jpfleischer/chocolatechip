# src/chocolatechip/model_training/coco_eval.py
from __future__ import annotations
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Dict, Tuple


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




