from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Literal
from pathlib import Path
import json
from statistics import mean

CsvStyle = Literal["generic", "per-class", "none"]

def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def _load_coco(gt_json: str, det_json: str):
    gt = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    dets = json.loads(Path(det_json).read_text(encoding="utf-8"))

    # category id ↔ name
    cat_name_by_id = {c["id"]: c["name"] for c in gt["categories"]}
    cat_ids_sorted = sorted(cat_name_by_id.keys())
    class_names = [cat_name_by_id[cid] for cid in cat_ids_sorted]
    idx_by_cat_id = {cid: i for i, cid in enumerate(cat_ids_sorted)}

    # images
    img_ids = {im["id"] for im in gt["images"]}

    # GT per image
    gts_by_img: Dict[int, List[Dict[str, Any]]] = {iid: [] for iid in img_ids}
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        gts_by_img.setdefault(ann["image_id"], []).append({
            "bbox": ann["bbox"],  # [x,y,w,h]
            "cat_id": ann["category_id"],
        })

    # dets per image (keep all; we’ll threshold later)
    dets_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for d in dets:
        iid = d["image_id"]
        dets_by_img.setdefault(iid, []).append({
            "bbox": d["bbox"],
            "cat_id": d["category_id"],
            "score": d.get("score", 0.0),
        })

    return class_names, idx_by_cat_id, gts_by_img, dets_by_img

def compute_confusion_from_coco(
    gt_json: str,
    det_json: str,
    iou_thresh: float = 0.50,
    conf_thresh: float = 0.50,
    *,
    # new options:
    csv_style: CsvStyle = "generic",                      # "generic" (recommended), "per-class", or "none"
    write_json_path: Optional[str] = None,                # if set, write a JSON sidecar with full details
    json_indent: int = 2                                  # indentation for the JSON file (if written)
) -> Dict[str, Any]:
    """
    Confusion at an operating point:
      - Only detections with score >= conf_thresh are considered.
      - Greedy one-to-one matching by highest IoU (category-agnostic).
      - If classes match: TP[class]++.
      - If classes differ: FP[pred_class]++ and FN[true_class]++ (misclassification).
      - Unmatched detections: FP[pred_class]++.
      - Unmatched GTs: FN[true_class]++.

    Returns a dict with:
      - params, classes, matrix, per_class, totals, micro, macro
      - csv_cols: stable CSV columns depending on csv_style
      - json_text: the full JSON payload (string) if write_json_path is None
    """
    (class_names, idx_by_cat_id, gts_by_img, dets_by_img) = _load_coco(gt_json, det_json)
    K = len(class_names)

    # integer matrix [pred, gt], class-to-class
    matrix = [[0 for _ in range(K)] for _ in range(K)]
    TP = [0]*K; FP = [0]*K; FN = [0]*K
    GT_total = [0]*K

    # count GT per class for later recall/FN
    for gts in gts_by_img.values():
        for g in gts:
            GT_total[idx_by_cat_id[g["cat_id"]]] += 1

    for iid, gts in gts_by_img.items():
        preds = [d for d in dets_by_img.get(iid, []) if d["score"] >= conf_thresh]
        if not gts and not preds:
            continue

        used_gt = [False]*len(gts)
        used_pr = [False]*len(preds)

        # precompute IoUs
        iou_pairs: List[Tuple[float, int, int]] = []
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                iou = _iou_xywh(p["bbox"], g["bbox"])
                if iou >= iou_thresh:
                    iou_pairs.append((iou, pi, gi))
        # greedy assignment by IoU desc
        iou_pairs.sort(key=lambda t: t[0], reverse=True)

        for iou, pi, gi in iou_pairs:
            if used_pr[pi] or used_gt[gi]:
                continue
            used_pr[pi] = True
            used_gt[gi] = True
            pcls = idx_by_cat_id[preds[pi]["cat_id"]]
            gcls = idx_by_cat_id[gts[gi]["cat_id"]]
            if pcls == gcls:
                TP[pcls] += 1
                matrix[pcls][gcls] += 1
            else:
                # misclassification: penalize both sides
                FP[pcls] += 1
                FN[gcls] += 1
                matrix[pcls][gcls] += 1

        # leftover preds: FP
        for pi, p in enumerate(preds):
            if not used_pr[pi]:
                FP[idx_by_cat_id[p["cat_id"]]] += 1
        # leftover gts: FN
        for gi, g in enumerate(gts):
            if not used_gt[gi]:
                FN[idx_by_cat_id[g["cat_id"]]] += 1

    # per-class metrics
    per_class = []
    for i, name in enumerate(class_names):
        tp, fp, fn = TP[i], FP[i], FN[i]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        
        
        # NEW: Jaccard / IoU from confusion counts
        denom = tp + fp + fn
        jacc = tp / denom if denom > 0 else 0.0
        # or, equivalently: jacc = f1 / (2 - f1) if denom > 0

        per_class.append({
            "class": name,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "jaccard": jacc,
            "GT_total": GT_total[i],
        })

    # totals + micro/macro
    TP_sum, FP_sum, FN_sum = sum(TP), sum(FP), sum(FN)
    def _safe(n, d): return (n / d) if d else None

    micro_prec = _safe(TP_sum, TP_sum + FP_sum)
    micro_rec  = _safe(TP_sum, TP_sum + FN_sum)
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) \
                 if (micro_prec and micro_rec and (micro_prec + micro_rec)) else None

    # NEW: micro Jaccard
    micro_jacc = _safe(TP_sum, TP_sum + FP_sum + FN_sum)

    macro_prec = mean([c["precision"] for c in per_class]) if per_class else None
    macro_rec  = mean([c["recall"]    for c in per_class]) if per_class else None
    macro_f1   = mean([c["f1"]        for c in per_class]) if per_class else None
    # NEW: macro Jaccard
    macro_jacc = mean([c["jaccard"]   for c in per_class]) if per_class else None

    # CSV columns (dataset-agnostic by default)
    csv_cols: Dict[str, Any] = {}
    if csv_style == "generic":
        csv_cols = {
            "CM_IoU_Thresh": iou_thresh,
            "CM_Conf_Thresh": conf_thresh,
            "CM_Classes": len(per_class),
            "CM_TotalTP": TP_sum,
            "CM_TotalFP": FP_sum,
            "CM_TotalFN": FN_sum,
            "CM_MicroPrecision": micro_prec,
            "CM_MicroRecall": micro_rec,
            "CM_MicroF1": micro_f1,
            "CM_MacroPrecision": macro_prec,
            "CM_MacroRecall": macro_rec,
            "CM_MacroF1": macro_f1,
        }
    elif csv_style == "per-class":
        # original behavior (not recommended for multi-dataset aggregation)
        for i, name in enumerate(class_names):
            key = name.replace(" ", "_")
            csv_cols[f"CM_TP_{key}"]   = TP[i]
            csv_cols[f"CM_FP_{key}"]   = FP[i]
            csv_cols[f"CM_FN_{key}"]   = FN[i]
            csv_cols[f"CM_Prec_{key}"] = round(per_class[i]["precision"], 4)
            csv_cols[f"CM_Rec_{key}"]  = round(per_class[i]["recall"], 4)
            csv_cols[f"CM_F1_{key}"]   = round(per_class[i]["f1"], 4)
    # elif "none": leave empty

    payload = {
        "params": {"iou_thresh": iou_thresh, "conf_thresh": conf_thresh},
        "classes": class_names,
        "matrix": matrix,
        "per_class": per_class,
        "totals": {"TP": TP_sum, "FP": FP_sum, "FN": FN_sum},
        "micro": {
            "precision": micro_prec,
            "recall": micro_rec,
            "f1": micro_f1,
            "jaccard": micro_jacc,
        },
        "macro": {
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": macro_f1,
            "jaccard": macro_jacc,
        },
        "csv_cols": csv_cols,
    }

    if write_json_path:
        Path(write_json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=json_indent), encoding="utf-8")
        payload["json_path"] = write_json_path
        payload["json_text"] = None
    else:
        payload["json_path"] = None
        payload["json_text"] = json.dumps(payload, ensure_ascii=False, indent=json_indent)

    return payload
