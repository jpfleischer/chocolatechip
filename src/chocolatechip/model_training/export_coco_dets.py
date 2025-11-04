# src/chocolatechip/model_training/export_coco_dets.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import argparse, json, sys, subprocess, os

# ---------- Common helpers ----------

def _load_gt_index(ann_json: str):
    """Return (img_id_by_name, cat_id_by_name) maps from COCO GT JSON."""
    with open(ann_json, "r", encoding="utf-8") as f:
        gt = json.load(f)
    img_id_by_name = {Path(im["file_name"]).name: im["id"] for im in gt["images"]}
    cat_id_by_name = {c["name"]: c["id"] for c in gt["categories"]}
    return img_id_by_name, cat_id_by_name

def _read_images_list(images_txt: str | None, ann_json: str) -> List[str]:
    """If a txt list is provided, use it; else use file_name from GT JSON."""
    if images_txt and os.path.isfile(images_txt):
        with open(images_txt, "r", encoding="utf-8", errors="ignore") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    with open(ann_json, "r", encoding="utf-8") as f:
        gt = json.load(f)
    # return as paths; if GT has relative names, caller must ensure cwd makes sense
    return [im["file_name"] for im in gt["images"]]

# ---------- Ultralytics (.pt) → COCO results JSON ----------

def export_ultra_detections(
    *,
    weights: str,
    ann_json: str,
    out_json: str,
    images_txt: str | None = None,
    conf: float = 0.001,
    iou: float = 0.6,
    imgsz: int | tuple[int, int] | None = None,
    device: str | int | list[int] | None = None,
    batch: int | None = 16,
) -> None:
    """Run an Ultralytics model on each image and write COCO-format det JSON."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[ultra] Ultralytics not installed: {e}", file=sys.stderr)
        raise

    img_id_by_name, cat_id_by_name = _load_gt_index(ann_json)
    images = _read_images_list(images_txt, ann_json)

    model = YOLO(weights)
    # Build predict kwargs (avoid passing None)
    pred_kwargs: Dict[str, Any] = dict(
        source=images, stream=True, conf=conf, iou=iou, device=device, verbose=False, save=False
    )
    if batch is not None:
        pred_kwargs["batch"] = batch
    if imgsz is not None:
        pred_kwargs["imgsz"] = imgsz

    preds_iter = model.predict(**pred_kwargs)
    name_by_idx = model.names  # {idx: "class_name"}

    results: List[Dict[str, Any]] = []
    for img_path, r in zip(images, preds_iter):
        if r.boxes is None:
            continue
        fname = Path(img_path).name
        img_id = img_id_by_name.get(fname) or img_id_by_name.get(Path(fname).name)
        if img_id is None:
            continue

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), ci, sc in zip(xyxy, cls, confs):
            w = x2 - x1
            h = y2 - y1
            cls_name = name_by_idx[int(ci)]
            cat_id = cat_id_by_name.get(cls_name)
            if cat_id is None:
                # Names mismatch between model and GT categories; skip or map explicitly.
                continue
            results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(sc),
            })

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"[ultra] wrote detections: {out_json} ({len(results)} boxes)")

# ---------- Darknet → COCO results JSON ----------

def _run_darknet_list(
    *,
    darknet_bin: str,
    data_path: str,
    cfg_path: str,
    weights_path: str,
    images_txt: str,
    out_json_raw: str,
    thresh: float = 0.001,
    letter_box: bool = True,
) -> None:
    """
    Use hank-ai/AB-compatible JSON output: `-out result.json` reading an image list from stdin.
    This produces a per-image JSON we will convert to COCO results.
    """
    #
    # jp is unsure about this
    #
    flags = ["-dont_show", "-thresh", f"{thresh:.3f}", "-out", out_json_raw]
    if letter_box:
        flags.append("-letter_box")
    cmd = [darknet_bin, "detector", "test", data_path, cfg_path, weights_path, *flags]

    with open(images_txt, "r", encoding="utf-8", errors="ignore") as fin, \
        open("darknet_export.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, stdin=fin, check=True, text=True,
                    stdout=log, stderr=log)   # or DEVNULL if you don’t want a log
    #
    #
    #


def _convert_darknet_json_to_coco(
    *,
    dk_json_path: str,
    ann_json: str,
    out_json: str,
) -> None:
    # Load Darknet -out JSON (list of frames)
    with open(dk_json_path, "r", encoding="utf-8") as f:
        dk = json.load(f)

    # Load GT maps and image metadata
    with open(ann_json, "r", encoding="utf-8") as f:
        gt = json.load(f)
    img_id_by_name = {Path(im["file_name"]).name: im["id"] for im in gt["images"]}
    meta_by_name = {
        Path(im["file_name"]).name: (im["id"], float(im["width"]), float(im["height"]))
        for im in gt["images"]
    }
    meta_by_name_lc = {k.lower(): v for k, v in meta_by_name.items()}
    cat_id_by_name = {c["name"]: c["id"] for c in gt["categories"]}

    miss_id = miss_cat = bad_wh = 0
    results: List[Dict[str, Any]] = []

    for frame in dk:
        fname = Path(frame.get("filename", "")).name

        # tolerant width/height parse
        try:
            W = float(frame.get("width") or 0.0)
            H = float(frame.get("height") or 0.0)
        except Exception:
            W = H = 0.0

        meta = meta_by_name.get(fname) or meta_by_name_lc.get(fname.lower())
        if (W <= 0 or H <= 0) and meta:
            _, W, H = meta  # fallback to GT dims

        if W <= 0 or H <= 0:
            bad_wh += 1
            continue

        img_id = (meta[0] if meta else img_id_by_name.get(fname))
        if img_id is None:
            miss_id += 1
            continue

        for obj in frame.get("objects", []) or []:
            name = (obj.get("name") or "").strip()
            cat_id = cat_id_by_name.get(name)
            if cat_id is None:
                alt = name.replace("_", " ").lower()
                cat_id = next((cid for n, cid in cat_id_by_name.items() if n.lower() == alt), None)
            if cat_id is None:
                miss_cat += 1
                continue

            rel = obj.get("relative_coordinates") or {}
            cx = float(rel.get("center_x", 0.0)); cy = float(rel.get("center_y", 0.0))
            w  = float(rel.get("width", 0.0));    h  = float(rel.get("height", 0.0))
            x = (cx - w/2.0) * W; y = (cy - h/2.0) * H; bw = w * W; bh = h * H

            results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "score": float(obj.get("confidence", 0.0) or 0.0),
            })

    print(f"[darknet] mapping summary: frames_bad_wh={bad_wh}, frames_missing_id={miss_id}, objs_missing_cat={miss_cat}")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"[darknet] wrote detections: {out_json} ({len(results)} boxes)")


def export_darknet_detections(
    *,
    darknet_bin: str,
    data_path: str,
    cfg_path: str,
    weights_path: str,
    ann_json: str,
    out_json: str,
    images_txt: str,
    thresh: float = 0.001,
    letter_box: bool = True,
) -> None:
    """Run Darknet on each image (from images_txt) and write COCO-format det JSON."""
    raw_json = str(Path(out_json).with_suffix(".raw_darknet.json"))
    _run_darknet_list(
        darknet_bin=darknet_bin,
        data_path=data_path,
        cfg_path=cfg_path,
        weights_path=weights_path,
        images_txt=images_txt,
        out_json_raw=raw_json,
        thresh=thresh,
        letter_box=letter_box,
    )
    _convert_darknet_json_to_coco(
        dk_json_path=raw_json,
        ann_json=ann_json,
        out_json=out_json,
    )

# ---------- CLI (optional) ----------

def main():
    ap = argparse.ArgumentParser(description="Export COCO-format detections for COCOeval.")
    ap.add_argument("--backend", required=True, choices=["ultralytics", "darknet"])
    ap.add_argument("--ann", required=True, help="COCO GT annotations JSON for *val*")
    ap.add_argument("--out_json", required=True, help="COCO results JSON to write")
    ap.add_argument("--images_txt", help="Optional TXT listing val images (else uses ann.images)")
    ap.add_argument("--conf", type=float, default=0.001)
    # ultralytics
    ap.add_argument("--weights", help=".pt weights (Ultralytics)")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    # darknet (hank-ai/AB-style)
    ap.add_argument("--darknet_bin", default="darknet")
    ap.add_argument("--data", help="darknet .data")
    ap.add_argument("--cfg", help="darknet .cfg")
    ap.add_argument("--dk_weights", help="darknet .weights")
    ap.add_argument("--letter_box", action="store_true", default=True)
    ap.add_argument("--no_letter_box", dest="letter_box", action="store_false")
    args = ap.parse_args()

    if args.backend == "ultralytics":
        if not args.weights:
            ap.error("--weights is required for backend=ultralytics")
        export_ultra_detections(
            weights=args.weights,
            ann_json=args.ann,
            out_json=args.out_json,
            images_txt=args.images_txt,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
        )
    else:
        for req in ("data", "cfg", "dk_weights"):
            if getattr(args, req) is None:
                ap.error(f"--{req} is required for backend=darknet")
        if not args.images_txt:
            ap.error("--images_txt is required for backend=darknet")
        export_darknet_detections(
            darknet_bin=args.darknet_bin,
            data_path=args.data,
            cfg_path=args.cfg,
            weights_path=args.dk_weights,
            ann_json=args.ann,
            out_json=args.out_json,
            images_txt=args.images_txt,
            thresh=args.conf,
            letter_box=args.letter_box,
        )

if __name__ == "__main__":
    main()
