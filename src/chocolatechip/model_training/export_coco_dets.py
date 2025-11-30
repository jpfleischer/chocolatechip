# src/chocolatechip/model_training/export_coco_dets.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import argparse, json, sys, subprocess, os

from PIL import Image, ImageDraw

# Simple class -> color mapping for visualizations (RGB)
CLASS_COLORS = {
    "blue":   (0, 0, 255),
    "red":    (255, 0, 0),
    "green":  (0, 255, 0),
    "yellow": (255, 255, 0),
    # fallback for any other class
}

DEFAULT_COLOR = (255, 255, 255)  # white outline for unknown classes
TEXT_COLOR = (0, 0, 0)           # black text



def _draw_and_save_vis_image(
    *,
    image_path: str,
    detections: List[Dict[str, Any]],
    vis_root: Path,
    vis_scale: float = 3.0,
    score_thresh: float = 0.5,
) -> None:
    """
    Draw boxes + labels on image_path and save to vis_root/<stem>_pred.png.

    detections: list of dicts with keys:
        - x1, y1, x2, y2 (float pixel coords)
        - cls_name (str)
        - score   (float)
    """
    if not detections:
        return

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[vis] failed to open {image_path}: {e}")
        return

    draw = ImageDraw.Draw(img)

    for det in detections:
        score = float(det.get("score", 0.0) or 0.0)
        if score < score_thresh:
            continue

        x1 = float(det["x1"]); y1 = float(det["y1"])
        x2 = float(det["x2"]); y2 = float(det["y2"])
        cls_name = str(det["cls_name"])

        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        # box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"{cls_name} {score:.2f}"
        # text background
        try:
            tb = draw.textbbox((0, 0), label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(label) * 7, 12

        tx1, ty1 = x1, max(0, y1 - th)
        tx2, ty2 = x1 + tw, ty1 + th

        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        draw.text((tx1, ty1), label, fill=TEXT_COLOR)

    # upscale at the end
    if vis_scale and vis_scale != 1.0:
        w_im, h_im = img.size
        new_size = (int(w_im * vis_scale), int(h_im * vis_scale))
        img = img.resize(new_size, Image.NEAREST)

    out_img = vis_root / f"{Path(image_path).stem}_pred.png"
    try:
        img.save(out_img)
    except Exception as e:
        print(f"[vis] failed to save {out_img}: {e}")

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
    batch: int | None = 16,  # kept for API compat, ignored
    save_vis: bool = False,
    vis_dir: str | None = None,
    vis_scale: float = 3.0,
) -> None:
    from ultralytics import YOLO
    img_id_by_name, cat_id_by_name = _load_gt_index(ann_json)
    images = _read_images_list(images_txt, ann_json)

    print(f"[ultra_export] device={device} batch={batch} (ignored) imgsz={imgsz} n_images={len(images)}")


    model = YOLO(weights)
    name_by_idx = model.names
    results: List[Dict[str, Any]] = []

    # where visualization PNGs go, if enabled
    vis_root = None
    if save_vis:
        base = Path(vis_dir) if vis_dir else Path(out_json).parent
        vis_root = base / "ultra_vis"
        vis_root.mkdir(parents=True, exist_ok=True)
        print(f"[ultra_vis] saving annotated images under: {vis_root}")

    for img_path in images:
        preds = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            device=device,
            imgsz=imgsz,
            verbose=False,
            save=False,
        )
        if not preds:
            continue
        r = preds[0]
        if r.boxes is None:
            continue

        fname = Path(img_path).name
        img_id = img_id_by_name.get(fname) or img_id_by_name.get(Path(fname).name)
        if img_id is None:
            continue

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        vis_dets: List[Dict[str, Any]] = []  # for visualization only

        for (x1, y1, x2, y2), ci, sc in zip(xyxy, cls, confs):
            w = x2 - x1
            h = y2 - y1
            cls_name = name_by_idx[int(ci)]
            cat_id = cat_id_by_name.get(cls_name)
            if cat_id is None:
                continue

            # COCO det JSON (all boxes)
            results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(sc),
            })

            # stash for vis; score filter happens in helper
            if save_vis and vis_root is not None:
                vis_dets.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cls_name": cls_name,
                    "score": float(sc),
                })

        # draw once per image
        if save_vis and vis_root is not None and vis_dets:
            _draw_and_save_vis_image(
                image_path=img_path,
                detections=vis_dets,
                vis_root=vis_root,
                vis_scale=vis_scale,
                score_thresh=0.5,
            )


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
    save_vis: bool = False,
    vis_dir: str | None = None,
    vis_scale: float = 3.0,
    vis_conf_thresh: float = 0.5,
) -> None:
    # Load Darknet -out JSON (list of frames)
    with open(dk_json_path, "r", encoding="utf-8") as f:
        dk = json.load(f)

    # Load GT maps and image metadata
    with open(ann_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    vis_root = None
    if save_vis:
        if vis_dir:
            vis_root = Path(vis_dir)
        else:
            vis_root = Path(out_json).parent / "darknet_vis"
        vis_root.mkdir(parents=True, exist_ok=True)
        print(f"[darknet_vis] saving annotated images under: {vis_root}")

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
        full_fname = frame.get("filename", "")

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


        vis_dets: List[Dict[str, Any]] = []

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
            x1, y1, x2, y2 = x, y, x + bw, y + bh

            score = float(obj.get("confidence", 0.0) or 0.0)

            results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "score": float(score),
            })

            if save_vis and vis_root is not None:
                vis_dets.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cls_name": name,
                    "score": float(score),
                })

        if save_vis and vis_root is not None and vis_dets and full_fname:
            _draw_and_save_vis_image(
                image_path=full_fname,
                detections=vis_dets,
                vis_root=vis_root,
                vis_scale=vis_scale,
                score_thresh=vis_conf_thresh,
            )

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
    save_vis: bool = False,
    vis_dir: str | None = None,
    vis_scale: float = 3.0,
    vis_conf_thresh: float = 0.5,
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
        save_vis=save_vis,
        vis_dir=vis_dir,
        vis_scale=vis_scale,
        vis_conf_thresh=vis_conf_thresh,
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
    ap.add_argument("--save_vis", action="store_true",
                    help="If set, save PNG/JPGs with predicted boxes during export")  # NEW
    ap.add_argument("--vis_dir",
                    help="Optional root dir for Ultralytics visualizations (default: next to out_json)")  # NEW
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
            save_vis=args.save_vis,
            vis_dir=args.vis_dir,
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
            save_vis=args.save_vis,
            vis_dir=args.vis_dir,
            # optional: could also expose vis_scale / vis_conf_thresh as CLI flags later
        )

if __name__ == "__main__":
    main()
