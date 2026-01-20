# src/chocolatechip/model_training/coco_build_gt.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
import json

# We assume Pillow is installed (per your Dockerfiles)
from PIL import Image


@dataclass(frozen=True)
class CocoIds:
    """Holds numeric ids used in the COCO file."""
    image_id: int
    ann_id: int


def _read_names_file(names_path: Optional[str]) -> Optional[List[str]]:
    if not names_path:
        return None
    p = Path(names_path)
    if not p.is_file():
        raise FileNotFoundError(f"names file not found: {names_path}")
    names = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            names.append(s)
    if not names:
        raise ValueError(f"no class names in: {names_path}")
    return names


def _load_per_image_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_image_paths(stem: str, parent: Path) -> Iterable[Path]:
    # Try common image extensions by stem in the same directory as the JSON
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = parent / f"{stem}{ext}"
        if p.is_file():
            yield p


def _get_img_wh_from_json_or_disk(js: dict, json_path: Path, stem: str) -> Tuple[int, int, Optional[str]]:
    # Prefer explicit size if present in the JSON
    img_info = js.get("image") or {}
    w = int(img_info.get("width") or 0)
    h = int(img_info.get("height") or 0)
    file_name = None

    # Try to infer image file next to the JSON
    for cand in _candidate_image_paths(stem, json_path.parent):
        file_name = cand.name
        if w <= 0 or h <= 0:
            try:
                with Image.open(cand) as im:
                    w, h = im.size
            except Exception:
                pass
        break  # first hit wins

    # Fallback: if still missing width/height, try any file_name embedded in JSON (uncommon)
    if (w <= 0 or h <= 0) and isinstance(img_info.get("file_name"), str):
        file_name = Path(img_info["file_name"]).name
        # as last resort, attempt to read it from disk (relative to JSON dir)
        disk = (json_path.parent / file_name)
        if disk.is_file():
            try:
                with Image.open(disk) as im:
                    w, h = im.size
            except Exception:
                pass

    if w <= 0 or h <= 0:
        raise ValueError(f"Missing width/height for {json_path}; couldn't infer from disk.")

    # If still no filename, synthesize one from stem with .jpg (COCO requires a string)
    if not file_name:
        file_name = f"{stem}.jpg"

    return int(w), int(h), file_name


def _points_to_bbox(points: List[dict], w: int, h: int) -> Tuple[float, float, float, float]:
    """
    Accepts a list of points; will use pixel ints if present (int_x/int_y),
    otherwise normalized floats (x/y).
    Returns COCO x,y,w,h (float).
    """
    xs: List[float] = []
    ys: List[float] = []

    # Prefer pixel ints if *all* points have int_x/int_y
    has_all_ints = all(("int_x" in p and "int_y" in p) for p in points if isinstance(p, dict))
    if has_all_ints:
        for p in points:
            xs.append(float(p["int_x"]))
            ys.append(float(p["int_y"]))
    else:
        for p in points:
            # fall back to normalized floats; multiply by image size
            x = float(p.get("x", 0.0)) * float(w)
            y = float(p.get("y", 0.0)) * float(h)
            xs.append(x); ys.append(y)

    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(w), max(xs))
    y_max = min(float(h), max(ys))

    bw = max(0.0, x_max - x_min)
    bh = max(0.0, y_max - y_min)
    return (float(x_min), float(y_min), float(bw), float(bh))


def _collect_jsons(ann_root: str) -> Dict[str, Path]:
    """
    Return {stem: json_path} for all *.json found under ann_root.
    stem = filename without extension (used to match image basenames).
    """
    root = Path(ann_root)
    if not root.is_dir():
        raise NotADirectoryError(f"ann_root not a directory: {ann_root}")
    out: Dict[str, Path] = {}
    for p in root.rglob("*.json"):
        out[p.stem] = p
    return out


def _read_list_file(list_path: Optional[str]) -> Optional[List[str]]:
    if not list_path:
        return None
    p = Path(list_path)
    if not p.is_file():
        raise FileNotFoundError(f"list file not found: {list_path}")
    items = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        # Accept absolute paths or relative paths; we use *basename stem* to match JSONs
        items.append(Path(s).stem)
    return items


def _read_list_paths(list_path: Optional[str]) -> Optional[List[Path]]:
    """
    Read a train/valid .txt list file.
    Returns a list of Paths (absolute or relative as written).
    Keeps full paths (does NOT reduce to stem).
    """
    if not list_path:
        return None
    p = Path(list_path)
    if not p.is_file():
        raise FileNotFoundError(f"list file not found: {list_path}")

    items: List[Path] = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        items.append(Path(s))
    return items


def _yolo_txt_to_coco_bbox(line: str, w: int, h: int) -> Optional[Tuple[int, float, float, float, float]]:
    """
    YOLO line: <cls> <xc> <yc> <bw> <bh>  (normalized floats)
    Returns (class_id0, x_min, y_min, bw_px, bh_px) in pixels.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        cls0 = int(float(parts[0]))
        xc = float(parts[1]) * w
        yc = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
    except Exception:
        return None

    # Convert center->top-left
    x = xc - bw / 2.0
    y = yc - bh / 2.0

    # Clamp to image bounds
    x = max(0.0, min(float(w), x))
    y = max(0.0, min(float(h), y))
    bw = max(0.0, min(float(w) - x, bw))
    bh = max(0.0, min(float(h) - y, bh))

    if bw <= 0.0 or bh <= 0.0:
        return None

    return (cls0, float(x), float(y), float(bw), float(bh))


def build_coco_gt_from_yolo_lists(
    *,
    list_file: str,
    out_json: Optional[str] = None,
    names_path: Optional[str] = None,
) -> dict:
    """
    Build COCO GT from a train/valid list file where images have YOLO .txt labels
    next to them (same stem).
    """
    img_paths = _read_list_paths(list_file) or []
    if not img_paths:
        raise ValueError(f"empty list_file: {list_file}")

    names_list = _read_names_file(names_path)  # may be None

    # Categories: prefer names.txt ordering; else infer max class id later
    categories: List[dict] = []
    if names_list:
        categories = [{"id": i, "name": n} for i, n in enumerate(names_list)]


    images: List[dict] = []
    annotations: List[dict] = []
    image_id = 1
    ann_id = 1
    max_cls0_seen = -1

    for raw_img in img_paths:
        img = raw_img
        if not img.is_absolute():
            # interpret relative paths relative to the list_file directory
            img = Path(list_file).parent / img

        if not img.is_file():
            # Robustness: skip missing images
            continue

        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            continue

        images.append({
            "id": image_id,
            "file_name": img.name,   # basename is fine for eval if your det export uses basenames too
            "width": int(w),
            "height": int(h),
        })

        label = img.with_suffix(".txt")
        if label.is_file():
            for ln in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parsed = _yolo_txt_to_coco_bbox(ln, w, h)
                if not parsed:
                    continue
                cls0, x, y, bw, bh = parsed
                max_cls0_seen = max(max_cls0_seen, cls0)

                cat_id = cls0


                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                ann_id += 1

        image_id += 1

    # If no names.txt, infer categories from observed max class id
    if not categories:
        k = max_cls0_seen + 1
        categories = [{"id": i, "name": f"class_{i}"} for i in range(max(0, k))]

    coco: Dict[str, object] = {
        "info": {"description": "YOLO txt → COCO GT", "version": "1.0", "year": 2025},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    if out_json:
        outp = Path(out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(coco), encoding="utf-8")
        print(f"[coco_gt] wrote {outp}  (images={len(images)}, anns={len(annotations)}, classes={len(categories)})")

    return coco


def _infer_categories(used_obj_names: Iterable[str], names_list: Optional[List[str]]) -> Tuple[List[dict], Dict[str, int]]:
    """
    Build COCO categories and a mapping from object name -> category_id.
    If names_list is provided, category order follows it; else use sorted set.
    """
    if names_list:
        uniq = list(dict.fromkeys(names_list))  # keep order, dedupe
    else:
        uniq = sorted(set(used_obj_names))

    categories = [{"id": i, "name": name} for i, name in enumerate(uniq)]
    cat_id_by_name = {c["name"]: c["id"] for c in categories}
    return categories, cat_id_by_name


def build_coco_gt(
    ann_root: str,
    out_json: Optional[str] = None,
    list_file: Optional[str] = None,
    names_path: Optional[str] = None,
) -> dict:
    """
    Build a COCO-style GT dict from per-image JSONs.

    - ann_root: directory containing (recursively) per-image JSONs
    - out_json: if provided, the COCO GT is written to this path
    - list_file: optional txt listing validation images to include (any form; we use basenames)
    - names_path: optional names.txt — class names to lock category ids (one per line)

    Expected per-image JSON minimal schema:
      {
        "image": {"width": <int>, "height": <int>},     # optional if an image file is present next to JSON
        "mark": [
          {"name": "classA",
           "points": [{"int_x": <int>, "int_y": <int>} ...]  # OR {"x": <0..1>, "y": <0..1>}
          }, ...
        ],
        "completely_empty": <bool>   # optional
      }
    """
    json_by_stem = _collect_jsons(ann_root)

    # If we have any per-image JSONs, use the existing JSON-based pipeline.
    # Otherwise (e.g., Leather), fall back to YOLO .txt labels next to images listed in list_file.
    if not json_by_stem:
        if not list_file:
            raise ValueError("No *.json annotations found and no list_file provided for YOLO-txt fallback.")
        return build_coco_gt_from_yolo_lists(
            list_file=list_file,
            out_json=out_json,
            names_path=names_path,
        )

    stems = set(json_by_stem.keys())


    # If a list file is provided, restrict to those stems
    restrict = _read_list_file(list_file)
    if restrict is not None:
        requested = set(restrict)
        missing = requested - stems
        # Not fatal; drop missing with a warning-ish print
        if missing:
            print(f"[coco_gt] warning: {len(missing)} list items not found as JSON: {sorted(list(missing))[:5]}{' ...' if len(missing)>5 else ''}")
        stems = stems & requested

    # First pass: gather class names that actually occur
    used_names: List[str] = []
    for stem in sorted(stems):
        js = _load_per_image_json(json_by_stem[stem])
        for obj in js.get("mark", []) or []:
            name = obj.get("name")
            if isinstance(name, str) and name:
                used_names.append(name)

    categories_list, cat_id_by_name = _infer_categories(used_names, _read_names_file(names_path))

    images: List[dict] = []
    annotations: List[dict] = []
    image_id = 1
    ann_id = 1

    for stem in sorted(stems):
        jpath = json_by_stem[stem]
        js = _load_per_image_json(jpath)

        try:
            w, h, file_name = _get_img_wh_from_json_or_disk(js, jpath, stem)
        except Exception as e:
            print(f"[coco_gt] skip {jpath.name}: {e}")
            continue

        # COCO image entry
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": int(w),
            "height": int(h),
        })

        # Objects → annotations
        for obj in js.get("mark", []) or []:
            name = obj.get("name")
            if not isinstance(name, str) or not name:
                continue
            cat_id = cat_id_by_name.get(name)
            if cat_id is None:
                # New class unseen in first pass (rare). Add it on the fly.
                cat_id = max(cat_id_by_name.values(), default=0) + 1
                cat = {"id": cat_id, "name": name}
                categories_list.append(cat)
                cat_id_by_name[name] = cat_id

            pts = obj.get("points") or []
            x, y, bw, bh = _points_to_bbox(pts, w, h)

            # Skip degenerate boxes (optional)
            if bw <= 0 or bh <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

        image_id += 1
        
    coco: Dict[str, object] = {
        "info": {
            "description": "DarkMark → COCO GT",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
    }


    if out_json:
        outp = Path(out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(coco), encoding="utf-8")
        print(f"[coco_gt] wrote {outp}  (images={len(images)}, anns={len(annotations)}, classes={len(categories_list)})")

    return coco


class CocoGTBuilder:
    """
    Small OO wrapper if you prefer an object with reusable config.
    """
    def __init__(self, ann_root: str, names_path: Optional[str] = None):
        self.ann_root = ann_root
        self.names_path = names_path

    def build(self, list_file: Optional[str] = None, out_json: Optional[str] = None) -> dict:
        return build_coco_gt(
            ann_root=self.ann_root,
            out_json=out_json,
            list_file=list_file,
            names_path=self.names_path,
        )
