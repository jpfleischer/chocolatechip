from __future__ import annotations
from pathlib import Path
import argparse, random, json, re
from typing import Dict, List, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def collect_images_by_subdir(root: Path, subdirs: List[str], exts=IMG_EXTS) -> Dict[str, List[Path]]:
    images: Dict[str, List[Path]] = {}
    for d in subdirs:
        p = (root / d)
        if not p.is_dir():
            raise FileNotFoundError(f"Missing dataset subdir: {p}")
        items = [x.resolve() for x in sorted(p.iterdir())
                 if x.is_file() and x.suffix.lower() in exts]
        if not items:
            raise RuntimeError(f"No images found in {p}")
        images[d] = items
    return images

def write_list(path: Path, paths: List[Path]) -> None:
    path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")

def write_darknet_data_file(out: Path, *, classes: int, train: Path, valid: Path, names: Path, backup: Path) -> None:
    txt = (
        f"classes = {classes}\n"
        f"train = {train}\n"
        f"valid = {valid}\n"
        f"names = {names}\n"
        f"backup = {backup}\n"
    )
    out.write_text(txt, encoding="utf-8")

def choose_split(
    images_by_subdir: Dict[str, List[Path]],
    neg_subdirs: List[str],
    val_frac: float,
    seed: int,
) -> tuple[List[Path], List[Path], dict]:
    """Return (train, valid, stats) with a deterministic 'random' split."""
    rng = random.Random(seed)

    negatives: List[Path] = []
    positives: List[Path] = []
    for d, lst in images_by_subdir.items():
        (negatives if d in neg_subdirs else positives).extend(lst)

    negatives.sort()
    positives.sort()

    rng.shuffle(negatives)
    rng.shuffle(positives)

    n_neg, n_pos = len(negatives), len(positives)
    n_total = n_neg + n_pos
    if n_total == 0:
        raise RuntimeError("No images found overall.")

    n_val = max(1, round(val_frac * n_total))
    n_val_neg = min(n_neg, round(n_val * (n_neg / n_total))) if n_neg else 0
    n_val_pos = n_val - n_val_neg
    n_val_pos = min(n_pos, n_val_pos)

    val_neg = negatives[:n_val_neg]
    val_pos = positives[:n_val_pos]
    valid = sorted(val_neg + val_pos)

    train = sorted(negatives[n_val_neg:] + positives[n_val_pos:])

    inter = set(train).intersection(valid)
    assert not inter, f"Train/valid overlap detected: {list(inter)[:3]}"

    stats = {
        "mode": "ratio",
        "seed": seed,
        "val_frac": val_frac,
        "counts": {
            "valid_total": len(valid),
            "train_total": len(train),
            "negatives_total": n_neg,
            "positives_total": n_pos,
        },
        "neg_subdirs": neg_subdirs,
    }
    return train, valid, stats

# ---------- LEGOS SPECIAL MODE ----------

_SCENE_RX = re.compile(r'(DSCN\d+)', re.IGNORECASE)

def _scene_key(p: Path) -> str:
    """Return the scene id (e.g., 'DSCN2164') if present, else the stem prefix before first '_'."""
    m = _SCENE_RX.search(p.stem)
    if m:
        return m.group(1)
    # fallback: take up to first underscore to avoid grouping different frames together
    return p.stem.split('_', 1)[0]

def choose_split_legos(
    images_by_subdir: Dict[str, List[Path]],
    neg_subdirs: List[str],
) -> Tuple[List[Path], List[Path], dict]:
    """
    Deterministic validation composition for the Lego experiment:
      - set_03: one image per scene (scene inferred from 'DSCN####')
      - set_02 (negatives): one image total (from provided neg_subdirs)
      - set_01: one image total (any)
      - everything else goes to train
    """
    # Find subdir keys case-insensitively
    def _find_key(name: str) -> str | None:
        name = name.lower()
        for k in images_by_subdir.keys():
            if name in k.lower():
                return k
        return None

    key_set01 = _find_key("set_01")
    key_set03 = _find_key("set_03")

    if key_set03 is None:
        raise RuntimeError("LEGOS mode expects a subdir containing 'set_03'.")

    # Deterministic (sorted) lists
    set03_imgs = list(sorted(images_by_subdir[key_set03]))
    set01_imgs = list(sorted(images_by_subdir.get(key_set01, [])))

    neg_imgs: List[Path] = []
    pos_other: List[Path] = []

    for d, lst in images_by_subdir.items():
        if d == key_set03 or d == key_set01:
            continue
        if d in neg_subdirs:
            neg_imgs.extend(lst)
        else:
            pos_other.extend(lst)

    neg_imgs.sort()
    pos_other.sort()

    # ---- Build VALID set deterministically ----
    valid: List[Path] = []

    # 1) set_03: pick first image per scene id
    seen_scenes = set()
    per_scene_picks: List[Path] = []
    for p in set03_imgs:
        sk = _scene_key(p)
        if sk not in seen_scenes:
            seen_scenes.add(sk)
            per_scene_picks.append(p)
    valid.extend(per_scene_picks)

    # 2) negatives: pick one (first) if present
    neg_pick = neg_imgs[0:1]
    valid.extend(neg_pick)

    # 3) set_01: pick one (first) if present
    set01_pick = set01_imgs[0:1]
    valid.extend(set01_pick)

    valid = sorted(valid)

    # ---- TRAIN = everything else ----
    all_imgs = []
    for lst in images_by_subdir.values():
        all_imgs.extend(lst)
    all_imgs = sorted(set(all_imgs))  # unique

    train = sorted(set(all_imgs) - set(valid))

    # Safety
    inter = set(train).intersection(valid)
    assert not inter, f"Train/valid overlap detected: {list(inter)[:3]}"

    # Stats
    stats = {
        "mode": "legos",
        "counts": {
            "valid_total": len(valid),
            "train_total": len(train),
            "valid_set03_scenes": len(per_scene_picks),
            "valid_negatives": len(neg_pick),
            "valid_set01": len(set01_pick),
        },
        "details": {
            "set03_scenes": sorted({_scene_key(p) for p in set03_imgs}),
            "neg_subdirs": neg_subdirs,
            "set01_present": key_set01 is not None,
        },
    }
    return train, valid, stats
# ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Prepare disjoint train/valid lists and a Darknet .data file.")
    ap.add_argument("--root", required=True, help="Dataset root (e.g., /ultralytics/LegoGears_v2)")
    ap.add_argument("--sets", nargs="+", required=True, help="Subdirectories with images (e.g., set_01 set_02_empty set_03)")
    ap.add_argument("--classes", type=int, required=True, help="Number of classes")
    ap.add_argument("--names", default="LegoGears.names", help="Names filename (inside root)")
    ap.add_argument("--prefix", default="LegoGears", help="Prefix for generated files")
    ap.add_argument("--val-frac", type=float, default=0.20, help="Validation fraction (default 0.20)")
    ap.add_argument("--seed", type=int, default=9001, help="Random seed for deterministic split")
    ap.add_argument("--neg-subdirs", nargs="*", default=None,
                    help="Subdirs to treat as negatives. If omitted, any subdir containing 'empty' or 'neg' is treated as negative.")
    ap.add_argument("--exts", nargs="*", default=[".jpg"], help="Image extensions to include")
    ap.add_argument("--legos", action="store_true",
                    help="Special deterministic split for LegoGears: 1 per scene from set_03, 1 negative from set_02, 1 from set_01.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    images_by_subdir = collect_images_by_subdir(root, args.sets, tuple(e.lower() for e in args.exts))

    # Auto-detect negatives if not provided
    if args.neg_subdirs is None:
        neg_subdirs = [d for d in args.sets if ("empty" in d.lower() or "neg" in d.lower())]
    else:
        neg_subdirs = args.neg_subdirs

    if args.legos:
        train_paths, valid_paths, stats = choose_split_legos(
            images_by_subdir,
            neg_subdirs=neg_subdirs,
        )
    else:
        train_paths, valid_paths, stats = choose_split(
            images_by_subdir,
            neg_subdirs=neg_subdirs,
            val_frac=args.val_frac,
            seed=args.seed,
        )

    train_file = root / f"{args.prefix}_train.txt"
    valid_file = root / f"{args.prefix}_valid.txt"
    data_file  = root / f"{args.prefix}.data"
    names_file = root / args.names
    manifest   = root / f"{args.prefix}_split.json"

    write_list(train_file, train_paths)
    write_list(valid_file, valid_paths)
    write_darknet_data_file(
        data_file,
        classes=args.classes,
        train=train_file,
        valid=valid_file,
        names=names_file,
        backup=root,
    )
    manifest.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    ctrain, cvalid = len(train_paths), len(valid_paths)
    print(
        "[setup] Wrote:\n"
        f"  {train_file}  ({ctrain} images)\n"
        f"  {valid_file}  ({cvalid} images)\n"
        f"  {data_file}\n"
        f"  {manifest}\n"
        f"[setup] Mode={stats.get('mode','ratio')}  Neg subdirs={neg_subdirs}"
    )

if __name__ == "__main__":
    main()
