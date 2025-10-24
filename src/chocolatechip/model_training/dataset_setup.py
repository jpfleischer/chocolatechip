from __future__ import annotations
from pathlib import Path
import argparse, random, json, re
from typing import Dict, List, Tuple
import yaml

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

def _label_path_for(img: Path) -> Path:
    # always .txt regardless of image extension/case
    return img.with_suffix(".txt")

def _derive_ratio_tag_from_prefix(prefix: str) -> str | None:
    """
    Extract trailing pattern like '_v10' or '_v15' from a prefix such as 'LegoGears_v10'.
    Returns 'v10' or None if absent.
    """
    m = re.search(r'_(v\d{2})$', prefix)
    return m.group(1) if m else None

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

def write_ultralytics_yaml(
    root: Path,
    prefix: str,
    classes: int,
    names_file: str,
    train_file: Path,
    valid_file: Path,
    ratio_tag: str | None = None,
) -> Path:
    # Resolve name list
    names_path = root / names_file
    class_names = [
        line.strip()
        for line in names_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # The prefix you pass (e.g., 'LegoGears_v10') already carries the ratio;
    # don’t append ratio_tag again to avoid '..._v10_v10.yaml'.
    yaml_path = root / f"{prefix}.yaml"

    # Build minimal ultralytics spec
    data = {
        "train": str(train_file),
        "val": str(valid_file),
        "nc": classes,
        "names": class_names,
    }

    # Write
    # Properly write YAML to file
    yaml_str = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=1000)
    yaml_path.write_text(yaml_str, encoding="utf-8")
    return yaml_path

def choose_split(
    images_by_subdir: Dict[str, List[Path]],
    neg_subdirs: List[str],
    val_frac: float,
    seed: int,
) -> tuple[List[Path], List[Path], dict]:
    """Return (train, valid, stats) with deterministic split after filtering unlabeled images."""
    rng = random.Random(seed)
    neg_set = set(s.lower() for s in neg_subdirs)

    kept, _, filter_stats = _partition_and_filter_all_labeled(images_by_subdir)

    # separate negatives vs positives by subdir name after filtering
    negatives: List[Path] = []
    positives: List[Path] = []
    for p in kept:
        # subdir is the first component under root; faster than scanning
        subdir = p.parent.name
        (negatives if subdir.lower() in neg_set else positives).append(p)

    rng.shuffle(negatives)
    rng.shuffle(positives)

    n_neg, n_pos = len(negatives), len(positives)
    n_total = n_neg + n_pos
    if n_total == 0:
        raise RuntimeError("No images left after filtering for adjacent .txt labels.")

    n_val = max(1, round(val_frac * n_total))
    n_val_neg = min(n_neg, round(n_val * (n_neg / max(1, n_total)))) if n_neg else 0
    n_val_pos = min(n_pos, n_val - n_val_neg)

    val_neg = negatives[:n_val_neg]
    val_pos = positives[:n_val_pos]
    valid = sorted(val_neg + val_pos)
    train = sorted(negatives[n_val_neg:] + positives[n_val_pos:])

    assert not (set(train) & set(valid)), "Train/valid overlap detected."

    stats = {
        "mode": "ratio",
        "seed": seed,
        "val_frac": val_frac,
        "counts": {
            "valid_total": len(valid),
            "train_total": len(train),
            "negatives_total": n_neg,
            "positives_total": n_pos,
            "dropped_missing_label_total": filter_stats["totals"]["dropped_total"],
        },
        "neg_subdirs": neg_subdirs,
        "filtering": filter_stats,
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

def _partition_and_filter_all_labeled(
    images_by_subdir: Dict[str, List[Path]],
) -> tuple[List[Path], List[Path], dict]:
    """
    Require an adjacent .txt for *all* images, regardless of subdir.
    Returns (negatives, positives, stats) where 'negatives' and 'positives'
    are separated purely by subdir membership decided later by caller.
    """
    kept: List[Path] = []
    dropped_missing_label: Dict[str, int] = {}
    kept_per_dir: Dict[str, int] = {}

    for d, imgs in images_by_subdir.items():
        for img in imgs:
            lp = _label_path_for(img)
            if lp.exists():
                kept.append(img)
                kept_per_dir[d] = kept_per_dir.get(d, 0) + 1
            else:
                dropped_missing_label[d] = dropped_missing_label.get(d, 0) + 1

    stats = {
        "filtered": {
            "dropped_missing_label": dropped_missing_label,
            "kept_per_dir": kept_per_dir,
        },
        "totals": {
            "kept_total": len(kept),
            "dropped_total": sum(dropped_missing_label.values()),
        },
        "policy": {"require_label_for_all": True},
    }
    return kept, [], stats  # second list unused (kept for signature parity)


def collect_images_flat(root: Path, flat_dir: str, exts=IMG_EXTS) -> List[Path]:
    """Return images in root/flat_dir that have an adjacent .txt."""
    p = (root / flat_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Missing flat dir: {p}")
    imgs = [x.resolve() for x in sorted(p.iterdir())
            if x.is_file() and x.suffix.lower() in exts]
    kept = []
    dropped = 0
    for img in imgs:
        if _label_path_for(img).exists():
            kept.append(img)
        else:
            dropped += 1
    if not kept:
        raise RuntimeError(f"No labeled images found in {p} (dropped={dropped})")
    return kept

def choose_split_flat(
    all_imgs: List[Path],
    val_frac: float,
    seed: int,
) -> tuple[List[Path], List[Path], dict]:
    rng = random.Random(seed)
    imgs = list(all_imgs)
    rng.shuffle(imgs)
    n = len(imgs)
    n_val = max(1, round(val_frac * n))
    valid = sorted(imgs[:n_val])
    train = sorted(imgs[n_val:])

    # stats: count positives vs negatives via label content
    pos = neg = 0
    for im in imgs:
        txt = _label_path_for(im)
        try:
            # negative if label file exists but has no non-empty lines
            has = any(line.strip() for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines())
            if has: pos += 1
            else:   neg += 1
        except Exception:
            pass

    stats = {
        "mode": "flat",
        "seed": seed,
        "val_frac": val_frac,
        "counts": {
            "valid_total": len(valid),
            "train_total": len(train),
            "positives_total": pos,
            "negatives_total": neg,
            "pool_total": n,
        },
        "flat_dir": str(all_imgs[0].parent if all_imgs else "<empty>"),
    }
    return train, valid, stats


# ---------- LEGOS SPECIAL MODE (after filtering) ----------
def choose_split_legos(
    images_by_subdir: Dict[str, List[Path]],
    neg_subdirs: List[str],
) -> Tuple[List[Path], List[Path], dict]:
    """
    Deterministic validation composition for the Lego experiment,
    after requiring adjacent .txt for all images.
    """
    kept, _, filter_stats = _partition_and_filter_all_labeled(images_by_subdir)

    # bucket kept images back by subdir
    bydir: Dict[str, List[Path]] = {d: [] for d in images_by_subdir.keys()}
    for p in kept:
        bydir[p.parent.name].append(p)

    def _find_key(name: str) -> str | None:
        name = name.lower()
        for k in bydir.keys():
            if name in k.lower():
                return k
        return None

    key_set01 = _find_key("set_01")
    key_set03 = _find_key("set_03")
    if key_set03 is None:
        raise RuntimeError("LEGOS mode expects a subdir containing 'set_3' after filtering.")

    set03_imgs = list(sorted(bydir[key_set03]))
    set01_imgs = list(sorted(bydir.get(key_set01, [])))
    neg_set = set(s.lower() for s in neg_subdirs)

    neg_imgs: List[Path] = []
    pos_other: List[Path] = []
    for d, lst in bydir.items():
        if d == key_set03 or d == key_set01:
            continue
        (neg_imgs if d.lower() in neg_set else pos_other).extend(lst)

    neg_imgs.sort(); pos_other.sort()

    valid: List[Path] = []
    seen_scenes = set()
    per_scene_picks: List[Path] = []
    for p in set03_imgs:
        sk = _scene_key(p)
        if sk not in seen_scenes:
            seen_scenes.add(sk)
            per_scene_picks.append(p)
    valid.extend(per_scene_picks)
    valid.extend(neg_imgs[0:1])
    valid.extend(set01_imgs[0:1])
    valid = sorted(valid)

    all_kept = sorted(set(kept))
    train = sorted(set(all_kept) - set(valid))

    assert not (set(train) & set(valid)), "Train/valid overlap detected."

    stats = {
        "mode": "legos",
        "counts": {
            "valid_total": len(valid),
            "train_total": len(train),
            "valid_set03_scenes": len(per_scene_picks),
            "valid_negatives": len(neg_imgs[0:1]),
            "valid_set01": len(set01_imgs[0:1]),
            "dropped_missing_label_total": filter_stats["totals"]["dropped_total"],
        },
        "details": {
            "set03_scenes": sorted({_scene_key(p) for p in set03_imgs}),
            "neg_subdirs": neg_subdirs,
            "set01_present": key_set01 is not None,
        },
        "filtering": filter_stats,
    }
    return train, valid, stats
# ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Prepare disjoint train/valid lists and a Darknet .data file.")
    ap.add_argument("--root", required=True, help="Dataset root (e.g., /ultralytics/LegoGears_v2)")
    ap.add_argument("--sets", nargs="+", required=False, default=[], help="Subdirectories with images (e.g., set_01 set_02_empty set_03)")
    ap.add_argument("--classes", type=int, required=True, help="Number of classes")
    ap.add_argument("--names", default="LegoGears.names", help="Names filename (inside root)")
    ap.add_argument("--prefix", default="LegoGears", help="Prefix for generated files")
    ap.add_argument("--val-frac", type=float, default=0.20, help="Validation fraction (default 0.20)")
    ap.add_argument("--seed", type=int, default=9001, help="Random seed for deterministic split")
    ap.add_argument("--neg-subdirs", nargs="*", default=None,
                    help="Subdirs to treat as negatives. If omitted, any subdir containing 'empty' or 'neg' is treated as negative.")
    ap.add_argument("--exts", nargs="*", default=list(IMG_EXTS), help="Image extensions to include")
    ap.add_argument("--flat-dir", default=None,
                    help="If set, use a single directory (relative to --root) that already mixes positive and negative samples (each with adjacent .txt). Ignores --sets/--neg-subdirs.")
    ap.add_argument("--legos", action="store_true",
                    help="Special deterministic split for LegoGears: 1 per scene from set_03, 1 negative from set_02, 1 from set_01.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    mode_str = "ratio"
    neg_subdirs_for_print = args.neg_subdirs

    # ---------------- FLAT MODE ----------------
    if args.flat_dir:
        # requires you added: collect_images_flat() and choose_split_flat()
        exts = tuple(e.lower() for e in args.exts)
        all_imgs = collect_images_flat(root, args.flat_dir, exts)
        train_paths, valid_paths, stats = choose_split_flat(
            all_imgs, val_frac=args.val_frac, seed=args.seed
        )
        mode_str = "flat"
        neg_subdirs_for_print = []  # ignored in flat mode

    # --------------- HIERARCHICAL MODE ---------------
    else:
        if not args.sets:
            raise SystemExit("--sets is required unless you use --flat-dir")
        images_by_subdir = collect_images_by_subdir(root, args.sets, tuple(e.lower() for e in args.exts))

        # Auto-detect negatives if not provided
        if args.neg_subdirs is None:
            neg_subdirs = [d for d in args.sets if ("empty" in d.lower() or "neg" in d.lower())]
        else:
            neg_subdirs = args.neg_subdirs
        neg_subdirs_for_print = neg_subdirs

        if args.legos:
            train_paths, valid_paths, stats = choose_split_legos(
                images_by_subdir,
                neg_subdirs=neg_subdirs,
            )
            mode_str = "legos"
        else:
            train_paths, valid_paths, stats = choose_split(
                images_by_subdir,
                neg_subdirs=neg_subdirs,
                val_frac=args.val_frac,
                seed=args.seed,
            )
            mode_str = "ratio"

    # --------------- WRITE ARTIFACTS ---------------
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

    yaml_path = write_ultralytics_yaml(
        root=root,
        prefix=args.prefix,
        classes=args.classes,
        names_file=args.names,
        train_file=train_file,
        valid_file=valid_file,
        # derive from prefix (e.g., 'LegoGears_v10' -> 'v10'); safe if absent
        ratio_tag=_derive_ratio_tag_from_prefix(args.prefix),
    )
    print(f"  {yaml_path}  (Ultralytics YAML)")

    ctrain, cvalid = len(train_paths), len(valid_paths)
    print(
        "[setup] Wrote:\n"
        f"  {train_file}  ({ctrain} images)\n"
        f"  {valid_file}  ({cvalid} images)\n"
        f"  {data_file}\n"
        f"  {manifest}\n"
        f"[setup] Mode={mode_str}  Neg subdirs={neg_subdirs_for_print}"
    )

if __name__ == "__main__":
    main()
