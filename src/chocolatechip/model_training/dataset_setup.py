# src/chocolatechip/model_training/dataset_setup.py
from __future__ import annotations
from pathlib import Path
import argparse, random, json
from typing import Dict, List

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
    """Return (train, valid, stats) with a deterministic 'random' split.

    - negatives are any subdir in `neg_subdirs`
    - validation keeps the global negative/positive ratio (rounded)
    """
    rng = random.Random(seed)

    negatives: List[Path] = []
    positives: List[Path] = []
    for d, lst in images_by_subdir.items():
        (negatives if d in neg_subdirs else positives).extend(lst)

    negatives.sort()
    positives.sort()

    # Shuffle deterministically, then slice
    rng.shuffle(negatives)
    rng.shuffle(positives)

    n_neg, n_pos = len(negatives), len(positives)
    n_total = n_neg + n_pos
    if n_total == 0:
        raise RuntimeError("No images found overall.")

    n_val = max(1, round(val_frac * n_total))
    # keep the negative/positive ratio in val
    n_val_neg = min(n_neg, round(n_val * (n_neg / n_total))) if n_neg else 0
    n_val_pos = n_val - n_val_neg
    n_val_pos = min(n_pos, n_val_pos)

    val_neg = negatives[:n_val_neg]
    val_pos = positives[:n_val_pos]
    valid = sorted(val_neg + val_pos)

    train = sorted(negatives[n_val_neg:] + positives[n_val_pos:])

    # safety checks
    inter = set(train).intersection(valid)
    assert not inter, f"Train/valid overlap detected: {list(inter)[:3]}"

    stats = {
        "seed": seed,
        "val_frac": val_frac,
        "counts": {
            "total": n_total,
            "negatives_total": n_neg,
            "positives_total": n_pos,
            "valid_total": len(valid),
            "valid_negatives": len(val_neg),
            "valid_positives": len(val_pos),
            "train_total": len(train),
        },
        "neg_subdirs": neg_subdirs,
    }
    return train, valid, stats

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
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    images_by_subdir = collect_images_by_subdir(root, args.sets, tuple(e.lower() for e in args.exts))

    # Auto-detect negatives if not provided
    if args.neg_subdirs is None:
        neg_subdirs = [d for d in args.sets if ("empty" in d.lower() or "neg" in d.lower())]
    else:
        neg_subdirs = args.neg_subdirs

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

    c = stats["counts"]
    print(
        "[setup] Wrote:\n"
        f"  {train_file}  ({c['train_total']} images)\n"
        f"  {valid_file}  ({c['valid_total']} images; {c['valid_negatives']} neg / {c['valid_positives']} pos)\n"
        f"  {data_file}\n"
        f"  {manifest}\n"
        f"[setup] Seed={stats['seed']}  Neg subdirs={stats['neg_subdirs']}"
    )

if __name__ == "__main__":
    main()
