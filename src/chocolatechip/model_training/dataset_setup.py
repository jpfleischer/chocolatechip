# src/chocolatechip/model_training/dataset_setup.py
from __future__ import annotations
from pathlib import Path
import argparse

def collect_images(root: Path, subdirs: list[str], exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> list[Path]:
    imgs: list[Path] = []
    for d in subdirs:
        p = (root / d)
        if not p.is_dir():
            raise FileNotFoundError(f"Missing dataset subdir: {p}")
        for x in sorted(p.iterdir()):
            if x.is_file() and x.suffix.lower() in exts:
                imgs.append(x.resolve())
    if not imgs:
        raise RuntimeError(f"No images found under {root} in {subdirs}")
    return imgs

def write_list(path: Path, paths: list[Path]) -> None:
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

def main():
    ap = argparse.ArgumentParser(description="Prepare train/valid lists and a Darknet .data file.")
    ap.add_argument("--root", required=True, help="Dataset root (e.g., /ultralytics/LegoGears_v2)")
    ap.add_argument("--sets", nargs="+", required=True, help="Subdirectories containing images (e.g., set_01 set_02_empty set_03)")
    ap.add_argument("--classes", type=int, required=True, help="Number of classes")
    ap.add_argument("--names", default="LegoGears.names", help="Names filename (inside root)")
    ap.add_argument("--prefix", default="LegoGears", help="Prefix for generated files")
    ap.add_argument("--exts", nargs="*", default=[".jpg"], help="Image extensions to include")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    imgs = collect_images(root, args.sets, tuple(e.lower() for e in args.exts))

    train_file = root / f"{args.prefix}_train.txt"
    valid_file = root / f"{args.prefix}_valid.txt"
    data_file  = root / f"{args.prefix}.data"
    names_file = root / args.names

    # For now, use the same list for train and valid (matches your current behavior)
    write_list(train_file, imgs)
    write_list(valid_file, imgs)
    write_darknet_data_file(
        data_file,
        classes=args.classes,
        train=train_file,
        valid=valid_file,
        names=names_file,
        backup=root,
    )

    print(f"[setup] Wrote:\n  {train_file}\n  {valid_file}\n  {data_file}")

if __name__ == "__main__":
    main()
