from pathlib import Path
from chocolatechip.model_training.coco_build_gt import build_coco_gt
from chocolatechip.model_training.profiles import DatasetSpec

def build_coco_gt_for_dataset(*, dataset: DatasetSpec, valid_list: Path, out_json: Path) -> None:
    if not valid_list.is_file():
        raise FileNotFoundError(f"valid.txt not found: {valid_list}")

    names_path = Path(dataset.root) / dataset.names
    if not names_path.is_file():
        raise FileNotFoundError(f"names file not found: {names_path}")

    ann_root = Path(dataset.root) / (dataset.flat_dir or "")
    build_coco_gt(
        ann_root=str(ann_root),
        out_json=str(out_json),
        list_file=str(valid_list),
        names_path=str(names_path),
    )
