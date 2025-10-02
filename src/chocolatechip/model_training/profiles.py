from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class TrainProfile:
    name: str
    backend: str              # "darknet" or "ultralytics"

    # dataset-specific (darknet)
    data_path: str
    cfg_out: str

    # training knobs
    width: int
    height: int
    batch_size: int
    subdivisions: int
    iterations: int
    learning_rate: float

    # choose ONE of the following per run mode:
    template: str | None = None           # single-template mode
    templates: Tuple[str, ...] = tuple()  # multi-template mode

    # ultralytics (ignored for darknet)
    ultra_args: str = "task=detect mode=train data=LG_v2.yaml model=yolo11n.pt epochs=200 imgsz=640 batch=16"

PROFILES = {
    "LegoGears": TrainProfile(
        name="LegoGears",
        backend="darknet",
        data_path="/workspace/LegoGears_v2/LegoGears.data",
        cfg_out="/workspace/LegoGears_v2/LegoGears.cfg",
        width=224, height=160,
        batch_size=64, subdivisions=8,
        iterations=3000, learning_rate=0.00261,
        # run both templates in one go:
        templates=("yolov4-tiny", "yolov7-tiny"),
    ),
    # If you want single-template profiles too, you can add:
    # "LegoGears_v4tiny": TrainProfile(..., template="yolov4-tiny"),
    # "LegoGears_v7tiny": TrainProfile(..., template="yolov7-tiny"),
}

def get_profile(key: str) -> TrainProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(PROFILES)}")
    return PROFILES[key]
