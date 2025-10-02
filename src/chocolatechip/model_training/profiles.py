from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class TrainProfile:
    name: str
    backend: str              # "darknet" or "ultralytics"

    # dataset-specific (darknet; ignored by ultralytics)
    data_path: str
    cfg_out: str

    # training knobs (ultralytics ignores subdivisions/cfg_out/data_path)
    width: int
    height: int
    batch_size: int
    subdivisions: int
    iterations: int           # Darknet max_batches; used to derive Ultralytics epochs
    learning_rate: float

    # choose ONE in darknet mode
    template: str | None = None
    templates: Tuple[str, ...] = tuple()

    # ultralytics only (no ultra_args)
    epochs: int | None = None          # if None, auto-derived from iterations+dataset size
    ultra_data: str = "LG_v2.yaml"
    ultra_model: str = "yolo11n.pt"

PROFILES = {
    "LegoGearsDarknet": TrainProfile(
        name="LegoGearsDarknet",
        backend="darknet",
        data_path="/workspace/LegoGears_v2/LegoGears.data",
        cfg_out="/workspace/LegoGears_v2/LegoGears.cfg",
        width=224, height=160,
        batch_size=64, subdivisions=8,
        iterations=3000, learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny"),
    ),
    "LegoGearsUltra": TrainProfile(
        name="LegoGearsUltra",
        backend="ultralytics",
        data_path="", cfg_out="",
        width=224, height=160,
        batch_size=64, subdivisions=8,
        iterations=3000, learning_rate=0.00261,
        templates=(),
        epochs=None,               # auto-derive from iterations and dataset size
        ultra_data="LG_v2.yaml",
        ultra_model="yolo11n.pt",
    ),
}

def get_profile(key: str) -> TrainProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(PROFILES)}")
    return PROFILES[key]
