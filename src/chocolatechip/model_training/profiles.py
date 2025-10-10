from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class DatasetSpec:
    root: str
    sets: Tuple[str, ...]
    classes: int
    names: str
    prefix: str
    seed: int = 9001
    neg_subdirs: Tuple[str, ...] = tuple()
    exts: Tuple[str, ...] = (".jpg",)
    legos: bool = False  # special lego split

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

    # sweep control (darknet only; ultralytics ignores unless you use your own splitter)
    val_fracs: Tuple[float, ...] = (0.20,)

    # dataset recipe for (re)generating splits per val_frac
    dataset: DatasetSpec | None = None

    # ultralytics only
    epochs: int | None = None
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
        iterations=4000, learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny"),
        val_fracs=(0.10, 0.15, 0.20),
        dataset=DatasetSpec(
            root="/workspace/LegoGears_v2",
            sets=("set_01", "set_02_empty", "set_03"),
            classes=5,
            names="LegoGears.names",
            prefix="LegoGears",
            seed=9001,
            neg_subdirs=("set_02_empty",),
            exts=(".jpg",),
            legos=False,
        ),
    ),
    "LegoGearsUltra": TrainProfile(
        name="LegoGearsUltra",
        backend="ultralytics",
        data_path="", cfg_out="",
        width=224, height=160,
        batch_size=64, subdivisions=8,
        iterations=4000, learning_rate=0.00261,
        templates=(),
        val_fracs=(0.20,),   # ignored unless you hook a splitter
        dataset=None,
        epochs=None,
        ultra_data="LG_v2.yaml",
        ultra_model="yolo11n.pt",
    ),
}

def get_profile(key: str) -> TrainProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(PROFILES.keys())}")
    return PROFILES[key]
