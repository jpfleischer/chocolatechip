from __future__ import annotations
from dataclasses import dataclass, replace, field
from typing import Tuple, Dict, Optional, Any
from pathlib import Path
import json
import math

@dataclass(frozen=True)
class DatasetSpec:
    root: str
    sets: Tuple[str, ...]
    classes: int
    names: str
    prefix: str
    split_seed: int = 9001
    neg_subdirs: Tuple[str, ...] = tuple()
    exts: Tuple[str, ...] = (".jpg",)
    flat_dir: str | None = None
    legos: bool = False  # special lego split
    url: str | None = None
    sha256: str | None = None 
    # Do not create/download/normalize this dataset; fail fast if missing.
    require_existing: bool = False

@dataclass(frozen=True)
class TrainProfile:
    name: str
    backend: str              # "darknet" or "ultralytics"
    data_path: str
    cfg_out: str

    # training knobs
    width: int
    height: int
    batch_size: int
    subdivisions: int
    iterations: int
    learning_rate: float

    # darknet template selection
    template: str | None = None
    templates: Tuple[str, ...] = tuple()
    val_fracs: Tuple[float, ...] = (0.20,)

    # SINGLE color knob: None -> keep template HSV; otherwise a preset name or "s,e,h"
    color_preset: Optional[str] = None
    color_presets: Tuple[Optional[str], ...] = (None,) # sweep list, e.g. (None, "preserve")

    tag_color_preset: bool = False

    # mAP evaluation knobs (darknet)
    map_thresh: float | None = None
    iou_thresh: float | None = None
    map_points: int | None = None

    # dataset recipe for split regen
    dataset: DatasetSpec | None = None

    # ultralytics only
    epochs: int | None = None
    ultra_data: str = "LG_v2.yaml"
    ultra_model: str = "yolo11n.pt"

    # ultralytics-only training RNG; ignored by Darknet
    training_seed: Optional[int] = None

    # request N GPUs (slice of visible devices). None => use all visible.
    num_gpus: Optional[int] = None


    sweep_keys: Tuple[str, ...] = tuple()
    sweep_values: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)

# ---------------- equalization helpers (profiles-level policy) ----------------

# cache target_epochs per profile name so the first split sets the baseline
_equalize_cache: Dict[tuple, float] = {}

def _manifest_from_data_path(data_path: str) -> Path:
    """'/path/LegoGears_v15.data' -> '/path/LegoGears_v15_split.json'."""
    p = Path(data_path)
    return p.with_name(p.stem + "_split.json")

def read_split_counts_from_data(data_path: str) -> Tuple[int, int]:
    """
    Return (n_train, n_valid) using the split manifest produced by dataset_setup.py.
    """
    mpath = _manifest_from_data_path(data_path)
    js = json.loads(Path(mpath).read_text(encoding="utf-8"))
    c = js.get("counts", {})
    return int(c.get("train_total", 0)), int(c.get("valid_total", 0))

def equalize_for_split(profile: TrainProfile, *, data_path: str, mode: str = "iterations") -> TrainProfile:
    """
    Returns a new TrainProfile where either iterations or batch_size has been
    adjusted so that approx_epochs ≈ constant across splits.

      approx_epochs ≈ (iterations * batch_size) / train_images

    mode:
      - "iterations" (recommended): keep batch the same, solve iterations.
      - "batch": keep iterations the same, solve batch (must be multiple of subdivisions).
    """
    global _equalize_cache

    # Count training images for this split (via manifest)
    try:
        T, _ = read_split_counts_from_data(data_path)
    except Exception:
        T = 0
    if T <= 0:
        return replace(profile, data_path=data_path)

    # Establish target epochs on first call for this profile name
    key = (profile.name, getattr(profile, "template", None), getattr(profile, "color_preset", None))
    if key not in _equalize_cache:
        _equalize_cache[key] = (profile.iterations * profile.batch_size) / max(1, T)
        if _equalize_cache[key] <= 0:
            _equalize_cache[key] = 1.0

    target_epochs = _equalize_cache[key]

    if mode == "iterations":
        new_iter = int(math.ceil(target_epochs * T / max(1, profile.batch_size)))
        new_iter = max(100, new_iter)
        return replace(profile, data_path=data_path, iterations=new_iter)

    elif mode == "batch":
        k = max(1, profile.subdivisions)
        raw = target_epochs * T / max(1, profile.iterations)
        new_batch = int(round(raw / k) * k)
        new_batch = min(max(new_batch, k), 1024)
        return replace(profile, data_path=data_path, batch_size=new_batch)

    # Unknown mode; only inject the data_path
    return replace(profile, data_path=data_path)

# ---------------- registered profiles ----------------

PROFILES = {
    "LegoGearsDarknet": TrainProfile(
        name="LegoGearsDarknet",
        backend="darknet",
        data_path="/workspace/LegoGears_v2/LegoGears.data",
        cfg_out="/workspace/LegoGears_v2/LegoGears.cfg",
        width=224, height=160,
        batch_size=64, subdivisions=1,
        iterations=6000, learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny",),
        # templates=("yolov7-tiny",),
        # templates=("yolov4-tiny",),
        val_fracs=(0.10, 0.15, 0.20),
        # val_fracs=(0.20,),
        sweep_keys=("templates", "val_fracs", "num_gpus"),
        
        sweep_values={"num_gpus": (1,)},
        dataset=DatasetSpec(
            root="/workspace/LegoGears_v2",
            sets=("set_01", "set_02_empty", "set_03"),
            classes=5,
            names="LegoGears.names",
            prefix="LegoGears",
            split_seed=9001,
            neg_subdirs=("set_02_empty",),
            exts=(".jpg",),
            legos=False,
            url="https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip",
            sha256="126980d3e43986bbd3d785ac16f6430e9bf3b726e65a30574bb3c9ba06a4462e",
        ),
    ),
    "LeatherDarknet": TrainProfile(
        name="LeatherDarknet",
        backend="darknet",
        data_path="/workspace/leather/leather.data",
        cfg_out="/workspace/leather/leather.cfg",
        width=256, height=256,
        # width=480, height=480,
        batch_size=64, subdivisions=1,
        iterations=7000, learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny"),
        # templates=("yolov7-tiny"),
        val_fracs=(0.20,),
        color_presets=(None, "preserve"),
        tag_color_preset=True,   # <— ONLY Leather opts in
        sweep_keys=("templates", "color_presets"),
        sweep_values={},
        dataset=DatasetSpec(
            root="/workspace/leather",
            sets=("color", "cut", "fold", "glue", "poke", "good_1", "good_2"),
            classes=5,
            names="leather.names",
            prefix="leather",
            split_seed=9001,
            neg_subdirs=("good_1", "good_2"),
            exts=(".jpg", ".png"),
            url="https://g-665dcc.55ba.08cc.data.globus.org/leather_oct_25.zip",
            sha256="87fba3c49bce7342af51e1fe6df5a470862f201c0e8e25bf3ea80a0c6f238d8c",
            flat_dir="darkmark_image_cache/resize",
        ),
        # map_thresh=0.50,
        # iou_thresh=0.60,
        # map_points=None,
    ),

    "LegoGearsUltra": TrainProfile(
        name="LegoGearsUltra",
        backend="ultralytics",
        data_path="", cfg_out="",
        width=224, height=160,
        batch_size=64, subdivisions=1,
        iterations=6000, learning_rate=0.00261,
        templates=(),
        val_fracs=(0.20,),
        dataset=DatasetSpec(
            root="/workspace/LegoGears_v2",
            sets=("set_01", "set_02_empty", "set_03"),
            classes=5,
            names="LegoGears.names",
            prefix="LegoGears",
            split_seed=9001,
            neg_subdirs=("set_02_empty",),
            exts=(".jpg",),
            legos=False,
            url="https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip",
            sha256="126980d3e43986bbd3d785ac16f6430e9bf3b726e65a30574bb3c9ba06a4462e",
        ),
        epochs=None,
        ultra_data="",
        ultra_model="yolo11n.pt",
        training_seed = 42,
    ),

    "LeatherUltra": TrainProfile(
        name="LeatherUltra",
        backend="ultralytics",
        data_path="", cfg_out="",
        width=256, height=256,
        batch_size=64, subdivisions=1,
        iterations=7000, learning_rate=0.00261,
        templates=(),
        tag_color_preset=True,
        val_fracs=(0.20,),
        dataset=DatasetSpec(
            root="/workspace/leather",
            sets=("color", "cut", "fold", "glue", "poke", "good_1", "good_2"),
            classes=5,
            names="leather.names",
            prefix="leather",
            split_seed=9001,
            neg_subdirs=("good_1", "good_2"),
            exts=(".jpg", ".png"),
            url="https://g-665dcc.55ba.08cc.data.globus.org/leather_oct_25.zip",
            sha256="87fba3c49bce7342af51e1fe6df5a470862f201c0e8e25bf3ea80a0c6f238d8c",
        ),
        epochs=None,
        ultra_data="",
        ultra_model="yolo11n.pt",
        training_seed = 42,
    ),

    "FisheyeTrafficDarknetLocal": TrainProfile(
        name="FisheyeTrafficDarknetLocal",
        backend="darknet",
        data_path="/blue/ranka/j.fleischer/annotation_data/combined.data",
        cfg_out="/blue/ranka/j.fleischer/annotation_data/combined.cfg",
        width=960, height=736,
        batch_size=64, subdivisions=16,
        iterations=8000, learning_rate=0.00261,
        # templates=("yolov7-tiny",),
        templates=("yolov4", "yolov7"),
        val_fracs=(0.10,),
        sweep_keys=("templates",),
        sweep_values={},
        dataset=DatasetSpec(
            root="/blue/ranka/j.fleischer/annotation_data",
            sets=tuple(),                # not needed for local-only
            classes=5,                   # keep in sync with your names file
            names="obj.names",           # adjust if your names file differs
            prefix="combined",
            split_seed=9001,
            neg_subdirs=tuple(),
            exts=(".jpg", ".png"),
            url=None,
            sha256=None,
            require_existing=True,       # <- key behavior
            # If you want optional re-splitting from a flat cache, set:
            flat_dir="darkmark_image_cache/resize",
        ),
    ),

}

def get_profile(key: str) -> TrainProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(PROFILES.keys())}")
    return PROFILES[key]
