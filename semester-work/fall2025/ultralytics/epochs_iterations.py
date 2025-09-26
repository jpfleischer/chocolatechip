import math
from typing import Iterable, List, Dict, Any

def schedule_from_darknet(
    *,
    max_batches: int,
    batch: int,
    dataset_size: int,
    steps: Iterable[int] = (),
    burn_in: int = 0,
    subdivisions: int = 1,
    round_up_ultralytics: bool = True,
) -> Dict[str, Any]:
    """
    Map a Darknet schedule to Ultralytics (epochs) and report Darknet iterations.

    Returns a dict with:
      - darknet: total_iterations, images_per_iteration, microbatch, steps_iter, burn_in_iter
      - epochs: epochs, step_epochs, warmup_epochs, iters_per_epoch (float and ceil)
    """
    if dataset_size <= 0 or batch <= 0 or max_batches <= 0:
        raise ValueError("dataset_size, batch, and max_batches must be positive")

    # ----- Darknet (native) -----
    total_iterations = int(max_batches)
    images_per_iteration = int(batch)
    microbatch = batch / max(1, subdivisions)  # size of each forward pass when using subdivisions
    steps_iter = list(map(int, steps))
    burn_in_iter = int(burn_in)

    # ----- Conversions to Ultralytics (epochs) -----
    iters_per_epoch_f = dataset_size / batch  # can be fractional
    epochs_f = total_iterations / iters_per_epoch_f
    step_epochs_f = [s / iters_per_epoch_f for s in steps_iter]
    warmup_epochs_f = burn_in_iter / iters_per_epoch_f if burn_in_iter else 0.0

    if round_up_ultralytics:
        epochs = math.ceil(epochs_f)
        step_epochs = [math.ceil(x) for x in step_epochs_f]
        warmup_epochs = math.ceil(warmup_epochs_f) if burn_in_iter else 0
        iters_per_epoch_ceil = math.ceil(iters_per_epoch_f)
    else:
        epochs = epochs_f
        step_epochs = step_epochs_f
        warmup_epochs = warmup_epochs_f
        iters_per_epoch_ceil = iters_per_epoch_f  # keep fractional

    return {
        "darknet": {
            "total_iterations": total_iterations,
            "images_per_iteration": images_per_iteration,
            "microbatch": microbatch,  # = batch/subdivisions
            "steps_iter": steps_iter,
            "burn_in_iter": burn_in_iter,
        },
        "epochs": {
            "epochs": epochs,
            "step_epochs": step_epochs,
            "warmup_epochs": warmup_epochs,
            "iters_per_epoch_float": iters_per_epoch_f,
            "iters_per_epoch_ceil": iters_per_epoch_ceil,
        },
    }

# Example with your cfg: batch=64, subdivisions=8, max_batches=3000, steps=2400,2700, burn_in=1000
if __name__ == "__main__":
    ds = 90  # put your actual image count here
    info = schedule_from_darknet(
        max_batches=3000,
        batch=64,
        dataset_size=ds,
        steps=(2400, 2700),
        burn_in=1000,
        subdivisions=8,
        round_up_ultralytics=True,
    )
    from pprint import pprint
    pprint(info)
