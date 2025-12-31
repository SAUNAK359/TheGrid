# hybriddetector/trainer/scheduler.py

from __future__ import annotations

from typing import Sequence, Union

from torch.optim.lr_scheduler import OneCycleLR


def get_scheduler(
    optimizer,
    max_lr: Union[float, Sequence[float]],
    epochs: int,
    steps_per_epoch: int,
):
    """OneCycleLR scheduler.

    Supports a single `max_lr` (float) or per-parameter-group `max_lr` (list/tuple).
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
