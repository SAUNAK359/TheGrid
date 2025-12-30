# hybriddetector/trainer/scheduler.py

from torch.optim.lr_scheduler import OneCycleLR

def get_scheduler(optimizer, max_lr, epochs, steps_per_epoch):
    """
    OneCycleLR scheduler
    """
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    return scheduler
