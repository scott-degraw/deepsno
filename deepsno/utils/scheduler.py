import math
import warnings

from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """
    Sets the learning rate of each parameter group using a linear warmup followed by a cosine annealing schedule.
    """

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, warmup_start_lr: float = 0.0, eta_min: float = 0.0, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lrs = [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / max(1, self.warmup_epochs)
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch <= self.max_epochs:
            # Cosine annealing
            # Decay from max_lr (which is base_lr) to eta_min
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lrs = [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
        else:
            # Constant after max_epochs
            lrs = [self.eta_min for _ in self.base_lrs]
            
        return lrs
