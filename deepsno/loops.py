import math
from pathlib import Path
from typing import Iterable
import time

import torch
import torch.distributed as dist
import tqdm
import uproot
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import _pytree as pytree
from torch.utils import data

from deepsno.metrics.metric_monitor import MetricMonitor


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap a DistributedDataParallel model to get the underlying module."""
    if isinstance(model, DDP):
        return model.module
    return model


TQDM_KWARGS = {
    "bar_format": "{desc:<10} {percentage:>5.1f}% |[{bar}]{r_bar}",
    "ascii": " =",
    "unit": "batch",
    "dynamic_ncols": True,
    "smoothing": 0.3,
}


def to_device(d: dict, device: str | torch.device) -> dict:
    def to(x):
        try:
            return x.to(device)
        except AttributeError:
            return x

    return pytree.tree_map(lambda x: to(x), d)


def detach_to_cpu(d: dict) -> dict:
    def to(x):
        try:
            return x.detach().cpu()
        except AttributeError:
            return x

    return pytree.tree_map(lambda x: to(x), d)


def bench_dataloader(dataloader: data.DataLoader, num_steps: int | None = None) -> None:
    """Iterate a dataloader and print throughput statistics."""
    batch_times = []
    samples_per_batch = []
    t_start = time.perf_counter()
    t_batch = t_start

    for i, (inputs, _) in enumerate(tqdm.tqdm(dataloader, total=num_steps, desc="Bench", **TQDM_KWARGS)):
        t_now = time.perf_counter()
        batch_times.append(t_now - t_batch)
        t_batch = t_now
        samples_per_batch.append(len(next(iter(inputs.values()))))
        if num_steps is not None and i + 1 >= num_steps:
            break

    total_time = time.perf_counter() - t_start
    n_batches = len(batch_times)
    n_samples = sum(samples_per_batch)
    batch_times_t = torch.tensor(batch_times)

    print(f"\n--- Dataloader benchmark ({n_batches} batches, {n_samples} samples) ---")
    print(f"  Total time     : {total_time:.2f} s")
    print(f"  Throughput     : {n_samples / total_time:.1f} samples/s  |  {n_batches / total_time:.2f} batches/s")
    print(
        f"  Batch time     : mean {batch_times_t.mean():.3f} s  "
        f"std {batch_times_t.std():.3f} s  "
        f"min {batch_times_t.min():.3f} s  "
        f"max {batch_times_t.max():.3f} s"
    )


@torch.inference_mode()
def predict(
    model: nn.Module,
    dataloader: data.DataLoader,
    file: uproot.WritableFile,
    device: str | torch.device,
    keys: Iterable[str] | None = None,
    predict_name: str = "predict",
    truth_name: str = "truth",
):
    model.to(device)
    model.eval()
    _unwrap(model).output_unnorm = True

    first_batch = True
    for inputs, entries in tqdm.tqdm(dataloader, desc="Test", **TQDM_KWARGS):
        inputs = to_device(inputs, device)

        predicts = model(**inputs)
        if keys is not None:
            predicts = {k: predicts[k] for k in keys}

        predicts = pytree.tree_map(lambda x: x.detach().cpu().numpy(), predicts)
        for key, value in entries.items():
            try:
                entries[key] = value.detach().cpu().numpy()
            except AttributeError:
                pass

        if first_batch:
            file[predict_name] = predicts
            file[truth_name] = entries
            first_batch = False
        else:
            file[predict_name].extend(predicts)
            file[truth_name].extend(entries)


@torch.inference_mode()
def validate(
    dataloader: data.DataLoader,
    device: str | torch.device,
    model: nn.Module,
    metric: object,
    global_step: int | None = None,
    metric_monitor: MetricMonitor | None = None,
    normalize_truth: bool = True,
) -> float:
    model.to(device)
    model.eval()
    _unwrap(model).output_unnorm = not normalize_truth

    if metric_monitor is not None:
        if global_step is None:
            raise ValueError("'global_step' must be given a value that is not 'None'")
        metric_monitor.reset()

    metric.reset()

    for inputs, truth in tqdm.tqdm(dataloader, desc="Validation", leave=False, **TQDM_KWARGS):
        inputs = to_device(inputs, device)
        truth = to_device(truth, device)

        predict = model(**inputs)

        if normalize_truth:
            truth = _unwrap(model).output_normalize(truth)
        metric.update(predict, truth)

        if metric_monitor is not None:
            metric_monitor.update(predict=predict, truth=truth)

    if metric_monitor is not None:
        metric_monitor.compute(global_step)

    val = metric.compute()

    if dist.is_available() and dist.is_initialized():
        val_tensor = torch.tensor(val, dtype=torch.float64, device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
        val = val_tensor.item()

    return val


def train(
    checkpoint_dir: str | Path,
    run: wandb.Run,
    log_interval: int,
    model: nn.Module,
    device: str | torch.device,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    val_metric: object,
    val_metric_is_inverted: bool,
    val_num_steps: int,
    num_steps: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    max_grad_norm: float = 0.0,
    metric_monitor: MetricMonitor | None = None,
    rank: int = 0,
    initial_step: int = 0,
    initial_sub_epoch: int = 0,
    train_norm: bool = True,
    val_norm: bool = True,
):
    is_main = rank == 0
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model.to(device)
    model.train()
    _unwrap(model).output_unnorm = not train_norm
    loss_fn.to(device)

    sub_epoch = initial_sub_epoch
    training = True
    step_num = initial_step
    rolling_loss = 0.0
    dset_size = 0
    first_dset_print = True
    with tqdm.tqdm(
        desc="Train", total=num_steps, initial=initial_step, disable=not is_main, **TQDM_KWARGS
    ) as progress_bar:
        while training:
            for inputs, truth in train_dataloader:
                dset_size += len(next(iter(inputs.values())))
                step_num += 1
                log_this_step = step_num % log_interval == 0

                progress_bar.update()

                optimizer.zero_grad()

                inputs = to_device(inputs, device)
                truth = to_device(truth, device)

                if train_norm:
                    truth = _unwrap(model).output_normalize(truth)

                predict = model(**inputs)

                loss = loss_fn(predict, truth)
                rolling_loss += loss.item()
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite")
                if log_this_step and is_main:
                    run.log({"Loss/train": rolling_loss / log_interval}, step=step_num)
                    rolling_loss = 0.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    if log_this_step and is_main:
                        run.log({"learning_rate": scheduler.get_last_lr()[0]}, step=step_num)

                if step_num % val_num_steps == 0:
                    del inputs, truth, predict, loss
                    log_this_step = True
                    val_loss = validate(
                        val_dataloader,
                        device=device,
                        model=model,
                        metric=val_metric,
                        global_step=step_num,
                        metric_monitor=metric_monitor if is_main else None,
                        normalize_truth=val_norm,
                    )

                    model.train()
                    _unwrap(model).output_unnorm = not train_norm

                    if not math.isfinite(val_loss):
                        raise ValueError("Validation loss is not finite")

                    if is_main:
                        run.log({"Loss/val": val_loss}, step=step_num)

                    if val_metric_is_inverted:
                        val_loss = -val_loss

                    if is_main:
                        state_dict = {
                            "sub_epoch": sub_epoch,
                            "step_num": step_num,
                            "model": detach_to_cpu(_unwrap(model).state_dict()),
                            "optimizer": detach_to_cpu(optimizer.state_dict()),
                            "scheduler": None if scheduler is None else detach_to_cpu(scheduler.state_dict()),
                        }

                        filename = f"sub_epoch={sub_epoch}_val_loss={val_loss}.pt"

                        torch.save(state_dict, checkpoint_dir / filename)

                    sub_epoch += 1

                if log_this_step and is_main:
                    run.log({}, step=step_num, commit=True)

                if step_num >= num_steps:
                    training = False
                    break

            if first_dset_print and is_main:
                print(f"Dataset size: {dset_size}")
                first_dset_print = False

    if is_main:
        print("Training completed")
