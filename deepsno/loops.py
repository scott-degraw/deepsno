import math
from pathlib import Path

import torch
import tqdm
import uproot
import wandb
from torch import nn
from torch.utils import _pytree as pytree
from torch.utils import data

from deepsno.metrics.metric_monitor import MetricMonitor
from deepsno.utils.profiling import LoopProfiler

TQDM_KWARGS = {
    "bar_format": "{desc:<10} {percentage:>5.1f}% |[{bar}]{r_bar}",
    "ascii": " =",
    "unit": "batch",
    "dynamic_ncols": True,
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


@torch.inference_mode()
def predict(
    model: nn.Module,
    dataloader: data.DataLoader,
    file: uproot.WritableFile,
    device: str | torch.device,
    predict_name: str = "predict",
    truth_name: str = "truth",
):
    model.to(device)
    model.eval()
    model.output_unnorm = True

    first_batch = True
    for inputs, entries in tqdm.tqdm(dataloader, desc="Test", **TQDM_KWARGS):
        inputs = pytree.tree_map(lambda x: x.to(device), inputs)

        predicts = model(**inputs)

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
    val_norm: bool = False,
) -> float:
    model.to(device)
    model.eval()
    model.output_unnorm = not val_norm

    if metric_monitor is not None:
        if global_step is None:
            raise ValueError("'global_step' must be given a value that is not 'None'")
        metric_monitor.reset()

    metric.reset()

    for inputs, truth in tqdm.tqdm(dataloader, desc="Validation", leave=False, **TQDM_KWARGS):
        inputs = to_device(inputs, device)
        truth = to_device(truth, device)

        predict = model(**inputs)

        if not model.output_unnorm:
            truth = model.output_normalize(truth)

        metric.update(predict, truth)

        if metric_monitor is not None:
            metric_monitor.update(predict=predict, truth=truth)

    if metric_monitor is not None:
        metric_monitor.compute(global_step)

    return metric.compute()


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
    num_epochs: int | None = None,
    num_steps: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    train_unnorm: bool = False,
    val_norm: bool = False,
    max_grad_norm: float = 0.0,
    memory_unit: str = "MiB",
    profile: bool = False,
    profiling_unit: str = "ms",
    metric_monitor: MetricMonitor | None = None,
    autocast_dtype: torch.dtype = torch.float32,
):
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    profiler = LoopProfiler(
        writer=run,
        profiles=[
            "train_data_load",
            "data_to_device",
            "forward_pass",
            "loss_calc",
            "backward_pass",
            "step_total",
            "validation",
            "model_save",
        ],
        cuda_sync="cuda" in device.type,
        profiling_unit=profiling_unit,
        disable=not profile,
    )

    if (num_epochs is not None) and (num_steps is not None):
        raise ValueError("Only 'num_epochs' or 'num_steps' can be given, not both.")
    if (num_epochs is None) and (num_steps is None):
        raise ValueError("Either 'num_epochs' or 'num_steps' must be provided.")

    if num_steps is not None:
        num_epochs = (num_steps - 1) // len(train_dataloader) + 1

    print(f"Performing {num_epochs} epochs through the training dataset", flush=True)

    model.to(device)
    model.train()
    model.output_unnorm = train_unnorm
    loss_fn.to(device)

    sub_epoch = 0
    training = True
    step_num = 0
    rolling_loss = 0.0
    with tqdm.tqdm(desc="Train", total=num_steps, **TQDM_KWARGS) as progress_bar:
        while training:
            profiler.start("step_total")
            profiler.start("train_data_load")
            for inputs, truth in train_dataloader:
                profiler.stop("train_data_load")
                step_num += 1
                log_this_step = step_num % log_interval == 0

                if num_steps is not None and step_num == num_steps:
                    training = False
                    break

                progress_bar.update()

                optimizer.zero_grad()

                profiler.start("data_to_device")
                inputs = to_device(inputs, device)
                truth = to_device(truth, device)
                profiler.stop("data_to_device")

                if not model.output_unnorm:
                    truth = model.output_normalize(truth)

                profiler.start("forward_pass")
                predict = model(**inputs)
                profiler.stop("forward_pass")

                profiler.start("loss_calc")
                loss = loss_fn(predict, truth)
                rolling_loss += loss.item()
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite")
                profiler.stop("loss_calc")
                if log_this_step:
                    run.log({"Loss/train": rolling_loss / log_interval}, step=step_num)
                    rolling_loss = 0.0

                profiler.start("backward_pass")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                profiler.stop("backward_pass")

                if scheduler is not None:
                    scheduler.step()
                    if log_this_step:
                        run.log({"learning_rate": scheduler.get_last_lr()[0]}, step=step_num)

                profiler.stop("step_total")

                if step_num % val_num_steps == 0:
                    del inputs, truth, predict, loss
                    log_this_step = True
                    profiler.start("validation")
                    val_loss = validate(
                        val_dataloader,
                        device=device,
                        model=model,
                        metric=val_metric,
                        global_step=step_num,
                        metric_monitor=metric_monitor,
                        val_norm=val_norm,
                    )
                    profiler.stop("validation")

                    model.train()
                    model.output_unnorm = train_unnorm

                    if not math.isfinite(val_loss):
                        raise ValueError("Validation loss is not finite")
                    run.log({"Loss/val": val_loss}, step=step_num)

                    if val_metric_is_inverted:
                        val_loss = -val_loss

                    profiler.start("model_save")
                    state_dict = {
                        "sub_epoch": sub_epoch,
                        "model": detach_to_cpu(model.state_dict()),
                        "optimizer": detach_to_cpu(optimizer.state_dict()),
                        "scheduler": None if scheduler is None else detach_to_cpu(scheduler.state_dict()),
                    }

                    filename = f"sub_epoch={sub_epoch}_val_loss={val_loss}.pt"

                    torch.save(state_dict, checkpoint_dir / filename)
                    profiler.stop("model_save")

                    sub_epoch += 1

                if log_this_step:
                    run.log({}, step=step_num, commit=True)
                profiler.start("step_total")
                profiler.start("train_data_load")

    print("Training completed")
