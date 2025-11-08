from pathlib import Path

import h5py
import torch
import tqdm
from torch import nn
from torch.utils import _pytree as pytree
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from deepsno.metrics.metric_monitor import MetricMonitor
from deepsno.utils.profiling import LoopProfiler
from deepsno.utils.train import convert_byte_units

TQDM_KWARGS = {
    "bar_format": "{desc:<10} {percentage:>5.1f}% |[{bar}]{r_bar}",
    "ascii": " =",
    "unit": "batch",
    "dynamic_ncols": True,
}


@torch.inference_mode()
def test(
    model: nn.Module,
    dataloader: data.DataLoader,
    group: h5py.File | h5py.Group,
    dataset_len: int,
    device: str | torch.device,
    predict_key: str | None = None,
):
    model.to(device)
    model.eval()
    model.output_unnorm = True

    inputs, _ = next(iter(dataloader))
    inputs = pytree.tree_map(lambda x: x.to(device), inputs)
    predicts = model(**inputs)
    if predict_key is not None:
        predicts = predicts[predict_key]

    batch_size = predicts.shape[0]
    data_shape = predicts.shape[1:]
    dataset_shape = (dataset_len, *data_shape)
    dataset_dtype = predicts.cpu().numpy().dtype

    predict_dset = group.create_dataset("predict", shape=dataset_shape, dtype=dataset_dtype)

    start_row = 0
    for inputs, truth in tqdm.tqdm(dataloader, desc="Test", **TQDM_KWARGS):
        inputs = pytree.tree_map(lambda x: x.to(device), inputs)
        predicts = model(**inputs)
        if predict_key is not None:
            predicts = predicts[predict_key]

        batch_size = min(start_row + predicts.shape[0], dataset_len) - start_row
        predict_dset[start_row : start_row + batch_size] = predicts[:batch_size].cpu().numpy()

        start_row += batch_size
        if start_row >= dataset_len:
            break


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
        inputs = pytree.tree_map(lambda x: x.to(device), inputs)
        truth = pytree.tree_map(lambda x: x.to(device), truth)

        predict = model(**inputs)

        if metric_monitor is not None:
            metric_monitor.update(predict=predict, truth=truth)

        metric.update(predict, truth)

    if metric_monitor is not None:
        metric_monitor.compute(global_step)

    return metric.compute()


def train(
    checkpoint_dir: str | Path,
    writer: SummaryWriter,
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
):
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    profiler = LoopProfiler(
        writer=writer,
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

    if "cuda" in device.type:
        writer.add_scalar(
            f"GPU/total_memory-{memory_unit}",
            convert_byte_units(torch.cuda.mem_get_info()[1], memory_unit),
            new_style=True,
        )

    sub_epoch = 0
    training = True
    step_num = 0
    with tqdm.tqdm(desc="Train", total=num_steps, **TQDM_KWARGS) as progress_bar:
        while training:
            profiler.start("step_total")
            profiler.start("train_data_load")
            for inputs, truth in train_dataloader:
                profiler.stop("train_data_load")

                if num_steps is not None and step_num == num_steps:
                    training = False
                    break

                progress_bar.update()

                if "cuda" in device.type:
                    writer.add_scalar(
                        "GPU/memory_allocated-MiB",
                        convert_byte_units(torch.cuda.max_memory_reserved(), memory_unit),
                        step_num,
                        new_style=True,
                    )
                    torch.cuda.reset_peak_memory_stats()

                optimizer.zero_grad()

                profiler.start("data_to_device")
                inputs = pytree.tree_map(lambda x: x.to(device), inputs)
                truth = pytree.tree_map(lambda x: x.to(device), truth)
                profiler.stop("data_to_device")

                if not model.output_unnorm:
                    truth = model.output_normalize(truth)

                profiler.start("forward_pass")
                predict = model(**inputs)
                profiler.stop("forward_pass")

                profiler.start("loss_calc")
                loss = loss_fn(predict, truth)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite")
                profiler.stop("loss_calc")
                writer.add_scalar("Loss/train", loss.detach().item(), step_num, new_style=True)

                profiler.start("backward_pass")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                profiler.stop("backward_pass")

                if scheduler is not None:
                    scheduler.step()
                    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], step_num, new_style=True)

                profiler.stop("step_total")

                step_num += 1

                if step_num % val_num_steps == 0:
                    del inputs, truth, predict, loss
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
                    writer.add_scalar("Loss/val", val_loss, step_num, new_style=True)

                    if val_metric_is_inverted:
                        val_loss = -val_loss

                    profiler.start("model_save")
                    state_dict = {
                        "sub_epoch": sub_epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }

                    state_dict["scheduler"] = None if scheduler is None else scheduler.state_dict()

                    filename = f"sub_epoch={sub_epoch}_val_loss={val_loss}.pt"

                    torch.save(state_dict, checkpoint_dir / filename)
                    profiler.stop("model_save")

                    sub_epoch += 1

                profiler.log_all(step_num)
                profiler.start("step_total")
                profiler.start("train_data_load")

    print("Training completed")
