from copy import deepcopy
from pathlib import Path

import h5py
import torch
from torch import nn
from torch.utils import _pytree as pytree
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


def test(
    model: nn.Module,
    dataloader: data.DataLoader,
    h5_group: h5py.File | h5py.Group,
    dataset_length: int,
    device: str | torch.device,
):
    model.to(device)
    model.eval()

    _, truth = next(iter(dataloader))
    truth = truth.numpy()
    dataset_shape = (dataset_length, truth.shape[1])
    dataset_dtype = truth.dtype

    position_group = h5_group.create_group("position")
    truth_dset = position_group.create_dataset("truth", shape=dataset_shape, dtype=dataset_dtype)
    predict_dset = position_group.create_dataset("predict", shape=dataset_shape, dtype=dataset_dtype)

    start_row = 0
    with torch.no_grad():
        for inputs, truth in dataloader:
            inputs = pytree.tree_map(lambda x: x.to(device), inputs)
            predicts = model(**inputs)

            batch_size = truth.shape[0]
            batch_slice = slice(start_row, start_row + batch_size)
            truth_dset[batch_slice] = truth.numpy()
            predict_dset[batch_slice] = predicts.cpu().numpy()

            start_row += batch_size


def validate(
    dataloader: data.DataLoader,
    device: str | torch.device,
    model: nn.Module,
    loss_fn: nn.Module,
) -> float:
    model = model.to(device)
    model = model.eval()
    model.output_unnorm = True

    with torch.no_grad():
        val_metric_sum: float = 0
        n_data_points: int = 0
        for inputs, truth in dataloader:
            inputs = pytree.tree_map(lambda x: x.to(device), inputs)
            truth = pytree.tree_map(lambda x: x.to(device), truth)

            batch_size = truth.shape[0]
            n_data_points += batch_size

            predict = model(**inputs)

            metric = batch_size * loss_fn(predict, truth).item()
            val_metric_sum += metric

        val_metric = val_metric_sum / n_data_points
        return val_metric


def train(
    checkpoint_dir: str | Path,
    writer: SummaryWriter,
    model: nn.Module,
    device: str | torch.device,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    val_loss_fn: nn.Module,
    val_loss_is_inverted: bool,
    val_num_steps: int,
    position_means: torch.Tensor,
    position_rmsds: torch.Tensor,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    max_grad_norm: float = 0.0,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model.to(device)
    model.train()

    position_means = torch.from_numpy(position_means).to(device)
    position_rmsds = torch.from_numpy(position_rmsds).to(device)

    it_num = 0
    sub_epoch = 0
    for _ in range(num_epochs):
        for inputs, truth in train_dataloader:
            optimizer.zero_grad()
            inputs = pytree.tree_map(lambda x: x.to(device), inputs)
            truth = pytree.tree_map(lambda x: x.to(device), truth)

            truth = (truth - position_means) / position_rmsds

            predict = model(**inputs)

            loss = loss_fn(predict, truth)
            writer.add_scalar("Loss/train", loss.item(), it_num, new_style=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], it_num, new_style=True)

            it_num += 1

            if it_num % val_num_steps == 0:
                val_loss = validate(val_dataloader, device=device, model=model, loss_fn=val_loss_fn)
                model.output_unnorm = False
                model.train()

                writer.add_scalar("Loss/val", val_loss, it_num, new_style=True)

                if val_loss_is_inverted:
                    val_loss = -val_loss

                state_dict = {
                    "sub_epoch": sub_epoch,
                    "model": deepcopy(model.state_dict()),
                    "optimizer": deepcopy(optimizer.state_dict()),
                }

                if scheduler is not None:
                    state_dict["scheduler"] = deepcopy(scheduler.state_dict())
                state_dict["scheduler"] = None

                filename = f"sub_epoch={sub_epoch}_val_loss={val_loss}.pt"

                torch.save(state_dict, checkpoint_dir / filename)

                sub_epoch += 1
