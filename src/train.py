from typing import Type

import torch
from torch import nn
from torch.utils import _pytree as pytree
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


def validate(
    dataloader: data.DataLoader,
    device: str | torch.device,
    model: nn.Module,
    metric_fn: nn.Module,
) -> float:
    model = model.to(device)
    model = model.eval()
    model.output_unnorm = True

    with torch.no_grad():
        val_metric_sum = 0
        n_data_points = 0
        for inputs, truth in dataloader:
            inputs = pytree.tree_map(lambda x: x.to(device), inputs)
            truth = pytree.tree_map(lambda x: x.to(device), truth)

            batch_size = truth.shape[0]
            n_data_points += batch_size

            predict = model(**inputs)

            metric = batch_size * metric_fn(predict, truth)
            val_metric_sum += metric

        val_metric = val_metric_sum / n_data_points
        return val_metric.item()


class VaswamiLRSchedule:
    def __init__(self, d_model: int, n_warmup_steps: int):
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps

    def __call__(self, step_num: int) -> float:
        return (1 / torch.sqrt(self.d_model)) * min(1 / torch.sqrt(step_num), step_num * self.n_warmup_steps ** (-1.5))


def train(
    model: nn.Module,
    device: str | torch.device,
    dataset: data.Dataset,
    train_val_split: float,
    batch_size: int,
    shuffle: bool,
    num_epochs: int,
    optimizer_class: Type[torch.optim.Optimizer],
    optimizer_kwargs: dict,
    loss_fn: nn.Module,
    val_metric_fn: nn.Module,
    val_batch_size: int | None = None,
):
    if val_batch_size is None:
        val_batch_size = batch_size

    train_set, val_set = data.random_split(dataset, [train_val_split, 1 - train_val_split])

    train_dataloader = data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

    position_means = dataset.position_means
    position_rmsds = dataset.position_rmsds
    model.add_input_norm(hit_time_mean=dataset.hit_time_mean, hit_time_rmsd=dataset.hit_time_rmsd)
    model.add_output_unnorm(
        position_means=torch.from_numpy(position_means),
        position_rmsds=torch.from_numpy(position_rmsds),
        output_unnorm=False,
    )

    model = model.to(device)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    optimizer.zero_grad()

    writer = SummaryWriter()

    writer.add_scalar("Number of training events", len(train_set))
    writer.add_scalar("Number of validation events", len(val_set))
    writer.add_scalar("Number of training batches", len(train_dataloader))

    position_means = torch.from_numpy(position_means).to(device)
    position_rmsds = torch.from_numpy(position_rmsds).to(device)

    it_num = 0
    for epoch_num in range(num_epochs):
        for inputs, truth in train_dataloader:
            inputs = pytree.tree_map(lambda x: x.to(device), inputs)
            truth = pytree.tree_map(lambda x: x.to(device), truth)

            truth = (truth - position_means) / position_rmsds

            predict = model(**inputs)

            loss = loss_fn(predict, truth)
            writer.add_scalar("Loss/train", loss.item(), it_num)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            it_num += 1

            if it_num % 500 == 0:
                val_metric = validate(val_dataloader, device=device, model=model, metric_fn=val_metric_fn)
                model.output_unnorm = False
                model.train()

                writer.add_scalar("Loss/val", val_metric, it_num)

    writer.flush()
    writer.close()
