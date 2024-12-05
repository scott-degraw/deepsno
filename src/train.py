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
        val_metric_sum: float = 0
        n_data_points: int = 0
        for inputs, truth in dataloader:
            inputs = pytree.tree_map(lambda x: x.to(device), inputs)
            truth = pytree.tree_map(lambda x: x.to(device), truth)

            batch_size = truth.shape[0]
            n_data_points += batch_size

            predict = model(**inputs)

            metric = batch_size * metric_fn(predict, truth).item()
            val_metric_sum += metric

        val_metric = val_metric_sum / n_data_points
        return val_metric


def train(
    writer: SummaryWriter,
    model: nn.Module,
    device: str | torch.device,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    val_metric_fn: nn.Module,
    val_num_steps: int,
    position_means: torch.Tensor,
    position_rmsds: torch.Tensor,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    max_grad_norm: float = 0.0,
):
    model = model.to(device)
    model.train()

    position_means = torch.from_numpy(position_means).to(device)
    position_rmsds = torch.from_numpy(position_rmsds).to(device)

    it_num = 0
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
                val_metric = validate(val_dataloader, device=device, model=model, metric_fn=val_metric_fn)
                model.output_unnorm = False
                model.train()

                writer.add_scalar("Loss/val", val_metric, it_num, new_style=True)
