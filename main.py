#!/usr/bin/env python3

import importlib
from datetime import datetime
from pathlib import Path
from typing import Type

import torch
from jsonargparse import ArgumentParser, Namespace
from jsonargparse import typing as ptyping
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from src.train import train


def check_instantiate_keys(namespace: Namespace, object_name: str):
    if "class_path" not in namespace:
        raise KeyError(f"'class_path' not found in {object_name} config object")
    if "init_args" not in namespace:
        raise KeyError(f"'init_args' not found in {object_name} config object")


def get_class(class_path: str) -> Type:
    if "." in class_path:
        module_path, class_str = class_path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
    else:
        module = importlib.import_module(__name__)

    return getattr(module, class_str)


if __name__ == "__main__":
    train_parser = ArgumentParser(prog="app")
    train_parser.add_argument("--checkpoint_dir", type=ptyping.Path_dc, required=True)
    train_parser.add_argument("--device", type=str, required=True)
    train_parser.add_argument("--dataset", type=torch.utils.data.Dataset)
    train_parser.add_argument("--batch_size", type=int, required=True)
    train_parser.add_argument("--val_batch_size", type=int, required=True)
    train_parser.add_argument("--shuffle", type=bool, required=True)
    train_parser.add_argument("--num_workers", type=int, default=0)
    train_parser.add_argument("--num_epochs", type=int, required=True)
    train_parser.add_argument("--train_val_split", type=float, required=True)

    train_parser.add_argument("--model", type=nn.Module, required=True)
    train_parser.add_argument("--loss_fn", type=nn.Module, required=True)
    train_parser.add_argument("--optimizer", type=dict, required=True)
    train_parser.add_argument("--scheduler", type=dict, required=False)
    train_parser.add_argument("--max_grad_norm", type=float, default=0.0)

    train_parser.add_argument("--val_metric_fn", type=nn.Module, required=True)
    train_parser.add_argument("--val_num_steps", type=int, required=False)

    test_parser = ArgumentParser()

    parser = ArgumentParser(prog="app", description="")
    parser.add_argument("-c", "--config", action="config")
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", train_parser)
    subcommands.add_subcommand("test", test_parser)

    cfg = parser.parse_args()

    if cfg.subcommand == "train":
        # TODO: Have to figure out good way of saving train config / splitting this from test config
        save_cfg = cfg
        cfg = parser.instantiate_classes(cfg)
        cfg = cfg.train
        checkpoint_dir: Path = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        datetime_string = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

        checkpoint_dir = checkpoint_dir / datetime_string

        checkpoint_dir.mkdir()

        parser.save(save_cfg, checkpoint_dir / "config.yaml")

        # Instantiate the optimizer
        check_instantiate_keys(cfg.optimizer, "optimizer")
        optimizer_class = get_class(cfg.optimizer["class_path"])

        optimizer = optimizer_class(cfg.model.parameters(), **cfg.optimizer["init_args"])

        # Instantiate the scheduler

        if "scheduler" in cfg:
            check_instantiate_keys(cfg.scheduler, "scheduler")
            scheduler_class = get_class(cfg.scheduler["class_path"])

            scheduler = scheduler_class(optimizer, **cfg.scheduler["init_args"])
        else:
            scheduler = None

        # Instantiate the dataloaders

        train_set, val_set = data.random_split(cfg.dataset, [cfg.train_val_split, 1 - cfg.train_val_split])

        train_dataloader = data.DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers
        )
        val_dataloader = data.DataLoader(
            val_set, batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        position_means = cfg.dataset.position_means
        position_rmsds = cfg.dataset.position_rmsds
        cfg.model.add_input_norm(hit_time_mean=cfg.dataset.hit_time_mean, hit_time_rmsd=cfg.dataset.hit_time_rmsd)
        cfg.model.add_output_unnorm(
            position_means=torch.from_numpy(position_means),
            position_rmsds=torch.from_numpy(position_rmsds),
            output_unnorm=False,
        )

        if cfg.val_num_steps is None:
            cfg.val_num_steps = len(train_dataloader)

        writer = SummaryWriter()

        writer.add_scalar("Number of training events", len(train_set))
        writer.add_scalar("Number of validation events", len(val_set))
        writer.add_scalar("Number of training batches", len(train_dataloader))

        train(
            writer=writer,
            model=cfg.model,
            device=cfg.device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=cfg.num_epochs,
            optimizer=optimizer,
            loss_fn=cfg.loss_fn,
            scheduler=scheduler,
            val_metric_fn=cfg.val_metric_fn,
            val_num_steps=cfg.val_num_steps,
            position_means=position_means,
            position_rmsds=position_rmsds,
            max_grad_norm=cfg.max_grad_norm,
        )
