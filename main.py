#!/usr/bin/env python3

import importlib
from datetime import datetime
from pathlib import Path

import h5py
import jsonargparse
import torch
from jsonargparse import ArgumentParser, Namespace
from jsonargparse import typing as ptyping
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from src.train import test, train
from src.utils.utils import get_best_ckpt, write_config_to_h5


def check_instantiate_keys(namespace: Namespace, object_name: str):
    if "class_path" not in namespace:
        raise KeyError(f"'class_path' not found in {object_name} config object")
    if "init_args" not in namespace:
        raise KeyError(f"'init_args' not found in {object_name} config object")


def get_class(class_path: str) -> type:
    if "." in class_path:
        module_path, class_str = class_path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
    else:
        module = importlib.import_module(__name__)

    return getattr(module, class_str)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

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

    train_parser.add_argument("--loss_fn", type=nn.Module, required=True)
    train_parser.add_argument("--optimizer", type=dict, required=True)
    train_parser.add_argument("--scheduler", type=dict, required=False)
    train_parser.add_argument("--max_grad_norm", type=float, default=0.0)

    train_parser.add_argument("--val_loss_fn", type=nn.Module, required=True)
    train_parser.add_argument("--val_num_steps", type=int, required=False)
    train_parser.add_argument("--val_loss_is_inverted", type=bool, default=False)

    predict_parser = ArgumentParser()
    predict_parser.add_argument("--ckpt", type=ptyping.path_type("dr") | ptyping.Path_fr, required=True)
    predict_parser.add_argument("--ckpt_config", type=ptyping.Path_fr, required=True)
    predict_parser.add_argument("--output_file", type=ptyping.Path_fc, required=True)
    predict_parser.add_argument("--device", type=str, required=True)
    predict_parser.add_argument("--dataset", type=torch.utils.data.Dataset)
    predict_parser.add_argument("--batch_size", type=int, required=True)
    predict_parser.add_argument("--num_workers", type=int, default=0)

    parser = ArgumentParser(prog="app", description="")
    parser.add_argument("-c", "--config", action="config")
    parser.add_argument("--model", type=nn.Module, required=True)
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", train_parser)
    subcommands.add_subcommand("predict", predict_parser)

    cfg = parser.parse_args()

    if cfg.subcommand == "train":
        cfg = jsonargparse.namespace_to_dict(cfg)
        cfg: dict = {"model": cfg["model"], "train": cfg["train"]}

        save_cfg: dict = cfg
        cfg: Namespace = parser.instantiate_classes(cfg)
        model: Namespace = cfg.model
        cfg: Namespace = cfg.train

        model_save_dir: Path = Path(cfg.checkpoint_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        datetime_string = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

        model_save_dir = model_save_dir / datetime_string

        model_save_dir.mkdir()

        # Instantiate the optimizer
        check_instantiate_keys(cfg.optimizer, "optimizer")
        optimizer_class = get_class(cfg.optimizer["class_path"])

        optimizer = optimizer_class(model.parameters(), **cfg.optimizer["init_args"])

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
        model.add_input_norm(hit_time_mean=cfg.dataset.hit_time_mean, hit_time_rmsd=cfg.dataset.hit_time_rmsd)
        model.add_output_unnorm(
            position_means=torch.from_numpy(position_means),
            position_rmsds=torch.from_numpy(position_rmsds),
            output_unnorm=False,
        )

        save_cfg["model"]["init_args"]["norm_dict"] = {
            "input_norms": {
                "hit_time_mean": float(cfg.dataset.hit_time_mean),
                "hit_time_rmsd": float(cfg.dataset.hit_time_rmsd),
            },
            "output_norms": {
                "position_means": position_means.tolist(),
                "position_rmsds": position_rmsds.tolist(),
            },
        }

        if cfg.val_num_steps is None:
            cfg.val_num_steps = len(train_dataloader)

        parser.save(save_cfg, model_save_dir / "config.yaml")

        writer = SummaryWriter(log_dir=model_save_dir)

        writer.add_scalar("Number of training events", len(train_set))
        writer.add_scalar("Number of validation events", len(val_set))
        writer.add_scalar("Number of training batches", len(train_dataloader))

        train(
            checkpoint_dir=model_save_dir / "ckpt",
            writer=writer,
            model=model,
            device=torch.device(cfg.device),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=cfg.num_epochs,
            optimizer=optimizer,
            loss_fn=cfg.loss_fn,
            scheduler=scheduler,
            val_loss_fn=cfg.val_loss_fn,
            val_loss_is_inverted=cfg.val_loss_is_inverted,
            val_num_steps=cfg.val_num_steps,
            position_means=position_means,
            position_rmsds=position_rmsds,
            max_grad_norm=cfg.max_grad_norm,
        )

    if "predict" in cfg:
        save_cfg = {"predict": jsonargparse.namespace_to_dict(cfg.predict)}
        cfg = cfg.predict

        ckpt_cfg: dict = jsonargparse.namespace_to_dict(parser.parse_path(cfg.ckpt_config))
        ckpt_model_cfg = {"model": ckpt_cfg["model"]}

        save_cfg = save_cfg | ckpt_cfg

        cfg = predict_parser.instantiate_classes(cfg)

        model: torch.nn.Module = parser.instantiate_classes(ckpt_model_cfg)["model"]

        ckpt_path: Path = Path(cfg.ckpt)

        if ckpt_path.is_dir():
            ckpt_path = get_best_ckpt(ckpt_path)

        state_dict = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        model_state_dict = state_dict["model"]
        del state_dict

        model.load_state_dict(model_state_dict)

        dataloader: data.DataLoader = data.DataLoader(
            cfg.dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )

        with h5py.File(cfg.output_file, "w") as h5_file:
            write_config_to_h5(h5_group=h5_file, config_obj=save_cfg)
            test(
                model=model,
                dataloader=dataloader,
                h5_group=h5_file,
                dataset_length=len(cfg.dataset),
                device=cfg.device,
            )
