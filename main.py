#!/usr/bin/env -S python3 -u

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable
from warnings import warn

import jsonargparse
import torch
from jsonargparse import ArgumentParser, Namespace, set_loader
from jsonargparse import typing as ptyping
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from deepsno.loops import predict, train
from deepsno.metrics.metric_monitor import MonitorCollection
from deepsno.metrics.metrics import Metric
from deepsno.utils import jinja as jinja_utils
from deepsno.utils.config_parse import check_instantiate_keys, get_class
from deepsno.utils.train import get_best_ckpt


def initialize_norm_dict(model_cfg: dict):
    # Instantiate the norm class if it is given

    if "norm_dict" in model_cfg["init_args"]:
        norm_dict_cfg = model_cfg["init_args"]["norm_dict"]
        if norm_dict_cfg is not None and "class_path" in norm_dict_cfg:
            check_instantiate_keys(model_cfg["init_args"]["norm_dict"], "norm_dict")
            norm_dict_class = get_class(norm_dict_cfg["class_path"])
            norm_dict = norm_dict_class(**norm_dict_cfg["init_args"])
            model_cfg["init_args"]["norm_dict"] = dict(norm_dict)


class UncommitedChangesError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


class UncommitedChangesWarning(RuntimeWarning):
    def __init__(self, message: str):
        super().__init__(message)


class MismatchedGitHash(RuntimeError):
    pass


def get_git_hash(raise_exception: bool = False) -> str:
    class UncommitedChangesError(RuntimeError):
        def __init__(self, message: str):
            super().__init__(message)

    repo_directory = Path(__file__).parent.resolve()

    is_working_tree_clean = subprocess.run(["git", "diff", "--quiet"], cwd=repo_directory).returncode != 0
    if raise_exception and is_working_tree_clean:
        raise UncommitedChangesError("Working tree is not clean. Please commit all changes.")

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo_directory, capture_output=True, text=True, check=True
    ).stdout

    git_hash = git_hash.strip()
    return git_hash


if __name__ == "__main__":
    try:
        loader = "jinja_yaml"
        set_loader(loader, loader_fn=jinja_utils.jinja_yaml_loader, exceptions=jinja_utils.get_exceptions())
        train_parser = ArgumentParser(parser_mode=loader)
        train_parser.add_argument("--seed", type=int, default=0)
        train_parser.add_argument("--checkpoint_dir", type=Path, required=True)
        train_parser.add_argument("--device", type=str, required=True)
        train_parser.add_argument("--train_dataset", type=torch.utils.data.Dataset)
        train_parser.add_argument("--val_dataset", type=torch.utils.data.Dataset)
        train_parser.add_argument("--batch_size", type=int, required=True)
        train_parser.add_argument("--val_batch_size", type=int, required=True)
        train_parser.add_argument("--shuffle", type=bool, required=True)
        train_parser.add_argument("--num_workers", type=int, default=0)
        train_parser.add_argument("--num_epochs", type=int, required=False)
        train_parser.add_argument("--num_steps", type=int, required=False)

        train_parser.add_argument("--ckpt", type=ptyping.path_type("dr") | ptyping.Path_fr, required=False)
        train_parser.add_argument("--ckpt_keys", type=str, nargs="+", required=False)

        train_parser.add_argument("--loss_fn", type=nn.Module, required=True)
        train_parser.add_argument("--optimizer", type=dict, required=True)
        train_parser.add_argument("--scheduler", type=dict, required=False)
        train_parser.add_argument("--max_grad_norm", type=float, default=0.0)
        train_parser.add_argument("--train_unnorm", action="store_true")
        train_parser.add_argument("--val_norm", action="store_true")

        train_parser.add_argument("--val_metric", type=Metric, required=True)
        train_parser.add_argument("--val_num_steps", type=int, required=False)
        train_parser.add_argument("--val_metric_is_inverted", action="store_true")
        train_parser.add_argument("--metric_monitors", type=dict | Iterable[dict], required=False)

        train_parser.add_argument("--dry_run", action="store_true")
        train_parser.add_argument("--profile", action="store_true")

        predict_parser = ArgumentParser(parser_mode=loader)
        predict_parser.add_argument("--ckpt", type=ptyping.path_type("dr") | ptyping.Path_fr, required=True)
        predict_parser.add_argument("--ckpt_config", type=ptyping.Path_fr, required=False)
        predict_parser.add_argument("--predict_keys", type=list, required=False)
        predict_parser.add_argument("--truth_keys", type=list, required=False)
        predict_parser.add_argument("--output_path", type=ptyping.Path_fc, required=False)
        predict_parser.add_argument("--device", type=str, required=True)
        predict_parser.add_argument("--dataset", type=torch.utils.data.Dataset)
        predict_parser.add_argument("--batch_size", type=int, required=True)
        predict_parser.add_argument("--num_workers", type=int, default=0)
        predict_parser.add_argument("--dataset_len", type=int, required=False)

        parser = ArgumentParser(prog="app", description="", parser_mode=loader)
        parser.add_argument("-c", "--config", action="config")
        parser.add_argument("--model", type=nn.Module, required=True)
        parser.add_argument("--force", action="store_true")
        parser.add_argument(
            "--git_hash", type=str, required=False, help="If given, will check if current repository matches this hash."
        )
        subcommands = parser.add_subcommands()
        subcommands.add_subcommand("train", train_parser)
        subcommands.add_subcommand("predict", predict_parser)

        cfg = parser.parse_args().as_dict()

        if cfg["subcommand"] == "predict":
            ckpt = Path(cfg["predict"]["ckpt"]).resolve()
            if ckpt.name == "ckpt":
                checkpoint_dir = ckpt.parent
            elif ckpt.is_file():
                checkpoint_dir = ckpt.parent.parent
            else:
                checkpoint_dir = ckpt
                ckpt = ckpt / "ckpt"
                cfg["predict"]["ckpt"] = ckpt

            if cfg["predict"]["ckpt_config"] is None:
                cfg["predict"]["ckpt_config"] = checkpoint_dir / "config.yaml"
            elif not Path(cfg["predict"]["ckpt_config"]).is_absolute():
                cfg["predict"]["ckpt_config"] = checkpoint_dir / cfg["predict"]["ckpt_config"]

            if cfg["predict"]["output_path"] is None:
                cfg["predict"]["output_path"] = checkpoint_dir / "predict.root"
            elif not Path(cfg["predict"]["output_path"]).is_absolute():
                cfg["predict"]["output_path"] = checkpoint_dir / cfg["predict"]["output_path"]

            # Merge the values from the ckpt config but override ckpt config with config from --config
            ckpt_cfg = parser.parse_path(cfg["predict"]["ckpt_config"]).as_dict()
            cfg = ckpt_cfg | cfg

        if cfg["force"]:
            warn(
                "Running in 'force' mode. Git commit hash may not reflect state of working tree.",
                UncommitedChangesWarning,
            )
            git_hash = get_git_hash(raise_exception=False)
        else:
            git_hash = get_git_hash(raise_exception=True)
            if cfg["git_hash"] is not None and git_hash != cfg["git_hash"]:
                raise MismatchedGitHash(
                    f"Git hash '{cfg['git_hash']}' does not match the git hash of the current working tree: '{git_hash}'"
                )

        cfg["git_hash"] = git_hash
        # These are the keys for the config that will be used for all subcommands
        cfg_keys = ["model", "force", "git_hash"]

        if cfg["subcommand"] == "train":
            cfg_keys.append("train")

            cfg = {key: cfg[key] for key in cfg_keys}

            torch.manual_seed(cfg["train"]["seed"])

            if cfg["train"]["dry_run"]:
                cfg["train"]["num_epochs"] = None
                cfg["train"]["num_steps"] = 3
                cfg["train"]["val_num_steps"] = 2
                cfg["train"]["checkpoint_dir"] = Path(tempfile.gettempdir()) / "dry_run"

            # Create the model save directory

            model_save_dir = Path(cfg["train"]["checkpoint_dir"])
            model_save_dir.mkdir(parents=True)

            print(
                f"Saving model config and checkpoints to {str(model_save_dir.resolve())}"
            )  # Instantiate the optimizer

            initialize_norm_dict(cfg["model"])

            # Instantiate the model and other classes

            save_cfg: dict = cfg
            cfg: Namespace = parser.instantiate_classes(cfg)
            model: nn.Module = cfg["model"]

            cfg: Namespace = cfg["train"]

            # Instantiate the optimizer

            check_instantiate_keys(cfg["optimizer"], "optimizer")
            optimizer_class = get_class(cfg["optimizer"]["class_path"])

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params}")
            optimizer: optim.Optimizer = optimizer_class(model.parameters(), **cfg["optimizer"]["init_args"])

            # Instantiate the scheduler

            if cfg["scheduler"] is not None:
                check_instantiate_keys(cfg["scheduler"], "scheduler")
                scheduler_class = get_class(cfg["scheduler"]["class_path"])

                scheduler: optim.lr_scheduler.LRScheduler = scheduler_class(optimizer, **cfg["scheduler"]["init_args"])
            else:
                scheduler = None

            # if ckpt is given, load the state dicts

            if cfg["ckpt"] is not None:
                ckpt = Path(cfg["ckpt"])
                if ckpt.is_dir():
                    ckpt = get_best_ckpt(ckpt)
                state_dict = torch.load(ckpt, map_location=cfg["device"], weights_only=True)
                ckpt_keys = cfg["ckpt_keys"]
                if "model" in ckpt_keys:
                    model.load_state_dict(state_dict["model"], strict=True)
                    model.to(cfg["device"])
                if "optimizer" in ckpt_keys:
                    optimizer.load_state_dict(state_dict["optimizer"])
                if "scheduler" in ckpt_keys:
                    scheduler.load_state_dict(state_dict["scheduler"])

            # Instantiate the dataloaders

            if (cfg["num_epochs"] is not None) and (cfg["num_steps"] is not None):
                raise ValueError("Only 'train.num_epochs' or 'train.num_steps' can be given, not both.")
            if (cfg["num_epochs"] is None) and (cfg["num_steps"] is None):
                raise ValueError("Either 'train.num_epochs' or 'train.num_steps' must be provided.")

            print(f"Training set size: {len(cfg['train_dataset']):,}")
            print(f"Validation set size: {len(cfg['val_dataset']):,}")

            train_dataloader = data.DataLoader(
                cfg["train_dataset"],
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"],
                shuffle=cfg["shuffle"],
                drop_last=True,
            )
            val_dataloader = data.DataLoader(
                cfg["val_dataset"],
                batch_size=cfg["val_batch_size"],
                shuffle=False,
                num_workers=1,
                drop_last=False,
            )

            if cfg.val_num_steps is None:
                cfg["val_num_steps"] = len(val_dataloader)

            # Save the config file

            parser.save(save_cfg, model_save_dir / "config.yaml")

            writer = SummaryWriter(log_dir=model_save_dir)

            writer.add_scalar("Number of training events", len(cfg["train_dataset"]))
            writer.add_scalar("Number of validation events", len(cfg["val_dataset"]))
            writer.add_scalar("Number of training batches", len(train_dataloader))

            # Instantiate the metric monitor
            if isinstance(cfg["metric_monitors"], dict):
                check_instantiate_keys(cfg["metric_monitors"], "metric_monitors")
                metric_monitor_class = get_class(cfg["metric_monitors"]["class_path"])
                if "init_args" in cfg["metric_monitors"]:
                    metric_monitor = metric_monitor_class(writer, **cfg["metric_monitors"]["init_args"])
                else:
                    metric_monitor = metric_monitor_class(writer)

            elif isinstance(cfg["metric_monitors"], Iterable):
                monitors = []
                for monitor_dict in cfg["metric_monitors"]:
                    check_instantiate_keys(monitor_dict, "metric_monitors")
                    metric_monitor_class = get_class(monitor_dict["class_path"])
                    if "init_args" in monitor_dict:
                        metric_monitor = metric_monitor_class(writer, **monitor_dict["init_args"])
                    else:
                        metric_monitor = metric_monitor_class(writer)
                    monitors.append(metric_monitor)

                metric_monitor = MonitorCollection(monitors)
            else:
                metric_monitor = None

            train(
                checkpoint_dir=model_save_dir / "ckpt",
                writer=writer,
                model=model,
                device=torch.device(cfg["device"]),
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=cfg["num_epochs"],
                num_steps=cfg["num_steps"],
                optimizer=optimizer,
                loss_fn=cfg["loss_fn"],
                scheduler=scheduler,
                train_unnorm=cfg["train_unnorm"],
                val_norm=cfg["val_norm"],
                val_metric=cfg["val_metric"],
                val_metric_is_inverted=cfg["val_metric_is_inverted"],
                val_num_steps=cfg["val_num_steps"],
                max_grad_norm=cfg["max_grad_norm"],
                metric_monitor=metric_monitor,
                profile=cfg["profile"],
            )

        elif cfg["subcommand"] == "predict":
            import uproot

            cfg_keys.append("predict")

            cfg = {key: cfg[key] for key in cfg_keys}

            initialize_norm_dict(cfg["model"])

            save_cfg: dict = cfg

            cfg: jsonargparse.Namespace = parser.instantiate_classes(cfg)

            model = cfg.model

            ckpt_path: Path = Path(cfg.predict.ckpt)

            if ckpt_path.is_dir():
                ckpt_path = get_best_ckpt(ckpt_path)

            state_dict = torch.load(ckpt_path, map_location=cfg.predict.device, weights_only=True)

            model.load_state_dict(state_dict["model"])

            dataloader: data.DataLoader = data.DataLoader(
                cfg.predict.dataset,
                batch_size=cfg.predict.batch_size,
                num_workers=cfg.predict.num_workers,
                shuffle=False,
            )

            # dataset_len = len(cfg.predict.dataset) if cfg.predict.dataset_len is None else cfg.predict.dataset_len
            # if dataset_len > len(cfg.predict.dataset):
            #     raise ValueError(
            #         (
            #             f"The value of 'dataset_len' is larger than the length of the dataset: {len(cfg.predict.dataset)}. "
            #             "'dataset_len' must be less than or equal to the length of the dataset."
            #         )
            #     )

            predict_cfg_path = Path(cfg.predict.output_path).with_suffix(".yaml")
            parser.save(save_cfg, predict_cfg_path, overwrite=True)

            with uproot.recreate(cfg.predict.output_path) as file:
                predict(
                    model=model,
                    dataloader=dataloader,
                    file=file,
                    device=cfg.predict.device,
                    predict_keys=cfg.predict.predict_keys,
                    truth_keys=cfg.predict.truth_keys,
                )

    except Exception as e:
        print(f"Critical error occured: {e}")
        print("Exiting")
        sys.exit(1)
