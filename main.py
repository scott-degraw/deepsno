#!/usr/bin/env -S python3 -u

import importlib
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from warnings import warn

import h5py
import jsonargparse
import torch
from jsonargparse import ArgumentParser, Namespace
from jsonargparse import typing as ptyping
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from src.loops import test, train
from src.utils.train import get_best_ckpt


def check_instantiate_keys(cfg_obj: Namespace | dict, object_name: str):
    if "class_path" not in cfg_obj:
        raise KeyError(f"'class_path' not found in {object_name} config object")
    if "init_args" not in cfg_obj:
        raise KeyError(f"'init_args' not found in {object_name} config object")


def get_class(class_path: str) -> type:
    if "." in class_path:
        module_path, class_str = class_path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
    else:
        module = importlib.import_module(__name__)

    return getattr(module, class_str)


def initialize_norm_dict(model_cfg: dict):
    # Instantiate the norm class if it is given

    if "norm_dict" in model_cfg["init_args"]:
        norm_dict_cfg = model_cfg["init_args"]["norm_dict"]
        if "class_path" in norm_dict_cfg:
            check_instantiate_keys(norm_dict_cfg, "norm_dict")
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
    # TODO: put this in the config
    torch.set_float32_matmul_precision("high")

    train_parser = ArgumentParser(prog="DeepSNO")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--checkpoint_dir", type=ptyping.Path_dc, required=True)
    train_parser.add_argument("--device", type=str, required=True)
    train_parser.add_argument("--dataset", type=torch.utils.data.Dataset)
    train_parser.add_argument("--batch_size", type=int, required=True)
    train_parser.add_argument("--val_batch_size", type=int, required=True)
    train_parser.add_argument("--shuffle", type=bool, required=True)
    train_parser.add_argument("--num_workers", type=int, default=0)
    train_parser.add_argument("--num_epochs", type=int, required=False)
    train_parser.add_argument("--num_steps", type=int, required=False)
    train_parser.add_argument("--val_len", type=int | float, required=True)

    train_parser.add_argument("--ckpt", type=ptyping.path_type("dr") | ptyping.Path_fr, required=False)
    train_parser.add_argument("--ckpt_keys", type=list[str], required=False)

    train_parser.add_argument("--loss_fn", type=nn.Module, required=True)
    train_parser.add_argument("--optimizer", type=dict, required=True)
    train_parser.add_argument("--scheduler", type=dict, required=False)
    train_parser.add_argument("--max_grad_norm", type=float, default=0.0)

    train_parser.add_argument("--val_loss_fn", type=nn.Module, required=True)
    train_parser.add_argument("--val_num_steps", type=int, required=False)
    train_parser.add_argument("--val_loss_is_inverted", type=bool, default=False)

    train_parser.add_argument("--dry_run", action="store_true")

    predict_parser = ArgumentParser()
    predict_parser.add_argument("--ckpt", type=ptyping.path_type("dr") | ptyping.Path_fr, required=True)
    predict_parser.add_argument("--ckpt_config", type=ptyping.Path_fr, required=False)
    predict_parser.add_argument("--output_path", type=ptyping.Path_fc, required=False)
    predict_parser.add_argument("--device", type=str, required=True)
    predict_parser.add_argument("--dataset", type=torch.utils.data.Dataset)
    predict_parser.add_argument("--batch_size", type=int, required=True)
    predict_parser.add_argument("--num_workers", type=int, default=0)
    predict_parser.add_argument("--dataset_len", type=int, required=False)

    parser = ArgumentParser(prog="app", description="")
    parser.add_argument("-c", "--config", action="config")
    parser.add_argument("--model", type=nn.Module, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--git_hash", type=str, required=False, help="If given, will check if current repository matches this hash."
    )
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", train_parser)
    subcommands.add_subcommand("predict", predict_parser)

    cfg = parser.parse_args()

    cfg: dict = jsonargparse.namespace_to_dict(cfg)

    if cfg["subcommand"] == "predict":
        ckpt = Path(cfg["predict"]["ckpt"]).resolve()
        checkpoint_dir = ckpt.parent
        if ckpt.is_file():
            checkpoint_dir = checkpoint_dir.parent

        if cfg["predict"]["ckpt_config"] is None:
            cfg["predict"]["ckpt_config"] = checkpoint_dir / "config.yaml"
        elif not Path(cfg["predict"]["ckpt_config"]).is_absolute():
            cfg["predict"]["ckpt_config"] = checkpoint_dir / cfg["predict"]["ckpt_config"]

        if cfg["predict"]["output_path"] is None:
            cfg["predict"]["output_path"] = checkpoint_dir / "test_result.h5"
        elif not Path(cfg["predict"]["output_path"]).is_absolute():
            cfg["predict"]["output_path"] = checkpoint_dir / cfg["predict"]["output_path"]

        # Merge the values from the ckpt config but override ckpt config with config from --config
        ckpt_cfg = jsonargparse.namespace_to_dict(parser.parse_path(cfg["predict"]["ckpt_config"]))
        cfg = ckpt_cfg | cfg

    if cfg["force"]:
        warn(
            "Running in 'force' mode. Git commit hash may not reflect state of working tree.", UncommitedChangesWarning
        )
        git_hash = get_git_hash(raise_exception=False)
    else:
        git_hash = get_git_hash(raise_exception=True)
        if cfg["git_hash"] is not None and git_hash != cfg["git_hash"]:
            raise MismatchedGitHash(
                f"Git hash '{cfg["git_hash"]}' does not match the git hash of the current working tree: '{git_hash}'"
            )

    cfg["git_hash"] = git_hash
    cfg_keys = ["model", "force", "git_hash"]  # These are the keys for the config that will be used for all subcommands

    if cfg["subcommand"] == "train":
        cfg_keys.append("train")

        cfg = {key: cfg[key] for key in cfg_keys}

        torch.manual_seed(cfg["train"]["seed"])

        datetime_string = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

        if cfg["train"]["dry_run"]:
            cfg["train"]["num_epochs"] = None
            cfg["train"]["num_steps"] = 3
            cfg["train"]["val_num_steps"] = 2
            cfg["train"]["val_len"] = int(1.5 * cfg["train"]["val_batch_size"])
            cfg["train"]["checkpoint_dir"] = Path(tempfile.gettempdir()) / f"dry_run_{datetime_string}"

        # Create the model save directory

        model_save_dir: Path = Path(cfg["train"]["checkpoint_dir"])
        model_save_dir.mkdir(parents=True, exist_ok=True)

        model_save_dir = model_save_dir / datetime_string

        model_save_dir.mkdir()

        print(f"Saving model config and checkpoints to {str(model_save_dir.resolve())}")  # Instantiate the optimizer

        initialize_norm_dict(cfg["model"])

        # Instantiate the model and other classes

        save_cfg: dict = cfg
        cfg: Namespace = parser.instantiate_classes(cfg)
        model: nn.Module = cfg["model"]
        cfg: Namespace = cfg["train"]

        # Instantiate the optimizer

        check_instantiate_keys(cfg["optimizer"], "optimizer")
        optimizer_class = get_class(cfg["optimizer"]["class_path"])

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
            if "optimizer" in ckpt_keys:
                optimizer.load_state_dict(state_dict["optimizer"], strict=True)
            if "scheduler" in ckpt_keys:
                scheduler.load_state_dict(state_dict["scheduler"], strict=True)

        # Instantiate the dataloaders

        if (cfg["num_epochs"] is not None) and (cfg["num_steps"] is not None):
            raise ValueError("Only 'train.num_epochs' or 'train.num_steps' can be given, not both.")
        if (cfg["num_epochs"] is None) and (cfg["num_steps"] is None):
            raise ValueError("Either 'train.num_epochs' or 'train.num_steps' must be provided.")

        if isinstance(cfg["val_len"], float):
            lengths = [1 - cfg["val_len"], cfg["val_len"]]
        else:
            lengths = [len(cfg["dataset"]) - cfg["val_len"], cfg["val_len"]]

        train_set, val_set = data.random_split(cfg["dataset"], lengths)

        train_dataloader = data.DataLoader(
            train_set, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], num_workers=cfg["num_workers"]
        )
        val_dataloader = data.DataLoader(
            val_set, batch_size=cfg["val_batch_size"], shuffle=False, num_workers=cfg["num_workers"]
        )

        if cfg.val_num_steps is None:
            cfg["val_num_steps"] = len(train_dataloader)

        # Save the config file

        parser.save(save_cfg, model_save_dir / "config.yaml")

        writer = SummaryWriter(log_dir=model_save_dir)

        writer.add_scalar("Number of training events", len(train_set))
        writer.add_scalar("Number of validation events", len(val_set))
        writer.add_scalar("Number of training batches", len(train_dataloader))

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
            val_loss_fn=cfg["val_loss_fn"],
            val_loss_is_inverted=cfg["val_loss_is_inverted"],
            val_num_steps=cfg["val_num_steps"],
            max_grad_norm=cfg["max_grad_norm"],
        )

    elif cfg["subcommand"] == "predict":
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
            cfg.predict.dataset, batch_size=cfg.predict.batch_size, num_workers=cfg.predict.num_workers
        )

        dataset_len = len(cfg.predict.dataset) if cfg.predict.dataset_len is None else cfg.predict.dataset_len
        if dataset_len > len(cfg.predict.dataset):
            raise ValueError(
                (
                    f"The value of 'dataset_len' is larger than the length of the dataset: {len(cfg.predict.dataset)}. "
                    "'dataset_len' must be less than or equal to the length of the dataset."
                )
            )

        predict_cfg_path = Path(cfg.predict.output_path).with_suffix(".yaml")
        parser.save(save_cfg, predict_cfg_path, overwrite=True)

        with h5py.File(cfg.predict.output_path, "w") as h5_file:
            test(
                model=model,
                dataloader=dataloader,
                group=h5_file,
                dataset_len=dataset_len,
                device=cfg.predict.device,
            )
