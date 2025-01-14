#!/usr/bin/env -S python3 -u

import importlib
import subprocess
from datetime import datetime
from pathlib import Path
from warnings import warn

import h5py
import jsonargparse
import torch
from jsonargparse import ArgumentParser, Namespace
from jsonargparse import typing as ptyping
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from src.train import test, train
from src.utils.train import get_best_ckpt, write_config_to_h5


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

    if not raise_exception and subprocess.run(["git", "diff", "--quiet"]).returncode != 0:
        raise UncommitedChangesError("Working tree is not clean. Please commit all changes.")

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout

    git_hash = git_hash.strip()
    return git_hash


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    train_parser = ArgumentParser(prog="app")
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--git_hash", type=str, required=False, help="If given, will check if current repository matches this hash."
    )
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", train_parser)
    subcommands.add_subcommand("predict", predict_parser)

    cfg = parser.parse_args()

    cfg = jsonargparse.namespace_to_dict(cfg)

    if cfg["force"]:
        warn(
            "Running in 'force' mode. Git commit hash may not reflect state of working tree.", UncommitedChangesWarning
        )
        git_hash = get_git_hash(raise_exception=True)
    else:
        git_hash = get_git_hash()

    # TODO: This part may need some testing and some thought
    if cfg["git_hash"] is not None:
        if git_hash != cfg["git_hash"]:
            raise MismatchedGitHash(
                f"Git hash: {cfg["git_hash"]} does not match the git hash of the current working tree: {git_hash}"
            )

    cfg["git_hash"] = git_hash

    cfg_keys = ["model", "force", "git_hash"]  # These are the keys for the config that will be used for all subcommands

    if cfg["subcommand"] == "train":
        cfg_keys.append("train")

        cfg = {key: cfg[key] for key in cfg_keys}

        torch.manual_seed(cfg["train"]["seed"])

        # Create the model save directory

        model_save_dir: Path = Path(cfg["train"]["checkpoint_dir"])
        model_save_dir.mkdir(parents=True, exist_ok=True)

        datetime_string = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

        model_save_dir = model_save_dir / datetime_string

        model_save_dir.mkdir()

        print(f"Saving model config and checkpoints to {str(model_save_dir.resolve())}")  # Instantiate the optimizer

        # Substitute custom tags
        # TODO: this needs to be written better
        cfg["train"]["dataset"]["init_args"]["delays_save_path"] = cfg["train"]["dataset"]["init_args"][
            "delays_save_path"
        ].replace(r"<ckpt_dir>", str(model_save_dir))

        initialize_norm_dict(cfg["model"])

        # Instantiate the model and other classes

        save_cfg: dict = cfg
        cfg: Namespace = parser.instantiate_classes(cfg)
        model: Namespace = cfg["model"]
        cfg: Namespace = cfg["train"]

        # Instantiate the optimizer

        check_instantiate_keys(cfg["optimizer"], "optimizer")
        optimizer_class = get_class(cfg["optimizer"]["class_path"])

        optimizer = optimizer_class(model.parameters(), **cfg["optimizer"]["init_args"])

        # Instantiate the scheduler

        if "scheduler" in cfg:
            check_instantiate_keys(cfg["scheduler"], "scheduler")
            scheduler_class = get_class(cfg["scheduler"]["class_path"])

            scheduler = scheduler_class(optimizer, **cfg["scheduler"]["init_args"])
        else:
            scheduler = None

        # Instantiate the dataloaders

        if (cfg["train"]["num_epochs"] is not None) and (cfg["train"]["num_steps"] is not None):
            raise ValueError("Only 'train.num_epochs' or 'train.num_steps' can be given, not both.")
        if (cfg["train"]["num_epochs"] is None) and (cfg["train"]["num_steps"] is None):
            raise ValueError("Either 'train.num_epochs' or 'train.num_steps' must be provided.")

        train_set, val_set = data.random_split(cfg["dataset"], [cfg["train_val_split"], 1 - cfg["train_val_split"]])

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

        save_cfg = cfg

        ckpt_cfg: dict = jsonargparse.namespace_to_dict(parser.parse_path(cfg["predict"]["ckpt_config"]))
        ckpt_model_cfg = {"model": ckpt_cfg["model"]}

        initialize_norm_dict(ckpt_model_cfg["model"])

        save_cfg = save_cfg | ckpt_cfg

        predict_cfg = predict_parser.instantiate_classes(cfg["predict"])

        model: torch.nn.Module = parser.instantiate_classes(ckpt_model_cfg)["model"]

        ckpt_path: Path = Path(predict_cfg["ckpt"])

        if ckpt_path.is_dir():
            ckpt_path = get_best_ckpt(ckpt_path)

        state_dict = torch.load(ckpt_path, map_location=predict_cfg["device"], weights_only=True)
        model_state_dict = state_dict["model"]
        del state_dict

        model.load_state_dict(model_state_dict)

        dataloader: data.DataLoader = data.DataLoader(
            predict_cfg["dataset"], batch_size=predict_cfg["batch_size"], num_workers=predict_cfg["num_workers"]
        )

        with h5py.File(predict_cfg["output_file"], "w") as h5_file:
            # TODO: perhaps have to rethink if this is the best way to do it
            write_config_to_h5(h5_group=h5_file, config_obj=save_cfg)
            test(
                model=model,
                dataloader=dataloader,
                h5_group=h5_file,
                dataset_length=len(predict_cfg["dataset"]),
                device=predict_cfg["device"],
            )
