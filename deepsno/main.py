#!/usr/bin/env -S python3 -u

import os
import shutil
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import wandb
from jsonargparse import ArgumentParser, set_loader
from jsonargparse import typing as ptyping
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from deepsno.loops import predict, train
from deepsno.metrics import metric_monitor
from deepsno.metrics.metrics import Metric
from deepsno.utils import jinja as jinja_utils
from deepsno.utils.config_parse import check_instantiate_keys, get_class, instantiate
from deepsno.utils.train import get_best_ckpt, get_latest_ckpt

LOADER = "jinja_yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Tee:
    """Write to both a file and another stream simultaneously."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def fileno(self):
        return self._stream.fileno()


class UncommittedChangesError(RuntimeError):
    pass


class MismatchedGitHash(RuntimeError):
    pass


def get_git_hash(raise_exception: bool = False) -> str:
    """Return the short git hash of HEAD.

    If *raise_exception* is ``True`` and the working tree has uncommitted
    changes, an :class:`UncommittedChangesError` is raised.
    """
    repo_directory = Path(__file__).parent.resolve()

    has_uncommitted = subprocess.run(["git", "diff", "--quiet"], cwd=repo_directory).returncode != 0
    if raise_exception and has_uncommitted:
        raise UncommittedChangesError("Working tree is not clean. Please commit all changes.")

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo_directory, capture_output=True, text=True, check=True
    ).stdout.strip()

    return git_hash


def initialize_norm_dict(model_cfg: dict):
    """Instantiate the norm class in-place if it is given in the model config."""
    if "norm_dict" not in model_cfg["init_args"]:
        return
    norm_dict_cfg = model_cfg["init_args"]["norm_dict"]
    if norm_dict_cfg is not None and "class_path" in norm_dict_cfg:
        check_instantiate_keys(norm_dict_cfg, "norm_dict")
        norm_dict_class = get_class(norm_dict_cfg["class_path"])
        norm_dict = norm_dict_class(**norm_dict_cfg["init_args"])
        model_cfg["init_args"]["norm_dict"] = dict(norm_dict)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_train_parser() -> ArgumentParser:
    """Build and return the ``train`` subcommand parser."""
    p = ArgumentParser(prog="deepsno", parser_mode=LOADER)

    # wandb
    p.add_argument("--entity", type=str, required=True)
    p.add_argument("--project", type=str, required=True)
    p.add_argument("--tags", type=str, nargs="+", required=False)
    p.add_argument("--wandb_disable", action="store_true", default=False)
    p.add_argument("--log_interval", type=int, required=False, default=50)

    # training
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--device", type=str, required=True)
    p.add_argument("--train_dataloader", type=dict)
    p.add_argument("--val_dataloader", type=dict)
    p.add_argument("--num_steps", type=int, required=False)

    # checkpoint resume
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--ckpt", type=ptyping.path_type("dr") | ptyping.Path_fr, required=False)
    p.add_argument("--ckpt_keys", type=str, nargs="+", required=False)

    # loss / optimizer / scheduler
    p.add_argument("--loss_fn", type=nn.Module, required=True)
    p.add_argument("--optimizer", type=dict, required=True)
    p.add_argument("--scheduler", type=dict, required=False)
    p.add_argument("--max_grad_norm", type=float, default=0.0)

    # normalisation
    p.add_argument("--train_norm", type=bool, default=True)
    p.add_argument("--val_norm", type=bool, default=True)

    # validation
    p.add_argument("--val_metric", type=Metric, required=True)
    p.add_argument("--val_num_steps", type=int, required=False)
    p.add_argument("--val_metric_is_inverted", action="store_true")
    p.add_argument("--metric_monitors", type=dict | Iterable[dict], required=False)

    # misc
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--profile", action="store_true")

    return p


def _build_predict_parser() -> ArgumentParser:
    """Build and return the ``predict`` subcommand parser."""
    p = ArgumentParser(parser_mode=LOADER)
    p.add_argument("--keys", type=str, nargs="+")
    p.add_argument("--ckpt", type=ptyping.path_type("dr") | ptyping.Path_fr, required=True)
    p.add_argument("--ckpt_config", type=ptyping.Path_fr, required=False)
    p.add_argument("--output_path", type=ptyping.Path_fc, required=False)
    p.add_argument("--device", type=str, required=True)
    p.add_argument("--dataloader", type=dict, required=True)
    p.add_argument("--dataset_len", type=int, required=False)
    return p


def build_parser() -> ArgumentParser:
    """Build the top-level CLI parser with train/predict subcommands."""
    set_loader(LOADER, loader_fn=jinja_utils.jinja_yaml_loader, exceptions=jinja_utils.get_exceptions())

    parser = ArgumentParser(prog="app", description="", parser_mode=LOADER)
    parser.add_argument("-c", "--config", action="config")
    parser.add_argument("--model", type=nn.Module, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--git_hash",
        type=str,
        required=False,
        help="If given, will check if current repository matches this hash.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        required=False,
        default=None,
        help="If given, redirect stdout and stderr to this file.",
    )

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", _build_train_parser())
    subcommands.add_subcommand("predict", _build_predict_parser())

    return parser


# ---------------------------------------------------------------------------
# Git hash resolution
# ---------------------------------------------------------------------------


def resolve_git_hash(cfg: dict) -> str:
    """Validate and return the git hash."""

    git_hash = get_git_hash(raise_exception=not cfg.get("force", False))
    if cfg["git_hash"] is not None and git_hash != cfg["git_hash"]:
        raise MismatchedGitHash(
            f"Git hash '{cfg['git_hash']}' does not match the git hash of the current working tree: '{git_hash}'"
        )
    return git_hash


# ---------------------------------------------------------------------------
# Predict-specific config resolution
# ---------------------------------------------------------------------------


def _resolve_predict_paths(cfg: dict) -> None:
    """Resolve checkpoint, config, and output paths for the predict subcommand (in-place)."""
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


# ---------------------------------------------------------------------------
# Subcommand runners
# ---------------------------------------------------------------------------


def _instantiate_monitors(cfg: dict, run: wandb.sdk.wandb_run.Run) -> metric_monitor.MetricMonitor | None:
    """Instantiate metric monitors from the config, if any."""
    monitors_cfg = cfg["metric_monitors"]

    if monitors_cfg is None:
        return None

    if isinstance(monitors_cfg, dict):
        monitors_cfg = [monitors_cfg]

    monitors = []
    for monitor_dict in monitors_cfg:
        check_instantiate_keys(monitor_dict, "metric_monitors")
        monitor_class = get_class(monitor_dict["class_path"])
        if "init_args" in monitor_dict:
            init_args = instantiate(monitor_dict["init_args"])
            monitors.append(monitor_class(run, **init_args))
        else:
            monitors.append(monitor_class(run))

    return metric_monitor.MonitorCollection(monitors)


def run_train(cfg: dict, parser: ArgumentParser) -> None:
    """Run the training workflow."""
    cfg_keys = ["model", "force", "git_hash", "train"]
    cfg = {key: cfg[key] for key in cfg_keys}

    train_cfg = cfg["train"]

    # DDP setup — triggered automatically when launched with torchrun.
    if dist.is_available() and "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", device_id=local_rank)
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = None
        device = torch.device(train_cfg["device"])

    is_main = rank == 0

    torch.manual_seed(train_cfg["seed"] + rank)

    # Dry-run overrides
    if train_cfg["dry_run"]:
        train_cfg["num_steps"] = 3
        train_cfg["val_num_steps"] = 2
        train_cfg["checkpoint_dir"] = Path(tempfile.gettempdir()) / "dry_run"
        train_cfg["wandb_disable"] = True
        if is_main:
            shutil.rmtree(train_cfg["checkpoint_dir"], ignore_errors=True)

    model_save_dir = Path(train_cfg["checkpoint_dir"])
    if is_main:
        model_save_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # ensure directory exists before all ranks proceed

    # Instantiate the model
    initialize_norm_dict(cfg["model"])
    save_cfg: dict = cfg
    cfg = parser.instantiate_classes(cfg)
    model: nn.Module = cfg["model"]

    train_cfg = cfg["train"]

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        for key in ("train_dataloader", "val_dataloader"):
            dl_cfg = train_cfg[key]
            bs = dl_cfg["init_args"]["batch_size"]
            if bs % world_size != 0:
                raise ValueError(f"{key} batch_size {bs} is not divisible by world_size {world_size}")
            dl_cfg["init_args"]["batch_size"] = bs // world_size
            nw = dl_cfg["init_args"]["num_workers"]
            if nw % world_size != 0:
                raise ValueError(f"{key} num_workers {nw} is not divisible by world_size {world_size}")
            dl_cfg["init_args"]["num_workers"] = nw // world_size

    train_cfg["train_dataloader"] = instantiate(train_cfg["train_dataloader"])
    train_cfg["val_dataloader"] = instantiate(train_cfg["val_dataloader"])

    # Optimizer
    check_instantiate_keys(train_cfg["optimizer"], "optimizer")
    optimizer_class = get_class(train_cfg["optimizer"]["class_path"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        next(iter(train_cfg["loss_fn"].parameters()))
        param_groups = [
            {"params": model.parameters()},
            {"params": train_cfg["loss_fn"].parameters()},
        ]
    except StopIteration:
        param_groups = model.parameters()

    optimizer: optim.Optimizer = optimizer_class(param_groups, **train_cfg["optimizer"]["init_args"])

    # Scheduler
    if train_cfg["scheduler"] is not None:
        check_instantiate_keys(train_cfg["scheduler"], "scheduler")
        scheduler_class = get_class(train_cfg["scheduler"]["class_path"])
        scheduler: optim.lr_scheduler.LRScheduler = scheduler_class(optimizer, **train_cfg["scheduler"]["init_args"])
    else:
        scheduler = None

    initial_step = 0
    initial_sub_epoch = 0

    # Resume from crash: load latest checkpoint and restore full training state
    if train_cfg["resume"]:
        ckpt_dir = model_save_dir / "ckpt"
        if ckpt_dir.is_dir() and any(ckpt_dir.iterdir()):
            ckpt = get_latest_ckpt(ckpt_dir)
            print(f"Resuming from checkpoint: {ckpt}")
            state_dict = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state_dict["model"], strict=True)
            optimizer.load_state_dict(state_dict["optimizer"])
            if scheduler is not None and state_dict.get("scheduler") is not None:
                scheduler.load_state_dict(state_dict["scheduler"])
            initial_step = state_dict.get("step_num", 0)
            initial_sub_epoch = state_dict["sub_epoch"] + 1
        else:
            print("No checkpoints found; starting from scratch.")

    # Load checkpoint state dicts (fine-tuning / partial load)
    elif train_cfg["ckpt"] is not None:
        ckpt = Path(train_cfg["ckpt"])
        if ckpt.is_dir():
            ckpt = get_best_ckpt(ckpt)
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
        ckpt_keys = train_cfg["ckpt_keys"]
        if "model" in ckpt_keys:
            model.load_state_dict(state_dict["model"], strict=True)
        if "optimizer" in ckpt_keys:
            optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in ckpt_keys:
            scheduler.load_state_dict(state_dict["scheduler"])

    # Move to device then wrap with DDP if multi-GPU.
    model.to(device)
    if local_rank is not None:
        model = DDP(model, device_ids=[local_rank])

    # Save the config (rank 0 only)
    if is_main:
        parser.save(save_cfg, model_save_dir / "config.yaml", overwrite=True)

    # Wandb — disabled on non-main ranks
    mode = "disabled" if (not is_main or train_cfg["wandb_disable"] or train_cfg["dry_run"]) else "online"

    with wandb.init(
        entity=train_cfg["entity"],
        project=train_cfg["project"],
        tags=train_cfg["tags"],
        dir=model_save_dir,
        config=save_cfg,
        mode=mode,
    ) as run:
        if is_main:
            package_root = Path(files("deepsno"))
            run.log_code(root=package_root, include_fn=lambda path: path.endswith(".py"))
            print(f"Saving model config and checkpoints to {model_save_dir.resolve()}")
            print(f"Number of trainable parameters: {num_params:,}")

        monitor = _instantiate_monitors(train_cfg, run) if is_main else None

        train(
            checkpoint_dir=model_save_dir / "ckpt",
            run=run,
            log_interval=train_cfg["log_interval"],
            model=model,
            device=device,
            train_dataloader=train_cfg["train_dataloader"],
            val_dataloader=train_cfg["val_dataloader"],
            num_steps=train_cfg["num_steps"],
            optimizer=optimizer,
            loss_fn=train_cfg["loss_fn"],
            scheduler=scheduler,
            val_metric=train_cfg["val_metric"],
            val_metric_is_inverted=train_cfg["val_metric_is_inverted"],
            val_num_steps=train_cfg["val_num_steps"],
            max_grad_norm=train_cfg["max_grad_norm"],
            metric_monitor=monitor,
            rank=rank,
            initial_step=initial_step,
            initial_sub_epoch=initial_sub_epoch,
            train_norm=train_cfg["train_norm"],
            val_norm=train_cfg["val_norm"],
        )


def run_predict(cfg: dict, parser: ArgumentParser) -> None:
    """Run the prediction workflow."""
    import uproot

    cfg_keys = ["model", "force", "git_hash", "predict"]
    cfg = {key: cfg[key] for key in cfg_keys}

    predict_cfg = cfg["predict"]

    # Merge ckpt config before instantiation so the checkpoint's model
    # definition takes precedence over anything from --config.
    ckpt_cfg = parser.parse_path(predict_cfg["ckpt_config"]).as_dict()
    cfg = cfg | ckpt_cfg

    initialize_norm_dict(cfg["model"])

    save_cfg: dict = cfg
    cfg = parser.instantiate_classes(cfg)

    model: nn.Module = cfg["model"]

    ckpt_path = Path(predict_cfg["ckpt"])
    if ckpt_path.is_dir():
        ckpt_path = get_best_ckpt(ckpt_path)

    state_dict = torch.load(ckpt_path, map_location=predict_cfg["device"], weights_only=True)
    model.load_state_dict(state_dict["model"])

    dataloader = instantiate(predict_cfg["dataloader"])

    predict_cfg_path = Path(predict_cfg["output_path"]).with_suffix(".yaml")
    parser.save(save_cfg, predict_cfg_path, overwrite=True)

    with uproot.recreate(predict_cfg["output_path"]) as file:
        predict(
            model=model,
            dataloader=dataloader,
            file=file,
            device=predict_cfg["device"],
            keys=predict_cfg["keys"],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    log_fh = None
    try:
        parser = build_parser()
        cfg = parser.parse_args().as_dict()

        # Optional log-file tee
        if cfg["log_file"] is not None:
            log_fh = open(cfg["log_file"], "w", buffering=1)  # line-buffered
            sys.stdout = Tee(sys.__stdout__, log_fh)
            sys.stderr = Tee(sys.__stderr__, log_fh)

        # Resolve predict paths before git-hash check (needs merged config)
        if cfg["subcommand"] == "predict":
            _resolve_predict_paths(cfg)

        cfg["git_hash"] = resolve_git_hash(cfg)

        if cfg["subcommand"] == "train":
            run_train(cfg, parser)
        elif cfg["subcommand"] == "predict":
            run_predict(cfg, parser)

    except UncommittedChangesError:
        print(
            "Error: working tree has uncommitted changes.\n"
            "Please commit or stash your changes before running, "
            "or pass --force to skip this check.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting.", file=sys.stderr)
        # Use os._exit to skip wandb cleanup (avoids hangs) and exit with
        # the conventional Ctrl+C code 130.  Close the log file first since
        # finally blocks are bypassed by os._exit.
        os._exit(130)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if log_fh is not None:
            log_fh.flush()
            log_fh.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
