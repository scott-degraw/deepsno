#!/usr/bin/env -S python3 -u

import copy
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from deepsno.loops import _unwrap, train
from deepsno.metrics import metric_monitor
from deepsno.utils.cli import MismatchedGitHash, Tee, UncommittedChangesError, resolve_git_hash
from deepsno.utils.hydra_cli import prepare_argv
from deepsno.utils.train import get_best_ckpt, get_latest_ckpt

# Distinct from a real crash/failure exit code: tells the Slurm wrapper script
# "ran out of time, please requeue" rather than "something went wrong".
REQUEUE_EXIT_CODE = 75


def create_training_snapshot(label: str) -> str:
    """Snapshot the working tree (including untracked files) as a tagged git commit.

    Does not modify the working tree or permanently alter the index.  If a
    previous snapshot exists with an identical tree, it is reused (O(1) lookup
    via ``training/tree/<tree_sha>``).  Otherwise a new commit is created and
    tagged with both ``training/<label>`` and ``training/tree/<tree_sha>``.

    Returns the snapshot commit SHA.
    """
    repo_dir = Path(__file__).parent.resolve()

    # Capture the current index so we can restore it afterwards.
    orig_tree = subprocess.run(
        ["git", "write-tree"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()

    try:
        # Stage *everything* (tracked modifications + untracked files).
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)

        snap_tree = subprocess.run(
            ["git", "write-tree"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
    finally:
        # Always restore the index to its original state.
        subprocess.run(["git", "read-tree", orig_tree], cwd=repo_dir, check=True)

    # O(1) dedup: if a snapshot with this exact tree already exists, reuse it.
    existing = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/training/tree/{snap_tree}"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        return existing.stdout.strip()

    parent = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()

    snapshot_sha = subprocess.run(
        ["git", "commit-tree", snap_tree, "-p", parent, "-m", f"training snapshot: {label}"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    subprocess.run(["git", "tag", f"training/{label}", snapshot_sha], cwd=repo_dir, check=True)
    subprocess.run(["git", "tag", f"training/tree/{snap_tree}", snapshot_sha], cwd=repo_dir, check=True)

    return snapshot_sha


def _instantiate_monitors(cfg: dict, run: wandb.sdk.wandb_run.Run) -> metric_monitor.MetricMonitor | None:
    """Instantiate metric monitors from the config, if any."""
    monitors_cfg = cfg["metric_monitors"]

    if monitors_cfg is None:
        return None

    if isinstance(monitors_cfg, dict):
        monitors_cfg = [monitors_cfg]

    monitors = [instantiate(monitor_cfg, run, _convert_="all") for monitor_cfg in monitors_cfg]

    return metric_monitor.MonitorCollection(monitors)


def run_train(cfg: dict) -> bool:
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

    dry_run = train_cfg["dry_run"]

    model_save_dir = Path(train_cfg["checkpoint_dir"])
    if is_main:
        model_save_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # ensure directory exists before all ranks proceed

    # Snapshot the current code state so every training run is reproducible.
    if is_main:
        label = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        snapshot_sha = create_training_snapshot(label)
        (model_save_dir / "snapshot.txt").write_text(f"{snapshot_sha}\n")
        print(f"Code snapshot: {snapshot_sha}")
    else:
        snapshot_sha = None

    # Instantiate the model
    save_cfg: dict = copy.deepcopy(cfg)
    if snapshot_sha is not None:
        save_cfg["git_hash"] = snapshot_sha

    model: nn.Module = instantiate(cfg["model"], _convert_="all")

    train_cfg = cfg["train"]

    if dist.is_available() and dist.is_initialized() and train_cfg["scale_by_world_size"]:
        world_size = dist.get_world_size()
        for key in ("train_dataloader", "val_dataloader"):
            dl_cfg = train_cfg[key]
            bs = dl_cfg["batch_size"]
            if bs % world_size != 0:
                raise ValueError(f"{key} batch_size {bs} is not divisible by world_size {world_size}")
            dl_cfg["batch_size"] = bs // world_size
            nw = dl_cfg["num_workers"]
            if nw % world_size != 0:
                raise ValueError(f"{key} num_workers {nw} is not divisible by world_size {world_size}")
            dl_cfg["num_workers"] = nw // world_size

    if train_cfg["test"]:
        for key in ("train_dataloader", "val_dataloader"):
            dl_cfg = train_cfg[key]
            dl_cfg["batch_size"] = 2
            dl_cfg["num_workers"] = 1

    for key in ("train_dataloader", "val_dataloader"):
        train_cfg[key]["generator"] = torch.Generator().manual_seed(train_cfg["seed"])
    train_dataloader = instantiate(train_cfg["train_dataloader"], _convert_="all")
    val_dataloader = instantiate(train_cfg["val_dataloader"], _convert_="all")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_fn = instantiate(train_cfg["loss_fn"], _convert_="all")
    try:
        next(iter(loss_fn.parameters()))
        param_groups = [
            {"params": model.parameters()},
            {"params": loss_fn.parameters()},
        ]
    except StopIteration:
        param_groups = model.parameters()

    val_metric = instantiate(train_cfg["val_metric"], _convert_="all")

    optimizer: optim.Optimizer = instantiate(train_cfg["optimizer"], param_groups, _convert_="all")

    # Scheduler
    if train_cfg["scheduler"] is not None:
        scheduler: optim.lr_scheduler.LRScheduler = instantiate(train_cfg["scheduler"], optimizer, _convert_="all")
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
            initial_sub_epoch = state_dict["sub_epoch"] + (1 if state_dict.get("epoch_complete", True) else 0)
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
        OmegaConf.save(config=OmegaConf.create(save_cfg), f=model_save_dir / "config.yaml")

    if dry_run in ("only", "before"):
        dry_run_dir = Path(tempfile.gettempdir()) / "dry_run"
        if is_main:
            shutil.rmtree(dry_run_dir, ignore_errors=True)
        initial_state = {
            "model": copy.deepcopy(_unwrap(model).state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        }
        with wandb.init(mode="disabled") as dry_run_wandb:
            train(
                checkpoint_dir=dry_run_dir / "ckpt",
                run=dry_run_wandb,
                log_interval=1,
                model=model,
                device=device,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_steps=3,
                steps_per_epoch=None,
                val_num_steps=3,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                val_metric=val_metric,
                val_metric_is_inverted=train_cfg["val_metric_is_inverted"],
                max_grad_norm=train_cfg["max_grad_norm"],
                rank=rank,
                train_norm=train_cfg["train_norm"],
                val_norm=train_cfg["val_norm"],
                tqdm_mininterval=train_cfg["tqdm_mininterval"],
            )
        _unwrap(model).load_state_dict(initial_state["model"])
        optimizer.load_state_dict(initial_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(initial_state["scheduler"])
        if is_main:
            print("Dry run passed.")

    if dry_run == "only":
        return False

    deadline = None
    requeue_buffer_seconds = train_cfg.get("requeue_buffer_seconds")
    if requeue_buffer_seconds is not None and "SLURM_JOB_END_TIME" in os.environ:
        deadline = float(os.environ["SLURM_JOB_END_TIME"]) - requeue_buffer_seconds

    # Wandb — disabled on non-main ranks
    mode = "disabled" if (not is_main or train_cfg["wandb_disable"]) else "online"

    # Persist the wandb run id so a requeued job continues the same run instead
    # of fragmenting the loss curve into a new run every ~3h.
    wandb_run_id_file = model_save_dir / "wandb_run_id.txt"
    wandb_id = wandb_run_id_file.read_text().strip() if wandb_run_id_file.is_file() else None

    with wandb.init(
        entity=train_cfg["entity"],
        project=train_cfg["project"],
        tags=train_cfg["tags"],
        dir=model_save_dir,
        config=save_cfg,
        mode=mode,
        id=wandb_id,
        resume="must" if wandb_id is not None else None,
    ) as run:
        if is_main and mode != "disabled" and wandb_id is None:
            wandb_run_id_file.write_text(f"{run.id}\n")

        if is_main:
            print(f"Saving model config and checkpoints to {model_save_dir.resolve()}")
            print(f"Number of trainable parameters: {num_params:,}")

        monitor = _instantiate_monitors(train_cfg, run) if is_main else None

        preempted = train(
            checkpoint_dir=model_save_dir / "ckpt",
            run=run,
            log_interval=train_cfg["log_interval"],
            model=model,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_steps=train_cfg["num_steps"],
            steps_per_epoch=train_cfg.get("steps_per_epoch"),
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            val_metric=val_metric,
            val_metric_is_inverted=train_cfg["val_metric_is_inverted"],
            val_num_steps=train_cfg.get("val_num_steps"),
            max_grad_norm=train_cfg["max_grad_norm"],
            metric_monitor=monitor,
            rank=rank,
            initial_step=initial_step,
            initial_sub_epoch=initial_sub_epoch,
            train_norm=train_cfg["train_norm"],
            val_norm=train_cfg["val_norm"],
            deadline=deadline,
            tqdm_mininterval=train_cfg["tqdm_mininterval"],
        )

    return preempted


@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig) -> None:
    log_fh = None
    try:
        cfg: dict = OmegaConf.to_container(cfg, resolve=True)

        if cfg.get("log_file") is not None:
            log_fh = open(cfg["log_file"], "w", buffering=1)  # line-buffered
            sys.stdout = Tee(sys.__stdout__, log_fh)
            sys.stderr = Tee(sys.__stderr__, log_fh)

        cfg["git_hash"] = resolve_git_hash(cfg)
        preempted = run_train(cfg)
        if preempted:
            sys.exit(REQUEUE_EXIT_CODE)

    except UncommittedChangesError:
        print(
            "Error: working tree has uncommitted changes.\n"
            "Please commit or stash your changes before running, "
            "or pass force=true to skip this check.",
            file=sys.stderr,
        )
        sys.exit(1)
    except MismatchedGitHash as e:
        print(f"Error: {e}", file=sys.stderr)
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
    sys.argv = [sys.argv[0], *prepare_argv(sys.argv[1:])]
    main()
