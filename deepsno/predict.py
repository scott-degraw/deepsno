#!/usr/bin/env -S python3 -u

import os
import sys
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from deepsno.loops import predict
from deepsno.utils.cli import MismatchedGitHash, Tee, UncommittedChangesError, resolve_git_hash
from deepsno.utils.hydra_cli import prepare_argv
from deepsno.utils.train import get_best_ckpt


def _resolve_predict_paths(cfg: dict) -> None:
    """Resolve checkpoint, config, and output paths for predict (in-place)."""
    ckpt = Path(cfg["predict"]["ckpt"]).resolve()
    if ckpt.name == "ckpt":
        checkpoint_dir = ckpt.parent
    elif ckpt.is_file():
        checkpoint_dir = ckpt.parent.parent
    else:
        checkpoint_dir = ckpt
        ckpt = ckpt / "ckpt"
        cfg["predict"]["ckpt"] = str(ckpt)

    if cfg["predict"]["ckpt_config"] is None:
        cfg["predict"]["ckpt_config"] = str(checkpoint_dir / "config.yaml")
    elif not Path(cfg["predict"]["ckpt_config"]).is_absolute():
        cfg["predict"]["ckpt_config"] = str(checkpoint_dir / cfg["predict"]["ckpt_config"])

    if cfg["predict"]["output_path"] is None:
        cfg["predict"]["output_path"] = str(checkpoint_dir / "predict.root")
    elif not Path(cfg["predict"]["output_path"]).is_absolute():
        cfg["predict"]["output_path"] = str(checkpoint_dir / cfg["predict"]["output_path"])


def run_predict(cfg: dict) -> None:
    """Run the prediction workflow."""
    import uproot

    cfg_keys = ["model", "force", "git_hash", "predict"]
    cfg = {key: cfg[key] for key in cfg_keys}

    predict_cfg = cfg["predict"]

    # Merge ckpt config before instantiation so the checkpoint's model
    # definition takes precedence over anything from --config.
    ckpt_cfg = OmegaConf.to_container(OmegaConf.load(predict_cfg["ckpt_config"]), resolve=True)
    cfg = cfg | ckpt_cfg

    save_cfg: dict = cfg
    model: nn.Module = instantiate(cfg["model"], _convert_="all")

    ckpt_path = Path(predict_cfg["ckpt"])
    if ckpt_path.is_dir():
        ckpt_path = get_best_ckpt(ckpt_path)

    state_dict = torch.load(ckpt_path, map_location=predict_cfg["device"], weights_only=True)
    model.load_state_dict(state_dict["model"])

    dataloader = instantiate(predict_cfg["dataloader"], _convert_="all")

    predict_cfg_path = Path(predict_cfg["output_path"]).with_suffix(".yaml")
    OmegaConf.save(config=OmegaConf.create(save_cfg), f=predict_cfg_path)

    with uproot.recreate(predict_cfg["output_path"]) as file:
        predict(
            model=model,
            dataloader=dataloader,
            file=file,
            device=predict_cfg["device"],
            keys=predict_cfg["keys"],
        )


@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig) -> None:
    log_fh = None
    try:
        cfg: dict = OmegaConf.to_container(cfg, resolve=True)

        if cfg.get("log_file") is not None:
            log_fh = open(cfg["log_file"], "w", buffering=1)  # line-buffered
            sys.stdout = Tee(sys.__stdout__, log_fh)
            sys.stderr = Tee(sys.__stderr__, log_fh)

        _resolve_predict_paths(cfg)
        cfg["git_hash"] = resolve_git_hash(cfg)
        run_predict(cfg)

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
