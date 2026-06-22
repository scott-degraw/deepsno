#!/usr/bin/env -S python3 -u

import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from deepsno.loops import bench_dataloader
from deepsno.utils.cli import MismatchedGitHash, UncommittedChangesError, resolve_git_hash
from deepsno.utils.hydra_cli import prepare_argv


def run_bench(cfg: dict) -> None:
    """Iterate the train dataloader and report throughput."""
    bench_cfg = cfg["bench"]
    dataloader = instantiate(bench_cfg["train_dataloader"], _convert_="all")
    bench_dataloader(dataloader, num_steps=bench_cfg.get("num_steps"))


@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig) -> None:
    try:
        cfg: dict = OmegaConf.to_container(cfg, resolve=True)
        cfg["git_hash"] = resolve_git_hash(cfg)
        run_bench(cfg)
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
        sys.exit(130)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *prepare_argv(sys.argv[1:])]
    main()
