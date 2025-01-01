#!/usr/bin/env python3

import os
import shutil
import subprocess
from pathlib import Path
import yaml

import htcondor

python_executable = str(Path("main.py").resolve())
conda_env_name = "deepsno"

config_path = "configs/cable_delays.yaml"

condor_root_dir = Path("condor_logs/train").resolve()
condor_root_dir.mkdir(parents=True, exist_ok=True)

arguments = f"run --name {conda_env_name} --no-capture-output {python_executable} --config {config_path} train"

job = htcondor.Submit(
    {
        "nice_user": "True",
        "batch_name": "train",
        "executable": shutil.which("conda"),
        "arguments": arguments,
        "output": str(condor_root_dir / "out.log"),
        "error": str(condor_root_dir / "out.log"),
        "log": str(condor_root_dir / "log.log"),
        "request_gpus": "1",
        "request_cpus": "64",
        "request_memory": "64GB",
    }
)

schedd = htcondor.Schedd()

submit_result = schedd.submit(job)

with open(config_path) as yaml_file:
    cfg = yaml.safe_load(yaml_file)

checkpoint_dir = cfg["train"]["checkpoint_dir"]

process = subprocess.run(
    ["pgrep", "-u", os.environ["USER"], "-f", f"tensorboard --logdir={checkpoint_dir}"], capture_output=True
)

if process.returncode != 0:
    tensorboard_process = subprocess.Popen(
        [
            "conda",
            "run",
            "--name",
            conda_env_name,
            "--no-capture-output",
            "tensorboard",
            f"--logdir={checkpoint_dir}",
            "--load_fast",
            "auto",
        ]
    )

print(submit_result)
