#!/usr/bin/env python3
import shutil
from pathlib import Path

import htcondor
import yaml

python_executable = str(Path("preprocessing/merge_and_preprocess.py").resolve())
conda_env_name = "deepsno"

input_paths = Path("/data/snoplus3/degraw/Po210_rat-7.0.8-9/h5_extract").glob("*.h5")
input_paths = [str(path) for path in input_paths]

config = {
    "input_paths": input_paths,
    "train_output_path": "/data/snoplus3/degraw/Po210_rat-7.0.8-9/train_dset.h5",
    "test_output_path": "/data/snoplus3/degraw/Po210_rat-7.0.8-9/test_dset.h5",
    "train_test_split": 0.8,
    "block_size": int(1e6),
    "seed": 47381,
}

condor_root_dir = Path("condor_logs/merge").resolve()
condor_root_dir.mkdir(parents=True, exist_ok=True)

config_path = condor_root_dir / "config.yaml"

with open(config_path, "w") as yaml_file:
    yaml.dump(config, yaml_file)

arguments = f"run --name {conda_env_name} --no-capture-output {python_executable} --config {config_path}"

job = htcondor.Submit(
    {
        "nice_user": "true",
        "batch_name": "merge",
        "executable": shutil.which("conda"),
        "arguments": arguments,
        "output": str(condor_root_dir / "out.log"),
        "error": str(condor_root_dir / "out.log"),
        "log": str(condor_root_dir / "log.log"),
        "max_materialize": "1",
        "request_cpus": "64",
        "request_memory": "64GB",
    }
)

schedd = htcondor.Schedd()

submit_result = schedd.submit(job)

print(submit_result)
