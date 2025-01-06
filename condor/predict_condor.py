import shutil
from pathlib import Path

import htcondor

python_executable = str(Path("main.py").resolve())
conda_env_name = "deepsno"

config_path = "configs/cable_delays_position_model.yaml"

condor_root_dir = Path("condor_logs/predict").resolve()
condor_root_dir.mkdir(parents=True, exist_ok=True)

arguments = (
    f"run --name {conda_env_name} --no-capture-output {python_executable} --config {config_path} --force predict "
)

job = htcondor.Submit(
    {
        "nice_user": "True",
        "batch_name": "predict",
        "executable": shutil.which("conda"),
        "arguments": arguments,
        "output": str(condor_root_dir / "out.log"),
        "error": str(condor_root_dir / "out.log"),
        "log": str(condor_root_dir / "log.log"),
        "request_gpus": "1",
        "request_cpus": "64",
        "request_memory": "64G",
    }
)

schedd = htcondor.Schedd()

submit_result = schedd.submit(job)

print(submit_result)
