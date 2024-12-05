import shutil
from pathlib import Path

import htcondor

python_executable = str(Path("main.py").resolve())
conda_env_name = "deepsno"

config_path = "src/configs/test.yaml"

condor_root_dir = Path("condor_logs/train").resolve()
condor_log_dir = (condor_root_dir / "logs").resolve()
stdout_dir = (condor_root_dir / "stdout").resolve()
err_dir = (condor_root_dir / "err").resolve()

condor_log_dir.mkdir(parents=True, exist_ok=True)
stdout_dir.mkdir(parents=True, exist_ok=True)
err_dir.mkdir(parents=True, exist_ok=True)

arguments = f"run --name {conda_env_name} --no-capture-output {python_executable} --config {config_path} train"

job = htcondor.Submit(
    {
        "nice_user": "True",
        "batch_name": "position_reco",
        "executable": shutil.which("conda"),
        "arguments": arguments,
        "output": str(stdout_dir / "out.log"),
        "error": str(err_dir / "err.log"),
        "log": str(condor_log_dir / "log.log"),
        "request_gpus": "1",
        "request_cpus": "33",
        "request_memory": "64G",
    }
)

schedd = htcondor.Schedd()

submit_result = schedd.submit(job)

print(submit_result)
