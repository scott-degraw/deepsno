import shutil
from pathlib import Path

import htcondor

python_executable = str(Path("preprocessing/ratds_extract.py").resolve())
root_macro = str(Path("preprocessing/ratds_extract.C").resolve())
conda_env_name = "snoplus"

min_hit_time = 0.0
max_hit_time = 800.0
context_window = 4096

input_paths = list(Path("/data/snoplus2/hewittc/lemon-type/pt-net-ratds").glob("*.root"))

output_dir = Path("/data/snoplus3/degraw/uniform_electron_energy/pt-net-h5/")
output_dir.mkdir(parents=True, exist_ok=True)

output_paths: list[Path] = []
for input_path in input_paths:
    output_paths.append((output_dir / input_path.name).with_suffix(".h5"))

condor_root_dir = Path("condor_logs/ratds_extract").resolve()
condor_log_dir = (condor_root_dir / "logs").resolve()
stdout_dir = (condor_root_dir / "stdout").resolve()
err_dir = (condor_root_dir / "err").resolve()

condor_log_dir.mkdir(parents=True, exist_ok=True)
stdout_dir.mkdir(parents=True, exist_ok=True)
err_dir.mkdir(parents=True, exist_ok=True)

itemdata = []
for input_path, output_path in zip(input_paths, output_paths):
    output_basename = output_path.name
    itemdata.append(
        {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "output_log": str((stdout_dir / output_basename).with_suffix(".out")),
            "error_log": str((err_dir / output_basename).with_suffix(".err")),
            "condor_log": str((condor_log_dir / output_basename).with_suffix(".log")),
        }
    )

arguments = (
    f"run --name {conda_env_name} --no-capture-output {python_executable} "
    f" -m {root_macro} -i $(input_file) -o $(output_file) "
    f" --min_hit_time {min_hit_time} --max_hit_time {max_hit_time} --context_window {context_window} "
)

print("Creating job")

job = htcondor.Submit(
    {
        "nice_user": "True",
        "batch_name": "ratds_extract",
        "getenv": "true",
        "executable": shutil.which("conda"),
        "arguments": arguments,
        "output": "$(output_log)",
        "error": "$(error_log)",
        "log": "$(condor_log)",
        "max_materialize": "200",
        "request_cpus": "1",
    }
)

schedd = htcondor.Schedd()

print(f"Submitting {len(itemdata)} jobs")
submit_result = schedd.submit(job, itemdata=iter(itemdata))
print(submit_result)
