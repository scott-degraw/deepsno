import shutil
from pathlib import Path

import htcondor

min_hit_time = 0.0
max_hit_time = 800.0

input_paths = list(
    Path("/data/snoplus3/SNOplusData/production/miniProd_RAT-7-0-14_ASCI_RATHS_newRecoordination").glob("*.root")
)

python_executable = str(Path("ratds_extract.py").resolve())

output_dir = Path("/data/snoplus3/degraw/extraction_test/")
output_dir.mkdir(parents=True, exist_ok=True)

output_paths: list[Path] = []
for input_path in input_paths:
    output_paths.append((output_dir / input_path.name).with_suffix(".h5"))

root_exec: str = shutil.which("root")

condor_root_dir = Path("condor_ratds_extract")
condor_log_dir = condor_root_dir / "logs"
stdout_dir = condor_root_dir / "stdout"
err_dir = condor_root_dir / "err"

condor_log_dir.mkdir(parents=True, exist_ok=True)
stdout_dir.mkdir(parents=True, exist_ok=True)
err_dir.mkdir(parents=True, exist_ok=True)

itemdata = []
for input_path, output_path in zip(input_paths, output_paths):
    output_basename = output_path.name
    itemdata.append(
        {
            "input_file": input_path.resolve().as_posix(),
            "output_file": output_path.resolve().as_posix(),
            "output_log": (stdout_dir / output_basename).with_suffix(".out").resolve().as_posix(),
            "error_log": (err_dir / output_basename).with_suffix(".err").resolve().as_posix(),
            "condor_log": (condor_log_dir / output_basename).with_suffix(".log").resolve().as_posix(),
        }
    )

arguments = f"{python_executable} -i $(input_file) -o $(output_file)"
arguments += f" --min_hit_time {min_hit_time} --max_hit_time {max_hit_time}"

job = htcondor.Submit(
    {
        "getenv": "true",
        "executable": shutil.which("python3"),
        "arguments": arguments,
        "output": "$(output_log)",
        "error": "$(error_log)",
        "log": "$(condor_log)",
        "request_cpus": "1",
        "request_memory": "4GB",
        "request_disk": "8GB",
    }
)

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))

print(submit_result)
