import shutil
from pathlib import Path

import htcondor

python_executable = str(Path("preprocessing/ratds_extract.py").resolve())
root_macro = str(Path("preprocessing/ratds_extract.C").resolve())
conda_env_name = "snoplus"

context_window = 512

input_paths = list(Path("/data/snoplus3/SNOplusData/production/rat-7-0-8-9/ratds/Po210").glob("*.root"))

output_dir = Path("/data/snoplus3/degraw/Po210_rat-7.0.8-9/h5_extract")
output_dir.mkdir(parents=True, exist_ok=True)

# Split these input_paths into groups

max_file_group_size: int = 5

input_file_groups = []
output_file_groups = []

input_file_group = []
output_file_group = []

for file_counter, input_path in enumerate(input_paths):
    input_file_group.append(str(input_path))
    output_file_group.append(str((output_dir / input_path.name).with_suffix(".h5")))

    if (file_counter + 1) % max_file_group_size == 0:
        input_file_groups.append(input_file_group)
        output_file_groups.append(output_file_group)
        input_file_group = []
        output_file_group = []

if input_file_group:
    input_file_groups.append(input_file_group)
    output_file_groups.append(output_file_group)

condor_root_dir = Path("condor_logs/ratds_extract").resolve()
condor_log_dir = (condor_root_dir / "logs").resolve()
stdouterr_dir = (condor_root_dir / "stdouterr").resolve()

condor_log_dir.mkdir(parents=True, exist_ok=True)
stdouterr_dir.mkdir(parents=True, exist_ok=True)

itemdata = []
for input_file_group, output_file_group in zip(input_file_groups, output_file_groups):
    input_files = [Path(input_file).name for input_file in input_file_group]
    output_files = [Path(output_file).name for output_file in output_file_group]
    transfer_output_remaps = [
        f"{output_file} = {output_path}" for output_file, output_path in zip(output_files, output_file_group)
    ]
    itemdata.append(
        {
            "input_files": " ".join(input_files),
            "output_files": " ".join(output_files),
            "output_log": f"{str(stdouterr_dir)}/$(ProcID).log",
            "error_log": f"{str(stdouterr_dir)}/$(ProcID).log",
            "condor_log": f"{str(condor_log_dir)}/$(ProcID).log",
            "transfer_input_files": f'{str(root_macro)},{",".join(input_file_group)}',
            "transfer_output_remaps": f'"{" ; ".join(transfer_output_remaps)}"',
        }
    )

arguments = (
    f"run --name {conda_env_name} --no-capture-output {python_executable} "
    f" -m {root_macro} -i $(input_files) -o $(output_files) "
    f" --context_window {context_window} "
)

print("Creating job")

job = htcondor.Submit(
    {
        "nice_user": "false",
        "batch_name": "ratds_extract",
        "executable": shutil.which("conda"),
        "arguments": arguments,
        "output": "$(output_log)",
        "error": "$(error_log)",
        "log": "$(condor_log)",
        "max_materialize": "200",
        "request_cpus": "1",
        "request_memory": "2GB",
        "should_transfer_files": "yes",
        "stream_error": "True",
        "stream_output": "True",
    }
)

schedd = htcondor.Schedd()

print(f"Submitting {len(itemdata)} jobs")
submit_result = schedd.submit(job, itemdata=iter(itemdata))
print(submit_result)
