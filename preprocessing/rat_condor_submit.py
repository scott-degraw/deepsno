# Script for submitting many rat jobs through condor

import shutil
from pathlib import Path

import htcondor

count = 10

rat_exec: str = shutil.which("rat")
rat_macro: Path = Path("electrons_2p2.mac")

rat_log_dir: Path = Path("rat_logs")
rat_error_dir: Path = Path("rat_err")
rat_condor_dir: Path = Path("rat_condor_logs")
rat_stdout_dir: Path = Path("rat_stdout")
rat_out_dir: Path = Path("rat_out")

rat_log_dir.mkdir(parents=True, exist_ok=True)
rat_error_dir.mkdir(parents=True, exist_ok=True)
rat_condor_dir.mkdir(parents=True, exist_ok=True)
rat_stdout_dir.mkdir(parents=True, exist_ok=True)
rat_out_dir.mkdir(parents=True, exist_ok=True)

arguments = rat_macro.as_posix() + " "
arguments += f"-o {(rat_out_dir / rat_macro.stem).as_posix()}-$(ProcId).root "
arguments += f"-l {(rat_log_dir / rat_macro.stem).as_posix()}-$(ProcId).log"

job = htcondor.Submit(
    {
        "getenv": "true",
        "executable": rat_exec,
        "arguments": arguments,
        "output": f"{(rat_stdout_dir / rat_macro.stem).as_posix()}-$(ProcId).out",
        "error": f"{(rat_error_dir / rat_macro.stem).as_posix()}-$(ProcId).err",
        "log": f"{(rat_condor_dir / rat_macro.stem).as_posix()}-$(ProcId).log",
        "request_cpus": "1",
        "request_memory": "8GB",
        "request_disk": "8GB",
    }
)

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, count=count)
print(submit_result)
