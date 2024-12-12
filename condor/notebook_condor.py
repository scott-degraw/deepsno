import shutil
import subprocess
from pathlib import Path

import htcondor

conda_env_name = "deepsno"

condor_root_dir = Path("condor_logs/notebook").resolve()
condor_root_dir.mkdir(parents=True, exist_ok=True)

port = "7777"
arguments = f"run --name {conda_env_name} --no-capture-output jupyter notebook --no-browser --port={port}"

job = htcondor.Submit(
    {
        "nice_user": "True",
        "batch_name": "plotting",
        "executable": shutil.which("conda"),
        "arguments": arguments,
        "output": str(condor_root_dir / "out.log"),
        "error": str(condor_root_dir / "err.log"),
        "log": str(condor_root_dir / "log.log"),
        "request_cpus": "1",
        "request_memory": "4G",
    }
)

schedd = htcondor.Schedd()

submit_result = schedd.submit(job)

cluster_id = submit_result.cluster()
print(cluster_id)

subprocess.run(
    ["condor_ssh_to_job", "-auto-retry", "-ssh", "ssh", str(cluster_id), "-NfL", f"localhost:{port}:localhost:{port}"]
)

print(submit_result)
