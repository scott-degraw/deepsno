from pathlib import Path

import DIRAC
from DIRAC.Core.Utilities.ReturnValues import returnValueOrRaise
from DIRAC.DataManagementSystem.Client.DataManager import DataManager
from DIRAC.Interfaces.API.Dirac import Dirac
from DIRAC.Interfaces.API.Job import Job

import tqdm

import jobinfo

DIRAC.initialize()

dirac = Dirac()

runlist = "/data/snoplus2/degraw/mlpca_data/bismsb_extract/runs.txt"
with open(runlist, "r") as f:
    runs = [int(line.strip()) for line in f.readlines()]

output_dir = Path("/data/snoplus2/degraw/mlpca_data/bismsb_extract")
base_lfn = Path("/snoplus.snolab.ca/user/processing/Analysis20_PARTCAL")

processed_sub_runs = jobinfo.get_processed_lfns(runs, base_lfn)

dm = DataManager()
for processed_sub_run in tqdm.tqdm(processed_sub_runs):
    pmt_file = processed_sub_run.path.with_suffix(".pmt.root")
    if dm.fileCatalog.exists(str(pmt_file)):
        print(returnValueOrRaise(
            dm.getFile(str(pmt_file), destinationDir=output_dir, diskOnly=True)
        ))
    else:
        print(f"File {pmt_file} does not exist")
