import csv
import re
from dataclasses import dataclass
from pathlib import Path

from gridman.jobexec.utils import gridutils
from gridman.jobs import jobdb, jobinfo
from gridman.jobs.session import Session

ntuple_filelist = Path(
    # "insitu_preproc/filelist.dat"
    # "/home/degraw/snopaul.worktrees/genericise/insitu_preproc/Silver_bisMSB_post_364311_april2025/Silver_bisMSB_post_364311_april2025.dat"
    # "/home/degraw/directionality/2p2/ntuple.dat"
    "/data/snoplus2/degraw/early_bismsb/filelist.dat"
    )
# ntuple_filelist = Path("insitu_preproc/test.dat")
jobdb_path = ntuple_filelist.parent / "jobs.db"

overwrite_output = True

min_ht = 0.0
max_ht = 500.0
min_qhs = 0.0
max_qhs = 500.0

dc_mask = "0xD82100000162C6"
filter_ = f"(fitValid == 1) && ((dcFlagged & {dc_mask}) == {dc_mask}) && (nhits > 20)"
filter_ += " && (posr_av < 4000)"
filter_ += " && (energy > 0.35) && (energy < 0.55)"
filter_ = f"'{filter_}'"

priority = 8

module_names = ["Processing_PARTCAL"]

extra_module_config = {
    "Processing_PARTCAL": {
        "type": "macro",
        "level": "sub_run",
        "path": str(Path("second_pass_select_pruned.mac").resolve()),
        "outputs": ["ratds"],
        "nhit_cut": -1,
    }
}

proc_config = jobinfo.ProcessingConfig(
    module_names=module_names,
    pass_number=0,
    ratdb_tag="fullscint-antinu-v3",
    ratdb_url="postgres://snoplus:dontestopmenow@pgsql.snopl.us:5400/ratdb",
    rat_version="8.0.1",
    output_dir="/snoplus.snolab.ca/user/degraw/insitu_pca/bm",
    container="/snoplus.snolab.ca/user/degraw/containers/rat_deepsno.sif",
    output_se="RAL-LCG2-ECHO-disk",
    extra_module_configs=extra_module_config,
)


def surl_fix(surl: str) -> str:
    return re.sub(
        r"srm://lcg-snopse1.sfu.computecanada.ca:8443",
        r"root://lcg-snopse1.sfu.computecanada.ca",
        surl,
    )


@dataclass
class JobFiles:
    ntuple: jobinfo.GridFile
    zdab: jobinfo.GridFile


jobfiles_by_run = {}

with open(ntuple_filelist, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        ntuple = jobinfo.GridFile(
            surl=surl_fix(row[2]),
            adler=row[3],
        )
        proc_subrun = jobinfo.ProcessedSubrun.from_file(ntuple.surl)
        if proc_subrun.run not in jobfiles_by_run:
            jobfiles_by_run[proc_subrun.run] = {}
        jobfiles_by_run[proc_subrun.run][proc_subrun.subrun] = JobFiles(
            ntuple=ntuple, zdab=None
        )
# first_run = next(iter(jobfiles_by_run))
# jobfiles_by_run = {first_run: jobfiles_by_run[first_run]}

runs = list(jobfiles_by_run.keys())
zdab_grid_files = jobinfo.get_raw_surls(runs)

for zdab in zdab_grid_files:
    run_subrun = jobinfo.RunSubrun.from_zdab(zdab.surl)

    jobfiles_by_run[run_subrun.run][run_subrun.subrun].zdab = zdab

for run, subrun_dict in jobfiles_by_run.items():
    for subrun, job_files in subrun_dict.items():
        if job_files.zdab is None:
            raise ValueError(f"Missing zdab for run {run}, subrun {subrun}")

all_job_files = []
for run, subrun_dict in jobfiles_by_run.items():
    for subrun, job_files in subrun_dict.items():
        all_job_files.append(job_files)
all_job_files.sort(key=lambda x: x.zdab.surl)

with Session(jobdb_path, delete_existing=True) as session:
    for job_files in all_job_files:
        proc_job_info = jobinfo.get_processing_job_info(
            grid_file=job_files.zdab, proc_config=proc_config
        )

        rat_args = "' '".join(proc_job_info.rat_args)
        rat_args = f"'{rat_args}'"

        arguments = [
            "--ratdb_url",
            proc_config.ratdb_url,
            "--container",
            Path(proc_config.container).name,
            "--ntuple",
            job_files.ntuple.surl,
            "--ntuple_adler",
            job_files.ntuple.adler,
            "--filter",
            filter_,
            "--zdab",
            job_files.zdab.surl,
            "--zdab_adler",
            job_files.zdab.adler,
            "--rat_args",
            rat_args,
            "--extract_input",
            Path(proc_job_info.output_data[0]).name,
            "--min_ht",
            min_ht,
            "--max_ht",
            max_ht,
            "--min_qhs",
            min_qhs,
            "--max_qhs",
            max_qhs,
        ]
        arguments = [str(arg) for arg in arguments]
        arguments = " ".join(arguments)

        output_data = proc_job_info.output_data
        output_data[0] = "LFN:" + str(Path(output_data[0]).with_suffix(".pmt.root"))

        input_sandbox = proc_job_info.input_sandbox
        input_sandbox += [
            gridutils.__file__,
            str(Path("ntuple_2_eventlist.C").resolve()),
            str(Path("ratds_extract.C").resolve()),
        ]

        job = jobdb.Job(
            status="unsubmitted",
            name=proc_job_info.name,
            executable=str(Path("deepsno_process_and_extract.py").resolve()),
            input_sandbox=proc_job_info.input_sandbox,
            output_sandbox=proc_job_info.logfiles,
            input_data="LFN:" + proc_config.container,
            output_data=output_data,
            output_se=proc_config.output_se,
            arguments=arguments,
            destination=proc_config.destination,
            priority=priority,
            overwrite_output=overwrite_output,
        )
        session.add(job)
