"""Fixed two-stage TTS host, currently disabled before all runtime side effects."""
from pathlib import Path
import json
import os
import platform
import sys
import time

from contract import require
from tts_common import safe_exception, atomic_json, child_environment, fsync_directory, isolated_command, sha, unaliased, write_once
from tts_gate import (HERE,ROOT,SOURCE_BRANCH,RESULT_BRANCH,verify_release,fresh_context,verify_clean_checkout,
                      read_json,read_only_preflight,public_reader)
from tts_worker import new_outcomes


def execute(env=None):
    env=dict(os.environ) if env is None else env
    release,source=verify_release()  # disabled here, before mkdir/network/subprocess/import
    event=read_json(env["GITHUB_EVENT_PATH"])
    identity=fresh_context(env,event)
    checkout=verify_clean_checkout(identity)
    require(platform.python_implementation()=="CPython" and platform.python_version()=="3.12.14" and
            platform.system()=="Linux" and platform.machine()=="x86_64" and sys.byteorder=="little","locked Python/platform")
    provision_start=float(env["K1_JOB_STARTED_UNIX_SECONDS"])
    require(0<=time.time()-provision_start<300,"remaining provisioning budget")
    provision_remaining=300-(time.time()-provision_start)
    get,preflight_http=public_reader(identity,time.monotonic()+min(120,provision_remaining))
    preflight=read_only_preflight(identity,get)
    require(time.time()-provision_start<=300,"provisioning budget before asset acquisition")
    runtime=unaliased(Path(env["RUNNER_TEMP"])/"k1-punctuation6-tts-v1")
    runtime.mkdir(mode=0o700)
    (runtime/"home").mkdir(mode=0o700)
    output=runtime/"publication-members";output.mkdir(mode=0o700)
    fsync_directory(runtime.parent)
    claim={"schema":"k1-punctuation6-host-claim-v1","phase":"tts","identity":identity,
           "source_branch":SOURCE_BRANCH,"result_branch":RESULT_BRANCH,
           "source_freeze_sha256":source["source_freeze_sha256"],"source_tree":checkout["tree"],
           "preflight_verified":True,"preflight":preflight,"maximum_tts_attempts":6}
    write_once(output/"claim.json",claim)
    write_once(output/"source.json",{"schema":"k1-punctuation6-tts-source-v1","identity":identity,
        "source_tree":checkout["tree"],**source,"source_files":read_json(HERE/"tts-source-freeze.json")["files"],
        "model_provenance":read_json(HERE/"model-provenance.json"),
        "plan_sha256":sha(HERE/"plan.proposed.json"),"runtime_setup":{"status":"not_available"},
        "runtime_model_load":{"status":"not_available"},"raw_definition":"wrapper-returned postdecoder float32, before our fixed resampling; not decoder-preclamp signal"})
    helpers=HERE.parent/"qwen6_tts";sys.path.insert(0,str(helpers))
    from supervision import supervise
    from setup_diagnostics import stderr_diagnostic
    started=time.monotonic();deadline=started+2400
    resources={"schema":"k1-punctuation6-tts-resources-v1","identity":identity,"status":"started",
        "preflight_http":preflight_http,"stages":{},"limits":{"setup_wall_seconds":1200,"generation_wall_seconds":1200,
        "setup_cpu_seconds_per_process":2400,"generation_cpu_seconds_per_process":3600,
        "setup_sampled_rss_bytes":4*1024**3,"generation_sampled_rss_bytes":7*1024**3,
        "affinity_cores":4,"job_stage_wall_seconds":2400},
        "cpu_accounting":"RLIMIT_CPU per process, not hard aggregate tree cap",
        "memory_disk_accounting":"sampled thresholds, not continuous hard limits"}
    atomic_json(output/"resources.json",resources)
    stages=[("setup",Path(sys.executable),helpers/"setup_locked.py",["--root",runtime],1200,2400,4*1024**3,True),
            ("generation",runtime/"venv/bin/python",HERE/"tts_worker.py",["--runtime",runtime,"--output",output],1200,3600,7*1024**3,False)]
    try:
        for name,python,script,args,wall,cpu,rss,is_setup in stages:
            errors=runtime/(name+".stderr")
            receipt=supervise(isolated_command(python,script,args),cwd=ROOT,
                env=child_environment(runtime,python),wall_seconds=min(wall,deadline-time.monotonic()),
                cpu_seconds=cpu,rss_bytes=rss,cores=4,workspace=runtime,installed=runtime/"venv",
                buildtmp=runtime/"buildtmp",log_path=runtime/(name+".log"),job_deadline=deadline,
                setup=is_setup,stderr_path=errors)
            receipt["stderr_diagnostic"]=stderr_diagnostic(errors)
            resources["stages"][name]=receipt
            atomic_json(output/"resources.json",resources)
            require(receipt["returncode"]==0,"supervised stage failed; no retry")
        require(read_json(output/"outcomes.json").get("status")=="completed_six_once","all six outcomes required")
        resources["status"]="completed_six_once"
    except BaseException as error:
        resources.update(status="stopped_no_retry",error_class=safe_exception(error))
        raise
    finally:
        from tts_public_schema import project_setup, project_load
        source_record=read_json(output/"source.json")
        for field,path,project in (("runtime_setup",runtime/"setup-receipt.json",lambda data:project_setup(data)),
                                   ("runtime_model_load",runtime/"model-load.private.json",lambda data:project_load(data,runtime))):
            try:
                source_record[field]=project(read_json(path,maximum=8*1024*1024)) if path.is_file() else {"status":"not_available"}
            except Exception as projection_error:
                source_record[field]={"status":"projection_failed","error_class":safe_exception(projection_error)}
                resources["evidence_projection_error_class"]=safe_exception(projection_error)
                resources["complete_provenance_available"]=False
        atomic_json(output/"source.json",source_record)
        resources["wall_seconds"]=time.monotonic()-started
        atomic_json(output/"resources.json",resources)
        # supervise always kills the process group and reaps before returning or raising.
        # No reconstruction promotes a cell or calls a model; original claims are retained.
        from tts_finalize import finalize_after_exit
        try:
            finalize_after_exit(output,runtime,worker_exited=True)
        except BaseException as finalization_error:
            resources["status"]="stopped_no_retry"
            resources.setdefault("error_class",safe_exception(finalization_error))
            resources["wall_seconds"]=time.monotonic()-started
            atomic_json(output/"resources.json",resources)
            raise
    return runtime


if __name__=="__main__":
    try: execute()
    except BaseException as error:
        print(json.dumps({"status":"stopped_no_retry","error_class":safe_exception(error)}),file=sys.stderr)
        raise SystemExit(1)
