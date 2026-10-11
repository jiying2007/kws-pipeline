"""Offline new-scope freeze; no network, models, publication or activation."""
from pathlib import Path
import ast
import json
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from campaign_contract import identity,write_once,require


def retained_pins(path):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RETAINED_PINS' for x in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('retained source roster missing')


def build(root=ROOT):
    root=Path(root);new='research/qwen12-native-admission-v1/';old='research/source-screen32-v1/';wh='research/source-screen32-whisper-v1/'
    native=['campaign_contract.py','cpu_budget.py','cpu_gate.py','native_worker.py','runtime_scope.py','setup_adapter.py','native_locks.py','whisper_binding.py','worker_receipts.py','optional_tools.py']
    common={new+n for n in native}
    common|={old+n for n in ('asr_worker.py','runtime_scope.py','sense_adapter.py','native_v2_adapter.py')}
    common|={'research/qwen6_asr/'+n for n in retained_pins(root/old/'asr_worker.py')}
    common|={wh+n for n in ('adapter.py','common.py','runtime-lock.json','source-lock.json','setup_runtime.py')}
    common|={'tools/asr_native_contracts.py','tools/speech_asr_consensus.py','tools/speech_label_admission.py'}
    setup=common|{wh+'attempt-3/'+n for n in ('bounded_host.py','bounded_connect.py','pip_acquire.py','pip_compatibility.py','phased_setup.py')}
    host=setup|{old+n for n in ('hosted_run.py','contract.py','container-lock.json')}
    host|={new+n for n in ('hosted_run.py','log_transport.py','branch_transport.py','evidence_publisher.py','emit_synthetic_branch_fixture.py','cleanup_lifecycle_smoke.py','public-inputs.json','acquisition-plan.json','workflow.yml','prepare_freeze.py')}
    # The actual workflow is reviewed and separately bound, never an arbitrary entrypoint.
    workflow='.github/workflows/qwen12-native-admission-v1.yml'
    if (root/workflow).exists():host.add(workflow)
    result={'schema':'qwen12-native-admission-source-freeze-v1','historical_outputs_reused':False}
    for k,names in (('inference_code',common),('setup_code',setup),('host_code',host)):
        result[k]={n:identity(root/n) for n in sorted(names)}
    require(not any('/evidence/' in n for n in host),'no old result evidence mounted')
    return result


if __name__=='__main__':
    write_once(HERE/'execution-freeze.json',build())
