"""Exact fixed diagnostic closures; offline stdlib measurement only."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD='research/k1-punctuation6-asr18-v1/'
NEW='research/k1-punctuation6-prepare-diagnostic-v1/'
if str(ROOT/OLD) not in sys.path:sys.path.insert(0,str(ROOT/OLD))
import prepare_freeze as retained
import campaign_contract as c
WORKER=('prepare_contract.py','prepare_worker.py','prepare-release.json')
HOST=('prepare_host.py','prepare_run.py','prepare_observer.py','prepare_freeze_diagnostic.py',
    'diagnostic_transport.py','diagnostic_publisher.py','publication-release.json','workflow.yml')


def build(root=ROOT):
    root=Path(root)
    old=retained.build(root)
    worker=dict(old['inference_code'])
    worker.update({NEW+name:c.identity(root/NEW/name) for name in WORKER})
    setup=dict(old['setup_code'])
    host=dict(old['host_code']);host.update(worker);host.update(setup)
    host.update({NEW+name:c.identity(root/NEW/name) for name in HOST})
    return {'schema':'k1-prepare-source-freeze-v1','experiment':'k1-punctuation6-prepare-diagnostic-r2',
        'worker_code':dict(sorted(worker.items())),'inference_code':dict(sorted(worker.items())),
        'setup_code':dict(sorted(setup.items())),'host_code':dict(sorted(host.items()))}


if __name__=='__main__':
    c.require(not sys.argv[1:],'fixed offline freeze takes no parameters')
    sys.stdout.buffer.write(c.canonical(build()))
