"""Source-pinned non-pickle readback of English weak research weights (not FAR)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english63_weak_train as pilot


def verify(input_path, directory):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    result_path=directory/'RESEARCH_RESULT.json'
    result=json.loads(result_path.read_text())
    pilot.require(result.get('schema')==pilot.POLICY and
                  result.get('exact_input_sha256')==pilot.INPUT_SHA256 and
                  result.get('training_script_sha256')==pilot.sha(Path(pilot.__file__).read_bytes()) and
                  result.get('status')=='EXECUTED_SYNTHETIC_WEAK_LABEL_TRAINING_NOT_KWS_QUALIFICATION' and
                  result.get('training_admitted') is False and
                  result.get('formal_promotion_allowed') is False and
                  result.get('human_gold') is False,
                  'EN_REPLAY_REPORT_PROVENANCE')
    rows, audio=pilot.load_strict_inputs(input_path)
    bank=pilot.mel_bank()
    feats={cell:pilot.features(v,bank) for cell,v in audio.items()}
    mean=np.asarray(result['train_only_frontend_norm']['mean32'],dtype=np.float32)
    std=np.asarray(result['train_only_frontend_norm']['std32'],dtype=np.float32)
    m,s=pilot.train_only_norm(feats,rows)
    pilot.require(np.array_equal(m,mean) and np.array_equal(s,std),
                  'EN_REPLAY_NORM_TRAIN_LEAK')
    features={cell:((x-mean)/std).astype(np.float32) for cell,x in feats.items()}
    roster={r['cell_id']:r for r in rows}
    pilot.require(type(result.get('models')) is list and
                  {m['architecture'] for m in result['models']} == {'gru','dsc_tcn'},
                  'EN_REPLAY_ARCHITECTURES')
    reports=[]
    for record in result['models']:
        kind=record['architecture']
        model=pilot.create_model(kind)
        path=directory/record['weights_file']
        pilot.require(path.is_file() and path.name==kind+'-weights.npz' and
                      pilot.sha(path.read_bytes())==record['weights_sha256'],
                      'EN_REPLAY_WEIGHT_DIGEST')
        with np.load(path,allow_pickle=False) as npz:
            pilot.require(set(npz.files)==set(model.state_dict()),
                          'EN_REPLAY_WEIGHT_ROSTER')
            sd={k:torch.from_numpy(npz[k]) for k in npz.files}
        model.load_state_dict(sd,strict=True)
        model.eval()
        saved=record['per_record']
        expected={v['cell_id']:v for v in saved}
        pilot.require(len(expected)==63, 'EN_REPLAY_REPORT_IDS')
        checks=[]
        for action in pilot.ROLES:
            selected=sorted([r for r in rows if r['action']==action],key=lambda r:r['cell_id'])
            checks+=pilot.score_model(model,selected,features)
        for row in checks:
            old=expected[row['cell_id']]
            pilot.require(row['source_family_id']==old['source_family_id'] and
                          row['role']==old['role'] and
                          row['predicted_class']==old['predicted_class'] and
                          row['requested_weak_intent_id']==old['requested_weak_intent_id'] and
                          all(abs(x-y)<2e-5 for x,y in zip(row['raw_clip_probabilities'],old['raw_clip_probabilities'])),
                          'EN_REPLAY_PREDICTION_DRIFT')
        reports.append({'kind':kind,'weight_sha256':record['weights_sha256'],
                        'replayed_cells':len(checks),'report_match':True,
                        'promotion_allowed':False})
    return reports


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-zip',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    args=p.parse_args()
    try:
        print(json.dumps({'verified':verify(args.input_zip,args.result_dir)},sort_keys=True))
    except (OSError,ValueError,RuntimeError,KeyError) as e:
        print('EN_REPLAY_REJECT: '+str(e),file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    raise SystemExit(main())
