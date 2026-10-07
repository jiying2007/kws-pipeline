#!/usr/bin/env python3
"""Build a public-safe inventory from already materialized, bounded archive inputs.

Reads hashes, WAV headers and decoded PCM bytes only. No inference, audio playback,
download, training or source mutation. All output references are logical, not local.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import wave

from quality import admission, load, require, sha

EXPECTED_PUBLIC_MANIFEST = 'd0c265888b8e4799101a226eb568f4592f5e63d824493fae4214b0d577b5ed01'
MODEL = 'a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d'
CODE_MANIFEST = '4a90303837f75183d55adf21bb3fd6bf2e7cf0c79a62e10d582e9f809612227e'
BASE_COMMIT = '804553286fd9caa294769cc4d44cc0c43dc66e88'
KEYWORDS = ['你好小窝', '小窝小窝']


def build(materialized, archive_root, public_manifest, source_manifest):
    require(sha(source_manifest) == CODE_MANIFEST, 'wrong immutable native source manifest')
    require(sha(public_manifest) == EXPECTED_PUBLIC_MANIFEST, 'wrong bounded public-file manifest')
    manifest = load(public_manifest)
    file_index = {f['path']: f for f in manifest['files']}
    index = load(archive_root / 'archive-index.json')
    require(sha(archive_root / 'archive-index.json') == manifest['archive_index_sha256'], 'archive index hash mismatch')
    scope_path = 'research/2026-10-03-native-a20/LICENSES/SCOPE.md'
    scope_sha = sha(archive_root / 'LICENSES/SCOPE.md')
    require(scope_sha == file_index[scope_path]['sha256'], 'license/provenance scope hash mismatch')
    scope_text = (archive_root / 'LICENSES/SCOPE.md').read_text(encoding='utf-8')
    require('29e01c4e8d000f4bcd70751be16fa94bf3d85a18' in scope_text, 'fixed12 generator revision source missing')
    assets = {}
    for shard in index['assets']:
        path = archive_root / shard['path']
        require(sha(path) == shard['sha256'], 'asset-index shard hash mismatch')
        public_path = 'research/2026-10-03-native-a20/' + shard['path']
        require(file_index[public_path]['sha256'] == shard['sha256'], 'public-file manifest mismatch')
        for asset in load(path)['assets']:
            require(asset['logical_path'] not in assets, 'duplicate logical asset')
            assets[asset['logical_path']] = asset
    def bound(logical):
        path = materialized / logical
        require(sha(path) == assets[logical]['sha256'], 'materialized public identity mismatch: ' + logical)
        return load(path)
    names = {'D20': 'training/F-A-D20/d20-manifest.public.json',
             'Serena18': 'evaluation/F-A-Serena18/source-mapping.public.json',
             'fixed12': 'provenance/fixed12-audio.json'}
    docs = {k: bound(v) for k, v in names.items()}
    restoration_path = 'training/F-A-D20/restoration-status.json'
    restoration = bound(restoration_path)
    require(restoration['exact_D20_source_audio_restored'] and restoration['trainer_protocol_manifest_restored'],
            'restoration status contradicts the inventoried bundle')
    rows = []
    for dataset, doc in docs.items():
        for r in doc.get('rows', doc.get('clips', [])):
            if dataset == 'fixed12':
                ident, logical = r['id'], r['wav_logical_path']
                wh, ph, frames = r['wav_sha256'], r['pcm_sha256'], r['samples']
                strength, binding, speaker, generator = 'unknown', 'not_publicly_bound', None, dict(
                    model='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                    revision='29e01c4e8d000f4bcd70751be16fa94bf3d85a18', evidence_ref=scope_path, evidence_sha256=scope_sha, voice='Qwen3-TTS-Serena-synthetic-reference',
                    seed=None, prompt_family='shared-synthetic-reference-fixed12')
                declared = []
                authority = 'Public byte provenance has no per-row label binding; aggregate human/ASR tiers are not transferable to individual rows'
            else:
                ident = r['source_id']
                logical = r.get('public_logical_name', r.get('expected_input_logical_name'))
                wh, ph, frames = r['source_wav_sha256'], r['source_pcm_sha256'], r['declared_pcm16_frames']
                presence = r['historical_weak_presence'] if dataset == 'D20' else r['weak_presence']
                strength, binding = ('weak', 'source_bound') if presence is not None else ('unknown', 'source_unknown')
                declared = [k for k, value in zip(KEYWORDS, presence or []) if value == 1]
                provenance = r['source_audio_provenance'] if dataset == 'D20' else r['provenance']
                speaker = r['speaker_group']
                generator = dict(model=provenance['generator_model'], revision=provenance['generator_revision'],
                                 voice=provenance['stock_voice'], evidence_ref=names[dataset], evidence_sha256=assets[names[dataset]]['sha256'], seed=provenance['generation_seed'],
                                 prompt_family='historical-'+dataset+'-weak-keyword-prompts')
                authority = 'Historical TTS/ASR weak label only' if presence is not None else 'Historical label unresolved; TTS intent is not truth'
            wav = materialized / logical
            require(sha(wav) == wh == assets[logical]['sha256'], 'WAV identity mismatch: ' + ident)
            with wave.open(str(wav), 'rb') as audio:
                require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) ==
                        (1, 2, 16000, frames), 'WAV format/frame identity mismatch: ' + ident)
                pcm = audio.readframes(frames)
            require(hashlib.sha256(pcm).hexdigest() == ph, 'decoded PCM identity mismatch: ' + ident)
            rows.append(dict(id=dataset+'-'+ident,logical_path=logical,wav_sha256=wh,pcm_sha256=ph,
                sample_rate_hz=16000,frames=frames,duration_s=frames/16000,
                source=dict(dataset=dataset,family=dataset+'-historical-synthetic',tier='synthetic',
                    evidence_ref=names[dataset],evidence_sha256=assets[names[dataset]]['sha256'],
                    speaker=speaker,generator=generator,session=ident,room=None,device=None,afe=None),
                license=dict(identifier='Recorded model/source Apache-2.0; output grant not established',
                    evidence_ref='research/2026-10-03-native-a20/LICENSES/SCOPE.md',
                    review_status='reviewed_research_scope',approved_roles=['inventory'],commercial_output_license='not_established'),
                role='inventory',label_strength=strength,label_binding=binding,label_authority=authority,
                declared_keywords=declared,exposures=['historical_training','results_observed'] if dataset=='D20' else ['results_observed'],
                exposure_review_complete=True,leakage_group=dataset+'-historical-family',derivation_family=ident,
                strata=dict(noise='unknown',far_field='unknown',hard_negative='weak_foil' if dataset=='D20' and not declared else 'unknown'),
                continuous=False,replayed_or_looped=False,positive_events=[],negative_intervals=[]))
    m=dict(schema_version=1,manifest_id='existing-50-inventory-only',keywords=KEYWORDS,
        protocol=dict(id='inventory-only-no-evaluation-protocol-approved',input_order=[a['id'] for a in rows],state_policy='reset_per_asset_preserve_state_within_asset',
            sha256=hashlib.sha256(b'inventory-only-no-evaluation-protocol-approved-v1').hexdigest(),
            model_sha256=MODEL,source_sha256=CODE_MANIFEST,decoder_config_sha256=None,
            split_frozen_before_predictions=False,heldout_open_authorized=False,
            matching='disjoint_half_open_event_windows_first_correct',confidence=.95,
            iid_binomial_assumption=False,stationary_poisson_assumption=False),assets=rows)
    report=admission(m)
    report['source_public_file_manifest_sha256']=EXPECTED_PUBLIC_MANIFEST
    report['baseline_source']=dict(commit=BASE_COMMIT,source_manifest_sha256=CODE_MANIFEST,
        source_manifest_url='https://github.com/jiying2007/kws-pipeline/blob/'+BASE_COMMIT+'/research/native_a20/SOURCE_MANIFEST.json')
    report['archive_index_sha256']=manifest['archive_index_sha256']
    report['publication_status']='Local bounded public-stage identities verified; remote archive completeness not asserted'
    report['public_reference_documents']=[dict(logical_path=scope_path,sha256=scope_sha,remote_presence_verified=False)]
    report['sources']=[dict(logical_path=name,public_sha256=assets[name]['sha256'],
        original_sha256=assets[name]['provenance']['original_sha256']) for name in [*names.values(),restoration_path]]
    report['cohorts']={name:dict(assets=sum(a['source']['dataset']==name for a in rows),
        seconds=round(sum(a['duration_s'] for a in rows if a['source']['dataset']==name),9),
        public_bound_label_tiers=dict(Counter(a['label_strength'] for a in rows if a['source']['dataset']==name))) for name in names}
    report['fixed12_reported_aggregate_only']=dict(human_confirmed=2,asr_weak=5,unknown=5,
        public_per_row_binding_available=False,not_scoring_truth=True,
        authority='Existing research review aggregate; not derivable from the public byte-only provenance used here')
    report['d20_restoration']=dict(source_audio_restored=True,trainer_protocol_manifest_restored=True,
        retrained=False,public_excerpt_standalone_retraining_revalidated=False,
        old_fresh_native_traces_restored=False)
    report['truthful_limits']=['All 50 are synthetic and historically exposed',
        'No independent FRR or continuous-negative FA/h denominator',
        'No true word-tail annotations; no latency qualification',
        'No new train/dev/heldout admission granted by this inventory',
        'Unknown source labels and unpublished row bindings remain distinct',
        'Replacement8 numeric DSP fixtures are not speech or negative hours']
    return m, report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialized-root',type=Path,required=True)
    parser.add_argument('--archive-root',type=Path,required=True)
    parser.add_argument('--public-file-manifest',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    m,r=build(args.materialized_root,args.archive_root,args.public_file_manifest,args.source_manifest)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,value in [('existing-50.manifest.json',m),('existing-50.summary.json',r)]:
        (args.output_dir/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
