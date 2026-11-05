#!/usr/bin/env python3
"""Bounded official FSMN2599 full-text weak-label consumption diagnostic."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time
import types
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'training'), str(ROOT / 'tools')]
import research_runtime
from research_job_lock import heavy_model_slot
from research_weak_train import (append_json, canonical, digest, json_read, require,
                                 save_json, sha_file, tensor_sha, torch_runtime, utc)
from research_weak_recipe import fraction_record, row_class

LOCK_PATH = ROOT / 'configs/runtime/fsmn-fulltext-sources.json'
SEED = 20261105
STAGES = ('head', 'tail')
HEAD = 'backbone.out_linear2.'
TAIL = ('backbone.fsmn.3.', 'backbone.out_linear1.')


def source_lock():
    return json_read(LOCK_PATH)


def verify_sources(directory):
    lock = source_lock()
    for entry in lock['files']:
        path = directory / entry['path']
        require(path.is_file() and path.stat().st_size == entry['bytes'] and sha_file(path) == entry['sha256'],
                'official pinned source changed: ' + entry['path'])
    return lock


def fetch(args):
    root = args.sources.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in source_lock()['files']:
        target = root / entry['path']
        if target.exists():
            require(target.stat().st_size == entry['bytes'] and sha_file(target) == entry['sha256'],
                    'refuse to replace mismatched cached source: ' + entry['path'])
            records.append({**entry, 'reused_verified_bytes': True})
            continue
        request = urllib.request.Request(entry['url'], headers={'User-Agent': 'kws-research-source-verification/1.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(10 * 1024 * 1024 + 1)
            status, final_url = response.status, response.url
        require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'],
                'downloaded official source identity mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        records.append({**entry, 'reused_verified_bytes': False, 'http_status': status, 'final_url': final_url})
    verify_sources(root)
    receipt = root / ('source-fetch-' + str(time.time_ns()) + '.json')
    save_json(receipt, {'schema': 'fsmn-source-fetch-v1', 'created_at': utc(), 'files': records, 'source_lock_sha256': sha_file(LOCK_PATH)})
    print(json.dumps({'status': 'verified', 'sources': str(root), 'files': len(records)}), flush=True)
    return 0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream(directory, torch):
    verify_sources(directory)
    fsmn = load_module('d0_official_fsmn', directory / 'upstream/fsmn.py')
    cmvn = load_module('d0_official_cmvn', directory / 'upstream/cmvn.py')
    path = directory / 'upstream/kaldi.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not
                 (isinstance(node, ast.Import) and any(alias.name == 'torchaudio' for alias in node.names)) and not
                 (isinstance(node, ast.FunctionDef) and node.name in ('mfcc', '_get_dct_matrix'))]
    require(not any(isinstance(node, ast.Name) and node.id == 'torchaudio' for node in ast.walk(tree)),
            'unexpected native torchaudio dependency in source-only fbank')
    namespace = {}
    exec(compile(tree, str(path), 'exec'), namespace)
    path = directory / 'upstream/processor.py'
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                node.name in ('compute_fbank', 'context_expansion', 'frame_skip')]
    require({node.name for node in selected} == {'compute_fbank', 'context_expansion', 'frame_skip'},
            'official frontend function set changed')
    processor = {'torch': torch, 'kaldi': types.SimpleNamespace(fbank=namespace['fbank'])}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), processor)
    return fsmn, cmvn, processor


def build_model(directory, torch):
    fsmn, cmvn, processor = load_upstream(directory, torch)
    state = torch.load(directory / 'avg_30.pt', weights_only=True, map_location='cpu')
    require(len(state) == 30 and tuple(state['backbone.out_linear2.linear.weight'].shape) == (2599, 140) and
            tuple(state['global_cmvn.mean'].shape) == (400,), 'official output/CMVN geometry changed')
    class OfficialDonor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_cmvn = cmvn.GlobalCMVN(state['global_cmvn.mean'].clone(), state['global_cmvn.istd'].clone(), True)
            self.backbone = fsmn.FSMN(400, 140, 4, 250, 128, 10, 2, 1, 1, 140, 2599)
            self.load_state_dict(state, strict=True)

        def forward(self, feature):
            logits, _ = self.backbone(self.global_cmvn(feature), None)
            return logits.transpose(0, 1)
    model = OfficialDonor()
    require(all(torch.equal(value, model.state_dict()[name]) for name, value in state.items()), 'official initial tensor mismatch')
    return model, processor


def vocabulary(raw):
    result = {}
    for line in raw.decode('utf-8').splitlines():
        token, number = line.split()
        output_id = int(number) - 1
        if 0 <= output_id < 2599:
            require(token not in result, 'duplicate vocabulary token')
            result[token] = output_id
    require(set(result.values()) == set(range(2599)) and result.get('<blk>') == 0,
            'official output vocabulary/blank changed')
    return result


def make_recipe(rows, vocab):
    require(len(rows) == 48 and len({row['source_id'] for row in rows}) == 48 and
            len({row['item_id'] for row in rows}) == 48, 'exact distinct FSMN48 roster required')
    classes = []
    for row in rows:
        require(row['consumer'] == 'fsmn-fulltext' and row['split'] == 'train' and
                row['purpose'] == 'technical-research-only' and row['human_gold'] is False and
                row['independent_test'] is False, 'FSMN view must be technical train-only')
        text, ids = row['full_text'], row['target_ids']
        require(bool(text) and all(char in vocab and vocab[char] > 0 for char in text) and
                ids == [vocab[char] for char in text] and all(type(value) is int for value in ids),
                'full text must be retained in official vocabulary; OOV-to-blank prohibited')
        require(row['ctc_mask'] is True and type(row['ctc_weight']) in (int, float) and
                math.isfinite(row['ctc_weight']) and 0 < row['ctc_weight'] <= 1, 'invalid CTC weight/mask')
        require(row['keyword_mask'] == [True, True] and row['keyword_targets'] ==
                [int(keyword in text) for keyword in ('你好小窝', '小窝小窝')], 'known keyword metadata differs from full weak text')
        classes.append(row_class(row))
    names = ('keyword-1-positive', 'keyword-2-positive', 'nearphone-negative')
    require([classes.count(name) for name in names] == [6, 6, 36], 'expected D0 classes differ')
    mass = {name: sum((Fraction(str(row['ctc_weight'])) for row, kind in zip(rows, classes) if kind == name), Fraction()) for name in names}
    require([mass[name] for name in names] == [Fraction(3), Fraction(11, 2), Fraction(57, 2)], 'fixed D0 ASR mass differs')
    scales = {names[0]: Fraction(1), names[1]: Fraction(6, 11), names[2]: Fraction(4, 171)}
    items = [{'row_index': index, 'item_id': row['item_id'], 'source_id': row['source_id'], 'class': kind,
              'asr_ctc_weight': row['ctc_weight'], 'class_multiplier': fraction_record(scales[kind]),
              'effective_ctc_weight': float(Fraction(str(row['ctc_weight'])) * scales[kind]),
              'label_binding_sha256': digest({key: value for key, value in row.items() if key != 'audio_path'})}
             for index, (row, kind) in enumerate(zip(rows, classes))]
    generator, orders = random.Random(SEED), {}
    for stage in STAGES:
        order = list(range(48))
        generator.shuffle(order)
        orders[stage] = order
    return {'schema': 'd0-official-fsmn-recipe-v1', 'seed': SEED, 'batch_size': 4, 'epochs_per_stage': 1,
            'stages': list(STAGES), 'stage_row_orders': orders, 'items': items, 'ctc_blank_id': 0,
            'vocabulary_size': 2599, 'fixed_ctc_normalizer': float(Fraction(5, 36)),
            'ctc_length_normalization': 'sequence NLL divided by full target token length',
            'loss_denominator': 'batch item count times fixed snapshot mean effective weight; never batch weight sum',
            'classes': {name: {'items': classes.count(name), 'raw_asr_mass': fraction_record(mass[name]),
                              'multiplier': fraction_record(scales[name]), 'effective_mass': fraction_record(mass[name] * scales[name])} for name in names},
            'nearphone_fraction_of_effective_ctc_mass': 0.1,
            'mass_definition': 'fixed objective coefficient mass; numerical loss contribution is measured separately',
            'stage_trainable_prefixes': {'head': [HEAD], 'tail': [HEAD, *TAIL]},
            'optimizer': {'name': 'AdamW', 'head_lr': 0.0001, 'tail_lr': 0.00001, 'betas': [0.9, 0.999],
                          'eps': 1e-8, 'weight_decay': 0.0, 'fresh_optimizer_each_stage': True},
            'gradient_clip_max_norm': 5.0, 'stage_sample_exposures': 48, 'stage_optimizer_steps': 12,
            'frontend': {'sample_rate': 16000, 'pcm_amplitude': 'PCM16 integer scale passed to official fbank',
                         'mel_bins': 80, 'frame_length_ms': 25, 'frame_shift_ms': 10, 'window': 'official default Povey',
                         'dither': 0.0, 'energy_floor': 0.0, 'snip_edges': True, 'left_context': 2, 'right_context': 2,
                         'frame_skip': 3, 'audio_padding_samples': 0, 'right_feature_frames_dropped': 2,
                         'augmentation': 'none; diagnostic dither0 and no SpecAugment, not original pretraining recipe'},
            'historical_b_or_a20_weights_used': False, 'ordinary_negative_items': 0, 'independent_development_items': 0,
            'human_gold': False, 'quality_winner': None, 'production_qualified': False,
            'training_set_readback_is_generalization': False}


def fulltext_ctc(logits, target, weight, *, normalizer):
    import torch
    require(logits.ndim == 3 and logits.shape[1:] == (1, 2599) and all(type(x) is int and 0 < x < 2599 for x in target),
            'FSMN full-text loss geometry/target mismatch')
    require(bool(target) and 0 < weight <= 1 and normalizer > 0, 'invalid full-text loss weights')
    raw = torch.nn.functional.ctc_loss(logits.log_softmax(2), torch.tensor(target, dtype=torch.long),
            torch.tensor([logits.shape[0]], dtype=torch.long), torch.tensor([len(target)], dtype=torch.long),
            blank=0, reduction='none', zero_infinity=False)[0] / len(target)
    return raw * (weight / normalizer), raw.detach()


def source_inventory():
    paths = ['training/research_fsmn_weak.py', 'training/research_weak_train.py', 'training/research_weak_recipe.py',
             'training/model.py', 'training/training_state.py', 'configs/runtime/fsmn-fulltext-sources.json',
             'configs/runtime/research-runtime.json', 'tools/research_runtime.py', 'tools/research_job_lock.py']
    return {path: sha_file(ROOT / path) for path in paths}


def prepare(args):
    manifest = research_runtime.load_manifest()
    research_runtime.check_interpreter(manifest)
    research_runtime.check_packages(manifest, 'core')
    sources, view = args.sources.resolve(), args.view.resolve()
    verify_sources(sources)
    require(sha_file(view / 'MANIFEST.json') == args.view_manifest_sha256, 'fixed view manifest changed')
    require(sha_file(view / 'evidence/fsmn-tokens.txt') == sha_file(sources / 'tokens.txt'), 'view and official vocabularies differ')
    loader_path = args.view_loader.resolve()
    data_root = loader_path.parents[1]
    data_paths = [loader_path, *[data_root / 'kws_data_contract' / name for name in ('identity.py', 'reference_text.py', 'source_registry.py')]]
    data_hashes = {str(path.relative_to(data_root)): sha_file(path) for path in data_paths}
    loader = load_module('d0_fsmn_verified_view_loader', loader_path)
    rows = list(loader.load_training_rows(view, 'fsmn-fulltext'))
    recipe = make_recipe(rows, vocabulary((sources / 'tokens.txt').read_bytes()))
    with heavy_model_slot(args.heavy_lock):
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        torch, _ = torch_runtime()
        import numpy as np
        from training_state import state_identity
        model, processor = build_model(sources, torch)
        model.eval()
        tensors, observations = [], []
        for index, row in enumerate(rows):
            path = Path(row['audio_path'])
            require(sha_file(path) == row['file_sha256'], 'training WAV changed')
            with wave.open(str(path), 'rb') as stream:
                require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) ==
                        (1, 2, 16000, row['samples']), 'training PCM format changed')
                pcm = stream.readframes(stream.getnframes())
            require(hashlib.sha256(pcm).hexdigest() == row['pcm_sha256'], 'training PCM changed')
            waveform = torch.from_numpy(np.frombuffer(pcm, dtype='<i2').astype(np.float32) / np.float32(32768)).unsqueeze(0)
            data = [{'key': row['item_id'], 'label': row['target_ids'], 'sample_rate': 16000, 'wav': waveform}]
            fbank = processor['compute_fbank'](data, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
            context = processor['context_expansion'](fbank, left=2, right=2)
            feature = next(processor['frame_skip'](context, skip_rate=3))['feat'].contiguous()
            fbank_frames = 1 + (row['samples'] - 400) // 160
            minimum = len(row['target_ids']) + sum(a == b for a, b in zip(row['target_ids'], row['target_ids'][1:]))
            require(tuple(feature.shape) == (fbank_frames // 3, 400) and len(feature) >= minimum and
                    bool(torch.isfinite(feature).all()), 'official no-padding CTC geometry infeasible')
            tensors.append(feature)
            observations.append({'item_id': row['item_id'], 'source_id': row['source_id'], 'samples': row['samples'],
                                 'fbank_frames': fbank_frames, 'model_frames': len(feature), 'ctc_minimum_frames': minimum,
                                 'feature_sha256': tensor_sha(feature), 'pcm_sha256': row['pcm_sha256'],
                                 'target_ids': row['target_ids'], 'label_binding_sha256': recipe['items'][index]['label_binding_sha256']})
        with torch.no_grad():
            probe = model(tensors[0].unsqueeze(0))
        require(tuple(probe.shape) == (len(tensors[0]), 1, 2599) and bool(torch.isfinite(probe).all()), 'official output probe failed')
        identity = state_identity(model.state_dict())
        with (output / 'features.pt').open('xb') as stream:
            torch.save({'features': tensors}, stream)
        with (output / 'source-rows.jsonl').open('xb') as stream:
            for row in rows:
                stream.write(canonical(row) + b'\n')
        save_json(output / 'recipe.json', recipe)
        save_json(output / 'feature-observations.json', observations)
        for relative, expected in data_hashes.items():
            raw = (data_root / relative).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == expected, 'data loader changed during preparation')
            target = output / 'data-source-snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        require(sha_file(view / 'MANIFEST.json') == args.view_manifest_sha256, 'view changed during preparation')
        record = {'schema': 'd0-official-fsmn-preparation-v1', 'status': 'complete', 'created_at': utc(),
                  'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_code': source_inventory(), 'sources': str(sources), 'source_lock_sha256': sha_file(LOCK_PATH),
                  'official_weight_sha256': sha_file(sources / 'avg_30.pt'), 'official_float_state_identity': identity,
                  'official_vocab_sha256': sha_file(sources / 'tokens.txt'), 'view': str(view), 'view_manifest_sha256': args.view_manifest_sha256,
                  'consumer_manifest_sha256': sha_file(view / 'fsmn-fulltext-train.jsonl'), 'data_sources': data_hashes,
                  'recipe_sha256': sha_file(output / 'recipe.json'), 'source_rows_sha256': sha_file(output / 'source-rows.jsonl'),
                  'feature_cache_sha256': sha_file(output / 'features.pt'), 'feature_observations_sha256': sha_file(output / 'feature-observations.json'),
                  'runtime': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'numpy': np.__version__,
                              'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()},
                  'items': 48, 'ctc_feasible_items': 48, 'input_pcm_samples': sum(row['samples'] for row in rows),
                  'model_feature_frames': sum(len(value) for value in tensors),
                  'source_families': dict(Counter(row['source_family_id'] for row in rows)),
                  'official_forward_probes': 1, 'probe_output_shape': list(probe.shape), 'optimizer_steps': 0,
                  'independent_development_items': 0, 'quality_winner': None}
        save_json(output / 'preparation.json', record)
    print(json.dumps({'status': 'complete', 'items': 48, 'feasible': 48, 'recipe_sha256': record['recipe_sha256'],
                      'source_revision': record['source_revision']}), flush=True)
    return 0


def verified_prepared(root):
    record = json_read(root / 'preparation.json')
    require(record['status'] == 'complete' and record['source_code'] == source_inventory(), 'fixed training code changed')
    verify_sources(Path(record['sources']))
    for filename, field in (('recipe.json', 'recipe_sha256'), ('source-rows.jsonl', 'source_rows_sha256'),
                            ('features.pt', 'feature_cache_sha256'), ('feature-observations.json', 'feature_observations_sha256')):
        require(sha_file(root / filename) == record[field], 'prepared input changed: ' + filename)
    rows = [json.loads(line) for line in (root / 'source-rows.jsonl').read_text().splitlines()]
    recipe = json_read(root / 'recipe.json')
    require(recipe == make_recipe(rows, vocabulary((Path(record['sources']) / 'tokens.txt').read_bytes())), 'fixed recipe changed')
    for row in rows:
        require(sha_file(row['audio_path']) == row['file_sha256'], 'bound training WAV changed')
    return record, rows, recipe


def worker(args):
    prepared, output = args.prepared.resolve(), args.output.resolve()
    record, rows, recipe = verified_prepared(prepared)
    with heavy_model_slot(args.heavy_lock):
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        status = {'schema': 'd0-official-fsmn-run-v1', 'stage': args.stage, 'status': 'running', 'started_at': utc(),
                  'optimizer_steps': 0, 'training_sample_exposures': 0, 'recipe_sha256': record['recipe_sha256'],
                  'source_revision': record['source_revision'], 'source_code': record['source_code'],
                  'official_weight_sha256': record['official_weight_sha256'], 'official_vocab_sha256': record['official_vocab_sha256'],
                  'runtime': record['runtime'], 'independent_development_items': 0, 'quality_winner': None, 'human_gold': False,
                  'training_set_readback_is_generalization': False, 'production_qualified': False}
        save_json(output / 'status.json', status)
        try:
            torch, _ = torch_runtime()
            research_runtime.check_packages(research_runtime.load_manifest(), 'core')
            require(sys.version.split()[0] == record['runtime']['python'] and
                    str(torch.__version__) == record['runtime']['torch'], 'worker runtime changed after preparation')
            from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
            random.seed(recipe['seed'])
            torch.manual_seed(recipe['seed'])
            model, _ = build_model(Path(record['sources']), torch)
            if args.stage == 'tail':
                payload, previous_sha = load_checkpoint_snapshot(args.previous_checkpoint)
                require(payload['stage'] == 'head' and payload['optimizer_steps'] == 12 and
                        payload['recipe_sha256'] == record['recipe_sha256'], 'tail must continue the fixed completed head stage')
                require(payload['source_code'] == record['source_code'] and payload['source_revision'] == record['source_revision'] and
                        payload['official_weight_sha256'] == record['official_weight_sha256'] and
                        payload['official_vocab_sha256'] == record['official_vocab_sha256'] and
                        payload['source_rows_sha256'] == record['source_rows_sha256'] and
                        payload['feature_cache_sha256'] == record['feature_cache_sha256'], 'head checkpoint provenance changed')
                model.load_state_dict(payload['state_dict'], strict=True)
                require(state_identity(model.state_dict()) == payload['float_state_identity'], 'head checkpoint changed on load')
                status['initial_checkpoint_sha256'] = previous_sha
            else:
                require(state_identity(model.state_dict()) == record['official_float_state_identity'], 'head did not start from official state')
                status['initial_checkpoint_sha256'] = record['official_weight_sha256']
            cached = torch.load(prepared / 'features.pt', weights_only=True, map_location='cpu')['features']
            observations = json_read(prepared / 'feature-observations.json')
            require(len(cached) == 48 and all(tensor_sha(value) == observation['feature_sha256'] for value, observation in zip(cached, observations)),
                    'cached feature bindings differ')
            selected, frozen = [], []
            for name, parameter in model.named_parameters():
                enabled = any(name.startswith(prefix) for prefix in recipe['stage_trainable_prefixes'][args.stage])
                parameter.requires_grad_(enabled)
                (selected if enabled else frozen).append(name)
            params = dict(model.named_parameters())
            groups = [{'params': [params[name] for name in selected if name.startswith(HEAD)], 'lr': recipe['optimizer']['head_lr']}]
            group_names = [[name for name in selected if name.startswith(HEAD)]]
            if args.stage == 'tail':
                group_names.append([name for name in selected if not name.startswith(HEAD)])
                groups.append({'params': [params[name] for name in group_names[-1]], 'lr': recipe['optimizer']['tail_lr']})
            optimizer = torch.optim.AdamW(groups, betas=tuple(recipe['optimizer']['betas']), eps=recipe['optimizer']['eps'],
                                          weight_decay=0.0, foreach=False)
            initial = state_identity(model.state_dict())
            frozen_names = set(model.state_dict()) - set(selected)
            frozen_before = {name: tensor_sha(model.state_dict()[name]) for name in frozen_names}
            status.update(initial_state_sha256=initial['sha256'], trainable_names=selected, frozen_names=sorted(frozen_names),
                          trainable_parameters=sum(params[name].numel() for name in selected),
                          all_parameters=sum(parameter.numel() for parameter in model.parameters()), optimizer=recipe['optimizer'])

            def save_checkpoint(label):
                identity = state_identity(model.state_dict())
                payload = {'schema': 'd0-official-fsmn-checkpoint-v1', 'stage': args.stage, 'state_dict': model.state_dict(),
                           'float_state_identity': identity, 'optimizer_state_dict': optimizer.state_dict(),
                           'optimizer_parameter_names': group_names, 'optimizer_steps': status['optimizer_steps'],
                           'training_sample_exposures': status['training_sample_exposures'], 'torch_rng_state': torch.get_rng_state(),
                           'python_rng_state': random.getstate(), 'recipe_sha256': record['recipe_sha256'],
                           'source_rows_sha256': record['source_rows_sha256'], 'feature_cache_sha256': record['feature_cache_sha256'],
                           'source_revision': record['source_revision'], 'source_code': record['source_code'],
                           'official_weight_sha256': record['official_weight_sha256'], 'official_vocab_sha256': record['official_vocab_sha256'],
                           'initial_checkpoint_sha256': status['initial_checkpoint_sha256'], 'trainable_names': selected,
                           'runtime': record['runtime'], 'quality_winner': None, 'human_gold': False, 'resume_supported': False}
                path = output / 'checkpoints' / (label + '.pt')
                atomic_save_checkpoint(payload, path)
                return {'path': str(path), 'sha256': sha_file(path), 'float_state_sha256': identity['sha256']}

            def loss_for(indices):
                terms, contributions = [], []
                for index in indices:
                    logits = model(cached[index].unsqueeze(0))
                    term, raw = fulltext_ctc(logits, rows[index]['target_ids'], recipe['items'][index]['effective_ctc_weight'],
                                            normalizer=recipe['fixed_ctc_normalizer'])
                    terms.append(term)
                    contributions.append({'row_index': index, 'raw_loss': raw.item(), 'weighted_contribution': term.detach().item() / len(indices)})
                loss = torch.stack(terms).mean()
                require(bool(torch.isfinite(loss)), 'nonfinite full-text loss; no automatic retry')
                return loss, contributions

            def readback(label):
                model.eval()
                by_class = defaultdict(float)
                with torch.no_grad():
                    for index in range(48):
                        _, contributions = loss_for([index])
                        by_class[recipe['items'][index]['class']] += contributions[0]['weighted_contribution'] / 48
                value = {'scope': 'same-training-set-loss-only', 'label': label, 'items': 48,
                         'loss': sum(by_class.values()), 'weighted_contribution_by_class': dict(by_class),
                         'generalization_metric': False, 'independent_development': False}
                append_json(output / 'readback.jsonl', value)
                model.train()
                return value

            status['initial_checkpoint'] = save_checkpoint('initial')
            status['initial_training_set_readback'] = readback('before-fixed-stage')
            order, exposures = recipe['stage_row_orders'][args.stage], Counter()
            for start in range(0, len(order), 4):
                require(time.monotonic() - started < args.seconds, 'stage budget exhausted')
                indices = order[start:start + 4]
                optimizer.zero_grad(set_to_none=True)
                loss, contributions = loss_for(indices)
                loss.backward()
                require(all(params[name].grad is not None and bool(torch.isfinite(params[name].grad).all()) for name in selected), 'missing/nonfinite selected gradient')
                require(all(params[name].grad is None for name in frozen), 'frozen parameter received gradient')
                norm = torch.nn.utils.clip_grad_norm_([params[name] for name in selected], recipe['gradient_clip_max_norm'], error_if_nonfinite=True, foreach=False)
                optimizer.step()
                require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), 'nonfinite updated state')
                status['optimizer_steps'] += 1
                status['training_sample_exposures'] += len(indices)
                draws = []
                for contribution in contributions:
                    index = contribution['row_index']
                    row, policy = rows[index], recipe['items'][index]
                    exposures[row['source_id']] += 1
                    draws.append({**contribution, **{name: row[name] for name in
                                 ('item_id', 'source_id', 'speaker_id', 'source_family_id', 'original_wav_sha256', 'pcm_sha256', 'target_ids', 'ctc_mask')},
                                  **{name: policy[name] for name in ('class', 'asr_ctc_weight', 'effective_ctc_weight', 'class_multiplier', 'label_binding_sha256')},
                                  'feature_sha256': observations[index]['feature_sha256']})
                append_json(output / 'consumption.jsonl', {'optimizer_step': status['optimizer_steps'], 'epoch': 1,
                            'loss': loss.item(), 'gradient_norm_before_clip': norm.item(), 'draws': draws})
                save_json(output / 'status.json', status)
            status['final_checkpoint'] = save_checkpoint('final')
            status['final_training_set_readback'] = readback('after-fixed-stage')
            require(all(tensor_sha(model.state_dict()[name]) == expected for name, expected in frozen_before.items()), 'frozen tensors changed')
            final = state_identity(model.state_dict())
            require(final['sha256'] != initial['sha256'], 'optimizer did not change model state')
            before_tensors = {value['name']: value['sha256'] for value in initial['tensors']}
            changed = [value['name'] for value in final['tensors'] if value['sha256'] != before_tensors[value['name']]]
            require(set(changed) <= set(selected), 'nonselected tensor changed')
            status.update(status='complete', finished_at=utc(), elapsed_seconds=round(time.monotonic() - started, 6),
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          final_state_sha256=final['sha256'], changed_tensor_names=changed, frozen_tensors_unchanged=True,
                          sample_exposures_by_source=dict(exposures), unique_training_sources=len(exposures),
                          training_feature_frame_exposures=sum(len(cached[index]) for index in order),
                          training_set_readback_forward_exposures=96)
            save_json(output / 'status.json', status)
            print(json.dumps({key: status[key] for key in ('stage', 'status', 'optimizer_steps', 'training_sample_exposures', 'elapsed_seconds', 'max_rss_kib')}), flush=True)
        except BaseException as error:
            status.update(status='failed', finished_at=utc(), error_type=type(error).__name__, error=str(error),
                          elapsed_seconds=round(time.monotonic() - started, 6))
            save_json(output / 'status.json', status)
            raise
    return 0


def run(args):
    record, _, _ = verified_prepared(args.prepared.resolve())
    with heavy_model_slot(args.heavy_lock) as lease:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        summary = {'schema': 'd0-official-fsmn-controller-v1', 'status': 'running', 'started_at': utc(), 'runs': {},
                   'recipe_sha256': record['recipe_sha256'], 'heavy_model_concurrency': 1,
                   'stage_seconds_cap': args.seconds, 'quality_retries': 0, 'quality_winner': None}
        save_json(output / 'summary.json', summary)
        for stage in STAGES:
            command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--stage', stage,
                       '--prepared', str(args.prepared.resolve()), '--output', str(output / stage),
                       '--heavy-lock', str(args.heavy_lock.resolve()), '--seconds', str(args.seconds)]
            if stage == 'tail':
                command += ['--previous-checkpoint', str(output / 'head/checkpoints/final.pt')]
            with (output / (stage + '.log')).open('x') as log:
                start = time.monotonic()
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=not lease.supervised,
                                           env=lease.child_environment(), pass_fds=lease.pass_fds)
                try:
                    code = process.wait(timeout=args.seconds)
                except subprocess.TimeoutExpired:
                    if lease.supervised:
                        # Stage workers execute only local Torch code and have
                        # no children. Keep them in the queue's process group
                        # so its outer deadline can terminate the entire job.
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = 124
                summary['runs'][stage] = {'exit_code': code, 'wall_seconds': round(time.monotonic() - start, 6)}
            if code != 0:
                status_path = output / stage / 'status.json'
                if status_path.exists():
                    status = json_read(status_path)
                    if status['status'] == 'running':
                        status.update(status='budget-exhausted' if code == 124 else 'failed', controller_exit_code=code, finished_at=utc())
                        save_json(status_path, status)
                break
            save_json(output / 'summary.json', summary)
            print(json.dumps({'stage': stage, **summary['runs'][stage]}), flush=True)
        summary.update(status='complete' if len(summary['runs']) == 2 and all(value['exit_code'] == 0 for value in summary['runs'].values()) else 'incomplete', finished_at=utc())
        save_json(output / 'summary.json', summary)
    return 0 if summary['status'] == 'complete' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fetcher = sub.add_parser('fetch')
    fetcher.add_argument('--sources', type=Path, required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--sources', type=Path, required=True)
    prep.add_argument('--view', type=Path, required=True)
    prep.add_argument('--view-manifest-sha256', required=True)
    prep.add_argument('--view-loader', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--heavy-lock', type=Path, required=True)
    for name in ('run', 'worker'):
        command = sub.add_parser(name)
        command.add_argument('--prepared', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--heavy-lock', type=Path, required=True)
        command.add_argument('--seconds', type=int, default=600)
        if name == 'worker':
            command.add_argument('--stage', choices=STAGES, required=True)
            command.add_argument('--previous-checkpoint', type=Path)
    args = parser.parse_args(argv)
    if args.command in ('run', 'worker') and not 1 <= args.seconds <= 600:
        parser.error('each fixed stage is capped at 600 seconds')
    if args.command != 'fetch':
        research_runtime.check_interpreter(research_runtime.load_manifest())
    return {'fetch': fetch, 'prepare': prepare, 'worker': worker, 'run': run}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
