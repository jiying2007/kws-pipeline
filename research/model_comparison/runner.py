"""Explicit research-only CPU training and continuous-WAV inference.

No imports of Torch or execution on import. Checkpoints are not .kwm exports.
Both architectures share this deliberately simple greedy CTC research decoder;
its events are not runtime/C-equivalent and cannot qualify a shipping model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
import wave

try:
    from .compare import validate_split, LINEAGE_FIELDS
except ImportError:
    from compare import validate_split, LINEAGE_FIELDS


SOURCE_PATHS = ('training/model.py', 'training/frontend.py', 'training/frontend_spec.py',
                'training/train_ctc.py', 'training/training_admission.py', 'tools/corpus_identity.py',
                'tools/source_identity.py', 'tools/source_lineage_admission.py',
                'research/candidate_encoders/contract.py', 'research/candidate_encoders/torch_models.py')
SOURCE_KEYS = frozenset((*SOURCE_PATHS, 'research_runner', 'comparison_contract'))
MAX_EPOCHS = 100
MAX_EXPOSURES = 100000
MAX_ROWS_PER_SPLIT = 10000
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_TRAIN_AUDIO_SECONDS = 30
MAX_SOURCE_AUDIO_SECONDS = 300
MAX_CHUNK_FRAMES = 256
MAX_STREAM_SECONDS = 24 * 3600
HASH_CHUNK_BYTES = 64 * 1024

def bounded_integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value

def source_hashes(repo):
    return {**{p: sha(repo / p) for p in SOURCE_PATHS},
            'research_runner': sha(__file__),
            'comparison_contract': sha(Path(__file__).with_name('compare.py'))}

def verify_source_hashes(value, repo):
    if not isinstance(value, dict) or set(value) != SOURCE_KEYS:
        raise ValueError('checkpoint source_hashes keys mismatch')
    for expected in value.values():
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('checkpoint source_hashes invalid digest')
    actual = source_hashes(repo)
    for path in SOURCE_KEYS:
        if value[path] != actual[path]:
            raise ValueError('checkpoint/source mismatch: ' + path)

def sample_order(size, epochs, seed):
    bounded_integer(size, 'training rows', 1, MAX_ROWS_PER_SPLIT)
    bounded_integer(epochs, 'epochs', 1, MAX_EPOCHS)
    if size * epochs > MAX_EXPOSURES:
        raise ValueError('sample exposure budget exceeds limit')
    rng = random.Random(seed)
    order = list(range(size))
    for _ in range(epochs):
        rng.shuffle(order)
        yield from order

def projected_identity(measured, raw, split):
    result = {key: measured[key] for key in ('speaker_id', 'session_id', 'source_id', 'pcm_sha256')}
    if 'file_sha256' in measured:
        result['file_sha256'] = measured['file_sha256']
    result['split'] = split
    for field in ('pcm_sha256', 'file_sha256'):
        if field in raw and raw[field] != measured.get(field):
            raise ValueError('declared audio identity differs from measured audio')
    links = raw.get('lineage_links', [])
    if not isinstance(links, list):
        raise ValueError('invalid raw lineage links')
    result['lineage_links'] = list(links)
    for record in (raw, raw.get('speech_like_provenance', {})):
        if not isinstance(record, dict):
            raise ValueError('invalid source provenance')
        result['lineage_links'].append({k: record[k] for k in LINEAGE_FIELDS if k in record})
    return result

def check_wav_payload(path):
    """Inspect RIFF chunk extents without loading PCM or using wave internals."""
    with Path(path).open('rb') as stream:
        file_end = stream.seek(0, 2)
        stream.seek(0)
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError('expected RIFF WAVE')
        limit = min(file_end, 8 + int.from_bytes(header[4:8], 'little'))
        position = 12
        while position + 8 <= limit:
            stream.seek(position)
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError('truncated WAV chunk header')
            size = int.from_bytes(chunk[4:], 'little')
            end = position + 8 + size
            if chunk[:4] == b'data':
                # wave.getnframes() rounds an odd PCM byte count down, so frame
                # coverage alone cannot validate this declared payload length.
                if size % 2:
                    raise ValueError('invalid PCM16 data size: incomplete sample')
                if end > limit:
                    raise ValueError('truncated PCM payload')
                return
            position = end + size % 2  # RIFF unknown chunks include word padding.
            if position > limit:
                raise ValueError('truncated WAV chunk payload')
        raise ValueError('missing or truncated WAV data chunk')


def check_wav_bound(path, seconds):
    with wave.open(str(path), 'rb') as wf:
        if (wf.getnchannels(), wf.getframerate(), wf.getsampwidth(), wf.getcomptype()) != (1,16000,2,'NONE'):
            raise ValueError('expected mono 16kHz PCM16')
        if wf.getnframes() > seconds * 16000:
            raise ValueError('audio duration exceeds bounded research limit')
    check_wav_payload(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class GreedyKeywordDecoder:
    """Collapsed CTC suffix matching with bounded age and per-word cooldown."""
    def __init__(self, keywords, max_frames=150, cooldown_frames=50):
        if not isinstance(keywords, list) or not 1 <= len(keywords) <= 16 or not 1 <= max_frames <= 1000 or not 0 <= cooldown_frames <= 1000:
            raise ValueError('invalid decoder settings')
        self.words = keywords
        for word in keywords:
            if not 1 <= len(word['tokens']) <= 16 or any(type(t) is not int or not 1 <= t <= 4 for t in word['tokens']):
                raise ValueError('keyword requires nonblank integer tokens')
            if not 0 < word['threshold'] <= 1:
                raise ValueError('invalid threshold')
        self.max_frames, self.cooldown = max_frames, cooldown_frames
        self.history, self.last_hit = [], {}
        self.previous, self.frame = 0, -1
        self.limit = max(len(w['tokens']) for w in keywords)

    def accept(self, token, probability):
        if type(token) is not int or token < 0 or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('invalid frame prediction')
        self.frame += 1
        self.history = [x for x in self.history if self.frame - x[2] < self.max_frames]
        events = []
        if token != 0 and token != self.previous:
            self.history.append((token, probability, self.frame))
            self.history = self.history[-self.limit:]
            for word in self.words:
                tail = self.history[-len(word['tokens']):]
                if [x[0] for x in tail] != word['tokens']:
                    continue
                score = min(x[1] for x in tail)
                wid = word['id']
                if score >= word['threshold'] and self.frame - self.last_hit.get(wid, -10**9) >= self.cooldown:
                    events.append({'keyword_id': wid, 'confidence': score,
                                   'time_s': (400 + self.frame * 320) / 16000})
                    self.last_hit[wid] = self.frame
        self.previous = token
        return events


def runtime(repo):
    # Import the product module first so its deterministic environment is applied
    # before any Torch import. It owns the one-time interop pool configuration.
    sys.path[:0] = [str(repo / 'training'), str(repo)]
    from model import TinyStreamingRNN
    import torch
    from train_ctc import Manifest
    from frontend import features
    from frontend_spec import FRONTEND_LOGMEL
    from research.candidate_encoders.contract import spec_from_dict
    from research.candidate_encoders.torch_models import build_encoder
    torch.use_deterministic_algorithms(True)
    return torch, TinyStreamingRNN, Manifest, features, FRONTEND_LOGMEL, spec_from_dict, build_encoder


def build(rt, spec):
    torch, RNN, _, _, _, parse, factory = rt
    parsed = parse(spec)
    if parsed.parameter_elements > 65536 or parsed.cache_elements > 16384 or parsed.weight_elements > 16384:
        raise ValueError('topology exceeds bounded CPU pilot limits')
    reference = RNN(32, 64, 5) if parsed.architecture == 'rnn64_control' else None
    return factory(parsed, rnn_model=reference)


def load_inventory(config, repo, rt, *, data_pair_details=False):
    torch, _, Manifest, _, frontend, _, _ = rt
    from kws_vocab import load_tokens
    from training_admission import verify_training_manifests
    from corpus_identity import canonical_audio_path, inspect_pcm16_wav
    tokens = load_tokens(Path(config['tokens']))
    if sorted(tokens.values()) != list(range(5)):
        raise ValueError('research runner requires exact five-token vocabulary')
    datasets, inventory, train_rows = {}, [], []
    for split in ('train', 'calibration', 'selection'):
        path = Path(config['manifests'][split]).resolve()
        if path.suffix != '.jsonl':
            raise ValueError('comparison requires JSONL identity-bearing manifests')
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError('manifest too large')
        raw_rows = []
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            row = data_pair_json(line) if data_pair_details else json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('manifest row must be an object')
            raw_rows.append(row)
            if data_pair_details and split == 'train':
                train_rows.append(dict(row))
            if data_pair_details:
                row = data_pair_parent(row)
                raw_rows[-1] = row
            if len(raw_rows) > MAX_ROWS_PER_SPLIT:
                raise ValueError('too many manifest rows')
            audio = Path(canonical_audio_path(row, str(path)))
            check_wav_bound(audio if audio.is_absolute() else path.parent/audio, MAX_TRAIN_AUDIO_SECONDS)
            if 'source_path' in row:
                source = Path(row['source_path'])
                source = source if source.is_absolute() else path.parent/source
                check_wav_bound(source, MAX_SOURCE_AUDIO_SECONDS)
                source_identity = inspect_pcm16_wav(source)
                source_pcm = source_identity['pcm_sha256']
                for field in ('source_pcm', 'source_pcm_sha256'):
                    if field in row and row[field] != source_pcm:
                        raise ValueError('source PCM differs from measured source')
                row['source_pcm_sha256'] = source_pcm
                if 'source_wav_sha256' in row and row['source_wav_sha256'] != source_identity['file_sha256']:
                    raise ValueError('source WAV differs from measured source')
                row['source_wav_sha256'] = source_identity['file_sha256']
            if 'split' in row and row['split'] != split:
                raise ValueError('manifest split differs from declared comparison split')
            if row.get('protected_evidence_used', False) is not False:
                raise ValueError('protected evidence cannot enter this research pilot')
        # Only training labels are consumed for optimizer updates. Calibration
        # and selection inventory is read for PCM/group leakage checks, not loss.
        if split == 'train':
            admission = verify_training_manifests([path], tokens,
                diagnostic=config['synthetic_diagnostic_only'])
        ds = Manifest([path], 32, 5, frontend)
        datasets[split] = ds
        if len(ds.identity_rows) != len(raw_rows):
            raise ValueError('raw/projected manifest row count mismatch')
        for measured, raw in zip(ds.identity_rows, raw_rows):
            inventory.append(projected_identity(measured, raw, split))
    split_hash = validate_split(inventory)
    if data_pair_details:
        return datasets['train'], split_hash, admission, {'inventory': inventory, 'train_rows': train_rows}
    return datasets['train'], split_hash, admission



def fixed_step_order(size, steps, seed):
    """Same optimizer-step count, not equal epochs or acoustic exposure."""
    bounded_integer(size, 'training rows', 1, MAX_ROWS_PER_SPLIT)
    bounded_integer(steps, 'optimizer steps', 1, MAX_EXPOSURES)
    bounded_integer(seed, 'seed', 0, 2**32 - 1)
    rng, order, emitted = random.Random(seed), list(range(size)), 0
    while emitted < steps:
        rng.shuffle(order)
        for index in order:
            if emitted == steps:
                return
            yield index
            emitted += 1


def canonical_row(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def data_pair_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('DATA_PAIR_DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('DATA_PAIR_NONFINITE_JSON')
        return result
    def constant(value):
        raise ValueError('DATA_PAIR_NONFINITE_JSON')
    return json.loads(text, object_pairs_hook=pairs, parse_float=finite, parse_constant=constant)


def data_pair_config(config):
    """Explicit additive-data experiment; no architecture or eval-set factor."""
    required = {'schema', 'development_only', 'protected_evidence_used',
                'synthetic_diagnostic_only', 'tokens', 'train_manifests',
                'evaluation_manifests', 'spec', 'optimizer_steps', 'seed',
                'learning_rate', 'max_training_seconds'}
    if type(config) is not dict or set(config) != required or config['schema'] != 'research-data-pair-v1':
        raise ValueError('DATA_PAIR_CONFIG')
    if (config['development_only'] is not True or config['protected_evidence_used'] is not False
            or type(config['synthetic_diagnostic_only']) is not bool):
        raise ValueError('DATA_PAIR_RESEARCH_SCOPE')
    bounded_integer(config['optimizer_steps'], 'optimizer steps', 1, MAX_EXPOSURES)
    bounded_integer(config['seed'], 'seed', 0, 2**32 - 1)
    bounded_integer(config['max_training_seconds'], 'training seconds', 1, 3600)
    lr = config['learning_rate']
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('DATA_PAIR_LEARNING_RATE')
    if type(config['spec']) is not dict or config['spec'].get('architecture') != 'rnn64_control':
        raise ValueError('DATA_PAIR_SINGLE_RNN64_SPEC')
    train_paths, evaluation = config['train_manifests'], config['evaluation_manifests']
    if (type(train_paths) is not dict or set(train_paths) != {'control', 'treatment'}
            or type(evaluation) is not dict or set(evaluation) != {'calibration', 'selection'}):
        raise ValueError('DATA_PAIR_MANIFEST_KEYS')
    paths = [config['tokens'], *train_paths.values(), *evaluation.values()]
    if any(type(p) is not str or not p for p in paths):
        raise ValueError('DATA_PAIR_PATH')
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('DATA_PAIR_INPUT_ALIAS')
    if resolved[1].parent != resolved[2].parent:
        raise ValueError('DATA_PAIR_COMMON_TRAIN_DIRECTORY')
    return {arm: {**config, 'manifests': {'train': train_paths[arm], **evaluation}}
            for arm in ('control', 'treatment')}


def data_pair_parent(row):
    """A declared parent file digest is not authenticated ancestry."""
    result = dict(row)
    value = row.get('parent_source_sha256')
    kind = row.get('parent_source_identity_kind')
    if 'parent_source_sha256' not in row and 'parent_source_identity_kind' not in row:
        return result
    if (kind != 'file-sha256' or type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('DATA_PAIR_PARENT_IDENTITY_KIND')
    links = list(result.get('lineage_links', []))
    links.append({'source_audio_sha256': value})
    result['lineage_links'] = links
    return result


def validate_data_pair_inventories(details):
    from collections import Counter
    a, b = (details[arm] for arm in ('control', 'treatment'))
    control = Counter(canonical_row(row) for row in a['train_rows'])
    treatment = Counter(canonical_row(row) for row in b['train_rows'])
    if not control or any(treatment[row] < n for row, n in control.items()):
        raise ValueError('DATA_PAIR_CONTROL_ROWS_CHANGED')
    additions = treatment - control
    if not additions or any(row in control or n != 1 for row, n in additions.items()):
        raise ValueError('DATA_PAIR_NO_UNIQUE_ADDITION')
    # Both arms are validated separately by load_inventory, then together.
    # This catches ancestor components bridged only through the other arm.
    rows = a['inventory'] + b['inventory']
    for detail in (a, b):
        pcm = [row['pcm_sha256'] for row in detail['inventory'] if row['split'] == 'train']
        if len(set(pcm)) != len(pcm):
            raise ValueError('DATA_PAIR_DUPLICATE_TRAIN_PCM')
    return validate_split(rows)


def initial_state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        metadata = canonical_row({'key': key, 'dtype': str(tensor.dtype),
                                  'shape': list(tensor.shape)}).encode()
        raw = tensor.numpy().tobytes()
        for block in (metadata, raw):
            digest.update(len(block).to_bytes(8, 'big'))
            digest.update(block)
    return digest.hexdigest()


def check_frozen_inputs(expected):
    for path, value in expected.items():
        if sha(path) != value:
            raise ValueError('DATA_PAIR_INPUT_DRIFT: ' + str(path))


def train_data(args):
    """Two admitted datasets, one model recipe, equal bounded optimizer steps.

    This does not run evaluation, select a winner, or authenticate source rights.
    """
    config_path = args.config.resolve()
    config_digest = sha(config_path)
    if config_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError('DATA_PAIR_CONFIG_SIZE')
    config = data_pair_json(config_path.read_text())
    arm_configs = data_pair_config(config)
    frozen = {str(config_path): config_digest}
    for path in (config['tokens'], *config['train_manifests'].values(),
                 *config['evaluation_manifests'].values()):
        frozen[str(Path(path).resolve())] = sha(path)
    check_frozen_inputs(frozen)
    rt = runtime(args.repo.resolve())
    torch = rt[0]
    datasets, splits, admissions, details = {}, {}, {}, {}
    # Both admissions must succeed before constructing either model/optimizer.
    for arm in ('control', 'treatment'):
        datasets[arm], splits[arm], admissions[arm], details[arm] = load_inventory(
            arm_configs[arm], args.repo, rt, data_pair_details=True)
    pair_split = validate_data_pair_inventories(details)
    if config['optimizer_steps'] < len(datasets['treatment']):
        raise ValueError('DATA_PAIR_STEPS_MUST_COVER_TREATMENT')
    check_frozen_inputs(frozen)
    identities = source_hashes(args.repo)
    args.output.mkdir(parents=True, exist_ok=False)
    results, init_digest = [], None
    steps = config['optimizer_steps']
    for arm in ('control', 'treatment'):
        check_frozen_inputs(frozen)
        torch.manual_seed(config['seed'])
        model = build(rt, config['spec'])
        current_init = initial_state_digest(model)
        if init_digest is not None and current_init != init_digest:
            raise ValueError('DATA_PAIR_INITIAL_STATE_DRIFT')
        init_digest = current_init
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        schedule = list(fixed_step_order(len(datasets[arm]), steps, config['seed']))
        schedule_sha = hashlib.sha256(canonical_row(schedule).encode()).hexdigest()
        started, losses, frames, exposures = time.monotonic(), [], 0, 0
        for index in schedule:
            if time.monotonic() - started >= config['max_training_seconds']:
                break
            x, y = datasets[arm][index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.unsqueeze(0))
            loss = criterion(logits.log_softmax(2), y, torch.tensor([x.shape[0]]),
                             torch.tensor([y.numel()]))
            if not bool(torch.isfinite(loss)):
                raise ValueError('DATA_PAIR_NONFINITE_LOSS')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            frames += int(x.shape[0])
            exposures += 1
        check_frozen_inputs(frozen)
        # Re-read original WAV bytes, source lineage and both admissions.
        refreshed = {}
        for name in ('control', 'treatment'):
            _, split, _, detail = load_inventory(arm_configs[name], args.repo, rt,
                                                 data_pair_details=True)
            if split != splits[name]:
                raise ValueError('DATA_PAIR_AUDIO_DRIFT')
            refreshed[name] = detail
        if validate_data_pair_inventories(refreshed) != pair_split or source_hashes(args.repo) != identities:
            raise ValueError('DATA_PAIR_SOURCE_DRIFT')
        elapsed = time.monotonic() - started
        status = 'TRAINED' if exposures == steps and elapsed <= config['max_training_seconds'] else 'INCOMPLETE_TIME_CAP'
        checkpoint = {'schema': 'research-paired-ctc-v1', 'product_export_compatible': False,
                      'frontend': 'logmel', 'spec': config['spec'], 'state_dict': model.state_dict(),
                      'synthetic_diagnostic_only': config['synthetic_diagnostic_only'],
                      'split_sha256': splits[arm], 'source_hashes': identities,
                      'input_hashes': frozen, 'data_pair_split_sha256': pair_split}
        target = args.output / (arm + '.research.pt')
        torch.save(checkpoint, target)
        results.append({'arm': arm, 'status': status, 'optimizer_steps': exposures,
                        'input_frames': frames, 'schedule_sha256': schedule_sha,
                        'initial_state_sha256': current_init, 'split_sha256': splits[arm],
                        'admission': admissions[arm], 'checkpoint_sha256': sha(target),
                        'training_seconds': elapsed,
                        'mean_training_loss': sum(losses) / len(losses) if losses else None})
        del model, optimizer
        if status != 'TRAINED':
            break  # Never present an unequal/incomplete pair as successful.
    check_frozen_inputs(frozen)
    complete = len(results) == 2 and all(r['status'] == 'TRAINED' for r in results)
    receipt = {'schema': 'research-data-pair-training-v1',
               'torch_version': torch.__version__,
               'status': 'TRAINED_PAIR' if complete else 'INCOMPLETE_PAIR',
               'config_sha256': config_digest, 'input_hashes': frozen,
               'pair_split_sha256': pair_split, 'source_hashes': identities,
               'arms': results, 'shipping_approved': False, 'effect_measured': False,
               'limitations': ['Equal optimizer steps, not equal audio frames or wall time.',
                              'Declared lineage and admitted labels do not authenticate rights.',
                              'No evaluation, winner selection or production qualification.']}
    with (args.output / 'data-pair-training-receipt.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    if not complete:
        raise ValueError('DATA_PAIR_INCOMPLETE')
    return receipt


def train(args):
    config = json.loads(args.config.read_text())
    if config.get('development_only') is not True or config.get('protected_evidence_used') is not False or type(config.get('synthetic_diagnostic_only')) is not bool:
        raise ValueError('explicit development-only and diagnostic policy required')
    bounded_integer(config['epochs'], 'epochs', 1, MAX_EPOCHS)
    bounded_integer(config['seed'], 'seed', 0, 2**32-1)
    bounded_integer(config['max_training_seconds'], 'training seconds', 1, 3600)
    lr = config['learning_rate']
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('invalid learning rate')
    # Never overwrite an existing model/result or product config.
    args.output.mkdir(parents=True, exist_ok=False)
    rt = runtime(args.repo.resolve())
    torch = rt[0]
    ds, split_hash, admission = load_inventory(config, args.repo, rt)
    if config['specs']['control']['architecture'] != 'rnn64_control':
        raise ValueError('control must be the existing RNN64')
    input_hashes = {str(Path(p).resolve()): sha(p) for p in [config['tokens'], *config['manifests'].values()]}
    exposures_expected = len(ds) * config['epochs']
    schedule_digest = hashlib.sha256()
    for index in sample_order(len(ds), config['epochs'], config['seed']):
        schedule_digest.update(index.to_bytes(4, 'little'))
    identities = source_hashes(args.repo)
    results = []
    # Same sample schedule, seed, optimizer, unpadded batch size 1 and CTC loss.
    # Each arm independently has the same time cap; incomplete arms cannot win.
    for arm in ('control', 'treatment'):
        torch.manual_seed(config['seed'])
        model = build(rt, config['specs'][arm])
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        start, losses, exposures = time.monotonic(), [], 0
        for index in sample_order(len(ds), config['epochs'], config['seed']):
            if time.monotonic() - start >= config['max_training_seconds']:
                break
            x, y = ds[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.unsqueeze(0))
            loss = criterion(logits.log_softmax(2), y, torch.tensor([x.shape[0]]), torch.tensor([y.numel()]))
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite loss; no successful receipt emitted')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            exposures += 1
            losses.append(float(loss.detach()))
        for p, expected in input_hashes.items():
            if sha(p) != expected:
                raise ValueError('input manifest or tokens changed during training')
        # Reopen the source audio and recheck group/PCM identities before saving.
        _, final_split_hash, _ = load_inventory(config, args.repo, rt)
        if final_split_hash != split_hash:
            raise ValueError('audio identity changed during training')
        checkpoint = {'schema': 'research-paired-ctc-v1', 'product_export_compatible': False, 'frontend': 'logmel',
                      'spec': config['specs'][arm], 'state_dict': model.state_dict(),
                      'synthetic_diagnostic_only': config['synthetic_diagnostic_only'],
                      'split_sha256': split_hash, 'source_hashes': identities, 'input_hashes': input_hashes}
        path = args.output / (arm + '.research.pt')
        torch.save(checkpoint, path)
        results.append({'arm': arm, 'status': 'TRAINED' if exposures == exposures_expected else 'INCOMPLETE_TIME_CAP',
                        'sample_exposures': exposures, 'training_seconds': time.monotonic() - start,
                        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        'mean_training_loss': sum(losses) / len(losses) if losses else None,
                        'checkpoint_sha256': sha(path)})
        del model, optimizer
    receipt = {'schema': 'research-paired-training-v1', 'shipping_approved': False,
               'config_sha256': sha(args.config), 'split_sha256': split_hash, 'input_hashes': input_hashes,
               'schedule_sha256': schedule_digest.hexdigest(),
               'source_hashes': identities, 'torch_version': str(torch.__version__),
               'admission': admission, 'arms': results,
               'limitations': ['Process RSS is cumulative high-water mark, not per-arm peak.',
                              'CTC-only research objective; not the product auxiliary-loss recipe.',
                              'Neither checkpoint can enter the product v2 exporter.']}
    (args.output / 'training-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    if any(r['status'] != 'TRAINED' for r in results):
        raise SystemExit('Incomplete paired run; results cannot be compared.')


def iter_wav_features(rt, wf, chunk_frames):
    """Complete logmel windows only, bounded PCM overlap; never EOF padding."""
    torch, _, _, features, frontend, _, _ = rt
    bounded_integer(chunk_frames, 'chunk frames', 1, MAX_CHUNK_FRAMES)
    pending = bytearray()
    expected_bytes = wf.getnframes() * 2  # Caller requires mono PCM16.
    received_bytes = 0
    while True:
        raw = wf.readframes(chunk_frames * 320)
        if not raw:
            if received_bytes != expected_bytes:
                raise ValueError('truncated PCM payload')
            return
        received_bytes += len(raw)
        if len(raw) % 2:
            raise ValueError('truncated PCM payload: incomplete PCM16 sample')
        pending.extend(raw)
        if len(pending) < 800:
            continue
        pcm = torch.frombuffer(bytearray(pending), dtype=torch.int16).float() / 32768
        x = features(pcm, 32, frontend=frontend)
        yield x
        del pending[:x.shape[0] * 320 * 2]


def infer(args):
    bounded_integer(args.chunk_frames, 'chunk frames', 1, MAX_CHUNK_FRAMES)
    check_wav_bound(args.wav, MAX_STREAM_SECONDS)
    if args.output.exists() or Path(str(args.output) + '.receipt.json').exists():
        raise ValueError('inference outputs already exist')
    rt = runtime(args.repo.resolve())
    torch, _, _, features, frontend, _, _ = rt
    if args.checkpoint.stat().st_size > 64 * 1024 * 1024 or args.keywords.stat().st_size > 65536:
        raise ValueError('checkpoint or keyword metadata exceeds size bound')
    # weights_only prevents arbitrary code execution in a pickle checkpoint.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('schema') != 'research-paired-ctc-v1' or ckpt.get('frontend') != 'logmel':
        raise ValueError('not a research checkpoint')
    verify_source_hashes(ckpt.get('source_hashes'), args.repo)
    model = build(rt, ckpt['spec'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    decoder = GreedyKeywordDecoder(json.loads(args.keywords.read_text()))
    state = model.init_state(1, device='cpu', dtype=torch.float32)
    # Exclusive output creation: cannot overwrite a previous evaluation.
    with wave.open(str(args.wav), 'rb') as wf, args.output.open('x') as out, torch.no_grad():
        if (wf.getnchannels(), wf.getframerate(), wf.getsampwidth(), wf.getcomptype()) != (1,16000,2,'NONE'):
            raise ValueError('expected mono 16kHz PCM16')
        for x in iter_wav_features(rt, wf, args.chunk_frames):
            logits, state = model.forward_chunk(x.unsqueeze(0), state)
            probs = logits[:, 0].softmax(1)
            values, ids = probs.max(1)
            for token, probability in zip(ids.tolist(), values.tolist()):
                for event in decoder.accept(token, probability):
                    out.write(json.dumps({'recording': args.recording, **event}) + '\n')
        # Partial final windows are intentionally not padded or flushed.
    receipt = {'schema': 'research-continuous-inference-v1', 'checkpoint_sha256': sha(args.checkpoint),
               'wav_sha256': sha(args.wav), 'keywords_sha256': sha(args.keywords),
               'detections_sha256': sha(args.output), 'frames': decoder.frame + 1,
               'decoder': 'greedy-ctc-research-only-v1', 'shipping_approved': False,
               'synthetic_diagnostic_only': ckpt['synthetic_diagnostic_only'],
               'c_runtime_equivalent': False}
    with Path(str(args.output) + '.receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config', type=Path, required=True)
    t.add_argument('--output', type=Path, required=True)
    d = sub.add_parser('train-data')
    d.add_argument('--config', type=Path, required=True)
    d.add_argument('--output', type=Path, required=True)
    i = sub.add_parser('infer')
    for name in ('checkpoint', 'wav', 'keywords', 'output'):
        i.add_argument('--' + name, type=Path, required=True)
    i.add_argument('--recording', required=True)
    i.add_argument('--chunk-frames', type=int, default=25)
    args = p.parse_args()
    {'train': train, 'train-data': train_data, 'infer': infer}[args.command](args)

if __name__ == '__main__':
    main()
