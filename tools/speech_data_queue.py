#!/usr/bin/env python3
"""Offline maintained speech queue. Never imports runtimes, invokes or admits models.

Inputs are supplied evidence, not authenticated execution/rights receipts. Outputs
are research-only machine candidates and proposed work, never product admission
or human gold. The separate research weak loader can consume screened examples.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import wave
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research/qwen6_asr/core'))
from asr_stage.decoding import qwen_transcript
from sense_binding import _contract, CONTRACT_SHA256, TOKENIZER_SHA256

from corpus_identity import canonical_hash, inspect_pcm16_wav, sha256_file
from speech_label_admission import PUNCTUATION, actual_tokens
from kws_vocab import load_tokens

KINDS = {'real_speech', 'environment', 'tts', 'clone', 'augmentation'}
OUTCOMES = {'success', 'content_failure', 'infra_failure'}
CALLS = {'not_called', 'called', 'unknown'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    require(isinstance(value, str) and bool(value.strip()), 'nonempty text required')
    return value


def normalize(value):
    require(isinstance(value, str), 'transcript must be text')
    return ''.join(c for c in value if c not in PUNCTUATION)


def sha(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'SHA256 required')
    return value


def confined(root, relative):
    path = Path(text(relative))
    require(not path.is_absolute() and '..' not in path.parts, 'relative contained path required')
    resolved = (root / path).resolve(strict=True)
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(), 'file escapes input root')
    return resolved


def validate(plan):
    require(plan.get('schema') == 'speech-data-queue-v1', 'wrong plan schema')
    require(type(plan.get('call_budget')) is int and plan['call_budget'] >= 0, 'invalid budget')
    recognizers = plan['recognizers']
    require(isinstance(recognizers, list) and len(recognizers) in (2, 3), 'two ASRs plus optional third')
    for field in ('id', 'family', 'model_sha256'):
        values = [r[field] for r in recognizers]
        require(len(set(values)) == len(values), 'independent recognizer identities required')
        for value in values:
            sha(value) if field == 'model_sha256' else text(value)
    for r in recognizers:
        text(r['model_id']); text(r['revision'])
        sha(r['config_sha256'])
        require(r.get('blind') is True, 'blind ASR required')
    require(isinstance(plan['keywords'], list) and plan['keywords'], 'keywords required')
    for keyword in plan['keywords']:
        require(normalize(text(keyword)) == keyword, 'keywords must already be normalized')
    text(plan['tokens_path']); sha(plan['tokens_sha256'])
    conditioned = any(r.get('protocol') == 'source-language-conditioned-asr-v1' for r in recognizers)
    if conditioned:
        require(all(r.get('protocol') == 'source-language-conditioned-asr-v1' for r in recognizers),
                'cannot silently mix auto and language-conditioned votes')
    ids = set()
    require(isinstance(plan['items'], list) and bool(plan['items']), 'items required')
    for item in plan['items']:
        item_id = text(item['id'])
        require(item_id not in ids, 'duplicate item')
        ids.add(item_id)
        require(item['source_kind'] in KINDS, 'unknown source kind')
        require(item['intent_kind'] in {'positive', 'negative', 'confusable', 'background'}, 'invalid intent')
        require(isinstance(item['intended_text'], str), 'intended text required, never actual')
        # Explicit original roots, not newly minted clone/augmentation slot names.
        require(isinstance(item['lineage_roots'], list) and bool(item['lineage_roots']), 'original lineage required')
        for root in item['lineage_roots']:
            text(root)
        require(item['split'] in {'train', 'validation', 'test'}, 'invalid split')
        sha(item['recipe_sha256'])  # freezes seed, prompt, reference and transform recipe externally
        require(isinstance(item['resources'], list), 'resource inventory required')
        for resource in item['resources']:
            text(resource['path']); sha(resource['sha256'])
        require(isinstance(item.get('rights'), dict), 'rights metadata required')
        require(item['rights'].get('research_use') in {'allowed', 'unknown', 'prohibited'}, 'rights decision required')
        text(item['rights']['path']); sha(item['rights']['sha256'])
        if conditioned and item['source_kind'] in {'tts', 'clone'}:
            require(item['rights'].get('review_schema') == 'internal-noncommercial-research-source-review-v1',
                    'conditioned generated source requires bounded source-rights evidence')
    return canonical_hash(plan)


def lineage_groups(items):
    """Connected components: any shared original speaker/reference root joins groups."""
    parent = {i['id']: i['id'] for i in items}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    roots = {}
    for item in items:
        for root in item['lineage_roots']:
            if root in roots:
                parent[find(item['id'])] = find(roots[root])
            roots[root] = item['id']
    groups = collections.defaultdict(list)
    for item in items:
        groups[find(item['id'])].append(item)
    result = {}
    for group in groups.values():
        identity = canonical_hash(sorted({r for i in group for r in i['lineage_roots']}))
        conflict = len({i['split'] for i in group}) != 1
        for item in group:
            result[item['id']] = (identity, conflict)
    return result


def validate_events(plan, events, plan_hash):
    histories = collections.defaultdict(list)
    known = {i['id'] for i in plan['items']}
    operations = {'generate', *(r['id'] for r in plan['recognizers'])}
    require(isinstance(events, list), 'events must be an ordered list')
    for event in events:
        require(event['plan_sha256'] == plan_hash, 'event belongs to another frozen plan')
        require(event['item_id'] in known and event['operation'] in operations, 'unknown event slot')
        key = (event['item_id'], event['operation'])
        previous = histories[key]
        require(type(event['attempt']) is int and event['attempt'] == len(previous) + 1, 'attempt sequence mismatch')
        if previous:
            last = previous[-1]
            require(last['call_state'] == 'not_called' and last['outcome'] == 'infra_failure',
                    'cannot retry completed, failed-content, called, or unknown invocation')
        require(event['call_state'] in CALLS and event['outcome'] in OUTCOMES, 'invalid attempt state')
        require(event['call_state'] == 'called' or event['outcome'] == 'infra_failure',
                'success/content require known invocation')
        # Retain raw outcomes separately, including failed_unbound output. Hash
        # references are validated during package capture, before interpretation.
        require(isinstance(event['raw_artifacts'], list) and bool(event['raw_artifacts']), 'raw evidence required')
        for artifact in event['raw_artifacts']:
            text(artifact['path']); sha(artifact['sha256'])
        if event['operation'] != 'generate' and event['outcome'] == 'success':
            require(event.get('binding') == 'verified', 'unbound output cannot vote')
            sha(event['pcm_sha256'])
            require(event.get('blind') is True, 'target-conditioned output cannot vote')
        histories[key].append(event)
    return histories


def generation_binding(item, history, audio, root):
    require(bool(history) and history[-1]['outcome'] == 'success', 'generation success evidence missing')
    event = history[-1]
    refs = {a['path']: a['sha256'] for a in event['raw_artifacts']}
    name = event['generation_receipt_path']
    require(name in refs, 'generation receipt not retained')
    data = confined(root, name).read_bytes()
    require(hashlib.sha256(data).hexdigest() == refs[name], 'generation receipt changed')
    receipt = json.loads(data)
    require(receipt['schema'] == 'sohee-two-generation-receipt-v1'
            and receipt['status'] == 'one_candidate_generated', 'unsupported generation receipt')
    require(len(receipt['generation_rows']) == 1, 'generation result association ambiguous')
    result = receipt['generation_rows'][0]
    require(result['source_id'] == item['id'] and result['status'] == 'generated_candidate', 'generation source identity mismatch')
    require(result['audio_sha256'] == audio['file_sha256'] and result['pcm_sha256'] == audio['pcm_sha256'], 'generation audio binding mismatch')
    require(result['termination']['ended_with_eos'] is True
            and result['termination']['hit_token_cap_without_eos'] is False
            and result['signal_flags'] == [], 'incomplete or unhealthy generated audio')


def native_vote(event, recognizer, audio, root):
    """Read saved native output; event-authored text never becomes a label."""
    if event.get('protocol') == 'source-language-conditioned-asr-v1':
        from language_conditioned_vote import native_vote as conditioned_vote
        return conditioned_vote(event, recognizer, audio, root)
    require(recognizer.get('protocol', 'anonymous-two-asr-raw-v1') == 'anonymous-two-asr-raw-v1',
            'recognizer/event protocol mismatch')
    refs = {a['path']: a['sha256'] for a in event['raw_artifacts']}
    def read(field):
        path = event[field]
        require(path in refs, 'native evidence must be captured')
        data = confined(root, path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == refs[path], 'native evidence hash mismatch')
        return json.loads(data)
    raw, declaration = read('raw_result_path'), read('declaration_path')
    require(raw['schema'] == 'anonymous-two-asr-raw-v1' and raw['status'] == 'success', 'native result not successful')
    require(raw['model_id'] == recognizer['model_id'] and raw['revision'] == recognizer['revision'], 'native identity mismatch')
    known = {'Qwen/Qwen3-ASR-0.6B': 'qwen3-asr', 'FunAudioLLM/SenseVoiceSmall': 'sensevoice'}
    require(known.get(raw['model_id']) == recognizer['family'], 'unsupported or relabeled native family')
    require(declaration['scope'] == 'single_decode_each_anonymous_audio_no_context'
            and declaration['model_id'] == raw['model_id'], 'blind declaration mismatch')
    require(declaration['asset_lock_sha256'] == recognizer['model_sha256'], 'native asset lock mismatch')
    require(canonical_hash(declaration) == recognizer['config_sha256'], 'native declaration changed')
    binding = raw['evidence']['input_binding']
    require(binding['raw_pcm16']['sha256'] == audio['pcm_sha256']
            and binding['wav']['sha256'] == audio['file_sha256'], 'native audio binding mismatch')
    require(raw['outcome']['completeness'] == 'complete' and raw['outcome']['quality_flags'] == [], 'native incomplete/quality flagged')
    if recognizer['family'] == 'qwen3-asr':
        require(raw['evidence']['prompt_context'] == '' and raw['evidence']['transcript_repair_applied'] is False,
                'conditioned/repaired output')
        extracted, _ = qwen_transcript(raw['evidence']['decoded_skip_special_tokens'], raw['evidence']['token_ids'])
        require(extracted == raw['outcome'], 'Qwen outcome differs from retained native decode')
    else:
        token = raw['evidence']['token_binding']
        require(token['raw_redecode_matches_capture'] is True and token['metadata_boundary_verified'] is True
                and token['lexical_unknown_present'] is False, 'Sense token binding failed')
        require(token['parser_contract_sha256'] == CONTRACT_SHA256
                and token['tokenizer']['sha256'] == TOKENIZER_SHA256, 'Sense parser/tokenizer changed')
        decoded = raw['evidence']['token_evidence']['decoded']
        require(hashlib.sha256(decoded.encode()).hexdigest() == token['raw_decoded_sha256'], 'Sense native text changed')
        require(raw['evidence']['token_evidence']['token_ids'] == token['raw_token_ids']
                and token['raw_token_ids'] == token['metadata_token_ids'] + token['lexical_token_ids'], 'Sense token boundary mismatch')
        extracted = _contract()['corrected_sense_extract'](decoded, token['lexical_token_ids'])
        require(all(extracted[k] == raw['outcome'][k] for k in ('raw_text', 'quality_flags', 'completeness')),
                'Sense outcome differs from retained native decode')
    return normalize(raw['outcome']['raw_text'])


def evaluate(plan, events, root):
    plan_hash = validate(plan)
    histories = validate_events(plan, events, plan_hash)
    groups = lineage_groups(plan['items'])
    from speech_history_evidence import load_history, verify_research_rights
    historical, history_error = [], None
    try:
        historical = load_history(plan, root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        history_error = str(exc)
    token_path = confined(root, plan['tokens_path'])
    require(hashlib.sha256(token_path.read_bytes()).hexdigest() == plan['tokens_sha256'], 'token vocabulary changed')
    token_map = load_tokens(token_path)
    consumed = sum(e['call_state'] in {'called', 'unknown'} for e in events)
    remaining = max(0, plan['call_budget'] - consumed)
    tasks, rows = [], []
    def propose(item, operation, audio=None):
        nonlocal remaining
        history = histories[(item['id'], operation)]
        if history and not (history[-1]['call_state'] == 'not_called' and history[-1]['outcome'] == 'infra_failure'):
            return
        if remaining <= 0:
            return
        task = {'item_id': item['id'], 'operation': operation, 'attempt': len(history) + 1,
                'plan_sha256': plan_hash, 'recipe_sha256': item['recipe_sha256'],
                'execution_authorized': False}
        if audio:
            task.update(audio_path=item['audio_path'], pcm_sha256=audio['pcm_sha256'])
        # ASR task contains no intended text, label, keyword, speaker or reference.
        if operation != 'generate':
            r = next(r for r in plan['recognizers'] if r['id'] == operation)
            task.update(model_sha256=r['model_sha256'], config_sha256=r['config_sha256'])
            task.pop('recipe_sha256')
            if r.get('protocol') == 'source-language-conditioned-asr-v1':
                task.update(protocol=r['protocol'],
                    source_language_condition=r['protocol_config']['source_language_condition'],
                    blindness_scope='target-text-blind-source-language-conditioned')
        tasks.append(task)
        remaining -= 1
    for item in plan['items']:
        group, conflict = groups[item['id']]
        row = {'item_id': item['id'], 'lineage_group': group, 'status': 'quarantine',
               'machine_text': None, 'human_gold': False,
               'intended_text': item['intended_text'], 'intent_kind': item['intent_kind'], 'product_training_admitted': False,
               'research_weak_allowed': False,
               'attempts': [{k: e[k] for k in ('operation', 'attempt', 'call_state', 'outcome')}
                            for e in events if e['item_id'] == item['id']]}
        rows.append(row)
        row['historical_attempts'] = [{k: e[k] for k in ('operation', 'attempt', 'call_state', 'outcome')}
                                    for e in historical if e['item_id'] == item['id']]
        if history_error:
            row['reason'] = 'historical_evidence_incomplete'; row['detail'] = history_error; continue
        if conflict:
            row['reason'] = 'lineage_split_conflict'; continue
        try:
            for event in events:
                if event['item_id'] != item['id']:
                    continue
                for ref in event['raw_artifacts']:
                    require(sha256_file(confined(root, ref['path'])) == ref['sha256'], 'raw artifact changed')
        except (ValueError, OSError) as exc:
            row['reason'] = 'evidence_capture_incomplete'; row['detail'] = str(exc); continue
        try:
            for resource in item['resources']:
                require(sha256_file(confined(root, resource['path'])) == resource['sha256'],
                        'resource hash mismatch')
            rights = item['rights']
            require(rights['research_use'] == 'allowed', 'research rights unavailable')
            require(hashlib.sha256(confined(root, rights['path']).read_bytes()).hexdigest() == rights['sha256'], 'rights evidence mismatch')
        except (ValueError, OSError) as exc:
            row['preflight_error'] = str(exc)
        if not item.get('audio_path'):
            row['reason'] = 'audio_pending'
            if item['source_kind'] in {'tts', 'clone', 'augmentation'} and not row.get('preflight_error'):
                propose(item, 'generate')
            continue
        try:
            audio = inspect_pcm16_wav(confined(root, item['audio_path']))
            require(audio['file_sha256'] == sha(item['file_sha256']), 'audio file mismatch')
            require(audio['pcm_sha256'] == sha(item['pcm_sha256']), 'audio PCM mismatch')
        except (ValueError, OSError, EOFError, wave.Error) as exc:
            row['reason'] = 'audio_integrity'; row['detail'] = str(exc); continue
        if item['source_kind'] in {'tts', 'clone'}:
            try:
                generated = histories[(item['id'], 'generate')] or [e for e in historical
                    if e['item_id'] == item['id'] and e['operation'] == 'generate']
                generation_binding(item, generated, audio, root)
                verify_research_rights(item, generated, audio, root)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                row['reason'] = 'generation_evidence_incomplete'; row['detail'] = str(exc); continue
        votes = []
        primary = plan['recognizers'][:2]
        for recognizer in primary:
            history = histories[(item['id'], recognizer['id'])]
            if not history:
                if not row.get('preflight_error'):
                    propose(item, recognizer['id'], audio)
                continue
            last = history[-1]
            if last['outcome'] == 'success' and last['pcm_sha256'] == audio['pcm_sha256']:
                try:
                    votes.append(native_vote(last, recognizer, audio, root))
                except (ValueError, KeyError, TypeError, OSError):
                    row['native_evidence_error'] = True
            elif last['call_state'] == 'not_called' and not row.get('preflight_error'):
                propose(item, recognizer['id'], audio)
        row['normalized_asr_votes'] = votes[:]
        protocols = [r.get('protocol', 'anonymous-two-asr-raw-v1') for r in primary]
        row['asr_protocols'] = protocols
        if 'source-language-conditioned-asr-v1' in protocols:
            row['language_is_model_independent_observation'] = False
            row['detected_language'] = None
            row['source_language_conditions'] = [r.get('protocol_config', {}).get('source_language_condition') for r in primary]
        if len(votes) < 2:
            row['reason'] = 'primary_asr_incomplete'; continue
        if votes[0] != votes[1]:
            row['reason'] = 'asr_disagreement'
            # Third model is configuration only: no scheduling or implicit call.
            if len(plan['recognizers']) == 3:
                third = plan['recognizers'][2]
                row['third_route'] = third['id']
                row['third_route_task'] = {'item_id': item['id'], 'operation': third['id'],
                    'audio_path': item['audio_path'], 'pcm_sha256': audio['pcm_sha256'],
                    'model_sha256': third['model_sha256'], 'config_sha256': third['config_sha256'],
                    'plan_sha256': plan_hash, 'execution_authorized': False,
                    'status': 'configuration_only_no_implicit_call'}
                history = histories[(item['id'], row['third_route'])]
                if history and history[-1]['outcome'] == 'success' and history[-1]['pcm_sha256'] == audio['pcm_sha256']:
                    try:
                        votes.append(native_vote(history[-1], plan['recognizers'][2], audio, root))
                    except (ValueError, KeyError, TypeError, OSError):
                        row['native_evidence_error'] = True
            winner, count = collections.Counter(votes).most_common(1)[0]
            if count < 2:
                continue
        else:
            winner = votes[0]
        row['machine_text'] = winner
        if not winner:
            row['reason'] = 'empty_asr_not_proof_of_no_speech'; continue
        # Conservative subsequence detection also quarantines interrupted wake paths.
        def contains_wake(keyword):
            chars = iter(winner)
            return all(any(c == k for c in chars) for k in keyword)
        if item['intent_kind'] in {'negative', 'confusable', 'background'} and any(contains_wake(k) for k in plan['keywords']):
            row['reason'] = 'negative_contains_keyword'; continue
        if item['intent_kind'] == 'background':
            row['reason'] = 'background_contains_speech'; continue
        try:
            names = actual_tokens(winner)
            targets = [token_map[name] for name in names]
            require(all(t > 0 for t in targets), 'nonblank targets required')
        except (ValueError, KeyError):
            row['reason'] = 'transcript_outside_training_vocabulary'; continue
        minimum = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
        if max(0, 1 + (audio['frames'] - 400) // 320) < minimum:
            row['reason'] = 'insufficient_ctc_frames'; continue
        # Derive class from actual machine text, never from requested generation intent.
        actual_kind = 'positive' if winner in plan['keywords'] else 'speech'
        if row.get('preflight_error'):
            row['reason'] = 'preflight_unavailable'; continue
        row.update(status='machine_candidate', reason='machine_agreement_only', research_weak_allowed=True,
                   actual_kind=actual_kind, target_ids=targets, tokens=names,
                   audio_path=item['audio_path'], file_sha256=audio['file_sha256'],
                   pcm_sha256=audio['pcm_sha256'], split=item['split'])
    accepted = sum(r['status'] == 'machine_candidate' for r in rows)
    return {'schema': 'speech-data-queue-report-v1', 'plan_sha256': plan_hash,
            'items': rows, 'proposed_tasks': tasks,
            'statistics': {'planned_items': len(rows), 'machine_candidates': accepted,
                           'quarantined_items': len(rows) - accepted,
                           'effective_yield': accepted / len(rows), 'attempt_records': len(events),
                           'known_calls': sum(e['call_state'] == 'called' for e in events),
                           'unknown_calls_reserved': sum(e['call_state'] == 'unknown' for e in events),
                           'content_failures': sum(e['outcome'] == 'content_failure' for e in events),
                           'infra_failures': sum(e['outcome'] == 'infra_failure' for e in events),
                           'budget_exceeded': consumed > plan['call_budget'],
                           'remaining_after_proposals': remaining},
            'historical_statistics': {'attempt_records': len(historical),
                'known_calls': sum(e['call_state'] == 'called' for e in historical),
                'infra_failures': sum(e['outcome'] == 'infra_failure' for e in historical),
                'scope': 'preserved-prior-experiment-not-current-call-budget'},
            'authority': 'unverified-offline-machine-candidates-only'}


def durable_write(path, data):
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def capture(plan_path, events_path, root, output):
    """Raw inputs first; fresh output directory only. No Library/authentication claims."""
    output.mkdir(parents=True, exist_ok=False)
    raw_plan, raw_events = plan_path.read_bytes(), events_path.read_bytes()
    durable_write(output / 'raw-plan.json', raw_plan)
    durable_write(output / 'raw-events.json', raw_events)
    plan, events = json.loads(raw_plan), json.loads(raw_events)
    # Preserve provided successful AND failed artifact bytes before validation.
    refs = [(i.get('audio_path'), i.get('file_sha256')) for i in plan['items'] if i.get('audio_path')]
    refs += [(a['path'], a['sha256']) for e in events for a in e['raw_artifacts']]
    refs += [(plan['tokens_path'], plan['tokens_sha256'])]
    refs += [(a['path'], a['sha256']) for a in plan.get('historical_evidence', {}).get('artifacts', [])]
    refs += [(a['path'], a['sha256']) for i in plan['items'] for a in i['rights'].get('supporting_artifacts', [])]
    # Retain rights evidence, not multi-GB model assets. Their exact inventory
    # and expected digests remain in raw-plan.json; missing assets quarantine.
    for item in plan['items']:
        for ref in [item['rights']]:
            try:
                confined(root, ref['path'])
            except (ValueError, OSError):
                continue
            refs.append((ref['path'], ref['sha256']))
    captured = []
    for index, (path, expected) in enumerate(refs):
        try:
            source = confined(root, path)
        except (ValueError, OSError) as exc:
            captured.append({'source_path': path, 'expected_sha256': expected, 'matches': False, 'error': str(exc)})
            continue
        target = output / f'artifact-{index:06d}{source.suffix}'
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
        actual = sha256_file(target)
        captured.append({'source_path': path, 'member': target.name, 'sha256': actual,
                         'expected_sha256': expected, 'matches': actual == expected})
    durable_write(output / 'capture.json', json.dumps(captured, ensure_ascii=False, indent=2).encode())
    report = evaluate(plan, events, root)
    report['capture_mismatches'] = [a for a in captured if not a['matches']]
    audio_members = {a['source_path']: a['member'] for a in captured if a.get('member') and a['matches']}
    weak = []
    for row in report['items']:
        if row['status'] != 'machine_candidate':
            continue
        weak.append({'audio': audio_members[row['audio_path']], 'target_ids': row['target_ids'],
                     'tokens': row['tokens'], 'actual_text': row['machine_text'],
                     'kind': row['actual_kind'], 'split': row['split'],
                     'speaker_id': row['lineage_group'], 'source_id': row['item_id'],
                     'file_sha256': row['file_sha256'], 'pcm_sha256': row['pcm_sha256'],
                     'label_source': 'weak-machine-agreement', 'purpose': 'noncommercial-research-only',
                     'human_gold': False, 'product_qualified': False,
                     'intended_text': next(i['intended_text'] for i in plan['items'] if i['id'] == row['item_id']),
                     'asr_protocols': row.get('asr_protocols', []),
                     'source_language_conditions': row.get('source_language_conditions'),
                     'detected_language': row.get('detected_language'),
                     'language_is_model_independent_observation': row.get('language_is_model_independent_observation')})
    for name, rows in [('weak-train.jsonl', [r for r in weak if r['split'] == 'train']),
                       ('weak-heldout.jsonl', [r for r in weak if r['split'] != 'train']),
                       ('quarantine.jsonl', [r for r in report['items'] if r['status'] == 'quarantine']),
                       ('third-route.jsonl', [r['third_route_task'] for r in report['items'] if r.get('third_route_task')])]:
        durable_write(output / name, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode())
    durable_write(output / 'report.json', json.dumps(report, ensure_ascii=False, indent=2).encode())
    manifest = {p.name: sha256_file(p) for p in sorted(output.iterdir())}
    durable_write(output / 'MANIFEST.json', json.dumps(manifest, indent=2).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('plan', 'events', 'input-root', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    capture(args.plan, args.events, args.input_root, args.output)


if __name__ == '__main__':
    main()
