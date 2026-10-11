"""Stdlib-only lexical comparison of independently verified public ASR evidence.

No inference, network access, native-trace validation, or training admission.
Existing inputs, scripts, and historical reports are never overwritten.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import wave

from compare import compare, normalized

HERE = pathlib.Path(__file__).resolve().parent
ROSTER = [f"clip-{i:06d}" for i in range(1, 17)]
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def normalized_punctuation_whitespace(text):
    return ''.join(character for character in normalized(text) if not character.isspace())


def add_punctuation_whitespace_comparison(result):
    """An explicit second policy; never silently change the historical policy."""
    categories = {key: [] for key in result['categories']}
    per_model_target_matches = {model: 0 for model in ['qwen06', 'sensevoice', 'whisper']}
    for row in result['rows']:
        votes = {model: normalized_punctuation_whitespace(text) for model, text in row['valid_literal_votes'].items()}
        target = normalized_punctuation_whitespace(row['target_text'])
        valid = len(votes) == 3
        agrees = len(set(votes.values())) == 1 if valid else None
        row['punctuation_whitespace_valid_votes'] = votes
        row['punctuation_whitespace_three_way_agreement'] = agrees
        row['punctuation_whitespace_per_asr_matches_target'] = {model: text == target for model, text in votes.items()}
        for model, match in row['punctuation_whitespace_per_asr_matches_target'].items():
            per_model_target_matches[model] += int(match)
        if not valid:
            category = 'incomplete_three_way'
        elif not agrees:
            category = 'three_way_disagreement'
        elif all(text == target for text in votes.values()):
            category = 'unanimous_matches_target'
        else:
            category = 'unanimous_differs_from_target'
        row['punctuation_whitespace_category'] = category
        categories[category].append(row['audio_id'])
    counts = {name: len(ids) for name, ids in categories.items()}
    counts['normalized_three_way_agreement'] = counts['unanimous_matches_target'] + counts['unanimous_differs_from_target']
    result['punctuation_whitespace_comparison'] = {
        'normalization': 'Remove Unicode category P punctuation and characters for which str.isspace() is true; preserve case, script, homophones, symbols, OOV and every other character. No Unicode normalization.',
        'fixed_denominator': 16,
        'observed_valid_triple_denominator': result['observed_valid_triple_denominator'],
        'counts': counts,
        'rates_fixed_16': {name: count / 16 for name, count in counts.items()},
        'agreement_rates_observed_valid_triples': {name: ratio(count, result['observed_valid_triple_denominator']) for name, count in counts.items() if name != 'incomplete_three_way'},
        'categories': categories,
        'per_asr_target_matches': per_model_target_matches,
        'additional_agreement_ids_vs_punctuation_only': [row['audio_id'] for row in result['rows'] if row['punctuation_whitespace_three_way_agreement'] and not row['normalized_three_way_agreement']],
    }


def authenticate_retained(data, authentication, root):
    """Recheck retained bytes and each unchanged original input against them."""
    for name, expected in authentication['verified_retained_files'].items():
        require(sha(root / name) == expected, 'retained file mismatch: ' + name)
    base = root / 'research/source-screen32-v1/evidence'
    old = base / 'run-38018029787'
    new = base / 'run-38021467257'
    archive_members = 0
    for run, kind, archive in [
        (old, 'generation', read(old / 'provenance.json')['archives']['generation']),
        (old, 'asr', read(old / 'provenance.json')['archives']['asr']),
        (new, 'asr', read(new / 'provenance.json')['archive']),
    ]:
        for name, metadata in archive['exact_members'].items():
            file = run / kind / name
            require(file.stat().st_size == metadata['bytes'], 'archive length: ' + name)
            require(sha(file) == metadata['sha256'], 'archive hash: ' + name)
            archive_members += 1
    sources = {
        'qwen06': read(old / 'asr/qwen06/terminal-outcomes.json'),
        'sensevoice': read(new / 'asr/sensevoice/terminal-outcomes.json'),
        'original_sensevoice': read(old / 'asr/sensevoice/terminal-outcomes.json'),
    }
    for model, rows in sources.items():
        require([r['opaque_id'] for r in rows] == ROSTER, 'source roster: ' + model)
    cells = {r['opaque_audio_id']: r for r in read(new / 'result-summary.json')['cells']}
    raw = read(new / 'asr/primary-raw.json')
    for index, row in enumerate(data['rows']):
        audio_id = row['audio_id']
        cell = cells[audio_id]
        for model, rows in sources.items():
            require(row[model] == rows[index], 'input changed from source: ' + model + '/' + audio_id)
        require(row['target_text'] == cell['intended_text'], 'target provenance: ' + audio_id)
        wav_path = root / row['public_wav_path']
        require(sha(wav_path) == row['wav_sha256'] == cell['wav_sha256'], 'WAV identity: ' + audio_id)
        with wave.open(str(wav_path), 'rb') as wav:
            require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000), 'PCM format')
            pcm = wav.readframes(wav.getnframes())
        require(hashlib.sha256(pcm).hexdigest() == row['pcm_sha256'] == cell['pcm_sha256'], 'PCM identity: ' + audio_id)
        for model, run in [('qwen06', old), ('sensevoice', new)]:
            record = row[model]
            receipt = run / f'asr/{model}/{model}/{audio_id}.receipt.json'
            literal = read(receipt)
            require(sha(receipt) == record['execution_receipt_sha256'], 'receipt hash: ' + audio_id)
            require(literal['raw_text'] == record['raw_text'] == cell[model + '_raw_text'], 'literal receipt: ' + audio_id)
            require(literal['wav_sha256'] == row['wav_sha256'], 'receipt WAV: ' + audio_id)
            original_raw = next(r for r in raw[model] if r['audio_id'] == audio_id)
            require(original_raw['raw_text'] == record['raw_text'] and original_raw['status'] == 'complete', 'raw provenance: ' + audio_id)
    return {
        'retained_files_rechecked': len(authentication['verified_retained_files']),
        'archive_members_rechecked': archive_members,
        'wav_pcm_identities_rechecked': 16,
        'successful_receipt_literal_bindings_rechecked': 32,
        'original_sensevoice_records_preserved': dict(collections.Counter(r['original_sensevoice']['status'] for r in data['rows'])),
        'private_human_observation_bodies_read_or_copied': False,
        'limitation': 'Byte-level authentication against retained public freezes and receipts, not independent attestation of original service execution.',
    }


def detailed_compare(data, terminal):
    require([r['audio_id'] for r in data['rows']] == ROSTER, 'fixed ordered 16-row roster required')
    require(len({r['wav_sha256'] for r in data['rows']}) == 16, 'distinct fixed WAV identities required')
    require(data['fixed_denominator'] == 16, 'fixed denominator must be 16')
    for row in data['rows']:
        for model in ['qwen06', 'sensevoice', 'original_sensevoice']:
            require(row[model]['wav_sha256'] == row['wav_sha256'], 'retained WAV binding: ' + model)
        require(row['role'] == 'source_screen_only' and row['human_gold'] is False and row['training_admitted'] is False, 'source role changed')
    # The native validator is external. These checks prevent accidental misuse of
    # lexical status fields; they do not substitute for native validation.
    if terminal is not None:
        require(isinstance(terminal, list), 'terminal must be an array')
        for record in terminal:
            if record.get('status') == 'completed_native_eot':
                require(bool(HEX64.fullmatch(record.get('raw_output_sha256', ''))), 'invalid raw-output SHA-256')
    result = compare(data, terminal)
    third = {r['audio_id']: r for r in terminal or []}
    native = sum(r.get('status') == 'completed_native_eot' and r.get('attempts') == 1 and r.get('stop_reason') == 'eot' and r.get('validation_errors') == [] for r in third.values())
    counts = result['counts']
    counts['third_terminal_rows_present'] = len(third)
    counts['third_native_eot_outputs'] = native
    counts['third_native_eot_nonlexical_outputs'] = native - counts['third_valid_outputs']
    observed = counts['all_three_valid_outputs']
    result['schema'] = 'exploratory-three-asr-comparison-v2'
    result['observed_valid_triple_denominator'] = observed
    result['agreement_rates_observed_valid_triples'] = {
        name: ratio(counts[name], observed)
        for name in ['literal_three_way_agreement', 'normalized_three_way_agreement', 'normalized_three_way_agreement_matches_target', 'normalized_three_way_agreement_differs_from_target', 'normalized_three_way_disagreement']
    }
    result['rates_fixed_16'] = {name: value / 16 for name, value in counts.items() if name != 'fixed_denominator'}
    result['rate_interpretation'] = 'Fixed-16 values are roster fractions. Observed agreement rates use only valid triples and are null with zero valid triples. Neither is genuine acoustic accuracy.'
    result['third_status_counts'] = dict(collections.Counter(r.get('status', 'missing') for r in third.values()))
    result['third_native_eot_coverage'] = {'count': native, 'denominator': 16, 'rate': native / 16}
    result['target_interpretation'] = 'Generated intended text is a separate comparison reference, not human truth and never a replacement for any ASR output.'
    result['working_truth_policy'] = 'Three independent ASR unanimous text may be a working-truth candidate only under provenance, rights, role and OOV gates. This retrospective screen does not implement or pass training admission.'
    result['original_human_observations'] = 'Preserved privately; neither consulted nor copied into this public-source comparison.'
    result['categories'] = {'unanimous_matches_target': [], 'unanimous_differs_from_target': [], 'three_way_disagreement': [], 'incomplete_three_way': []}
    per_model = {model: {'valid_outputs': 0, 'literal_target_matches': 0, 'normalized_target_matches': 0} for model in ['qwen06', 'sensevoice', 'whisper']}
    for original, row in zip(data['rows'], result['rows']):
        t = third.get(row['audio_id'], {})
        row['raw_texts'] = {'qwen06': original['qwen06'].get('raw_text'), 'sensevoice': original['sensevoice'].get('raw_text'), 'whisper': t.get('raw_text')}
        row['normalized_valid_votes'] = {model: normalized(text) for model, text in row['valid_literal_votes'].items()}
        row['normalized_target_text'] = normalized(row['target_text'])
        row['whisper_evidence'] = {name: t.get(name) for name in ['status', 'attempts', 'stop_reason', 'validation_errors', 'raw_output_sha256', 'decoder_trace_sha256', 'execution_receipt_sha256']}
        row['public_wav_path'] = original['public_wav_path']
        if row['normalized_three_way_agreement'] is None:
            category = 'incomplete_three_way'
        elif row['normalized_three_way_agreement'] is False:
            category = 'three_way_disagreement'
        elif all(match['normalized_match'] for match in row['per_asr_vs_target'].values()):
            category = 'unanimous_matches_target'
        else:
            category = 'unanimous_differs_from_target'
        row['category'] = category
        result['categories'][category].append(row['audio_id'])
        row['unanimous_normalized_text'] = next(iter(row['normalized_valid_votes'].values())) if row['normalized_three_way_agreement'] else None
        for model, match in row['per_asr_vs_target'].items():
            per_model[model]['valid_outputs'] += 1
            per_model[model]['literal_target_matches'] += int(match['literal_match'])
            per_model[model]['normalized_target_matches'] += int(match['normalized_match'])
    for record in per_model.values():
        record['literal_target_match_rate_observed'] = ratio(record['literal_target_matches'], record['valid_outputs'])
        record['normalized_target_match_rate_observed'] = ratio(record['normalized_target_matches'], record['valid_outputs'])
        record['normalized_target_match_fraction_fixed_16'] = record['normalized_target_matches'] / 16
    result['per_asr_vs_target_summary'] = per_model
    result['shipping_approved'] = False
    result['acoustic_verbatim_completeness'] = 'unknown'
    add_punctuation_whitespace_comparison(result)
    return result


def chinese_report(result):
    c = result['counts']
    p = result['provenance']
    whitespace = result['punctuation_whitespace_comparison']
    lines = [
        '原始 16 条合成筛查音频：三 ASR 离线逐字对照',
        '',
        f"Whisper 第四次运行：{p['whisper_run_url']}",
        f"执行提交：{p['whisper_head_sha']}",
        f"第三路原生 EOT 覆盖：{c['third_native_eot_outputs']}/16；可参与三路比较：{c['all_three_valid_outputs']}/16。",
        f"字面完全一致：{c['literal_three_way_agreement']}/16；仅去标点后一致：{c['normalized_three_way_agreement']}/16。",
        f"其中与生成目标一致 {c['normalized_three_way_agreement_matches_target']} 条，共同偏离生成目标 {c['normalized_three_way_agreement_differs_from_target']} 条；三路分歧 {c['normalized_three_way_disagreement']} 条；不完整 {c['incomplete_three_way']} 条。",
        f"若同时去掉标点和空白：三路一致 {whitespace['counts']['normalized_three_way_agreement']}/16（{whitespace['counts']['unanimous_matches_target']} 条匹配目标、{whitespace['counts']['unanimous_differs_from_target']} 条共同偏离目标），分歧 {whitespace['counts']['three_way_disagreement']}/16。额外一致的条目：{', '.join(whitespace['additional_agreement_ids_vs_punctuation_only']) or '无'}。",
        f"实际有效三路分母：{result['observed_valid_triple_denominator']}。固定分母始终为 16；缺失或失败不算票。",
        '规范化只删除 Unicode 标点，保留空白、大小写、简繁、同音字、符号和 OOV；不合并“窝/屋/吴”。',
        '原生 EOT 证明正常结束，不证明听对了所有字；声学逐字完整性仍为 unknown。',
        '下表顺序：生成目标；Qwen0.6 原文；SenseVoice 原文；Whisper 原文；去标点结论／去标点与空白结论。',
        '',
    ]
    labels = {'unanimous_matches_target': '三路一致且匹配目标', 'unanimous_differs_from_target': '三路一致但偏离目标', 'three_way_disagreement': '三路分歧', 'incomplete_three_way': '三路不完整'}
    for row in result['rows']:
        values = [row['target_text'], *(row['raw_texts'][model] for model in ['qwen06', 'sensevoice', 'whisper'])]
        texts = '；'.join(json.dumps(value, ensure_ascii=False) for value in values)
        lines.append(f"{row['audio_id']}：{texts}；{labels[row['category']]}／{labels[row['punctuation_whitespace_category']]}")
    lines += [
        '',
        '单路与生成目标的匹配（不能当作真实听感准确率）：',
        *(f"- {model}：规范化匹配 {record['normalized_target_matches']}/{record['valid_outputs']} 个有效输出（固定筛查分母 16）。" for model, record in result['per_asr_vs_target_summary'].items()),
        '',
        '边界：三独立 ASR 的一致文本可作为工作真值候选，但仍须满足来源、权利、角色与 OOV 门禁。这 16 条仍全部为 source_screen_only，human_gold=false，training_admitted=false，不自动进入训练或上线。生成目标不覆盖 ASR 原文；私人原始听感记录未读取、复制或修改。',
        '旧 SenseVoice 原始记录保持 1 failed_no_retry + 15 not_run；此次使用的是既有续跑的 16 个成功输出，没有重算或改写旧结果。',
        '',
        '公开原始来源：',
        *('- ' + url for url in p['source_run_urls'].values()),
        f"输入 SHA-256：{p['comparison_input_sha256']}",
        f"Whisper terminal SHA-256：{p['whisper_terminal_sha256']}",
        f"原有认证记录 SHA-256：{p['retained_authentication_sha256']}",
        f"本次验证记录 SHA-256：{p['whisper_validation_sha256']}",
        '全部逐条 WAV/PCM 身份、字面输出、状态、匹配判断和证据校验值见配套 JSON。',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal', type=pathlib.Path, required=True)
    parser.add_argument('--terminal-sha256', required=True)
    parser.add_argument('--validation', type=pathlib.Path, required=True)
    parser.add_argument('--validation-sha256', required=True)
    parser.add_argument('--source-root', type=pathlib.Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--head-sha', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--report', type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.report.exists(), 'will not overwrite an existing output')
    require(sha(args.terminal) == args.terminal_sha256, 'reviewed terminal SHA-256 mismatch')
    require(sha(args.validation) == args.validation_sha256, 'reviewed validation SHA-256 mismatch')
    validation = read(args.validation)
    require(str(validation['run_id']) == args.run_id and validation['head_sha'] == args.head_sha, 'verified run/head mismatch')
    require(validation['conclusion'] == 'success' and validation['terminal_validation_errors'] == 0, 'terminal verification must pass')
    require(validation['member_allowlist_hash_size_verified'] is True, 'artifact member verification required')
    require(validation['all_source_entries_verified_against_git_tree'] is True, 'source-tree verification required')
    verified_terminal = next(row for row in validation['artifact_members'] if row['path'] == 'terminal-outcomes.json')
    require(verified_terminal['sha256'] == args.terminal_sha256 and verified_terminal['bytes'] == args.terminal.stat().st_size, 'terminal not bound by verification receipt')
    terminal = read(args.terminal)
    require([row['audio_id'] for row in terminal] == [row['audio_id'] for row in validation['per_clip']], 'verified terminal roster mismatch')
    for row, verified in zip(terminal, validation['per_clip']):
        for key in ['wav_sha256', 'raw_text', 'raw_output_sha256', 'attempts']:
            require(row[key] == verified[key], 'verified per-clip binding mismatch: ' + key)
    data = read(HERE / 'comparison-input.json')
    auth = read(HERE / 'authentication.json')
    recheck = authenticate_retained(data, auth, args.source_root)
    result = detailed_compare(data, terminal)
    result['native_artifact_validation'] = 'PERFORMED_SEPARATELY_BY_FROZEN_HOST_VALIDATOR_AND_VERIFIER; see validation file and its hash'
    result['provenance'] = {
        'repository': data['repository'],
        'retained_source_commit': data['source_commit'],
        'retained_reviewed_tree': data['reviewed_tree'],
        'source_run_urls': {model: f'https://github.com/{data["repository"]}/actions/runs/{run}' for model, run in data['source_runs'].items()},
        'comparison_input_file': 'comparison-input.json',
        'comparison_input_sha256': sha(HERE / 'comparison-input.json'),
        'retained_authentication_file': 'authentication.json',
        'retained_authentication_sha256': sha(HERE / 'authentication.json'),
        'whisper_run_url': f'https://github.com/{data["repository"]}/actions/runs/{args.run_id}',
        'whisper_run_id': args.run_id,
        'whisper_head_sha': args.head_sha,
        'whisper_executed_tree': validation['executed_tree'],
        'whisper_artifact_id': validation['artifact_id'],
        'whisper_artifact_zip_sha256': validation['zip_sha256'],
        'whisper_terminal_file': str(args.terminal),
        'whisper_terminal_sha256': sha(args.terminal),
        'whisper_validation_file': str(args.validation),
        'whisper_validation_sha256': sha(args.validation),
        'comparison_script_sha256': sha(pathlib.Path(__file__)),
        'inherited_comparator_sha256': sha(HERE / 'compare.py'),
    }
    result['retained_reauthentication'] = recheck
    with args.output.open('x') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write('\n')
    with args.report.open('x') as file:
        file.write(chinese_report(result))
    print(json.dumps(result['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
