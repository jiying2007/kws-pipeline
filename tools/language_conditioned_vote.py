"""Offline checks for source-language-conditioned-asr-v1 archived observations.

No model/tokenizer import, downloads, inference, or runtime reconstruction.
The caller freezes actual archive identities in its PRIVATE plan. This checks
saved-byte consistency plus native token/prefix/PCM bindings; it does not prove
execution authenticity, re-decode tokens, or supply independent language ID.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import PurePosixPath
import struct
import wave
import zipfile

SCHEMA = 'source-language-conditioned-asr-v1'
PROTOCOL_SHA256 = 'c84748f9d18ab8f4aa6b0cd2a659b5a890b189f1cc3c84af86fcc7ad959bcfb8'
ADAPTER_SHA256 = 'b7380c981241d3433d24e10193c7f7b3ebde25b25690f2f5cf040c3e9f93696d'
MAINTAINED_SHA256 = '35b7c392610e78a4bb13cd1cfca57a94b8a3555f38740fc7c9320f04b2d70d3b'
MODELS = {'qwen3-asr': ('Qwen/Qwen3-ASR-0.6B', 'qwen06', 'Chinese'),
          'sensevoice': ('FunAudioLLM/SenseVoiceSmall', 'sensevoice', 'zh')}
MAX_ARCHIVE = 16 * 1024 * 1024
MAX_MEMBER = 8 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode())


def loads(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read_archive(raw, expected_sha256):
    require(len(raw) <= MAX_ARCHIVE and digest(raw) == expected_sha256,
            'archive differs from frozen private identity')
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        infos = bundle.infolist()
        names = [i.filename for i in infos]
        require(0 < len(names) <= 128 and len(set(names)) == len(names), 'invalid archive roster')
        require(sum(i.file_size for i in infos) <= MAX_ARCHIVE, 'oversized archive')
        for info in infos:
            p = PurePosixPath(info.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename
                    and not info.is_dir() and info.file_size <= MAX_MEMBER
                    and (info.external_attr >> 16) & 0o170000 != 0o120000,
                    'unsafe archive member')
        contents = {n: bundle.read(n) for n in names}
    manifest = loads(contents['PACKAGE-MANIFEST.json'])
    require(set(names) == {'PACKAGE-MANIFEST.json', *manifest['files']}, 'archive manifest roster mismatch')
    for n, meta in manifest['files'].items():
        require(meta == {'bytes': len(contents[n]), 'sha256': digest(contents[n])}, 'archive member mismatch')
    return manifest, contents


def token_ids(value, maximum=4096):
    require(type(value) is list and 0 < len(value) <= maximum
            and all(type(x) is int and x >= 0 for x in value), 'invalid native token IDs')
    return value


def verify_pcm(wav, binding, audio, opaque):
    with wave.open(io.BytesIO(wav), 'rb') as r:
        require(r.getparams()[:3] == (1, 2, 16000) and r.getcomptype() == 'NONE', 'PCM format mismatch')
        frames = r.getnframes(); pcm = r.readframes(frames)
    require(0 < frames <= 480000 and len(pcm) == frames * 2, 'invalid bounded PCM')
    require(digest(wav) == audio['file_sha256'] and digest(pcm) == audio['pcm_sha256']
            and frames == audio['frames'], 'archived input does not match actual audio')
    require(binding['opaque_id'] == opaque and binding['schema'] == 'cosyvoice12-pcm-binding-v1'
            and binding['operations'] == {'mix': False, 'resample': False, 'trim': False}, 'input descriptor identity mismatch')
    require(binding['wav'] == {'bytes': len(wav), 'format': 'RIFF-WAVE-fmt16-PCM1-data-only',
            'header_bytes': 44, 'sha256': digest(wav)}, 'input WAV descriptor mismatch')
    require(len(wav) == len(pcm) + 44, 'noncanonical WAV')
    floats = b''.join(struct.pack('<f', x[0] / 32768.0) for x in struct.iter_unpack('<h', pcm))
    for name, data, dtype, enc in [('raw_pcm16', pcm, '<i2', 'signed-int16-le'),
                                  ('float32_pcm', floats, '<f4', 'ieee754-binary32-le')]:
        expected = {'bytes': len(data), 'channels': 1, 'dtype': dtype, 'encoding': enc,
                    'frame_count': frames, 'sample_order': 'unchanged', 'sample_rate_hz': 16000,
                    'sha256': digest(data), 'shape': [frames]}
        if name == 'float32_pcm': expected['normalization'] = 'signed_int16 / 32768.0'
        require(binding[name] == expected and binding[name + '_descriptor_sha256'] == canonical(expected),
                'actual PCM descriptor mismatch')


def verify_qwen(raw, before, generation):
    ids = token_ids(raw['generated_token_ids'], 256)
    require(type(before['input_ids']) is list and len(before['input_ids']) == 1, 'Qwen actual input batch mismatch')
    prefix = token_ids(before['input_ids'][0])
    start = [151644,8948,198,151645,198,151644,872,198,151669]
    end = [151670,151645,198,151644,77091,198,11528,8453,151704]
    require(prefix[:len(start)] == start and prefix[-len(end):] == end
            and len(prefix) > len(start) + len(end)
            and all(i == 151676 for i in prefix[len(start):-len(end)]), 'Qwen language-only native token prefix mismatch')
    require(before['input_ids'] == [prefix] and raw['input_ids_after_generate'] == [prefix]
            and raw['input_ids_unchanged'] is True and raw['prefix_preserved'] is True,
            'Qwen input prefix changed')
    require(raw['prefix_token_count'] == len(prefix)
            and raw['returned_sequence_ids'] == [prefix + ids]
            and raw['returned_sequence_shape'] == [1, len(prefix) + len(ids)]
            and raw['generation_tail_only'] is True, 'Qwen generation tail/prefix mismatch')
    require(all(raw[k] == v for k, v in generation.items()), 'Qwen generation capture differs from raw')
    require(raw['context'] == '' and raw['force_language'] == 'Chinese', 'Qwen unexpected conditioning')
    template = '<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n'
    require(raw['empty_context_template'] == template
            and raw['prompt_prefix'] == template + 'language Chinese<asr_text>'
            and raw['messages'] == [{'role': 'system', 'content': ''},
                {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]}], 'Qwen actual prompt mismatch')
    require(raw['generation_options'] == {'do_sample': False, 'eos_token_id': [151645,151643], 'max_new_tokens': 256}, 'Qwen generation options changed')
    t = raw['decoded_tail']
    require(type(t) is str and 0 < len(t) <= 32768 and t.strip()
            and '<asr_text>' not in t and '<|' not in t
            and not any(0xD800 <= ord(c) <= 0xDFFF for c in t), 'Qwen invalid native tail')
    require(not any(i in (151645,151643) for i in ids[:-1]), 'Qwen premature EOS token')
    require(ids[-1] in (151645,151643), 'Qwen incomplete generated sequence')
    suffix = '<|im_end|>' if ids[-1] == 151645 else '<|endoftext|>'
    require(raw['batch_decode_skip_special_tokens'] == [t]
            and raw['decoded_with_special_tokens'] == t + suffix
            and raw['batch_decode_with_special_tokens'] == [t + suffix], 'Qwen decoder capture mismatch')
    packed = struct.pack('<' + 'q' * len(prefix), *prefix)
    require(before['processor_tensor_evidence']['input_ids'] == {
        'dtype': 'torch.int64', 'sha256': digest(packed), 'shape': [1,len(prefix)]}, 'Qwen input token tensor mismatch')
    return t


def verify_sense(raw, parsed, receipt, *, emotion_ids=None):
    from sense_binding import CONTRACT_SHA256, TOKENIZER_SHA256, _contract
    require(raw['inference_arguments'] == {'language': 'zh', 'use_itn': False, 'text_norm': 'woitn',
            'fs': 16000, 'data_type': 'sound', 'output_timestamp': False, 'ban_emo_unk': False, 'device': 'cpu'},
            'Sense language/inference options changed')
    require(raw['context'] == '' and raw['hotwords'] == [] and raw['rich_postprocess_applied'] is False
            and raw['transcript_repair_applied'] is False and raw['raw_result_python_type'] == 'tuple', 'Sense repaired/conditioned output')
    result = raw['raw_result']
    require(type(result) is list and len(result) == 2 and type(result[0]) is list and len(result[0]) == 1,
            'Sense native result shape mismatch')
    item = result[0][0]
    require(set(item) == {'key','text'} and item['key'] == raw['opaque_id'], 'Sense native opaque mismatch')
    decoded = item['text']; token = parsed['token_binding']; evidence = parsed['token_evidence']
    ids = token_ids(token['raw_token_ids'])
    require(raw['capture_calls'] == [{'decoded': decoded, 'token_ids': ids}]
            and raw['raw_decode_attempts'] == [{'decode_kwargs': {}, 'raw_decoded_result': decoded, 'raw_token_ids': ids}],
            'Sense native capture/token mismatch')
    require(evidence['decoded'] == decoded and evidence['token_ids'] == ids
            and token['raw_decoded_sha256'] == digest(decoded.encode()), 'Sense raw/parsed decode mismatch')
    require(token['raw_redecode_matches_capture'] is True and token['metadata_boundary_verified'] is True
            and token['lexical_unknown_present'] is False and token['parser_contract_sha256'] == CONTRACT_SHA256,
            'Sense token boundary/parser invalid')
    require(token['tokenizer'] == raw['tokenizer_identity'] == receipt['receipt']['tokenizer_identity']
            and token['tokenizer']['sha256'] == TOKENIZER_SHA256, 'Sense tokenizer identity mismatch')
    if emotion_ids is None:
        emotion_ids = {'NEUTRAL': 25004, 'EMO_UNKNOWN': 25009}
    metadata = parsed['metadata']
    require(type(metadata) is list and len(metadata) == 4 and metadata[1] in emotion_ids, 'unreviewed Sense metadata variant')
    require(ids == token['metadata_token_ids'] + token['lexical_token_ids'] and len(token['metadata_token_ids']) == 4
            and token['metadata_token_ids'] == [24884,emotion_ids[metadata[1]],24993,25017]
            and token['metadata_pieces'] == ['<|zh|>','<|' + metadata[1] + '|>','<|Speech|>','<|woitn|>'],
            'unsupported or mismatched Sense metadata token boundary')
    lexical = token_ids(token['lexical_token_ids'])
    require(0 not in lexical and all(x < 25055 for x in ids), 'Sense lexical unknown/invalid token')
    extracted = _contract()['corrected_sense_extract'](decoded, lexical)
    require(extracted['quality_flags'] == [] and extracted['completeness'] == 'complete'
            and all(extracted[k] == parsed[k] for k in ('raw_text','quality_flags','completeness','metadata')),
            'Sense parsed outcome mismatch')
    require(parsed['metadata'] == ['zh',metadata[1],'Speech','woitn'], 'Sense metadata mismatch')
    return extracted['raw_text']


def verify_contents(manifest, content, recognizer, audio, lock):
    model, model_key, language = MODELS[recognizer['family']]
    conf = recognizer['protocol_config']
    require(conf['schema'] == SCHEMA and conf['source_language_condition'] == language
            and canonical(conf) == recognizer['config_sha256'], 'conditioned protocol configuration mismatch')
    require(manifest['slot_id'] == lock['slot_id'] and manifest['experiment_id'] == lock['experiment_id'], 'archive experiment/slot mismatch')
    raw = loads(content['output/raw.json']); before = loads(content['output/input-evidence.json'])
    parsed = loads(content['output/parsed.json']); dec = loads(content['output/native-declaration.json'])
    receipt = loads(content['output/load-receipt.json']); task = loads(content['task.json'])
    slot = task['slot']; opaque = slot['opaque_id']
    require(slot == dec['slot'] and slot['slot_id'] == manifest['slot_id'] == model_key + '-' + opaque
            and slot['model_key'] == model_key and slot['model_id'] == model
            and slot['language_argument'] == language and slot['frame_count'] == audio['frames']
            and slot['wav_sha256'] == audio['file_sha256'] and slot['pcm_sha256'] == audio['pcm_sha256'], 'native slot binding mismatch')
    for data in (raw, before, parsed, receipt):
        require(data['schema'] == SCHEMA and data['model_id'] == model == recognizer['model_id'], 'native model/schema mismatch')
    require(raw['opaque_id'] == before['opaque_id'] == parsed['opaque_id'] == opaque
            and all(raw[k] == v for k,v in before.items()), 'raw/input observation mismatch')
    require(raw['forward_completion_confirmed'] is True, 'forward incomplete')
    verify_pcm(content['input.wav'], before['input_binding'], audio, opaque)
    require(dec['schema'] == 'source-language-single-slot-execution-v1'
            and dec['scope'] == 'one_existing_pcm_one_source_language_conditioned_decode'
            and dec['experiment_id'] == manifest['experiment_id'] and dec['model_id'] == model
            and dec['source_language_condition'] == language and dec['context'] == ''
            and dec['hotwords'] == [] and dec['target_text_provided_to_asr'] is False
            and dec['batch_size'] == dec['candidate_count'] == dec['max_inference_calls'] == 1
            and dec['retries'] == 0 and dec['model_dtype'] == 'float32', 'native declaration mismatch')
    for name in ('source_lock_sha256','expected_schema_sha256','decoder_manifest_sha256'):
        require(dec[name] == conf[name], 'native source/config identity drift')
    require(dec['asset_lock_sha256'] == recognizer['model_sha256'] and receipt['declaration'] == dec
            and receipt['maintained_adapter_sha256'] == MAINTAINED_SHA256, 'native loader declaration mismatch')
    assets = receipt['receipt']['assets']
    require(assets['model_id'] == model and assets['revision'] == recognizer['revision']
            and assets['asset_lock_sha256'] == recognizer['model_sha256'], 'native model asset identity mismatch')
    require(receipt['receipt']['device'] == 'cpu' and receipt['receipt']['dtype'] == 'float32', 'runtime dtype/device mismatch')
    require(receipt['receipt']['load']['complete_checkpoint_verification_passed'] is True, 'native checkpoint not verified')
    source_names = [PurePosixPath(r['path']).name for r in task['source_manifest']]
    require(all(source_names.count(n) == 1 for n in ('native_adapter.py','language_conditioned_protocol.py','adapters.py')), 'ambiguous native source freeze')
    source_hashes = {PurePosixPath(r['path']).name: r['sha256'] for r in task['source_manifest']}
    require(source_hashes['native_adapter.py'] == ADAPTER_SHA256
            and source_hashes['language_conditioned_protocol.py'] == PROTOCOL_SHA256
            and source_hashes['adapters.py'] == MAINTAINED_SHA256, 'native source freeze mismatch')
    require(task['config']['protocol_sha256'] == PROTOCOL_SHA256
            and task['config']['maintained_adapter_sha256'] == MAINTAINED_SHA256
            and task['config']['manifest_sha256'] == conf['decoder_manifest_sha256'], 'task source/config drift')
    require(parsed['raw_evidence_sha256'] == digest(content['output/raw.json'])
            and parsed['binding_verified'] is True and parsed['detected_language'] is None
            and parsed['language_is_model_independent_observation'] is False
            and parsed['source_language_condition'] == language and parsed['human_gold'] is False
            and parsed['label_scope'] == 'weak-research-only' and parsed['transcript_repair_applied'] is False
            and parsed['quality_flags'] == [] and parsed['completeness'] == 'complete', 'parsed scope/binding mismatch')
    worker = loads(content['output/worker-result.json'])
    require(worker['status'] == 'completed_bound' and worker['binding_verified'] is True
            and worker['result'] == parsed and worker['slot_id'] == slot['slot_id']
            and worker['raw_evidence_sha256'] == digest(content['output/raw.json'])
            and worker['parsed_sha256'] == digest(content['output/parsed.json']), 'worker outcome binding mismatch')
    terminal = loads(content['terminal.json'])
    require(terminal['returncode'] == 0 and terminal['termination_reason'] == 'normal_exit'
            and terminal['model_leader_reaped'] is True, 'native process did not complete')
    if recognizer['family'] == 'qwen3-asr':
        generation = loads(content['output/raw-generation.json'])
        require(raw['generation_evidence_sha256'] == digest(content['output/raw-generation.json']), 'raw generation hash mismatch')
        actual = verify_qwen(raw, before, generation)
    else:
        actual = verify_sense(raw, parsed, receipt)
    require(actual == parsed['raw_text'], 'native-derived text differs from parsed text')
    return actual


def native_vote(event, recognizer, audio, root):
    # Delayed import avoids a cycle; the legacy queue stays unchanged except dispatch.
    from speech_data_queue import confined, normalize
    require(event['protocol'] == SCHEMA and recognizer['protocol'] == SCHEMA, 'protocol must be explicit')
    lock = recognizer['archive_bindings'][event['slot_id']]
    require(lock['item_id'] == event['item_id'] and lock['operation'] == event['operation'], 'private association mismatch')
    refs = {r['path']:r['sha256'] for r in event['raw_artifacts']}
    require(len(refs) == len(event['raw_artifacts']), 'duplicate conditioned artifact paths')
    archive = event['archive_path']
    require(refs[archive] == lock['archive_sha256'], 'archive reference not frozen')
    require(event['source_language_condition'] == recognizer['protocol_config']['source_language_condition']
            and event['detected_language'] is None and event['language_is_model_independent_observation'] is False, 'event misrepresents conditioned language')
    raw_archive = confined(root, archive).read_bytes()
    manifest, content = read_archive(raw_archive, lock['archive_sha256'])
    require(set(event.get('delivery_sidecars', {})) == set(lock.get('sidecars', {})), 'event delivery sidecar roster changed')
    for name, path in event.get('delivery_sidecars', {}).items():
        require(refs[path] == lock['sidecars'][name], 'delivery sidecar identity not frozen')
        sidecar = loads(confined(root,path).read_bytes())
        if name == 'delivered.json':
            require(sidecar['archive_sha256'] == lock['archive_sha256'] and sidecar['slot_id'] == lock['slot_id']
                    and sidecar['all_outcomes_retained'] is True, 'supplied delivery archive mismatch')
        if name == 'delivery-proof.json':
            require(sidecar['sha256'] == lock['archive_sha256'] and sidecar['bytes'] == len(raw_archive), 'supplied readback byte mismatch')
    require(set(event['archive_members']) == set(content), 'captured member roster mismatch')
    for original, path in event['archive_members'].items():
        require(refs[path] == digest(content[original]) and confined(root,path).read_bytes() == content[original], 'captured sidecar differs from archive')
    return normalize(verify_contents(manifest, content, recognizer, audio, lock))
