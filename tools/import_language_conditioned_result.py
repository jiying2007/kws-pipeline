"""Import one frozen language-conditioned result, retaining original bytes first.

Private caller owns item association, archive identity, and optional delivery
sidecars. Neither an import nor a supplied delivery receipt authenticates a run.
No model/tokenizer imports, inference, network, training, or budget changes.
"""
import json
from pathlib import Path
from language_conditioned_vote import SCHEMA, digest, loads, read_archive, verify_contents
from speech_data_queue import durable_write, inspect_pcm16_wav, require


def import_slot(archive: Path, staging: Path, *, item_id, operation, recognizer,
                lock, sidecars=None):
    staging.mkdir(parents=True, exist_ok=False)
    raw = archive.read_bytes()
    durable_write(staging / 'original.zip', raw)
    event = {'item_id': item_id, 'operation': operation, 'attempt': 1,
             'call_state': 'unknown', 'outcome': 'infra_failure', 'protocol': SCHEMA,
             'archive_path': 'original.zip', 'slot_id': lock['slot_id'],
             'raw_artifacts': [{'path': 'original.zip', 'sha256': digest(raw)}],
             'archive_members': {}}
    try:
        require(lock['item_id'] == item_id and lock['operation'] == operation, 'frozen association mismatch')
        manifest, contents = read_archive(raw, lock['archive_sha256'])
        for index, (name, data) in enumerate(sorted(contents.items())):
            member = f'member-{index:04d}{Path(name).suffix}'
            durable_write(staging / member, data)
            event['archive_members'][name] = member
            event['raw_artifacts'].append({'path': member, 'sha256': digest(data)})
        # Record execution facts separately from this new offline validator.
        native = loads(contents.get('output/raw.json', b'{}'))
        terminal = loads(contents.get('terminal.json', b'{}'))
        worker = loads(contents.get('output/worker-result.json', b'{}'))
        if native.get('forward_completion_confirmed') is True:
            event['call_state'] = 'called'
            event['execution_observation'] = 'archived-native-forward-returned'
            if terminal.get('returncode') == 0 and worker.get('status') == 'completed_bound':
                event['outcome'] = 'success'
        event['import_validation'] = 'pending'
        # Optional supplied Library readback/delivery sidecars remain evidence,
        # never execution authority. Their private freeze is mandatory if supplied.
        expected = lock.get('sidecars', {})
        require(set(sidecars or {}) == set(expected), 'delivery sidecar roster mismatch')
        event['delivery_sidecars'] = {}
        for index, (name, path) in enumerate(sorted((sidecars or {}).items())):
            data = Path(path).read_bytes(); member = f'delivery-{index:04d}.json'
            durable_write(staging / member, data)
            event['raw_artifacts'].append({'path': member, 'sha256': digest(data)})
            event['delivery_sidecars'][name] = member
            require(digest(data) == expected[name], 'delivery sidecar changed')
        audio = inspect_pcm16_wav(staging / event['archive_members']['input.wav'])
        verify_contents(manifest, contents, recognizer, audio, lock)
        event.update(call_state='called', outcome='success', binding='verified', blind=True,
                     import_validation='verified-archived-byte-consistency',
                     blindness_scope='target-text-blind-source-language-conditioned',
                     source_language_condition=recognizer['protocol_config']['source_language_condition'],
                     detected_language=None, language_is_model_independent_observation=False,
                     pcm_sha256=audio['pcm_sha256'])
        durable_write(staging / 'import-event.json', json.dumps(event, ensure_ascii=False, indent=2).encode())
        return event
    except Exception as exc:
        event['import_validation'] = 'rejected'
        durable_write(staging / 'import-failure.json', json.dumps({
            'schema': SCHEMA, 'status': 'rejected-offline-import', 'exception_type': type(exc).__name__,
            'reason': str(exc), 'retained_event': event, 'model_calls': 0,
            'retry_authorized': False}, ensure_ascii=False, indent=2).encode())
        raise
