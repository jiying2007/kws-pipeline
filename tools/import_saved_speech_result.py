"""Read retained per-slot result ZIPs; no runtime imports, model calls or downloads.

The caller supplies item-to-opaque-slot association from its private manifest.
This validates saved-byte consistency, not cryptographic execution authenticity.
"""
import hashlib
import json
from pathlib import Path
import zipfile
from speech_data_queue import durable_write, require


def import_slot(archive: Path, staging: Path, *, item_id, operation, slot, attempt=1):
    staging.mkdir(parents=True, exist_ok=False)
    # Original archive is retained first, including failed/unbound evidence.
    archive_bytes = archive.read_bytes()
    durable_write(staging / 'original.zip', archive_bytes)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        require(len(set(names)) == len(names), 'duplicate archive members')
        manifest = json.loads(bundle.read('manifest.json'))
        require(manifest['slot'] == slot, 'saved slot identity mismatch')
        require(not manifest.get('missing'), 'incomplete saved package')
        require(set(names) == {'manifest.json', *manifest['files']}, 'archive roster mismatch')
        refs, mapped, contents = [], {}, {}
        for index, name in enumerate(sorted(manifest['files'])):
            meta = manifest['files'][name]
            raw = bundle.read(name)
            require(len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256'], 'saved member hash mismatch')
            member = f'member-{index:04d}{Path(name).suffix}'
            durable_write(staging/member, raw)
            mapped[name] = member; contents[name] = raw
            refs.append({'path': member, 'sha256': meta['sha256']})
    refs.append({'path': 'original.zip', 'sha256': hashlib.sha256(archive_bytes).hexdigest()})
    event = {'item_id': item_id, 'operation': operation, 'attempt': attempt,
             'call_state': 'unknown', 'outcome': 'infra_failure', 'raw_artifacts': refs}
    if operation == 'generate':
        require(manifest['stage'] == 'tts', 'generation stage mismatch')
        name = f'outputs/tts/{slot}/receipt.json'
        receipt = json.loads(contents[name])
        rows = receipt['generation_rows']
        require(len(rows) == 1 and rows[0]['source_id'] == slot, 'generation association mismatch')
        if rows[0]['status'] == 'generated_candidate':
            event.update(call_state='called', outcome='success')
        event['generation_observation'] = rows[0]
        event['generation_receipt_path'] = mapped[name]
    else:
        stage = manifest['stage']
        raw_name = f'outputs/{stage}/{slot}/{slot}.raw.json'
        dec_name = f'outputs/{stage}/{slot}/execution-declaration.json'
        raw = json.loads(contents[raw_name])
        require(raw['opaque_id'] == slot, 'raw opaque identity mismatch')
        event.update(raw_result_path=mapped[raw_name], declaration_path=mapped[dec_name])
        if raw['status'] == 'success':
            event.update(call_state='called', outcome='success', binding='verified', blind=True,
                         pcm_sha256=raw['evidence']['input_binding']['raw_pcm16']['sha256'])
        else:
            summary = json.loads(contents[f'outputs/{stage}/{slot}/summary.json'])
            claim = json.loads(contents[f'delivery-claims/{stage}.{slot}.json'])
            if (summary.get('attempted') == 1 and summary.get('stage') == stage
                    and claim.get('slot') == slot and claim.get('stage') == stage
                    and claim.get('budget_reserved') is True
                    and any(o.get('opaque_id') == slot and o.get('status') == raw['status']
                            for o in summary.get('outcomes', []))):
                event.update(call_state='called', outcome='infra_failure',
                             invocation_scope='recorded_adapter_attempt_not_successful_transcript')
        # A bound attempted=1 summary establishes the actual adapter invocation.
        # Without it retain unknown; neither case is eligible for automatic retry.
    return event


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'staging'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('item-id', 'operation', 'slot', 'plan-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    event = import_slot(args.archive, args.staging, item_id=args.item_id, operation=args.operation, slot=args.slot)
    event['plan_sha256'] = args.plan_sha256
    durable_write(args.staging/'event.json', json.dumps(event, ensure_ascii=False, indent=2).encode())


if __name__ == '__main__':
    main()
