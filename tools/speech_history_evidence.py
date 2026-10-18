"""Offline prior-experiment and bounded source-rights evidence checks."""
import copy
from pathlib import Path
from language_conditioned_vote import canonical, digest, loads, require

RIGHTS_SCHEMA = 'internal-noncommercial-research-source-review-v1'
TTS_MODEL = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
TTS_REVISION = '85e237c12c027371202489a0ec509ded67b5e4b5'
TTS_CARD_SHA256 = '1e3adeecc7a72d6756fdb77c2847f8e994195e105b51206a7a5c049b0dfa48a8'


def load_history(plan, root):
    from speech_data_queue import confined, validate, validate_events
    if 'historical_evidence' not in plan:
        return []
    h = plan['historical_evidence']
    refs = {a['path']: a['sha256'] for a in h['artifacts']}
    require(len(refs) == len(h['artifacts']), 'duplicate historical artifact')
    for path, sha in refs.items():
        require(digest(confined(root,path).read_bytes()) == sha, 'historical evidence changed')
    def read(field):
        p = h[field]
        require(p in refs, 'historical root evidence missing')
        return loads(confined(root,p).read_bytes())
    old_plan, events = read('plan_path'), read('events_path')
    old_hash = validate(old_plan)
    validate_events(old_plan, events, old_hash)
    original = {i['id']: i for i in old_plan['items']}
    for item in plan['items']:
        for key in ('source_kind','intent_kind','intended_text','lineage_roots','split','recipe_sha256','file_sha256','pcm_sha256'):
            require(item[key] == original[item['id']][key], 'historical source/intent/lineage was rewritten')
    translated = copy.deepcopy(events)
    prefix = h['input_prefix']
    require(not Path(prefix).is_absolute() and '..' not in Path(prefix).parts, 'invalid history prefix')
    for e in translated:
        for a in e['raw_artifacts']:
            a['path'] = prefix + '/' + a['path']
            require(refs.get(a['path']) == a['sha256'], 'prior raw artifact not retained')
        for field in ('generation_receipt_path','raw_result_path','declaration_path'):
            if field in e: e[field] = prefix + '/' + e[field]
    return translated


def verify_research_rights(item, history, audio, root):
    from speech_data_queue import confined
    rights = item['rights']
    if rights.get('review_schema') != RIGHTS_SCHEMA:
        return  # Preserve the existing legacy protocol behavior.
    require(rights['research_use'] == 'allowed', 'new source rights not allowed')
    review_path = Path(rights['path']); directory = review_path.parent
    review_raw = confined(root, str(review_path)).read_bytes()
    require(digest(review_raw) == rights['sha256'], 'rights review identity changed')
    review = loads(review_raw)
    require(review['schema'] == RIGHTS_SCHEMA
            and review['status'] == 'allowed_for_this_project_internal_noncommercial_research_only'
            and review['source_model_id'] == TTS_MODEL and review['source_revision'] == TTS_REVISION
            and review['voice'] == 'Sohee' and review['voice_kind'] == 'official_builtin_custom_voice'
            and review['user_reference_audio_supplied'] is False
            and review['public_audio_redistribution_rights'] == 'not_established_by_this_record'
            and review['commercial_use_rights'] == 'not_reviewed'
            and review['old_rights_unknown_record_preserved'] is True, 'source rights scope mismatch')
    refs = {a['path']: a['sha256'] for a in rights['supporting_artifacts']}
    data = {}
    for name, expected in review['evidence_files'].items():
        path = str(directory / name)
        raw = confined(root,path).read_bytes()
        require(refs.get(path) == digest(raw) == expected['sha256'] and len(raw) == expected['bytes'], 'rights supporting evidence mismatch')
        data[name] = raw
    require(digest(data['model-card.md']) == TTS_CARD_SHA256, 'unreviewed official model card')
    lock = loads(data['model-locks.json'])['qwen_tts']
    assets = lock['asset_lock']
    require(lock['model_id'] == assets['model_id'] == TTS_MODEL and assets['revision'] == TTS_REVISION
            and canonical(assets) == lock['asset_lock_sha256'], 'TTS model lock mismatch')
    card = [a for a in assets['files'] if a['filename'] == 'README.md']
    require(len(card) == 1 and card[0]['sha256'] == TTS_CARD_SHA256, 'model card not bound to asset lock')
    generation = loads(data['generation-plan.json'])
    require(generation['model_id'] == TTS_MODEL and generation['revision'] == TTS_REVISION, 'TTS generation plan model mismatch')
    request = [r for r in generation['requests'] if r['source_id'] == item['id']]
    require(len(request) == 1 and request[0]['voice'] == 'Sohee'
            and request[0]['intended_text'] == item['intended_text'], 'TTS source request mismatch')
    matches = [r for r in review['receipts'] if r['source_id'] == item['id']]
    require(len(matches) == 1 and len(history) == 1, 'ambiguous historical generation rights association')
    reference = matches[0]; raw = data[reference['receipt_file']]
    require(digest(raw) == reference['receipt_sha256']
            and raw == confined(root,history[0]['generation_receipt_path']).read_bytes(), 'rights receipt differs from retained generation')
    receipt = loads(raw)
    require(receipt['plan_sha256'] == digest(data['generation-plan.json']), 'generation plan bytes not bound')
    expected_assets = [{k: a[k] for k in ('filename','bytes','sha256')} for a in assets['files']]
    require(receipt['setup_verified']['model_assets']['qwen_tts'] == expected_assets, 'generated TTS assets differ from licensed source')
    rows = receipt['generation_rows']
    require(len(rows) == 1 and rows[0]['voice'] == reference['voice'] == 'Sohee'
            and rows[0]['source_id'] == item['id'] and rows[0]['intended_text'] == item['intended_text']
            and rows[0]['seed'] == request[0]['seed']
            and rows[0]['audio_sha256'] == reference['wav_sha256'] == audio['file_sha256']
            and rows[0]['pcm_sha256'] == reference['pcm_sha256'] == audio['pcm_sha256'], 'source rights not bound to actual PCM')
    require(any(a['sha256'] == reference['archive_sha256'] and a['path'].endswith('/original.zip')
                for a in history[0]['raw_artifacts']), 'original TTS archive identity missing')
