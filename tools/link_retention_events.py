#!/usr/bin/env python3
"""Validate bounded research traces; links are observations, never wake truth.

JSONL header, contiguous successful row transactions, footer. Episode callback
outputs are provisional until this function returns successfully. No model runs.
"""
import hashlib
import json
import math
import re
import struct

SCHEMA = 'retention-event-linkage-v1'
SHA = re.compile(r'[0-9a-f]{64}\Z')
BITS = re.compile(r'[0-9a-f]{16}\Z')
COUNTERS = ('episodes_started', 'low_repeated', 'strong_closed', 'expired', 'window_censored', 'active_episodes', 'peak_active_episodes')
PASSIVE = {'detect_attempt', 'detect_local_reset_done', 'detect_local_state_retained', 'detect_no_suffix', 'detect_window_begin', 'row_trial_begin', 'row_committed'}
MERGE = {'candidate_before_dedup', 'candidate_before_threshold', 'candidate_accepted', 'duplicate_rejected', 'threshold_rejected', 'refractory_rejected'}
CLOSE = {'retention_strong_closed': 'strong_closed', 'retention_expired': 'expired', 'retention_window_censored': 'window_censored'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def integer(value, name, low=0):
    require(type(value) is int and value >= low, 'invalid integer: ' + name)
    return value

def score(bits):
    require(isinstance(bits, str) and BITS.fullmatch(bits), 'invalid score_bits')
    value = struct.unpack('>d', bytes.fromhex(bits))[0]
    require(math.isfinite(value) and 0 <= value <= 1, 'invalid score value')
    return value

def key(record, row):
    require(integer(record.get('row'), 'event row') == row, 'event row mismatch')
    window = integer(record.get('window_start'), 'window_start')
    require(window <= row, 'window starts after decision row')
    keyword = integer(record.get('keyword_id'), 'keyword_id', 1)
    require(keyword <= 2, 'unknown keyword')
    nodes = record.get('node_frames')
    require(isinstance(nodes, list) and len(nodes) == 4, 'node shape')
    for node in nodes:
        integer(node, 'node_frame')
    require(nodes == sorted(nodes), 'unordered nodes')
    require(all(3 * window <= node <= 3 * row for node in nodes), 'nodes outside window/decision clock')
    require(5 <= nodes[-1] - nodes[0] <= 250, 'candidate duration out of range')
    score(record.get('score_bits'))
    return (row, window, keyword, tuple(nodes), record['score_bits'])

def _object(pairs):
    out = {}
    for name, value in pairs:
        require(name not in out, 'duplicate JSON key')
        out[name] = value
    return out

def validate_stream(stream, emit_episode=lambda episode: None, *, byte_limit=64 * 1024 * 1024, record_limit=1000000):
    """Binary input required. Memory is one bounded row plus <=3 active episodes.

Limits are caller ceilings, not permissions to run a scientific evaluation.
The callback must not publish provisional outputs before successful admission.
"""
    integer(byte_limit, 'byte_limit', 1); integer(record_limit, 'record_limit', 1)
    total = 0; digest = hashlib.sha256(); episode_digest = hashlib.sha256(); header = None; footer_seen = False
    def emit(ep):
        episode_digest.update((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
        emit_episode(ep)
    seq = 0; row_count = 0; event_count = 0; linked_count = 0
    counts = dict.fromkeys(COUNTERS, 0); active = {}
    max_row_bytes = min(byte_limit, 16 * 1024 * 1024)
    for raw in iter(lambda: stream.readline(max_row_bytes + 1), b''):
        require(isinstance(raw, bytes), 'binary stream required')
        require(len(raw) <= max_row_bytes, 'row byte limit')
        total += len(raw); digest.update(raw)
        require(total <= byte_limit, 'total byte limit')
        require(raw.endswith(b'\n'), 'unterminated JSONL line')
        require(not footer_seen, 'data after footer')
        data = json.loads(raw, object_pairs_hook=_object, parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        require(isinstance(data, dict), 'object required')
        if header is None:
            require(data.get('type') == 'header' and data.get('schema') == SCHEMA, 'header required')
            for name in ('run_id', 'recording_id', 'arm'):
                require(isinstance(data.get(name), str) and 0 < len(data[name]) <= 256, 'invalid identity')
            require(data['arm'] == 'candidate', 'candidate trace required')
            for name in ('pcm_sha256', 'plan_sha256', 'candidate_sha256', 'library_sha256', 'harness_sha256', 'ledger_abi_sha256'):
                require(isinstance(data.get(name), str) and SHA.fullmatch(data[name]), 'invalid hash: '+name)
            require(type(data.get('threshold')) in (float, int) and data['threshold'] == .7, 'fixed threshold')
            require(type(data.get('retention_rows')) is int and data['retention_rows'] == 5, 'fixed retention rows')
            integer(data.get('expected_rows'), 'expected_rows')
            for name in ('max_bytes', 'max_records', 'max_row_records'):
                integer(data.get(name), name, 1)
            require(data['max_bytes'] <= byte_limit and data['max_records'] <= record_limit and data['max_row_records'] <= 32768, 'declared limits exceed caller ceiling')
            header = data
            require(total <= header['max_bytes'], 'declared byte limit')
            continue
        require(total <= header['max_bytes'], 'declared byte limit')
        if data.get('type') == 'footer':
            require(data.get('valid') is True and data.get('sink_closed') is True, 'invalid or unclosed capture')
            for name in ('dropped', 'pending', 'exit_code'):
                require(type(data.get(name)) is int and data[name] == 0, 'failed footer: '+name)
            require(integer(data.get('rows_committed'), 'rows_committed') == row_count == header['expected_rows'], 'incomplete rows')
            require(integer(data.get('sequence_count'), 'sequence_count') == seq, 'sequence footer mismatch')
            summary = data.get('summary'); require(isinstance(summary, dict), 'summary required')
            for name in COUNTERS:
                require(integer(summary.get(name), name) == counts[name], 'counter mismatch: '+name)
            require(integer(summary.get('rows_committed'), 'summary rows') == row_count, 'summary rows mismatch')
            require(integer(data.get('events_emitted'), 'events_emitted') == event_count, 'event total mismatch')
            windows = data.get('active_windows'); require(isinstance(windows, list) and len(windows) == len(active), 'active window snapshot mismatch')
            snapshot = {}
            for item in windows:
                require(isinstance(item, dict), 'window object')
                w = integer(item.get('window_start'), 'snapshot window')
                require(w not in snapshot, 'duplicate active window')
                snapshot[w] = (integer(item.get('anchor_row'), 'snapshot anchor'), integer(item.get('deadline_row'), 'snapshot deadline'))
            require(snapshot == {w:(e['start_row'], e['deadline_row']) for w,e in active.items()}, 'active snapshot mismatch')
            for episode in active.values():
                emit(dict(episode, closure='active_at_input_end', closure_row=None, merge_disposition='not_applicable', emitted_event_id=None))
            footer_seen = True
            continue
        require(data.get('type') == 'row', 'unknown transaction')
        row = integer(data.get('row'), 'row')
        require(row == row_count and row_count < header['expected_rows'], 'row order/range')
        require(type(data.get('decoder_status')) is int and data['decoder_status'] == 0, 'failed row is inadmissible')
        records = data.get('records'); events = data.get('events')
        require(isinstance(records, list) and 1 <= len(records) <= header['max_row_records'], 'row record limit')
        require(isinstance(events, list) and len(events) <= 3, 'event bound')
        require(seq + len(records) <= header['max_records'], 'record limit')
        require(sum(r.get('kind') == 'row_committed' for r in records if isinstance(r,dict)) == 1, 'exactly one commit required')
        require(records[-1].get('kind') == 'row_committed', 'commit must terminate row')
        require(sum(r.get('kind') == 'row_trial_begin' for r in records if isinstance(r,dict)) == 1 and records[0].get('kind') == 'row_trial_begin', 'trial must begin row')
        candidates = {}; merges = {}; closed = []; repeated = set()
        for rec in records:
            require(isinstance(rec, dict), 'record object')
            require(integer(rec.get('sequence'), 'sequence') == seq, 'sequence gap')
            seq += 1
            require(integer(rec.get('row'), 'record row') == row, 'record row mismatch')
            kind = rec.get('kind')
            require(kind in PASSIVE | MERGE | set(CLOSE) | {'retention_started','retention_low_repeated','detect_candidate_before_local_reset'}, 'unknown ledger kind')
            if kind in {'row_trial_begin','row_committed'}:
                require(type(rec.get('decoder_status')) is int and rec['decoder_status'] == 0, 'ledger row failure')
            if kind == 'detect_candidate_before_local_reset':
                k = key(rec,row); w=k[1]
                require(w not in candidates and len(candidates) < 3, 'ambiguous/window candidate bound')
                candidates[w] = k
            elif kind in {'retention_started','retention_low_repeated'} | set(CLOSE):
                w = integer(rec.get('window_start'), 'retention window')
                if kind == 'retention_started':
                    require(w not in active and len(active)<3, 'duplicate start/active bound')
                    require(w in candidates and score(candidates[w][-1]) < .7, 'start lacks low candidate')
                    active[w] = {'episode_id':[header['run_id'],header['recording_id'],header['arm'],w,row], 'window_start':w,'start_row':row,'deadline_row':row+5,'low_repeat_count':0}
                    counts['episodes_started'] += 1; counts['active_episodes'] += 1
                    counts['peak_active_episodes'] = max(counts['peak_active_episodes'],len(active))
                else:
                    require(w in active, 'retention event without active episode')
                    ep = active[w]
                    require(ep['start_row'] <= row <= ep['deadline_row'], 'episode clock violation')
                    if kind == 'retention_low_repeated':
                        require(w not in repeated and row > ep['start_row'] and w in candidates and score(candidates[w][-1])<.7, 'repeat lacks unique low candidate')
                        repeated.add(w)
                        ep['low_repeat_count'] += 1; counts['low_repeated'] += 1
                    else:
                        closure = CLOSE[kind]
                        if closure == 'expired': require(row == ep['deadline_row'], 'premature expiry')
                        if closure == 'strong_closed': require(w in candidates and score(candidates[w][-1]) >= .7, 'strong close lacks strong candidate')
                        closed.append(dict(ep,closure=closure,closure_row=row, candidate_key=candidates.get(w) if closure=='strong_closed' else None))
                        counts[closure] += 1; counts['active_episodes'] -= 1; del active[w]
            elif kind in MERGE:
                k=key(rec,row); require(candidates.get(k[1]) == k,'merge without exact detect key')
                phase=merges.get(k)
                if kind=='candidate_before_dedup': require(phase is None,'duplicate merge'); merges[k]='dedup'
                elif kind=='candidate_before_threshold': require(phase=='dedup','threshold order'); merges[k]='threshold'
                elif kind=='duplicate_rejected': require(phase=='dedup','duplicate disposition order'); merges[k]='duplicate_rejected'
                else:
                    require(phase=='threshold','terminal disposition order')
                    if kind=='threshold_rejected': require(score(k[-1])<.7,'threshold rejection contradicts score')
                    else: require(score(k[-1])>=.7,'accept/refractory below threshold')
                    merges[k]=kind
        require(all(ep['deadline_row'] > row for ep in active.values()),'expired episode left active')
        require(len(merges)==len(candidates) and all(v in {'candidate_accepted','duplicate_rejected','threshold_rejected','refractory_rejected'} for v in merges.values()),'incomplete merge disposition')
        returned = {}
        for event in events:
            require(isinstance(event,dict),'event object'); k=key(event,row)
            require(k not in returned and merges.get(k)=='candidate_accepted','unmatched/duplicate returned event')
            require(integer(event.get('event_index'),'event_index')==event_count,'event index sequence')
            returned[k]=[header['recording_id'],header['arm'],event_count]; event_count+=1
        require(set(returned)=={k for k,v in merges.items() if v=='candidate_accepted'},'accepted/returned event mismatch')
        for ep in closed:
            k=ep.pop('candidate_key'); ep['merge_disposition']=merges[k] if k else 'not_applicable'
            ep['emitted_event_id']=returned.get(k)
            if ep['emitted_event_id'] is not None: linked_count+=1
            emit(ep)
        row_count+=1
    require(header is not None and footer_seen,'missing header/footer')
    return {'schema':'retention-linkage-validation-v1','valid':True,'trace_sha256':digest.hexdigest(),'episodes_sha256':episode_digest.hexdigest(),'plan_sha256':header['plan_sha256'],'pcm_sha256':header['pcm_sha256'],'recording_id':header['recording_id'],'bytes':total,'rows_committed':row_count,'records':seq,'events_emitted':event_count,'retained_path_events':linked_count,'summary':counts,'acoustic_truth_established':False,'causal_rescue_established':False,'production_qualified':False}


def compare_canonical_matches(bindings, references, control, candidate, episodes, admission, *, expected_trace_sha256, max_items=100000):
    """Compare already computed canonical matches; never perform acoustic matching.

Everything here is a consistency check over declared labels and execution
artifacts, not authentication of inference or individual causal necessity.
"""
    integer(max_items, 'max_items', 1)
    require(isinstance(bindings, dict), 'bindings required')
    for name in ('plan_sha256','pcm_sha256','references_sha256','source_split_sha256','reference_projection_sha256'):
        require(isinstance(bindings.get(name),str) and SHA.fullmatch(bindings[name]), 'invalid matcher binding')
    require(isinstance(admission,dict) and admission.get('schema')=='retention-linkage-validation-v1' and admission.get('valid') is True, 'admitted trace required')
    require(isinstance(expected_trace_sha256,str) and SHA.fullmatch(expected_trace_sha256) and admission.get('trace_sha256')==expected_trace_sha256, 'trace hash mismatch')
    require(all(admission.get(n)==bindings[n] for n in ('plan_sha256','pcm_sha256')), 'trace binding mismatch')
    require(isinstance(references,list) and len(references)<=max_items, 'reference bound')
    # Exact UTF-8 compact JSON list, sorted object keys, ASCII escaping, no newline.
    # The controller must separately establish derivation from references_sha256.
    projection=json.dumps(references,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')
    require(hashlib.sha256(projection).hexdigest()==bindings['reference_projection_sha256'], 'reference projection hash mismatch')
    refs={}
    for ref in references:
        require(isinstance(ref,dict), 'reference object')
        rid=ref.get('reference_id')
        require(isinstance(rid,str) and 0<len(rid)<=256 and rid not in refs, 'duplicate/invalid reference ID')
        require(ref.get('eligibility') in ('eligible','ambiguous','excluded'), 'explicit reference eligibility required')
        refs[rid]=ref['eligibility']
    recording=admission.get('recording_id')
    require(isinstance(recording,str) and recording, 'recording identity required')
    def event_id(value,arm):
        require(isinstance(value,list) and len(value)==3 and value[:2]==[recording,arm], 'event identity mismatch')
        integer(value[2],'event index')
        return tuple(value)
    maps={}; inventories={}
    for arm,report in (('control',control),('candidate',candidate)):
        require(isinstance(report,dict) and report.get('bindings')==bindings and report.get('matcher')=='canonical-reference-matcher-v1', 'canonical matcher binding mismatch')
        events=report.get('event_ids'); matches=report.get('matches')
        require(isinstance(events,list) and isinstance(matches,list) and len(events)<=max_items and len(matches)<=max_items,'matcher item bound')
        ids=[event_id(e,arm) for e in events]
        require(len(ids)==len(set(ids)), 'duplicate event ID')
        require({e[2] for e in ids} == set(range(len(ids))), 'noncontiguous event inventory')
        inventories[arm]=set(ids); used=set(); mapping={}
        for match in matches:
            require(isinstance(match,dict), 'match object')
            rid=match.get('reference_id'); eid=event_id(match.get('event_id'),arm)
            require(rid in refs and rid not in mapping and eid in inventories[arm] and eid not in used, 'unknown/duplicate match')
            mapping[rid]=eid; used.add(eid)
        maps[arm]=mapping
    require(len(inventories['candidate'])==admission.get('events_emitted'), 'candidate inventory disagrees with trace')
    require(isinstance(episodes,list) and len(episodes)<=max_items, 'episode bound')
    digest=hashlib.sha256(); linked=set()
    for ep in episodes:
        require(isinstance(ep,dict), 'episode object')
        digest.update((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
        if ep.get('emitted_event_id') is not None:
            eid=event_id(ep['emitted_event_id'],'candidate')
            require(eid in inventories['candidate'] and eid not in linked, 'unknown/duplicate linked event')
            linked.add(eid)
    require(digest.hexdigest()==admission.get('episodes_sha256'), 'episode digest mismatch')
    require(len(linked)==admission.get('retained_path_events'), 'linked count mismatch')
    result={n:[] for n in ('candidate_only','control_only','both','neither','retention_associated_candidate_only','excluded')}
    for rid,eligibility in refs.items():
        if eligibility!='eligible': result['excluded'].append(rid); continue
        c=rid in maps['control']; t=rid in maps['candidate']
        group='both' if c and t else 'control_only' if c else 'candidate_only' if t else 'neither'
        result[group].append(rid)
        if t and not c and maps['candidate'][rid] in linked: result['retention_associated_candidate_only'].append(rid)
    return {'schema':'declared-paired-match-comparison-v1','reference_ids':result,'counts':{k:len(v) for k,v in result.items()},'trace_sha256':expected_trace_sha256,'label_truth_verified':False,'individual_causal_necessity_proven':False,'production_qualified':False}
