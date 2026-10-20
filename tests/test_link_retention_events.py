import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from link_retention_events import validate_stream, compare_canonical_matches, key

def bits(x): return struct.pack('>d',x).hex()
def candidate(row,window=0,value=.2): return dict(row=row,window_start=window,keyword_id=1,node_frames=[3*row-6,3*row-4,3*row-2,3*row],score_bits=bits(value))
def fixture(end='accepted', rows=2):
    # Three blank rows precede the first physically valid duration-six candidate.
    rows += 3
    header={'type':'header','schema':'retention-event-linkage-v1','run_id':'synthetic','recording_id':'invented','arm':'candidate','threshold':.7,'retention_rows':5,'expected_rows':rows,'max_bytes':1000000,'max_records':10000,'max_row_records':100}
    header.update({n:'a'*64 for n in ('pcm_sha256','plan_sha256','candidate_sha256','library_sha256','harness_sha256','ledger_abi_sha256')})
    docs=[header]; seq=0
    for row in range(rows):
        records=[{'kind':'row_trial_begin','row':row,'decoder_status':0}]; events=[]
        if row==3:
            c=candidate(row); records += [dict(c,kind='detect_candidate_before_local_reset'),dict(kind='retention_started',row=row,window_start=0),dict(c,kind='candidate_before_dedup'),dict(c,kind='candidate_before_threshold'),dict(c,kind='threshold_rejected')]
        elif row==4 and end!='active':
            c=candidate(row,value=.8); records += [dict(c,kind='detect_candidate_before_local_reset'),dict(kind='retention_strong_closed',row=row,window_start=0),dict(c,kind='candidate_before_dedup')]
            if end=='duplicate': records += [dict(c,kind='duplicate_rejected')]
            else:
                records += [dict(c,kind='candidate_before_threshold'),dict(c,kind='candidate_accepted' if end=='accepted' else 'refractory_rejected')]
                if end=='accepted': events=[dict(c,event_index=0)]
        records.append({'kind':'row_committed','row':row,'decoder_status':0})
        for record in records: record['sequence']=seq; seq+=1
        docs.append({'type':'row','row':row,'decoder_status':0,'records':records,'events':events})
    active=end=='active'; summary=dict(episodes_started=1,low_repeated=0,strong_closed=0 if active else 1,expired=0,window_censored=0,active_episodes=int(active),peak_active_episodes=1,rows_committed=rows)
    docs.append(dict(type='footer',valid=True,dropped=0,pending=0,exit_code=0,sink_closed=True,rows_committed=rows,sequence_count=seq,summary=summary,events_emitted=int(end=='accepted'),active_windows=[dict(window_start=0,anchor_row=3,deadline_row=8)] if active else []))
    return docs

def encode(docs): return ('\n'.join(json.dumps(x) for x in docs)+'\n').encode()
def run(docs):
    episodes=[]; result=validate_stream(io.BytesIO(encode(docs)),episodes.append)
    return result,episodes

def resequence(docs):
    seq=0
    for row in docs[1:-1]:
        for record in row['records']: record['sequence']=seq; seq+=1
    docs[-1]['sequence_count']=seq

class LinkageTests(unittest.TestCase):
    def test_accepted_link_is_not_truth(self):
        result,eps=run(fixture()); self.assertEqual(result['retained_path_events'],1); self.assertEqual(eps[0]['emitted_event_id'],['invented','candidate',0]); self.assertFalse(result['causal_rescue_established'])
    def test_duplicate_and_refractory_closes_without_event(self):
        for end in ('duplicate','refractory'):
            result,eps=run(fixture(end)); self.assertIsNone(eps[0]['emitted_event_id']); self.assertEqual(result['retained_path_events'],0)
    def test_input_end_stays_active(self):
        result,eps=run(fixture('active')); self.assertEqual(eps[0]['closure'],'active_at_input_end'); self.assertEqual(result['summary']['active_episodes'],1)
    def test_failed_row_trial_not_admitted(self):
        d=fixture(); d[4]['decoder_status']=3
        with self.assertRaisesRegex(ValueError,'failed row'): run(d)
    def test_commit_missing_extra_or_failure(self):
        for variant in ('missing','extra','failed'):
            d=fixture()
            if variant=='missing': d[4]['records'].pop()
            elif variant=='extra': d[4]['records'].append(copy.deepcopy(d[4]['records'][-1]))
            else: d[4]['records'][-1]['decoder_status']=2
            resequence(d)
            with self.assertRaises(ValueError): run(d)
    def test_gap_or_overflow_or_bad_footer(self):
        for field,value in [('dropped',1),('pending',1),('exit_code',1),('valid',False),('sink_closed',False),('sequence_count',999)]:
            d=fixture(); d[-1][field]=value
            with self.assertRaises(ValueError): run(d)
        d=fixture(); d[4]['records'][2]['sequence']+=1
        with self.assertRaises(ValueError): run(d)
        with self.assertRaises(ValueError): run(fixture()[:-1])
    def test_counters_snapshot_and_event_total(self):
        d=fixture(); d[-1]['summary']['episodes_started']=2
        with self.assertRaises(ValueError): run(d)
        d=fixture('active'); d[-1]['active_windows'][0]['anchor_row']=1
        with self.assertRaises(ValueError): run(d)
        d=fixture(); d[-1]['events_emitted']=0
        with self.assertRaises(ValueError): run(d)
    def test_exact_score_bits_and_window_join(self):
        for name,value in [('score_bits',bits(.8000000000000002)),('window_start',1),('event_index',2)]:
            d=fixture(); d[5]['events'][0][name]=value
            with self.assertRaises(ValueError): run(d)
    def test_ambiguous_detect_and_missing_merge(self):
        d=fixture(); d[4]['records'].insert(2,copy.deepcopy(d[4]['records'][1])); resequence(d)
        with self.assertRaises(ValueError): run(d)
        d=fixture(); del d[5]['records'][-2]; resequence(d)
        with self.assertRaises(ValueError): run(d)
    def test_expiry_precedence_and_low_repeat(self):
        d=fixture('active',rows=6)
        row=d[9]; c=candidate(8)
        row['records'][1:1]=[dict(c,kind='detect_candidate_before_local_reset'),dict(kind='retention_low_repeated',row=8,window_start=0),dict(kind='retention_expired',row=8,window_start=0),dict(c,kind='candidate_before_dedup'),dict(c,kind='candidate_before_threshold'),dict(c,kind='threshold_rejected')]
        d[-1]['summary'].update(low_repeated=1,expired=1,active_episodes=0); d[-1]['active_windows']=[]; resequence(d)
        result,eps=run(d); self.assertEqual(eps[0]['closure'],'expired'); self.assertEqual(eps[0]['low_repeat_count'],1)
        row['records'][2]['kind']='retention_strong_closed'
        with self.assertRaises(ValueError): run(d)
    def test_window_censor_and_new_episode(self):
        d=fixture('active',rows=3); d[5]['records'].insert(1,dict(kind='retention_window_censored',row=4,window_start=0))
        c=candidate(5); d[6]['records'][1:1]=[dict(c,kind='detect_candidate_before_local_reset'),dict(kind='retention_started',row=5,window_start=0),dict(c,kind='candidate_before_dedup'),dict(c,kind='candidate_before_threshold'),dict(c,kind='threshold_rejected')]
        d[-1]['summary'].update(episodes_started=2,window_censored=1); d[-1]['active_windows']=[dict(window_start=0,anchor_row=5,deadline_row=10)]; resequence(d)
        _,eps=run(d); self.assertEqual([e['closure'] for e in eps],['window_censored','active_at_input_end']); self.assertNotEqual(eps[0]['episode_id'],eps[1]['episode_id'])
    def test_bounded_and_json_integrity(self):
        for name,value in [('max_bytes',10),('max_records',1),('max_row_records',1)]:
            d=fixture(); d[0][name]=value
            with self.assertRaises(ValueError): run(d)
        with self.assertRaises(ValueError): validate_stream(io.BytesIO(encode(fixture())),byte_limit=100)
        raw=encode(fixture()).replace(b'"valid": true',b'"valid": true, "valid": true')
        with self.assertRaises(ValueError): validate_stream(io.BytesIO(raw))
        with self.assertRaises(ValueError): validate_stream(io.BytesIO(encode(fixture())+b'{}\n'))
    def test_impossible_candidate_clocks_rejected(self):
        for change in ('future_window','future_node','prewindow_node','too_short','too_long'):
            d=fixture()
            for rec in d[5]['records'] + d[5]['events']:
                if 'score_bits' not in rec: continue
                if change=='future_window': rec['window_start']=5
                elif change=='future_node': rec['node_frames']=[6,8,10,13]
                elif change=='prewindow_node': rec['window_start']=3; rec['node_frames']=[6,8,10,12]
                elif change=='too_short': rec['node_frames']=[8,9,10,12]
                else: rec['node_frames']=[0,1,2,251]
            with self.assertRaises(ValueError): run(d)

    def test_candidate_duration_boundaries(self):
        for duration in (5,250):
            rec=candidate(100); rec['node_frames']=[0,1,2,duration]
            self.assertEqual(key(rec,100)[3][-1],duration)
        for duration in (4,251):
            rec=candidate(100); rec['node_frames']=[0,1,2,duration]
            with self.assertRaisesRegex(ValueError,'duration'): key(rec,100)

    def test_multi_window_same_dedup_score_identity(self):
        d=fixture('active',rows=2)
        for row in d[4:6]:
            existing=row['records'][1:-1]
            if row['row']==3:
                copied=[]
                for r in existing:
                    rr=copy.deepcopy(r); rr['window_start']=1; copied.append(rr)
                row['records'][-1:-1]=copied
            else:
                for w,terminal in [(0,'candidate_accepted'),(1,'duplicate_rejected')]:
                    c=candidate(4,w,.8)
                    entries=[dict(c,kind='detect_candidate_before_local_reset'),dict(kind='retention_strong_closed',row=4,window_start=w),dict(c,kind='candidate_before_dedup')]
                    if terminal=='candidate_accepted': entries.append(dict(c,kind='candidate_before_threshold')); row['events'].append(dict(c,event_index=0))
                    entries.append(dict(c,kind=terminal)); row['records'][-1:-1]=entries
        d[-1]['summary'].update(episodes_started=2,strong_closed=2,active_episodes=0,peak_active_episodes=2); d[-1]['active_windows']=[]; d[-1]['events_emitted']=1; resequence(d)
        _,eps=run(d); self.assertEqual(len(eps),2); self.assertEqual(sum(e['emitted_event_id'] is not None for e in eps),1)

class PairedJoinTests(unittest.TestCase):
    def setup_bundle(self):
        admission,eps=run(fixture())
        bindings={n:'a'*64 for n in ('plan_sha256','pcm_sha256','references_sha256','source_split_sha256')}
        refs=[dict(reference_id=n,eligibility='ambiguous' if n=='amb' else 'eligible') for n in ('rescue','lost','both','neither','amb')]
        bindings['reference_projection_sha256']=hashlib.sha256(json.dumps(refs,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')).hexdigest()
        control=dict(bindings=bindings,matcher='canonical-reference-matcher-v1',event_ids=[['invented','control',i] for i in range(2)],matches=[dict(reference_id='lost',event_id=['invented','control',0]),dict(reference_id='both',event_id=['invented','control',1])])
        # Trace fixture has one candidate event. Other reference matches remain
        # deliberately absent; both/ambiguous classification tested separately.
        candidate=dict(bindings=bindings,matcher='canonical-reference-matcher-v1',event_ids=[['invented','candidate',0]],matches=[dict(reference_id='rescue',event_id=['invented','candidate',0])])
        return bindings,refs,control,candidate,eps,admission
    def compare(self,bundle):
        return compare_canonical_matches(*bundle,expected_trace_sha256=bundle[-1]['trace_sha256'])
    def test_paired_categories_and_association(self):
        b=self.setup_bundle(); r=self.compare(b)
        self.assertEqual(r['counts']['candidate_only'],1); self.assertEqual(r['counts']['control_only'],2)
        self.assertEqual(r['reference_ids']['retention_associated_candidate_only'],['rescue'])
        self.assertEqual(r['counts']['excluded'],1)
        b[3]['matches'][0]['reference_id']='both'; r=self.compare(b)
        self.assertEqual(r['counts']['both'],1); self.assertEqual(r['counts']['candidate_only'],0)
        b[3]['matches'][0]['reference_id']='amb'; r=self.compare(b)
        self.assertEqual(r['counts']['candidate_only'],0); self.assertEqual(r['counts']['retention_associated_candidate_only'],0)
    def test_reject_partial_forged_or_unbound_links(self):
        for mutation in ('invalid','trace','episode','binding','unknown','duplicate','inventory'):
            b=self.setup_bundle()
            if mutation=='invalid': b[-1]['valid']=False
            elif mutation=='trace': b[-1]['trace_sha256']='bad'
            elif mutation=='episode': b[-2][0]['closure_row']=999
            elif mutation=='binding': b[3]['bindings']=dict(b[0],pcm_sha256='b'*64)
            elif mutation=='unknown': b[3]['matches'][0]['reference_id']='unknown'
            elif mutation=='duplicate': b[3]['matches'].append(copy.deepcopy(b[3]['matches'][0]))
            else: b[3]['event_ids']=[]
            with self.assertRaises(ValueError): self.compare(b)
    def test_reference_projection_tampering(self):
        b=self.setup_bundle(); b[1][0]['eligibility']='excluded'
        with self.assertRaisesRegex(ValueError,'projection hash'): self.compare(b)
        b=self.setup_bundle(); b[1][0]['reference_id']='changed'
        with self.assertRaisesRegex(ValueError,'projection hash'): self.compare(b)
    def test_item_bounds_and_unknown_event_id(self):
        b=self.setup_bundle(); b[3]['matches'][0]['event_id'][-1]=999
        with self.assertRaises(ValueError): self.compare(b)
        b=self.setup_bundle()
        with self.assertRaises(ValueError): compare_canonical_matches(*b,expected_trace_sha256=b[-1]['trace_sha256'],max_items=1)

if __name__=='__main__': unittest.main()
