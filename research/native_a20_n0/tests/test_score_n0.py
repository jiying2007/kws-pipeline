"""Pure invented-logit trace tests; no model/frontend/decoder/audio execution."""
import copy,gzip,hashlib,importlib.util,json,os,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=pathlib.Path(os.environ.get('A20_N0_DATA_ROOT',ROOT.parents[2]/'kws-data/research/2026-10-04-n0-deterministic-controls'))
spec=importlib.util.spec_from_file_location('n0score',ROOT/'src/score_n0.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
def h(x):return hashlib.sha256(x.encode()).hexdigest()
def fixture(events=None):
    events=events or {}; m=json.loads((DATA/'metadata/verified-inputs.json').read_text());g=json.loads(gzip.decompress((DATA/'metadata/geometry.json.gz').read_bytes()))
    bindings={k:h('PURE TOY '+k) for k in S.HASH_KEYS if k not in ['manifest_sha256','geometry_sha256']}
    raw=[{'kind':'run_start','schema':S.RAW_SCHEMA,**bindings,'manifest_sha256':hashlib.sha256(S.canonical_json_bytes(m)).hexdigest(),'geometry_sha256':hashlib.sha256(S.canonical_json_bytes(g)).hexdigest()}, {'kind':'model_loaded','weights_bytes':1565280,'state_bytes':333704,'wall_ns_since_run_start':0,'cpu_ns_since_run_start':0}]
    dr=ev=0
    for row,geo in zip(m['rows'],g['rows']):
        name=row['recording'];identity={k:row[k] for k in ['recording','frames','wav_sha256','pcm_sha256']}; raw.append({'kind':'clip_start',**identity});dec=events_n=mr=0
        for p in geo['callback_plan']:
            kw=events.get((name,p['call_index']),0); n=p['selected_rows'];decoded=4 if kw else n
            raw.append({'kind':'callback','recording':name,**p,'valid':1,'state':int(bool(kw)),'keyword':kw,'start_frame':mr*3 if kw else -1,'end_frame':mr*3+9 if kw else -1,'score':.5 if kw else 0.,'decoder_rows_decoded':decoded,'logits':[[0.,0.,0.,0.,0.,0.] for _ in range(n)],'callback_entry_wall_ns_since_run_start':0,'callback_entry_cpu_ns_since_run_start':0})
            mr+=n;dec+=decoded;events_n+=int(bool(kw))
            raw.append({'kind':'feed','recording':name,'feed_index':p['call_index'],'input_samples':4800,'cumulative_samples':p['available_samples'],'callbacks_delta':1,'service_wall_ns_including_callback_output':0,'service_cpu_ns_including_callback_output':0})
        raw.append({'kind':'finish','recording':name,'finish_calls':1,'callbacks_delta':0,'service_wall_ns_including_callback_output':0,'service_cpu_ns_including_callback_output':0})
        raw.append({'kind':'clip_end',**identity,**{k:geo[k] for k in S.COUNT_FIELDS},'finish_calls':1,'decoder_rows_decoded':dec,'event_count':events_n,'complete':True}); dr+=dec;ev+=events_n
    raw.append({'kind':'run_end','complete':True,'clips':3,**g['totals'],'decoder_rows_decoded':dr,'event_count':ev,'wall_ns':0,'process_cpu_ns':0,'maxrss_kib':0,'minor_faults':0,'major_faults':0,'block_input_ops':0,'block_output_ops':0})
    return raw,m,g,bindings

def first(raw,kind):return next(x for x in raw if x['kind']==kind)
class Tests(unittest.TestCase):
    def test_empty_explicit_conditional_domain_only(self):
        report=S.score(*fixture());self.assertEqual(report['aggregate']['activation_events'],0);self.assertTrue(all(x['events']==[] for x in report['streams']));self.assertAlmostEqual(report['conditional_zero_count_poisson95_upper_per_hour'],11.9829290942);self.assertEqual(report['probabilities'],'NOT_CAPTURED')
    def test_events_retained_and_search_rows_observed(self):
        d=fixture({('n0-low-triangular',0):1,('n0-low-triangular',2):2});r=S.score(*d)
        self.assertEqual(r['aggregate']['activation_events'],2);self.assertEqual(r['aggregate']['event_rate_per_constructed_hour'],8);self.assertEqual(r['streams'][0]['event_rate_per_constructed_hour'],24);self.assertEqual(r['observed_totals']['decoder_rows_decoded'],29986);self.assertIsNone(r['conditional_zero_count_poisson95_upper_per_hour']);self.assertEqual([x['keyword'] for x in r['streams'][0]['events']],[1,2])
    def test_reject_mutations(self):
        mutations={
        'missing_callback_timing':lambda d:first(d[0],'callback').pop('callback_entry_wall_ns_since_run_start'),
        'string_cpu_timing':lambda d:first(d[0],'feed').update(service_cpu_ns_including_callback_output='0'),
        'missing_resource':lambda d:d[0][-1].pop('maxrss_kib'),
        'negative_timing':lambda d:first(d[0],'finish').update(service_wall_ns_including_callback_output=-1),
        'nonmonotonic_timing':lambda d:first(d[0],'callback').update(callback_entry_wall_ns_since_run_start=1),
        'missing_callback':lambda d:d[0].remove(first(d[0],'callback')),
        'duplicate_callback':lambda d:d[0].insert(4,copy.deepcopy(first(d[0],'callback'))),
        'missing_finish':lambda d:d[0].remove(first(d[0],'finish')),
        'missing_model_load':lambda d:d[0].remove(first(d[0],'model_loaded')),
        'model_size':lambda d:first(d[0],'model_loaded').update(weights_bytes=1),
        'truncated':lambda d:d[0].pop(),
        'trailing':lambda d:d[0].append(copy.deepcopy(d[0][-1])),
        'input_hash':lambda d:first(d[0],'clip_start').update(pcm_sha256=h('wrong')),
        'protocol_hash':lambda d:d[0][0].update(protocol_sha256=h('wrong')),
        'nonfinite':lambda d:first(d[0],'callback')['logits'][0].__setitem__(0,float('nan')),
        'clock':lambda d:first(d[0],'callback').update(decoder_total_frames=99),
        'source_center':lambda d:first(d[0],'callback')['centers'].__setitem__(0,1),
        'lost_search':lambda d:first(d[0],'callback').update(decoder_rows_decoded=1),
        'bad_geometry':lambda d:first(d[0],'callback').update(wave_samples=0),
        'bad_boundary':lambda d:first(d[0],'feed').update(cumulative_samples=100),
        'extra_finish_callback':lambda d:first(d[0],'finish').update(callbacks_delta=1),
        'false_domain':lambda d:d[1]['rows'][0].update(kind='verified_realworld_negative'),
        'incomplete':lambda d:first(d[0],'clip_end').update(complete=False),
        'shortened_stream':lambda d:d[1]['rows'][0].update(frames=4799999),
        'duplicate_seed':lambda d:d[1]['rows'][0].update(seed_hex=d[1]['rows'][1]['seed_hex'])}
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                d=fixture();mutate(d)
                with self.assertRaises(S.ValidationError):S.score(*d)
if __name__=='__main__':unittest.main()
