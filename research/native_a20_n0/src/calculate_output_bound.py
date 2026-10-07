"""Conservative source-format bound: no collector/runtime/audio execution."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
g=json.loads((ROOT/'metadata/geometry.json').read_text())
# All integer fields use<=20 chars; float logits%.9g finite binary32<=15chars.
# score%.17g finite binary64 worst-case sign+significand+3digit exponent<=24chars.
# Full callback syntax is reproduced with placeholders of conservative width.
s='9'*20;f='-'+'9'*14;score='-'+'9'*23;name='n0-sparse-transients'
fields={'kind':'callback','recording':name,'phase':'finish','call_index':s,'available_samples':s,'call_samples':s,'waveform_samples':s,'fbank_rows':s,'splice_rows':s,'selected_rows':s,'is_final_short':s,'wave_samples':s,'feature_count':s,'offset':s,'centers':[s]*10,'valid':s,'state':s,'keyword':s,'start_frame':'-'+'9'*19,'end_frame':'-'+'9'*19,'score':score,'decoder_rows_decoded':s,'decoder_total_frames':s,'callback_entry_wall_ns_since_run_start':s,'callback_entry_cpu_ns_since_run_start':s,'logits':[[f]*6 for _ in range(10)]}
# Frozen geometry provides tighter non-timing integer bounds, independent of activation.
for k in ['call_index','call_samples','waveform_samples','fbank_rows','splice_rows','selected_rows','is_final_short','wave_samples','feature_count','offset']:
    fields[k]=max(call[k] for row in g['rows'] for call in row['callback_plan'])
fields['available_samples']=4800000
fields['centers']=[29994]*10
for k,v in {'valid':1,'state':1,'keyword':2,'start_frame':29997,'end_frame':29997,'decoder_rows_decoded':10,'decoder_total_frames':29997}.items():fields[k]=v
# JSON string placeholders have quotes where C emits bare numeric: keeping quotes adds slack.
cb=len(json.dumps(fields,separators=(',',':')).encode())+1
feed={'kind':'feed','recording':name,'feed_index':999,'input_samples':4800,'cumulative_samples':4800000,'callbacks_delta':1,'service_wall_ns_including_callback_output':s,'service_cpu_ns_including_callback_output':s}
fbytes=len(json.dumps(feed,separators=(',',':')).encode())+1
bound=cb*3000+fbytes*3000+65536
r={'schema':'a20-n0-output-bound-v1','method':'source printf syntax conservatively represented in compactJSON with quoted numeric placeholders (extra2bytes each); timings20chars; remaining integers bounded by fixed geometry; %.9g finite binary32 logits15chars, %.17g finite score24chars; maximum10rows eachcallback fixed by geometry','callback_bound_bytes':cb,'callback_count':3000,'feed_bound_bytes':fbytes,'feed_count':3000,'all_other_raw_records_reserve_bytes':65536,'raw_bound_bytes':bound,'raw_per_file_hard_limit_bytes':6291456,'fits_6MiB':bound<=6291456,'geometry_totals':g['totals'],'note':'Counts and finite values enforced; malformed/nonfinite callback fails. Integer digit widths bound C types, not extrapolated timing. Bound overstates 3first9row callbacks.'}
(ROOT/'metadata/output-bound.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps(r,indent=2))
