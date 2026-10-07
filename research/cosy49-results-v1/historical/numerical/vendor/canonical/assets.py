"""Weight-binding adapter; table and inert oracle loading bodies unchanged."""
import hashlib,json,math,re,sys,types
from pathlib import Path
import numpy as np
STAGE=Path(__file__).resolve().parents[2]
ROOT=STAGE/"vendor/historical/kws-native-a20-research-v1"
HISTORICAL_CONTRACT_SHA = '8cdaf377fa7bf4efaf2be72f9271659373e64c32f3181a3375365a915eefd23f'
PINS = {
    'frontend-diagnosis/decimal_dft.py': 'b48c28657aabb91cd99c75a625dfcc0a50363749e30f30b97ccae05c8c50f789',
    'oracle-corrected/oracle/decimal_fsmn.py': '198900485f2ae9dcdcb73fdf4a5c29932bd17b6e1065b6c7aab930eab9601f71',
    'export/manifest.json': 'fa10e0a11750e2b3e215c44f340e80bfdec1b374c68cd03eacd23b07b119dcea',
    'export/a20.f32': 'a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d',
    'native/donor_fbank/frontend_tables.h': '269c14d6aacbe189e9672ee58f207915f9fbb09e9db988340b1f2f494ac7ae0b',
    '../kws-alternate-fsmn-precheck-v1/reference/upstream/wekws/bin/stream_kws_ctc.py': '2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53',
    '../kws-alternate-fsmn-precheck-v1/decoder_tail_fix.py': '5fc55921a555c78cb08c12018850652d98d4283594f875366284dc1df0e71f4d',
}
TABLES = {
    'donor_hamming': ('<f4', 400, '30f68a101c19dd95343f14ddbbe94eceb18e1fb0be17a182b676f2bd28321cc0'),
    'donor_mel_weights': ('<f4', 501, 'd88295eef8478942df2226ba581d0a817925914d663b9192ff047624e9781337'),
    'donor_mel_offsets': ('<u2', 81, '1aba9b6be1bc2a64c01957c932adc5b2ba20796a0be4d70985e95d039db67d8d'),
    'donor_mel_bins': ('<u2', 501, 'a921dacf01825de80f75f364c9b2510d3a3ee6406f08ca2a39c066cff83e1538'),
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    return digest(Path(path).read_bytes())

def inert_module(name, relative):
    path = ROOT / relative
    assert sha(path) == PINS[relative]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module

def load_tables():
    path = ROOT / 'native/donor_fbank/frontend_tables.h'
    assert sha(path) == PINS[str(path.relative_to(ROOT))]
    text, out = path.read_text(), {}
    for name, (dtype, count, expected) in TABLES.items():
        pattern = r'\b' + name + r'\[' + str(count) + r'\]\s*=\s*\{([^}]*)\}'
        found = re.findall(pattern, text, re.S)
        assert len(found) == 1
        words = [v.strip() for v in found[0].split(',') if v.strip()]
        assert len(words) == count
        if dtype == '<f4':
            values = [float.fromhex(v.removesuffix('f')) for v in words]
            array = np.array(values, dtype=dtype)
            assert all(float(v) == f for v, f in zip(array, values))
        else:
            array = np.array([int(v) for v in words], dtype=dtype)
        assert digest(array.tobytes()) == expected
        array.flags.writeable = False
        out[name] = array
    weights, offsets, bins = (out['donor_mel_' + key] for key in ['weights', 'offsets', 'bins'])
    assert np.isfinite(out['donor_hamming']).all()
    assert np.all((out['donor_hamming'] > 0) & (out['donor_hamming'] <= 1))
    assert np.isfinite(weights).all() and np.all((weights > 0) & (weights <= 1))
    assert int(offsets[0]) == 0 and int(offsets[-1]) == 501
    delta = np.diff(offsets.astype(int))
    assert np.all(delta > 0) and np.max(delta) <= 16 and np.all(bins < 257)
    for m in range(80):
        row = bins[int(offsets[m]):int(offsets[m + 1])]
        assert len(set(map(int, row))) == len(row)
    return out

def check_pins():
    import common
    common.identities(common.RELEASE)
    result={}
    for relative,expected in PINS.items():
        p=ROOT/relative
        assert sha(p)==expected
        result[str(p.relative_to(STAGE))]=expected
    bundle=common.bundle()
    result['candidate_payload']=bundle['payload']['sha256']
    result['candidate_manifest']=bundle['manifest']['sha256']
    result['new_candidate_numerical_protocol']=common.CONTRACT_SHA
    return result

def contract_binding():
    import common
    common.identities(common.RELEASE)
    return dict(recovery_contract_sha256=common.CONTRACT_SHA,historical_v3_sha256=HISTORICAL_CONTRACT_SHA,
                semantics='unchanged recovered canonical authority; candidate weight identity extension',
                old_original_A20_numerical_verdict_inherited=False)

def load_state():
    import common
    b=common.bundle();manifest=json.loads(Path(b['manifest']['path']).read_text());raw=Path(b['payload']['path']).read_bytes()
    schema=json.loads((STAGE/'metadata/A20-TENSOR-SCHEMA.json').read_text())
    assert manifest['state_sha256']==common.RELEASE['terminal_state_sha256'] and manifest['symbols']==['<blank>','你','好','小','窝','屋']
    assert len(raw)==manifest['payload_bytes']==1565280 and digest(raw)==manifest['payload_sha256']==b['payload']['sha256']
    assert len(manifest['tensors'])==len(schema['tensors'])==30
    state={};h=hashlib.sha256();offset=0
    for e,original in zip(manifest['tensors'],schema['tensors']):
        assert e['name']==original['name'] and e['shape']==original['shape'] and e['bytes']==original['bytes'] and e['offset']==offset
        block=raw[offset:offset+e['bytes']];assert digest(block)==e['sha256']
        if e['name'].startswith('global_cmvn.'):assert e['sha256']==original['sha256']
        a=np.frombuffer(block,dtype='<f4').reshape(e['shape']);assert np.isfinite(a).all() and not a.flags.writeable
        state[e['name']]=a;h.update(e['name'].encode());h.update(block);offset+=len(block)
    assert offset==len(raw) and h.hexdigest()==manifest['state_sha256'] and np.all(state['global_cmvn.istd']>0)
    return state,manifest
