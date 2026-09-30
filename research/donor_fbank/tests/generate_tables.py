"""Generate fixed frontend constants from independently retained official golden tables."""
import argparse,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--golden-constants',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();x=json.loads(a.golden_constants.read_text())
window=x['hamming400'];banks=x['mel80x257'];assert len(window)==400 and len(banks)==80 and all(len(b)==257 and b[-1]==0 for b in banks)
bins=[];weights=[];offsets=[0]
for bank in banks:
 for i,w in enumerate(bank):
  if w!=0:assert w>0;bins.append(i);weights.append(w)
 offsets.append(len(weights))
def array(name,kind,values):
 out=['static const '+kind+' '+name+'['+str(len(values))+'] = {']
 for i in range(0,len(values),8):out.append('  '+', '.join(float(v).hex()+'f' if kind=='float' else str(v) for v in values[i:i+8])+',')
 return '\n'.join(out+['};'])
s='/* Source-derived float32 frontend constants; generated, do not edit.\n * Official oracle constants SHA256 '+hashlib.sha256(a.golden_constants.read_bytes()).hexdigest()+'\n * No learned model weights; Hamming and mel filters only. */\n'
s+='\n'.join([array('donor_hamming','float',window),array('donor_mel_offsets','uint16_t',offsets),array('donor_mel_bins','uint16_t',bins),array('donor_mel_weights','float',weights)])+'\n';a.output.write_text(s);print('tables',len(weights),'nonzero mel weights',hashlib.sha256(a.output.read_bytes()).hexdigest())
