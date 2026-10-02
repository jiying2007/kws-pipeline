"""Offline generic fixtures; never downloads or installs packages."""
import ast,hashlib,io,json,pathlib,sys,tempfile,unittest,zipfile
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import dependency_preflight as p
class Response(io.BytesIO):
    def __enter__(self):return self
    def __exit__(self,*a):self.close()
class Tests(unittest.TestCase):
    def proposal(self):return json.loads((ROOT/'proposal.freeze-unready.json').read_text())
    def test_fourteen_fixed_packages(self):
        x=self.proposal()['packages'];self.assertEqual(len(x),14)
        self.assertEqual([i['name'] for i in x if i['sha256'] is None],['torch','torchaudio'])
        self.assertTrue(all(i['version'] and i['filename'].endswith('.whl') for i in x))
    def test_no_gpu_or_model_dependencies(self):
        n={i['name'].lower() for i in self.proposal()['packages']}
        self.assertTrue(n.isdisjoint({'transformers','funasr','qwen-asr','huggingface-hub','triton'}))
        self.assertFalse(any(x.startswith(('nvidia','cuda')) for x in n))
    def test_pypi_hashes_match_manifest(self):
        data=(ROOT/'pypi-subset.cp311-linux-x86_64.requirements.txt').read_text()
        self.assertIn('--require-hashes',data);self.assertIn('--only-binary=:all:',data)
        for i in self.proposal()['packages']:
            if i['sha256']:self.assertIn(i['name']+'=='+i['version']+' --hash=sha256:'+i['sha256'],data)
    def test_cpu_index_exact_hash(self):
        item=self.proposal()['packages'][0];url=item['url'];h='a'*64
        with patch.object(p.urllib.request,'urlopen',return_value=Response(f'<a href="{url}#sha256={h}">x</a>'.encode())):
            self.assertEqual(p.get_cpu_index_entry(item),(url,h))
    def test_cpu_index_absent_hash(self):
        item=self.proposal()['packages'][0]
        with patch.object(p.urllib.request,'urlopen',return_value=Response(f'<a href="{item["url"]}">x</a>'.encode())):
            with self.assertRaises(RuntimeError):p.get_cpu_index_entry(item)
    def test_cpu_index_wrong_host(self):
        item=self.proposal()['packages'][0];url='https://example.invalid/'+item['filename']
        with patch.object(p.urllib.request,'urlopen',return_value=Response(f'<a href="{url}#sha256={"a"*64}">x</a>'.encode())):
            with self.assertRaises(RuntimeError):p.get_cpu_index_entry(item)
    def test_cpu_index_ambiguous_hash(self):
        item=self.proposal()['packages'][0];text=''.join(f'<a href="{item["url"]}#sha256={h*64}">x</a>' for h in ['a','b'])
        with patch.object(p.urllib.request,'urlopen',return_value=Response(text.encode())):
            with self.assertRaises(RuntimeError):p.get_cpu_index_entry(item)
    def test_cpu_index_wrong_python_ignored(self):
        item=self.proposal()['packages'][0];url=item['url'].replace('cp311','cp312')
        with patch.object(p.urllib.request,'urlopen',return_value=Response(f'<a href="{url}#sha256={"a"*64}">x</a>'.encode())):
            with self.assertRaises(RuntimeError):p.get_cpu_index_entry(item)
    def test_wheel_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            w=pathlib.Path(d)/'fixture.whl'
            with zipfile.ZipFile(w,'w') as z:z.writestr('fixture-1.dist-info/METADATA','Name: fixture\nVersion: 1\n')
            raw,m=p.read_metadata(w);self.assertEqual(m['Name'],'fixture');self.assertEqual(m['Version'],'1')
    def test_duplicate_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w=pathlib.Path(d)/'fixture.whl'
            with zipfile.ZipFile(w,'w') as z:
                for n in ['a','b']:z.writestr(n+'.dist-info/METADATA','Name: '+n)
            with self.assertRaises(RuntimeError):p.read_metadata(w)
    def test_source_has_explicit_boundaries(self):
        s=(ROOT/'dependency_preflight.py').read_text();ast.parse(s)
        self.assertIn("system_site_packages=False",s)
        self.assertIn("'--no-index'",s);self.assertIn("'--no-deps'",s)
        self.assertIn("'-I', '-m', 'pip'",s)
        self.assertIn("'model_or_audio_loaded':False",s)
        self.assertNotIn('torch.load(',s);self.assertNotIn('from_pretrained(',s)
    def test_execution_gate_default(self):
        with patch.object(sys,'argv',['dependency_preflight.py','--output','unused']):
            with self.assertRaises(SystemExit):p.main()
if __name__=='__main__':unittest.main()
