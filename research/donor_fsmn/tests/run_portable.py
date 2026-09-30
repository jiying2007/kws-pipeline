"""No downloads, Torch, NumPy, donor weights or audio assets required."""
import pathlib,subprocess,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='donor-fsmn-') as d:
 out=pathlib.Path(d)/'test'
 cmd=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off',str(ROOT/'tests/test_portable.c'),str(ROOT/'pcm.c'),str(ROOT/'splice.c'),str(ROOT.parent/'donor_fbank/donor_fbank.c'),'-lm','-o',str(out)]
 subprocess.run(cmd,check=True);subprocess.run([str(out)],check=True)
print('PASS portable synthetic model/cache/PCM contracts; not donor numerical qualification')
