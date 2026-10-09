"""Pure file/header negative tests; never execute a foreign ELF."""
import pathlib, struct, tempfile, unittest
from native_elf import read_elf, native_expected, require_native, check_elf

class NativeGuard(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.path=pathlib.Path(self.temp.name)/'mock';self.raw=bytearray(pathlib.Path('/proc/self/exe').read_bytes());self.expected=native_expected();self.endian='<' if self.expected['endian']=='little' else '>'
 def tearDown(self):self.temp.cleanup()
 def reject(self,raw):
  self.path.write_bytes(raw)
  with self.assertRaises(ValueError):require_native(self.path)
 def test_native_process(self):
  self.path.write_bytes(self.raw);self.assertEqual(require_native(self.path)['machine'],self.expected['machine'])
 def test_foreign_machine(self):
  struct.pack_into(self.endian+'H',self.raw,18,40 if self.expected['machine']!=40 else 62);self.reject(self.raw)
 def test_wrong_class(self):
  self.raw[4]=1 if self.raw[4]==2 else 2;self.reject(self.raw)
 def test_wrong_endian(self):
  self.raw[5]=2 if self.raw[5]==1 else 1;self.reject(self.raw)
 def test_wrong_osabi(self):self.raw[7]=255;self.reject(self.raw)
 def test_wrong_abi_version(self):self.raw[8]=(self.raw[8]+1)%256;self.reject(self.raw)
 def test_object_cannot_execute(self):struct.pack_into(self.endian+'H',self.raw,16,1);self.reject(self.raw)
 def test_truncated(self):self.reject(self.raw[:20])
 def test_not_elf(self):self.reject(b'#!/bin/sh\nexit 0\n')
 def test_bad_program_bounds(self):
  off=32 if self.raw[4]==2 else 28;fmt='Q' if self.raw[4]==2 else 'I';struct.pack_into(self.endian+fmt,self.raw,off,len(self.raw)*4);self.reject(self.raw)
 def test_arm_abi_flags(self):
  raw=bytearray(52);raw[:7]=b'\x7fELF\x01\x01\x01';struct.pack_into('<HHI',raw,16,1,40,1);struct.pack_into('<I',raw,36,0x05000000);struct.pack_into('<H',raw,40,52);self.path.write_bytes(raw)
  expected={'class':32,'endian':'little','machine':40,'osabi':0,'abi_version':0,'flags':0x04000000}
  with self.assertRaises(ValueError):check_elf(self.path,expected,(1,))
 def test_shared_library_must_be_dyn(self):
  self.path.write_bytes(self.raw);header=read_elf(self.path);struct.pack_into(self.endian+'H',self.raw,16,2);self.path.write_bytes(self.raw)
  with self.assertRaises(ValueError):require_native(self.path,(3,))
if __name__=='__main__':unittest.main()
