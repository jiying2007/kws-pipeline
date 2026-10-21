"""Small hand-derived arithmetic fixtures, never model/qualification evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'research/fsmn2599-stream-c'
spec = importlib.util.spec_from_file_location('binary32_reference', MODULE / 'binary32_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
NATIVE = False

# Literal expected bits, independently derived from powers of two and RNE.
# Half an ulp at 1 is 2^-24: even stays, odd increments. No host arithmetic.
ADD = (
    (0, 0, 0), (0x80000000, 0x80000000, 0x80000000),
    (0, 0x80000000, 0), (0x3f800000, 0xbf800000, 0),
    (0x3f800000, 0x33800000, 0x3f800000),
    (0x3f800001, 0x33800000, 0x3f800002),
    (0xbf800000, 0xb3800000, 0xbf800000),
    (1, 1, 2), (0x007fffff, 1, 0x00800000),
    (0x00800000, 0x80000001, 0x007fffff),
    (0x7f7fffff, 0x72800000, 0x7f7fffff),  # max + quarter ulp
)
MUL = (
    (0x80000000, 0xbf800000, 0), (0, 0xbf800000, 0x80000000),
    (0x3f800000, 0x3f800000, 0x3f800000),
    (1, 0x3f000000, 0), (0x80000001, 0x3f000000, 0x80000000),
    (3, 0x3f000000, 2), (0x007fffff, 0x3f800000, 0x007fffff),
    (0x00800000, 0x3f000000, 0x00400000),
    (0x00800000, 0x3f7fffff, 0x00800000),  # carry at subnormal tie
    (0x3f800001, 0x3f7ffffe, 0x3f800000),  # 1 - 2^-46 rounds to 1
    (0x7f7fffff, 0x3f800000, 0x7f7fffff),
)
# name, x, row-major rows, optional biases, expected outputs.
# All values are synthetic constants, ni <= 9, no <= 3.
AFFINE = (
    ('negative_zero', (0x80000000,), ((0x3f800000,),), None, (0,)),
    ('three_tail', (0x3f800000,)*3, ((0x3f800000,)*3,), None, (0x40400000,)),
    # (2^24 + 1) + (-2^24 - 1) = +0 after pair rounding;
    # serial reduction gives -1, cross-pairing also differs on the next case.
    ('tree_serial', (0x4b800000,0x3f800000,0xcb800000,0xbf800000),
     ((0x3f800000,)*4,), None, (0,)),
    ('tree_cross', (0x4b800000,0x3f800000,0xcb800000,0x3f800000),
     ((0x3f800000,)*4,), None, (0x3f800000,)),
    ('five_tail', (0x3f800000,)*5, ((0x3f800000,)*5,), None, (0x40a00000,)),
    # lane 0: -1 + RN((1+2^-23)*(1-2^-23)) = 0; FMA gives -2^-46.
    ('separate_product', (0x3f800000,0,0,0,0x3f800001),
     ((0xbf800000,0,0,0,0x3f7ffffe),), None, (0,)),
    # Injecting bias into lane 0 first loses 1 beside 2^24.
    ('bias_last', (0x4b800000,0,0,0,0xcb800000),
     ((0x3f800000,)*5,), (0x3f800000,), (0x3f800000,)),
    ('nine_three_rows', (0x3f800000,)*9,
     ((0x3f800000,)*9,(0xbf800000,)*9,(0,)*8+(0x40000000,)),
     (0,0x3f800000,0x3f800000), (0x41100000,0xc1000000,0x40400000)),
)


def affine(x, rows, biases):
    result = []
    for o, row in enumerate(rows):
        lanes = [0]*4
        for i, (value, weight) in enumerate(zip(x, row)):
            lanes[i % 4] = reference.add(lanes[i % 4], reference.mul(value, weight))
        result.append(reference.add(reference.add(reference.add(*lanes[:2]),
                                                 reference.add(*lanes[2:])),
                                    0 if biases is None else biases[o]))
    return tuple(result)


class Binary32Tests(unittest.TestCase):
    def test_literal_primitives(self):
        for operation, cases in ((reference.add, ADD), (reference.mul, MUL)):
            for a, b, expected in cases:
                with self.subTest(operation=operation.__name__, a=a, b=b):
                    self.assertEqual(operation(a, b), expected)

    def test_domain_and_overflow(self):
        invalid = (True, 1.0, None, -1, 0x100000000, 0x7f800000, 0xff800000, 0x7fc00001)
        for operation in (reference.add, reference.mul):
            for value in invalid:
                expected = TypeError if type(value) is not int else ValueError
                for args in ((value, 0), (0, value)):
                    with self.subTest(operation=operation.__name__, args=args):
                        with self.assertRaises(expected):
                            operation(*args)
        for operation, a, b in ((reference.add, 0x7f7fffff, 0x73000000),
                                (reference.add, 0xff7fffff, 0xf3000000),
                                (reference.mul, 0x7f7fffff, 0x40000000)):
            with self.assertRaises(OverflowError):
                operation(a, b)

    def test_literal_affine(self):
        self.assertLessEqual(len(AFFINE), 16)
        self.assertLessEqual(len(ADD) + len(MUL) + 32 + 3, 64)
        for name, x, rows, biases, expected in AFFINE:
            with self.subTest(name=name):
                self.assertLessEqual(len(x), 9)
                self.assertLessEqual(len(rows), 3)
                self.assertTrue(all(len(row) == len(x) for row in rows))
                self.assertEqual(affine(x, rows, biases), expected)

    @unittest.skipUnless('--execute-native' in sys.argv, 'explicit native opt-in required')
    def test_current_c_default_and_block2(self):
        # Exactly one current source region; never use a duplicate/cached C body.
        source = (MODULE / 'fsmn2599.c').read_text()
        start = '#ifdef KWS_AFFINE_BLOCK2\n'
        end = '\nstatic int finite_values('
        self.assertEqual(source.count(start), 1)
        self.assertEqual(source.count(end), 1)
        region = source.split(start)[1].split(end)[0]
        self.assertTrue(region.rstrip().endswith('#endif'))
        self.assertEqual(region.count('static void affine('), 2)
        body = '#include <stdint.h>\n#include <string.h>\n#include <float.h>\n#include <fenv.h>\n'
        body += '_Static_assert(sizeof(float)==4 && FLT_MANT_DIG==24 && FLT_RADIX==2 && FLT_EVAL_METHOD==0,"binary32 required");\n'
        body += start + region
        body += '\nint main(void) { if (fegetround()!=FE_TONEAREST) return 1;\n'
        body += 'volatile float tiny=FLT_MIN, half=0.5f; if (!(tiny*half>0.0f)) return 2;\n'
        def array(values):
            return '{' + ','.join('0x%08xu' % value for value in values) + '}'
        for index, (_, x, rows, biases, expected) in enumerate(AFFINE):
            weights = tuple(value for row in rows for value in row)
            body += '{ uint32_t xb[]='+array(x)+', wb[]='+array(weights)+', bb[]='+array(biases or (0,))+', expected[]='+array(expected)+';\n'
            body += 'float x[9], w[27], b[3], y[3]; uint32_t actual[3];\n'
            body += 'memcpy(x,xb,sizeof xb); memcpy(w,wb,sizeof wb); memcpy(b,bb,sizeof bb);\n'
            body += 'affine(x,y,w,%s,%d,%d); memcpy(actual,y,sizeof expected);\n' % ('b' if biases is not None else '0',len(x),len(rows))
            body += 'if(memcmp(actual,expected,sizeof expected)) return %d; }\n' % (10+index)
        body += 'return 0; }\n'
        with tempfile.TemporaryDirectory(prefix='binary32-fixture-') as tmp:
            path = Path(tmp)
            (path/'fixture.c').write_text(body)
            for mode in ('default', 'block2'):
                command = ['cc','-std=c11','-O2','-fno-fast-math','-ffp-contract=off']
                if mode == 'block2':
                    command += ['-DKWS_AFFINE_BLOCK2']
                command += ['fixture.c','-lm','-o',mode]
                # Fixed tiny source, no model calls, two compiles/runs; discard output.
                subprocess.run(command, cwd=path, check=True, timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                subprocess.run([str(path/mode)], check=True, timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    unittest.main(argv=[arg for arg in sys.argv if arg != '--execute-native'])
