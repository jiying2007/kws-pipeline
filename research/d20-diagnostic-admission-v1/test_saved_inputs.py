"""Ordinary valid-format byte/metadata fixtures only; no backend or codec probes."""
from contextlib import ExitStack
import hashlib
import inspect
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import contract
import saved_inputs as saved


def npy(dtype, shape, payload):
    header = repr(dict(descr=dtype, fortran_order=False, shape=shape)).encode('ascii')
    header += b' ' * (117 - len(header)) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + payload


class SavedInputTests(unittest.TestCase):
    def test_frozen_metadata_matches_existing_contract(self):
        self.assertEqual(set(saved.SOURCE_FILES), set(contract.FROZEN_SOURCES))
        self.assertEqual(tuple(size for _, size in saved.SOURCE_FILES.values()),
                         (301434, 442116, 8034, 1565280))
        self.assertEqual(tuple(inspect.signature(saved.bind_saved).parameters), ('root',))

    def test_regular_saved_byte_read(self):
        # This private helper verifies arbitrary small bytes, not D20 authority.
        raw = b'ordinary byte identity fixture\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            path.write_bytes(raw)
            self.assertEqual(saved._read_frozen(path, len(raw), hashlib.sha256(raw).hexdigest()), raw)

    def test_regular_file_identity_mismatch_stops_before_parsing(self):
        raw = b'ordinary valid byte identity fixture\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            path.write_bytes(raw)
            for size, digest in ((len(raw), '0' * 64),
                                 (len(raw) + 1, hashlib.sha256(raw).hexdigest())):
                with self.subTest(size=size, digest=digest), self.assertRaises(ValueError):
                    saved._read_frozen(path, size, digest)

    def test_fourth_source_failure_precedes_all_parsing_and_output(self):
        # An orchestration-only failure: no archive is constructed or decoded.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, output = Path(directory) / 'saved', Path(directory) / 'new-output'
            reads = stack.enter_context(mock.patch.object(saved, '_read_frozen', side_effect=[
                b'opaque first file', b'opaque second file', b'opaque third file',
                ValueError('fourth frozen identity mismatch')]))
            forbidden = [stack.enter_context(mock.patch.object(owner, name))
                         for owner, name in ((saved, '_arrays'), (saved.json, 'loads'),
                                             (saved, '_weights'), (saved, '_build_plan'),
                                             (saved.json, 'dumps'), (Path, 'mkdir'), (Path, 'open'))]
            receipt = stack.enter_context(mock.patch('builtins.print'))
            stack.enter_context(mock.patch('sys.argv', ['saved_inputs.py', '--saved-root', str(root),
                                                        '--output', str(output)]))
            with self.assertRaisesRegex(ValueError, 'fourth frozen identity mismatch'):
                saved.main()
            self.assertEqual(reads.call_args_list,
                             [mock.call(root / path, size, contract.FROZEN_SOURCES[name])
                              for name, (path, size) in saved.SOURCE_FILES.items()])
            for operation in forbidden:
                operation.assert_not_called()
            receipt.assert_not_called()
            self.assertFalse(output.exists())

    def test_positive_negative_zero_preserved(self):
        positive, negative = bytes.fromhex('00000000'), bytes.fromhex('00000080')
        raw = positive + negative + negative + positive
        payload, start = saved._npy_payload(npy('<f4', (2, 2), raw), '<f4', (2, 2))
        self.assertEqual(start, 128)
        self.assertEqual(payload, raw)
        first, first_offset = saved._row(payload, (2, 2), 0)
        last, last_offset = saved._row(payload, (2, 2), 1)
        self.assertEqual((first, first_offset), (positive + negative, 0))
        self.assertEqual((last, last_offset), (negative + positive, 8))
        self.assertNotEqual(saved._sha(first), saved._sha(last))

    def test_valid_counts_payload(self):
        raw = struct.pack('<qq', 9, 10)
        payload, offset = saved._npy_payload(npy('<i8', (2,), raw), '<i8', (2,))
        self.assertEqual((payload, offset), (raw, 128))

    def test_all_global_row_and_stage_mappings(self):
        self.assertEqual([saved._row_identity(row) for row in range(19)],
                         [(0, row) for row in range(9)] + [(1, row) for row in range(10)])
        for part in (0, 1):
            self.assertEqual(saved._input_key(part, 0), f'cmvn{part}')
            self.assertEqual([saved._input_key(part, stage) for stage in range(1, 21)],
                             [f'stage{part}_{stage}' for stage in range(20)])

    def test_observed_array_metadata(self):
        c, torch = saved._layout('c_npz'), saved._layout('torch_npz')
        self.assertEqual((len(c), len(torch)), (58, 52))
        self.assertEqual(set(c) - set(torch),
                         {'cmvn_same0', 'cmvn_same1', 'cache_before_0', 'cache_before_1',
                          'cache_after_0', 'cache_after_1'})
        self.assertEqual(c['cmvn0'], ('<f4', (9, 400)))
        self.assertEqual(c['cmvn1'], ('<f4', (10, 400)))
        self.assertEqual(c['cache_before_1'], ('<f4', (10, 128, 11, 4)))

    def test_weight_stage_and_fixed_offset_identities(self):
        offsets, offset = {}, 0
        for name, shape in saved._weight_inventory():
            offsets[name] = offset
            size = 4
            for dimension in shape:
                size *= dimension
            offset += size
        self.assertEqual((len(offsets), offset), (30, 1565280))
        self.assertEqual(offsets['backbone.in_linear1.linear.weight'], 3200)
        self.assertEqual(offsets['backbone.fsmn.0.1.conv_left.weight'], 496760)
        self.assertEqual(offsets['backbone.fsmn.3.1.conv_right.weight'], 1291312)
        self.assertEqual(offsets['backbone.out_linear2.linear.weight'], 1561896)
        self.assertEqual(offsets['backbone.out_linear2.linear.bias'], 1565256)
        for stage in (2, 6, 10, 14, 18):
            self.assertEqual(saved._stage_weights(stage), ())
        for layer, stage in enumerate((4, 8, 12, 16)):
            self.assertEqual(saved._stage_weights(stage),
                             (f'backbone.fsmn.{layer}.1.conv_left.weight',
                              f'backbone.fsmn.{layer}.1.conv_right.weight'))
        used = {name for stage in range(21) for name in saved._stage_weights(stage)}
        self.assertEqual(used, set(offsets) - {'global_cmvn.mean', 'global_cmvn.istd'})

    def test_plan_pair_slices_are_byte_identical_without_authenticity(self):
        # <1 MiB of opaque array slices, no arithmetic, archive codec or model.
        arrays = {}
        for part, count in enumerate((9, 10)):
            keys = [saved._input_key(part, stage) for stage in range(21)] + [f'cache_before_{part}']
            for key in keys:
                shape = saved._layout('c_npz')[key][1]
                width = 1
                for dimension in shape[1:]:
                    width *= dimension
                arrays[key] = b''.join(struct.pack('<I', row + 1 + 16 * part) * width
                                        for row in range(count))
        # No actual weights are supplied; this private mapper cannot authenticate.
        names = {name: {} for name, _ in saved._weight_inventory()}
        plan, payload = saved._build_plan(arrays, b'', names)
        result = contract.validate(plan)
        self.assertFalse(result['input_authenticity_verified'])
        self.assertFalse(result['execution_ready'])
        self.assertFalse(result['numerical_admission'])
        self.assertEqual(len(plan['jobs']), 798)
        for a, b in zip(plan['jobs'][::2], plan['jobs'][1::2]):
            self.assertEqual((a['backend'], b['backend']), ('C', 'Torch'))
            self.assertEqual(a['input_binding'], b['input_binding'])
            part, row = saved._row_identity(a['row'])
            binding = a['input_binding']
            source, offset = saved._row(arrays[binding['array_key']],
                                       saved._layout('c_npz')[binding['array_key']][1], row)
            self.assertEqual(binding['array_byte_offset'], offset)
            self.assertEqual(binding['member_byte_offset'], offset + 128)
            self.assertEqual(payload[binding['bundle_offset']:binding['bundle_offset'] + binding['bytes']], source)
            if a['stage'] in contract.MEMORY_STAGES:
                self.assertEqual(a['cache_binding'], b['cache_binding'])
                self.assertEqual(a['cache_binding']['array_key'], f'cache_before_{part}')
                self.assertEqual(a['cache_bytes'], 22528)
                self.assertEqual(a['cache_layer'], (a['stage'] - 4) // 4)


if __name__ == '__main__':
    unittest.main()
