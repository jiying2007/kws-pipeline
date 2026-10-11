#!/usr/bin/env python3
"""Stdlib source-shape arithmetic only: no Torch, tensors, models, audio or probes."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.candidate_encoders.contract import (
    EncoderSpec, MAX_SPEC_BYTES, causal_ds_tcn_spec, load_spec, loads_spec,
    research_export_layout, rnn64_spec, spec_from_dict, spec_sha256,
    static_costs, tensor_inventory,
)


class EncoderContractTests(unittest.TestCase):
    def setUp(self):
        # Toy fixture dimensions test arithmetic, never a selected recipe.
        self.c1 = causal_ds_tcn_spec(hidden_dim=4, kernels=(3, 1), dilations=(2, 7))

    def test_c0_is_existing_exact_reference_shape(self):
        c0 = rnn64_spec()
        self.assertEqual((c0.feature_dim, c0.hidden_dim, c0.vocab_size), (32, 64, 5))
        self.assertEqual(c0.history_frames, ())
        self.assertEqual(c0.state_shapes(2), ((2, 64),))
        self.assertEqual(c0.weight_elements, 6464)
        self.assertEqual(c0.bias_elements, 69)
        self.assertEqual(c0.parameter_elements, 6533)
        self.assertEqual(c0.cache_elements, 64)
        self.assertEqual([row['name'] for row in tensor_inventory(c0)], [
            'in_proj.weight', 'in_proj.bias', 'rec_proj.weight',
            'out_proj.weight', 'out_proj.bias'])

    def test_c1_declares_every_layer_and_zero_history(self):
        self.assertEqual(self.c1.history_frames, (4, 0))
        self.assertEqual(self.c1.state_shapes(3), ((3, 4, 4), (3, 4, 0)))
        self.assertEqual(self.c1.weight_elements, 196)
        self.assertEqual(self.c1.bias_elements, 25)
        self.assertEqual(self.c1.parameter_elements, 221)
        self.assertEqual(self.c1.cache_elements, 16)
        with self.assertRaises(TypeError):
            causal_ds_tcn_spec()

    def test_strict_numeric_types_and_shape_fields(self):
        good = self.c1.to_dict()
        for name in ('feature_dim', 'hidden_dim', 'vocab_size'):
            for value in (True, False, float(good[name]), str(good[name]), None, [], {}):
                bad = copy.deepcopy(good)
                bad[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    spec_from_dict(bad)
        for name in ('kernels', 'dilations'):
            for value in (True, 1.0, '1', None, [], {}):
                bad = copy.deepcopy(good)
                bad[name][0] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    spec_from_dict(bad)
            for value in ((), '1', None, {}, False):
                bad = copy.deepcopy(good)
                bad[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    spec_from_dict(bad)
        for batch in (True, 1.0, '1', 0, -1, 257, None):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                self.c1.state_shapes(batch)

    def test_closed_schema_and_architecture(self):
        good = self.c1.to_dict()
        for key in good:
            bad = copy.deepcopy(good)
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                spec_from_dict(bad)
        for key in ('execution_enabled', 'shipping_approved', 'right_context', 'pretrained'):
            bad = copy.deepcopy(good)
            bad[key] = True
            with self.subTest(extra=key), self.assertRaises(ValueError):
                spec_from_dict(bad)
        for value in ('causal_fsmn_ctc', 'C1', '', True, [], None):
            bad = copy.deepcopy(good)
            bad['architecture'] = value
            with self.subTest(architecture=value), self.assertRaises(ValueError):
                spec_from_dict(bad)
        for value in ('other', 1, True, None):
            bad = copy.deepcopy(good)
            bad['schema'] = value
            with self.subTest(schema=value), self.assertRaises(ValueError):
                spec_from_dict(bad)

    def test_no_c0_recipe_drift(self):
        for args in [ ('rnn64_control', 40, 64, 5), ('rnn64_control', 32, 32, 5),
                      ('rnn64_control', 32, 64, 6),
                      ('rnn64_control', 32, 64, 5, (1,), (1,)) ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                EncoderSpec(*args)

    def test_defensive_limits_are_not_resource_admission(self):
        cases = [dict(hidden_dim=0, kernels=(3,), dilations=(1,)),
                 dict(hidden_dim=1025, kernels=(3,), dilations=(1,)),
                 dict(hidden_dim=4, kernels=(), dilations=()),
                 dict(hidden_dim=4, kernels=(3,), dilations=(1, 2)),
                 dict(hidden_dim=4, kernels=(0,), dilations=(1,)),
                 dict(hidden_dim=4, kernels=(64,), dilations=(1,)),
                 dict(hidden_dim=4, kernels=(3,), dilations=(0,)),
                 dict(hidden_dim=4, kernels=(3,), dilations=(1025,)),
                 dict(hidden_dim=4, kernels=(1,) * 33, dilations=(1,) * 33),
                 dict(hidden_dim=1024, kernels=(1,), dilations=(1,)),
                 dict(hidden_dim=32, kernels=(63,), dilations=(1024,)),
                 dict(hidden_dim=4, kernels=[3], dilations=[1])]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                causal_ds_tcn_spec(**kwargs)
        cache_heavy = causal_ds_tcn_spec(hidden_dim=16, kernels=(63,), dilations=(1024,))
        self.assertEqual(cache_heavy.cache_elements, 1015808)
        with self.assertRaises(ValueError):
            cache_heavy.state_shapes(2)
        self.assertFalse(static_costs(self.c1)['resource_admitted'])

    def test_json_rejects_duplicate_keys_nonfinite_and_oversize(self):
        raw = json.dumps(self.c1.to_dict()).encode()
        self.assertEqual(loads_spec(raw), self.c1)
        bads = [raw.replace(b'"schema":', b'"schema": "ignored", "schema":'),
                raw.replace(b'"hidden_dim": 4', b'"hidden_dim": NaN'),
                raw.replace(b'"hidden_dim": 4', b'"hidden_dim": Infinity'),
                b'{}' + b' ' * MAX_SPEC_BYTES, b'\xff', b'[' * 2000, b'null']
        for bad in bads:
            with self.subTest(raw=bad[:70]), self.assertRaises(ValueError):
                loads_spec(bad)
        with self.assertRaises(ValueError):
            loads_spec(raw.decode())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'spec.json'
            path.write_bytes(raw)
            self.assertEqual(load_spec(path), self.c1)
            path.write_bytes(b' ' * (MAX_SPEC_BYTES + 1))
            with self.assertRaises(ValueError):
                load_spec(path)

    def test_identity_is_canonical_and_binds_structure(self):
        reordered = dict(reversed(list(self.c1.to_dict().items())))
        self.assertEqual(spec_sha256(self.c1), spec_sha256(spec_from_dict(reordered)))
        changed = causal_ds_tcn_spec(hidden_dim=4, kernels=(3, 1), dilations=(1, 7))
        self.assertNotEqual(spec_sha256(self.c1), spec_sha256(changed))
        self.assertEqual(len(spec_sha256(self.c1)), 64)

    def test_inventory_and_layout_are_float32_description_only(self):
        for spec in (rnn64_spec(), self.c1):
            layout = research_export_layout(spec)
            json.dumps(layout, allow_nan=False)
            self.assertEqual(layout['artifact_kind'], 'JSON-description-only-no-weight-payload')
            self.assertEqual(layout['hypothetical_float32_payload_bytes'], 4 * spec.parameter_elements)
            self.assertEqual(sum(row['bytes'] for row in layout['logical_tensors']), 4 * spec.parameter_elements)
            offset = 0
            for row in layout['logical_tensors']:
                self.assertEqual(row['hypothetical_payload_offset'], offset)
                self.assertEqual(offset % 4, 0)
                self.assertEqual(row['dtype'], 'float32')
                offset += row['bytes']
            for key in ('payload_written', 'abi_assigned', 'kwsp_compatible',
                        'candidate_c_runtime_available', 'execution_authorized',
                        'training_ready', 'product_qualified'):
                self.assertIs(layout[key], False)
            self.assertIsNone(layout['binary_header_bytes'])
            self.assertEqual(layout['numerical_parity'], 'NOT_RUN')
            self.assertEqual(layout['candidate_c_export_parity'], 'NOT_RUN')
        self.assertFalse(research_export_layout(rnn64_spec())['c0_reference']['supplied_instance_origin_verified'])
        self.assertEqual(research_export_layout(self.c1)['chunk_output_order'], 'T,B,V')
        self.assertEqual(research_export_layout(self.c1)['eof'], 'no-invented-frames-or-flush')

    def test_static_costs_separate_unknown_resources_and_non_macs(self):
        costs = static_costs(self.c1)
        self.assertEqual(costs['macs_per_frame'], 196)
        self.assertEqual(costs['macs_per_audio_second_at_assumed_hop'], 9800)
        self.assertEqual(costs['non_mac_ops_per_frame'], dict(
            bias_adds=25, residual_or_recurrent_adds=8, relu_elements=20, tanh_elements=0))
        self.assertEqual(costs['float32_state_bytes_per_stream'], 64)
        self.assertEqual(costs['encoder_receptive_field_frames'], 5)
        self.assertEqual(costs['encoder_right_context_frames'], 0)
        for key in ('quantized_weights_bytes', 'scale_bytes', 'scratch_bytes', 'frontend_ring_bytes',
                    'total_runtime_arena_bytes', 'measured_cpu', 'measured_latency_ms', 'measured_peak_rss_bytes'):
            self.assertIsNone(costs[key])
        c0 = static_costs(rnn64_spec())
        self.assertTrue(c0['recurrent_history_unbounded'])
        self.assertIsNone(c0['encoder_receptive_field_frames'])
        self.assertEqual(c0['non_mac_ops_per_frame'], dict(
            bias_adds=69, residual_or_recurrent_adds=64, relu_elements=0, tanh_elements=64))

    def test_contract_import_has_no_model_or_third_party_import(self):
        script = '''
import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'training'}:
            raise RuntimeError('forbidden import ' + fullname)
sys.meta_path.insert(0, Guard())
from research.candidate_encoders import rnn64_spec
from research.candidate_encoders.contract import research_export_layout
assert research_export_layout(rnn64_spec())['numerical_parity'] == 'NOT_RUN'
'''
        result = subprocess.run([sys.executable, '-S', '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
