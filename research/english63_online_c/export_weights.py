#!/usr/bin/env python3
"""Convert two fixed saved NPZs to exact FP32 C constants; no Torch execution."""
import argparse
import io
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT/'research/english63_weak_pilot'):
    sys.path.insert(0, str(directory))
import contract as c

PARAMETERS = {
    'gru': [('rnn.weight_ih_l0', (96, 32)), ('rnn.weight_hh_l0', (96, 32)),
            ('rnn.bias_ih_l0', (96,)), ('rnn.bias_hh_l0', (96,)),
            ('classifier.weight', (3, 64)), ('classifier.bias', (3,))],
    'dsc_tcn': [('proj.weight', (32, 32, 1)), ('proj.bias', (32,))]
        + [(f'blocks.{i}.{name}', shape) for i in range(3) for name, shape in
           [('dw.weight', (32, 1, 3)), ('dw.bias', (32,)), ('pw.weight', (32, 32, 1)), ('pw.bias', (32,))]]
        + [('classifier.weight', (3, 64)), ('classifier.bias', (3,))],
}


def source_result(root):
    root = Path(root)
    summary_path = root/'model/SUMMARY.json'
    c.require(summary_path.is_file() and c.digest(summary_path.read_bytes()) == c.SOURCE_SUMMARY_SHA,
              'C_EXPORT_ORIGINAL_TRAINING_SUMMARY')
    summary = c.read_json(summary_path)
    for kind, sha in c.SOURCE_WEIGHTS.items():
        path = root/'model'/(kind+'-weights.npz')
        c.require(path.is_file() and not path.is_symlink() and path.stat().st_size < 500000
                  and c.digest(path.read_bytes()) == sha, 'C_EXPORT_FIXED_WEIGHT_IDENTITY')
    return summary


def array(name, values, kind='float'):
    if kind == 'float':
        literals = [float(x).hex()+'f' for x in values]
    else:
        literals = [str(int(x)) for x in values]
    lines = [', '.join(literals[i:i+6]) for i in range(0, len(literals), 6)]
    return f'static const {kind} {name}[{len(values)}] = {{\n    '+',\n    '.join(lines)+'\n};\n'


def export(trained, output):
    import numpy as np
    import pcm_frontend_parity as frontend
    c.require(np.__version__ == '2.3.5', 'C_EXPORT_PINNED_NUMPY')
    source = source_result(trained)
    old_path = ROOT/'research/english63_weak_pilot/RESULT.json'
    c.require(c.digest(old_path.read_bytes()) == source['code_sha256']['research/english63_weak_pilot/RESULT.json'],
              'C_EXPORT_FROZEN_NORMALIZER')
    norm = c.read_json(old_path)['train_only_frontend_norm']
    mean, std = (np.asarray(norm[k], dtype=np.float32) for k in ('mean32', 'std32'))
    bank = frontend.mel_bank()
    window = np.hanning(400).astype(np.float32)
    starts, counts, offsets, weights = [], [], [0], []
    for row in bank:
        indices = np.flatnonzero(row)
        start, end = int(indices[0]), int(indices[-1])+1
        starts.append(start); counts.append(end-start)
        weights.extend(row[start:end].tolist()); offsets.append(len(weights))
    text = ['/* Exact saved FP32 weights, no optimization or quantization. */\n',
            '#ifndef EN63_GENERATED_H\n#define EN63_GENERATED_H\n#include "en63_stream.h"\n']
    for name, data in (('window', window), ('mean', mean), ('std', std), ('mel_weights', weights)):
        text.append(array('en63_'+name, data))
    for name, data in (('mel_start', starts), ('mel_count', counts), ('mel_offset', offsets)):
        text.append(array('en63_'+name, data, 'uint16_t'))
    models = []
    for kind in c.KINDS:
        path = Path(trained)/'model'/(kind+'-weights.npz')
        arrays = []
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
            c.require(set(archive.files) == {n for n, _ in PARAMETERS[kind]}, 'C_EXPORT_TENSOR_NAMES')
            for name, shape in PARAMETERS[kind]:
                value = archive[name]
                c.require(value.dtype == np.float32 and value.shape == shape and bool(np.isfinite(value).all()),
                          'C_EXPORT_TENSOR_VALUES')
                arrays.append(value.reshape(-1))
        flat = np.concatenate(arrays).astype('<f4')
        text.append(array('en63_'+kind+'_parameters', flat))
        tag = 'EN63_GRU' if kind == 'gru' else 'EN63_DSC_TCN'
        text.append(f'static const en63_model en63_{kind}_model = {{\n'
                    f'    {tag}, {len(flat)}, en63_{kind}_parameters, en63_window, en63_mean, en63_std,\n'
                    f'    en63_mel_start, en63_mel_count, en63_mel_offset, en63_mel_weights, {len(weights)},\n'
                    f'    "{c.SOURCE_WEIGHTS[kind]}"\n}};\n')
        models.append({'kind': kind, 'source_weights_sha256': c.SOURCE_WEIGHTS[kind],
                       'parameter_count': len(flat), 'parameter_bytes': len(flat)*4,
                       'parameter_payload_sha256': c.digest(flat.tobytes()),
                       'tensor_order': [{'name': n, 'shape': list(s)} for n, s in PARAMETERS[kind]]})
    table_bytes = 4*(400+64+len(weights))+2*(32+32+33)
    text.append(f'#define EN63_FRONTEND_TABLE_BYTES {table_bytes}u\n#endif\n')
    output = Path(output)
    c.require(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'C_EXPORT_CREATE_ONLY')
    output.mkdir()
    header = output/'en63_generated.h'
    header.write_text(''.join(text))
    report = {'schema': 'english63-exact-fp32-c-array-export-v1', 'source_training_run_id': 43595338774,
              'source_summary_sha256': c.SOURCE_SUMMARY_SHA, 'models': models,
              'frontend_table_bytes': table_bytes, 'header_sha256': c.digest(header.read_bytes()),
              'optimizer_updates': 0, 'model_forward_calls': 0, 'quantization_performed': False,
              'formal_training_admitted': False, 'model_promotion_allowed': False}
    c.write_json(output/'EXPORT.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained-result-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    export(args.trained_result_dir, args.output_dir)
