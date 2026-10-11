"""Native Whisper short decode, with an observational pre-finalize greedy trace.

Importing this module does not import any ML dependency. The runner first proves
its actual no-network container scope, exact source/runtime/asset bytes and input
identity. No transcribe(), window fallback, prompt, peer result, or target text.
"""
from dataclasses import asdict
from pathlib import Path
import os
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
from common import require, sha, canonical, file_hash, read_json, identity, verify_inventory, write_once
from asr_native_contracts import (WHISPER_REQUEST, EXECUTION, CONDITIONING_POLICY,
                                  canonical_sha256, validate_native_config, validate_native_capture)


def termination(trace, effective):
    """Interpret only the tokens actually observed before native finalize()."""
    require(type(trace) is list and bool(trace), 'at least one native greedy update')
    initial = effective['initial_token_ids']; previous = initial
    eot = effective['eot_token_id']; ended = False
    for index, step in enumerate(trace, 1):
        require(set(step) == {'step', 'token_ids', 'completed'} and step['step'] == index,
                'native update trace schema/order')
        ids = step['token_ids']
        require(not ended and type(ids) is list and ids[:-1] == previous and
                len(ids) == len(previous) + 1 and type(step['completed']) is bool,
                'one actual greedy token per update')
        require(step['completed'] == (ids[-1] == eot), 'native completion/EOT disagreement')
        previous = ids; ended = step['completed']
    generated = previous[len(initial):]
    stop = 'eot' if ended else ('context_limit' if len(previous) > effective['n_text_ctx'] else 'sample_limit')
    require(ended or len(trace) == effective['effective_sample_len'] or
            len(previous) > effective['n_text_ctx'], 'unexplained decoder return')
    return {'kind': 'whisper_greedy_pre_finalize', 'capture_stage': 'decoder_update_before_finalize',
            'initial_token_ids': initial, 'generated_token_ids': generated,
            'step_count': len(trace), 'pre_finalize_token_count': len(previous),
            'completed': ended, 'stop_reason': stop, 'truncated': not ended}



def flatten_pre_finalize(tokens):
    require(len(tokens.shape) == 3 and tuple(tokens.shape[:2]) == (1, 1),
            'one audio and one greedy group before finalize')
    return tokens[0, 0].detach().cpu().tolist()


def observed_effective(task, torch, model, tokenizer_sha):
    from whisper.decoding import GreedyDecoder, SuppressBlank, SuppressTokens
    require(type(task.decoder) is GreedyDecoder and task.decoder.temperature == 0.0,
            'native greedy decoder required')
    require(asdict(task.options) == WHISPER_REQUEST, 'actual native DecodingOptions differ')
    require(type(model).__module__ == 'whisper.model' and model.is_multilingual,
            'official multilingual model required')
    parameters = list(model.parameters())
    require(parameters and all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in parameters),
            'actual CPU float32 model parameters required')
    require(task.n_group == 1 and task.n_ctx == model.dims.n_text_ctx and
            task.sample_len == model.dims.n_text_ctx // 2, 'native greedy/sample geometry')
    filters = task.logit_filters
    require(len(filters) == 2 and type(filters[0]) is SuppressBlank and type(filters[1]) is SuppressTokens,
            'exact native no-timestamp logit filter sequence')
    tokenizer = task.tokenizer
    require(task.decoder.eot == tokenizer.eot and filters[0].tokenizer is tokenizer,
            'actual native EOT/blank tokenizer binding')
    require(tokenizer.language == 'zh' and tokenizer.task == 'transcribe' and
            filters[0].sample_begin == task.sample_begin, 'actual language/task/suppression onset')
    controls = {key: getattr(tokenizer, key) for key in
                ('transcribe', 'translate', 'sot', 'sot_prev', 'sot_lm', 'no_speech', 'no_timestamps')}
    controls['language'] = tokenizer.language_token
    return {'execution': dict(EXECUTION), 'native_request': asdict(task.options), 'decoder': 'greedy',
            'n_text_ctx': task.n_ctx, 'effective_sample_len': task.sample_len,
            'tokenizer_sha256': tokenizer_sha, 'non_speech_token_ids': sorted(tokenizer.non_speech_tokens),
            'control_token_ids': controls, 'resolved_suppress_token_ids': sorted(filters[1].suppress_tokens),
            'space_token_ids': tokenizer.encode(' '),
            'initial_blank_suppress_token_ids': sorted(set(tokenizer.encode(' ') + [tokenizer.eot])),
            'eot_token_id': tokenizer.eot, 'initial_token_ids': list(task.initial_tokens)}


class FixedWhisper:
    def __init__(self, runtime, output):
        self.runtime, self.output = Path(runtime), Path(output)
        source = self.runtime / 'source'
        source_lock = read_json(HERE / 'source-lock.json')
        expected = {r['path']: {'sha256': r['sha256'], 'bytes': r['size_bytes']} for r in source_lock['files']}
        verify_inventory(source, expected)
        installed = self.runtime / 'venv/lib/python3.12/site-packages'
        verify_inventory(installed, read_json(self.runtime / 'installed-inventory.json')['files'])
        asset = read_json(self.runtime / 'asset-lock.json')
        for name, row in asset['files'].items():
            path = self.runtime / 'model' / name if name == 'small.pt' else source / name
            require(identity(path) == row, 'loaded asset byte mismatch')
        from common import MODEL_SHA
        require(asset['files']['small.pt']['sha256'] == MODEL_SHA, 'small checkpoint only')
        sys.path.insert(0, str(source))
        # Every mutable cache is private scratch. No runtime/source writes are allowed.
        for key in ('HOME', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR', 'TORCH_HOME', 'TMPDIR'):
            target = Path('/scratch') / key.lower(); target.mkdir(exist_ok=True)
            os.environ[key] = str(target)
        import numpy as np
        import torch
        import whisper
        from whisper.decoding import DecodingTask, DecodingOptions
        require(Path(whisper.__file__).resolve() == source / 'whisper/__init__.py', 'loaded source origin')
        require(torch.version.cuda is None and not torch.cuda.is_available(), 'CPU-only wheel/runtime')
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        self.torch, self.np, self.whisper = torch, np, whisper
        self.model = whisper.load_model(str(self.runtime / 'model/small.pt'), device='cpu')
        self.model.eval()
        require(asdict(self.model.dims) == {'n_mels': 80, 'n_audio_ctx': 1500, 'n_audio_state': 768,
                'n_audio_head': 12, 'n_audio_layer': 12, 'n_vocab': 51865, 'n_text_ctx': 448,
                'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}, 'small native dimensions')
        self.make_task = lambda: DecodingTask(self.model, DecodingOptions(**WHISPER_REQUEST))
        probe_task = self.make_task()
        self.effective = observed_effective(probe_task, torch, self.model,
                                           file_hash(source / 'whisper/assets/multilingual.tiktoken'))
        self.config = {'schema_version': 2, 'evidence_class': 'speech-asr-native-config-v2',
            'adapter_id': 'openai-whisper-decode-short', 'adapter_version': 1, 'model_family': 'whisper',
            'source_lock_sha256': file_hash(HERE / 'source-lock.json'),
            'asset_lock_sha256': file_hash(self.runtime / 'asset-lock.json'),
            'conditioning_policy': CONDITIONING_POLICY, 'execution': dict(EXECUTION),
            'input_policy': {'sample_rate': 16000, 'channels': 1, 'max_audio_frames': 480000,
                'pcm_format': 'float32_pcm16_scale_32768', 'preprocessing': 'zero_pad_only_480000'},
            'native_request': dict(WHISPER_REQUEST), 'effective_config': self.effective,
            'effective_config_sha256': canonical_sha256(self.effective)}
        validate_native_config(self.config)
        # Effective config is observed and frozen after load, before any audio/model forward.
        write_once(self.output / 'native-config.json', self.config)
        write_once(self.output / 'load-receipt.json', {'status': 'loaded_before_first_audio',
            'model_dimensions': asdict(self.model.dims), 'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'source_lock_sha256': self.config['source_lock_sha256'],
            'asset_lock_sha256': self.config['asset_lock_sha256'],
            'installed_inventory_sha256': file_hash(self.runtime / 'installed-inventory.json'),
            'torch_version': torch.__version__, 'numpy_version': np.__version__, 'model_training': self.model.training,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'config_sha256': canonical_sha256(self.config), 'audio_calls': 0})

    def decode(self, pcm_bytes, frames, directory):
        np, torch = self.np, self.torch
        require(len(pcm_bytes) == frames * 2 and 0 < frames <= 480000, 'complete bounded PCM input')
        samples = np.frombuffer(pcm_bytes, dtype='<i2').astype(np.float32) / np.float32(32768.0)
        require(samples.shape == (frames,) and samples.dtype == np.float32, 'normalized PCM geometry')
        # Never call pad_or_trim: the explicit bound is checked before pad-only.
        padded = np.pad(samples, (0, 480000 - frames), mode='constant')
        require(np.array_equal(padded[:frames], samples) and not np.count_nonzero(padded[frames:]),
                'no source samples changed; zero padding only')
        mel = self.whisper.log_mel_spectrogram(padded, n_mels=self.model.dims.n_mels)
        require(tuple(mel.shape) == (80, 3000) and mel.dtype == torch.float32 and torch.isfinite(mel).all().item(),
                'complete native frontend tensor')
        input_capture = {'sample_rate': 16000, 'channels': 1, 'audio_frames': frames,
            'processed_start_frame': 0, 'processed_end_frame': frames, 'padding_frames': 480000 - frames,
            'trimmed_frames': 0, 'preprocessing': 'zero_pad_only_480000',
            'pcm_format': 'float32_pcm16_scale_32768'}
        write_once(directory / 'input-capture.json', {'native_input': input_capture,
            'pcm16_sha256': sha(pcm_bytes), 'float32_source_sha256': sha(samples.astype('<f4').tobytes()),
            'float32_padded_sha256': sha(padded.astype('<f4').tobytes()),
            'mel_float32_sha256': sha(mel.numpy().astype('<f4').tobytes()), 'mel_shape': list(mel.shape),
            'frontend_source_sha256': file_hash(self.runtime / 'source/whisper/audio.py'),
            'mel_filters_sha256': file_hash(self.runtime / 'source/whisper/assets/mel_filters.npz')})
        task = self.make_task()
        require(observed_effective(task, torch, self.model, self.effective['tokenizer_sha256']) == self.effective,
                'per-call actual effective configuration changed')
        trace = []; original_update, original_finalize = task.decoder.update, task.decoder.finalize
        final_tokens = []
        def capture_update(tokens, logits, sum_logprobs):
            new_tokens, completed = original_update(tokens, logits, sum_logprobs)
            require(new_tokens.shape[0] == 1, 'one blind utterance only')
            step = {'step': len(trace) + 1, 'token_ids': new_tokens[0].detach().cpu().tolist(),
                    'completed': bool(completed)}
            trace.append(step)
            # Durable per-step evidence survives timeout; no need to infer from finalized EOT.
            write_once(directory / f'update-{len(trace):03d}.json', step)
            return new_tokens, completed
        def capture_finalize(tokens, sum_logprobs):
            final_tokens.extend(flatten_pre_finalize(tokens))
            write_once(directory / 'pre-finalize.json', {'token_ids': final_tokens,
                'last_update_completed': trace[-1]['completed'] if trace else None,
                'trace_count': len(trace)})
            return original_finalize(tokens, sum_logprobs)
        task.decoder.update, task.decoder.finalize = capture_update, capture_finalize
        started = time.monotonic()
        with torch.no_grad():
            results = task.run(mel.unsqueeze(0))
        require(len(results) == 1 and trace and final_tokens == trace[-1]['token_ids'], 'pre-finalize/update binding')
        result = results[0]; stop = termination(trace, self.effective)
        actual_returned = list(result.tokens)
        generated = stop['generated_token_ids']
        expected_returned = generated[:generated.index(self.effective['eot_token_id'])] if self.effective['eot_token_id'] in generated else generated
        require(actual_returned == expected_returned and result.language == 'zh' and result.temperature == 0.0,
                'native result/token/language binding')
        require(result.text == task.tokenizer.decode(actual_returned).strip(), 'unaltered native output text')
        capture = {'schema_version': 2, 'evidence_class': 'speech-asr-native-capture-v2',
            'native_config_sha256': canonical_sha256(self.config),
            'source_lock_sha256': self.config['source_lock_sha256'], 'asset_lock_sha256': self.config['asset_lock_sha256'],
            'effective_config': self.effective, 'effective_config_sha256': self.config['effective_config_sha256'],
            'input': input_capture, 'termination': stop, 'transcript_postcorrection_applied': False,
            'target_context_received': False, 'peer_transcript_received': False,
            'acoustic_verbatim_completeness': 'unknown'}
        if stop['completed']:
            validate_native_capture(self.config, capture)
        features = result.audio_features.detach().cpu().contiguous()
        require(features.dtype == torch.float32 and tuple(features.shape) == (1500, 768) and
                result.language_probs is None, 'actual fixed native result tensor/language fields')
        raw = {'schema': 'whisper-screen-native-result-projection-v1',
               'projection_scope': 'All DecodingResult scalar/list fields; audio_features retained by exact float32 byte hash and shape.',
               'audio_features': {'shape': list(features.shape), 'dtype': str(features.dtype),
                   'device': str(features.device), 'float32_sha256': sha(features.numpy().astype('<f4').tobytes())},
               'language_probs': result.language_probs,
               'raw_text': result.text, 'returned_token_ids': actual_returned, 'language': result.language,
               'temperature': result.temperature, 'avg_logprob': result.avg_logprob,
               'no_speech_prob': result.no_speech_prob, 'compression_ratio': result.compression_ratio,
               'native_capture': capture, 'trace_files': len(trace), 'wall_seconds': time.monotonic() - started,
               'decoder_completion_validated': stop['completed'], 'full_input_processed': True,
               'acoustic_verbatim_completeness': 'unknown',
               'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        write_once(directory / 'raw-output.json', raw)
        return raw
