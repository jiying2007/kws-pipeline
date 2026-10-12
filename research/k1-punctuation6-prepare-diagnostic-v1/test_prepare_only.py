"""Offline only: contracts, source call graph and constructor sentinels, no ML."""
import ast
import hashlib
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare_contract as c
import prepare_worker as worker
import prepare_host as host


# Exact function/accessor source segments from the retained official Whisper
# files. Tests compile only these ASTs; they do not import Whisper or Torch.
LOCKED_TOKENIZER_FILE_SHA256 = '3b48e361a7e95b4ec0356ca6d72bba635778aa10269153136ee7bc34cae30b85'
LOCKED_TOKENIZER_ACCESSORS = 'class Tokenizer:\n    @cached_property\n    def transcribe(self) -> int:\n        return self.special_tokens["<|transcribe|>"]\n    def decode(self, token_ids: List[int], **kwargs) -> str:\n        token_ids = [t for t in token_ids if t < self.timestamp_begin]\n        return self.encoding.decode(token_ids, **kwargs)\n    @cached_property\n    def timestamp_begin(self) -> int:\n        return self.special_tokens["<|0.00|>"]\n'
LOCKED_TRANSCRIBE_FILE_SHA256 = '176f5ab30e961bec2a7872f5701a1f5bf3cbfc77946e50512f0f0cf00a1ceacf'
LOCKED_TRANSCRIBE_FUNCTION = 'def transcribe(\n    model: "Whisper",\n    audio: Union[str, np.ndarray, torch.Tensor],\n    *,\n    verbose: Optional[bool] = None,\n    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),\n    compression_ratio_threshold: Optional[float] = 2.4,\n    logprob_threshold: Optional[float] = -1.0,\n    no_speech_threshold: Optional[float] = 0.6,\n    condition_on_previous_text: bool = True,\n    initial_prompt: Optional[str] = None,\n    carry_initial_prompt: bool = False,\n    word_timestamps: bool = False,\n    prepend_punctuations: str = "\\"\'“¿([{-",\n    append_punctuations: str = "\\"\'.。,，!！?？:：”)]}、",\n    clip_timestamps: Union[str, List[float]] = "0",\n    hallucination_silence_threshold: Optional[float] = None,\n    **decode_options,\n):\n    """\n    Transcribe an audio file using Whisper\n\n    Parameters\n    ----------\n    model: Whisper\n        The Whisper model instance\n\n    audio: Union[str, np.ndarray, torch.Tensor]\n        The path to the audio file to open, or the audio waveform\n\n    verbose: bool\n        Whether to display the text being decoded to the console. If True, displays all the details,\n        If False, displays minimal details. If None, does not display anything\n\n    temperature: Union[float, Tuple[float, ...]]\n        Temperature for sampling. It can be a tuple of temperatures, which will be successively used\n        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.\n\n    compression_ratio_threshold: float\n        If the gzip compression ratio is above this value, treat as failed\n\n    logprob_threshold: float\n        If the average log probability over sampled tokens is below this value, treat as failed\n\n    no_speech_threshold: float\n        If the no_speech probability is higher than this value AND the average log probability\n        over sampled tokens is below `logprob_threshold`, consider the segment as silent\n\n    condition_on_previous_text: bool\n        if True, the previous output of the model is provided as a prompt for the next window;\n        disabling may make the text inconsistent across windows, but the model becomes less prone to\n        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.\n\n    word_timestamps: bool\n        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,\n        and include the timestamps for each word in each segment.\n\n    prepend_punctuations: str\n        If word_timestamps is True, merge these punctuation symbols with the next word\n\n    append_punctuations: str\n        If word_timestamps is True, merge these punctuation symbols with the previous word\n\n    initial_prompt: Optional[str]\n        Optional text to provide as a prompt for the first window. This can be used to provide, or\n        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns\n        to make it more likely to predict those word correctly.\n\n    carry_initial_prompt: bool\n        If carry_initial_prompt is True, `initial_prompt` is prepended to the prompt of each internal\n        `decode()` call. If there is not enough context space at the start of the prompt, it is\n        left-sliced to make space.\n\n    decode_options: dict\n        Keyword arguments to construct `DecodingOptions` instances\n\n    clip_timestamps: Union[str, List[float]]\n        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.\n        The last end timestamp defaults to the end of the file.\n\n    hallucination_silence_threshold: Optional[float]\n        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)\n        when a possible hallucination is detected\n\n    Returns\n    -------\n    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and\n    the spoken language ("language"), which is detected when `decode_options["language"]` is None.\n    """\n    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32\n    if model.device == torch.device("cpu"):\n        if torch.cuda.is_available():\n            warnings.warn("Performing inference on CPU when CUDA is available")\n        if dtype == torch.float16:\n            warnings.warn("FP16 is not supported on CPU; using FP32 instead")\n            dtype = torch.float32\n\n    if dtype == torch.float32:\n        decode_options["fp16"] = False\n\n    # Pad 30-seconds of silence to the input audio, for slicing\n    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)\n    content_frames = mel.shape[-1] - N_FRAMES\n    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)\n\n    if decode_options.get("language", None) is None:\n        if not model.is_multilingual:\n            decode_options["language"] = "en"\n        else:\n            if verbose:\n                print(\n                    "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"\n                )\n            mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)\n            _, probs = model.detect_language(mel_segment)\n            decode_options["language"] = max(probs, key=probs.get)\n            if verbose is not None:\n                print(\n                    f"Detected language: {LANGUAGES[decode_options[\'language\']].title()}"\n                )\n\n    language: str = decode_options["language"]\n    task: str = decode_options.get("task", "transcribe")\n    tokenizer = get_tokenizer(\n        model.is_multilingual,\n        num_languages=model.num_languages,\n        language=language,\n        task=task,\n    )\n\n    if isinstance(clip_timestamps, str):\n        clip_timestamps = [\n            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])\n        ]\n    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]\n    if len(seek_points) == 0:\n        seek_points.append(0)\n    if len(seek_points) % 2 == 1:\n        seek_points.append(content_frames)\n    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))\n\n    punctuation = "\\"\'“¿([{-\\"\'.。,，!！?？:：”)]}、"\n\n    if word_timestamps and task == "translate":\n        warnings.warn("Word-level timestamps on translations may not be reliable.")\n\n    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:\n        temperatures = (\n            [temperature] if isinstance(temperature, (int, float)) else temperature\n        )\n        decode_result = None\n\n        for t in temperatures:\n            kwargs = {**decode_options}\n            if t > 0:\n                # disable beam_size and patience when t > 0\n                kwargs.pop("beam_size", None)\n                kwargs.pop("patience", None)\n            else:\n                # disable best_of when t == 0\n                kwargs.pop("best_of", None)\n\n            options = DecodingOptions(**kwargs, temperature=t)\n            decode_result = model.decode(segment, options)\n\n            needs_fallback = False\n            if (\n                compression_ratio_threshold is not None\n                and decode_result.compression_ratio > compression_ratio_threshold\n            ):\n                needs_fallback = True  # too repetitive\n            if (\n                logprob_threshold is not None\n                and decode_result.avg_logprob < logprob_threshold\n            ):\n                needs_fallback = True  # average log probability is too low\n            if (\n                no_speech_threshold is not None\n                and decode_result.no_speech_prob > no_speech_threshold\n                and logprob_threshold is not None\n                and decode_result.avg_logprob < logprob_threshold\n            ):\n                needs_fallback = False  # silence\n            if not needs_fallback:\n                break\n\n        return decode_result\n\n    clip_idx = 0\n    seek = seek_clips[clip_idx][0]\n    input_stride = exact_div(\n        N_FRAMES, model.dims.n_audio_ctx\n    )  # mel frames per output token: 2\n    time_precision = (\n        input_stride * HOP_LENGTH / SAMPLE_RATE\n    )  # time per output token: 0.02 (seconds)\n    all_tokens = []\n    all_segments = []\n    prompt_reset_since = 0\n\n    remaining_prompt_length = model.dims.n_text_ctx // 2 - 1\n    if initial_prompt is not None:\n        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())\n        all_tokens.extend(initial_prompt_tokens)\n        remaining_prompt_length -= len(initial_prompt_tokens)\n    else:\n        initial_prompt_tokens = []\n\n    def new_segment(\n        *, start: float, end: float, tokens: torch.Tensor, result: DecodingResult\n    ):\n        tokens = tokens.tolist()\n        text_tokens = [token for token in tokens if token < tokenizer.eot]\n        return {\n            "seek": seek,\n            "start": start,\n            "end": end,\n            "text": tokenizer.decode(text_tokens),\n            "tokens": tokens,\n            "temperature": result.temperature,\n            "avg_logprob": result.avg_logprob,\n            "compression_ratio": result.compression_ratio,\n            "no_speech_prob": result.no_speech_prob,\n        }\n\n    # show the progress bar when verbose is False (if True, transcribed text will be printed)\n    with tqdm.tqdm(\n        total=content_frames, unit="frames", disable=verbose is not False\n    ) as pbar:\n        last_speech_timestamp = 0.0\n        # NOTE: This loop is obscurely flattened to make the diff readable.\n        # A later commit should turn this into a simpler nested loop.\n        # for seek_clip_start, seek_clip_end in seek_clips:\n        #     while seek < seek_clip_end\n        while clip_idx < len(seek_clips):\n            seek_clip_start, seek_clip_end = seek_clips[clip_idx]\n            if seek < seek_clip_start:\n                seek = seek_clip_start\n            if seek >= seek_clip_end:\n                clip_idx += 1\n                if clip_idx < len(seek_clips):\n                    seek = seek_clips[clip_idx][0]\n                continue\n            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)\n            window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)\n            segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)\n            mel_segment = mel[:, seek : seek + segment_size]\n            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE\n            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)\n\n            if carry_initial_prompt:\n                nignored = max(len(initial_prompt_tokens), prompt_reset_since)\n                remaining_prompt = all_tokens[nignored:][-remaining_prompt_length:]\n                decode_options["prompt"] = initial_prompt_tokens + remaining_prompt\n            else:\n                decode_options["prompt"] = all_tokens[prompt_reset_since:]\n\n            result: DecodingResult = decode_with_fallback(mel_segment)\n            tokens = torch.tensor(result.tokens)\n\n            if no_speech_threshold is not None:\n                # no voice activity check\n                should_skip = result.no_speech_prob > no_speech_threshold\n                if (\n                    logprob_threshold is not None\n                    and result.avg_logprob > logprob_threshold\n                ):\n                    # don\'t skip if the logprob is high enough, despite the no_speech_prob\n                    should_skip = False\n\n                if should_skip:\n                    seek += segment_size  # fast-forward to the next segment boundary\n                    continue\n\n            previous_seek = seek\n            current_segments = []\n\n            # anomalous words are very long/short/improbable\n            def word_anomaly_score(word: dict) -> float:\n                probability = word.get("probability", 0.0)\n                duration = word["end"] - word["start"]\n                score = 0.0\n                if probability < 0.15:\n                    score += 1.0\n                if duration < 0.133:\n                    score += (0.133 - duration) * 15\n                if duration > 2.0:\n                    score += duration - 2.0\n                return score\n\n            def is_segment_anomaly(segment: Optional[dict]) -> bool:\n                if segment is None or not segment["words"]:\n                    return False\n                words = [w for w in segment["words"] if w["word"] not in punctuation]\n                words = words[:8]\n                score = sum(word_anomaly_score(w) for w in words)\n                return score >= 3 or score + 0.01 >= len(words)\n\n            def next_words_segment(segments: List[dict]) -> Optional[dict]:\n                return next((s for s in segments if s["words"]), None)\n\n            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)\n            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]\n\n            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]\n            consecutive.add_(1)\n            if len(consecutive) > 0:\n                # if the output contains two consecutive timestamp tokens\n                slices = consecutive.tolist()\n                if single_timestamp_ending:\n                    slices.append(len(tokens))\n\n                last_slice = 0\n                for current_slice in slices:\n                    sliced_tokens = tokens[last_slice:current_slice]\n                    start_timestamp_pos = (\n                        sliced_tokens[0].item() - tokenizer.timestamp_begin\n                    )\n                    end_timestamp_pos = (\n                        sliced_tokens[-1].item() - tokenizer.timestamp_begin\n                    )\n                    current_segments.append(\n                        new_segment(\n                            start=time_offset + start_timestamp_pos * time_precision,\n                            end=time_offset + end_timestamp_pos * time_precision,\n                            tokens=sliced_tokens,\n                            result=result,\n                        )\n                    )\n                    last_slice = current_slice\n\n                if single_timestamp_ending:\n                    # single timestamp at the end means no speech after the last timestamp.\n                    seek += segment_size\n                else:\n                    # otherwise, ignore the unfinished segment and seek to the last timestamp\n                    last_timestamp_pos = (\n                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin\n                    )\n                    seek += last_timestamp_pos * input_stride\n            else:\n                duration = segment_duration\n                timestamps = tokens[timestamp_tokens.nonzero().flatten()]\n                if (\n                    len(timestamps) > 0\n                    and timestamps[-1].item() != tokenizer.timestamp_begin\n                ):\n                    # no consecutive timestamps but it has a timestamp; use the last one.\n                    last_timestamp_pos = (\n                        timestamps[-1].item() - tokenizer.timestamp_begin\n                    )\n                    duration = last_timestamp_pos * time_precision\n\n                current_segments.append(\n                    new_segment(\n                        start=time_offset,\n                        end=time_offset + duration,\n                        tokens=tokens,\n                        result=result,\n                    )\n                )\n                seek += segment_size\n\n            if word_timestamps:\n                add_word_timestamps(\n                    segments=current_segments,\n                    model=model,\n                    tokenizer=tokenizer,\n                    mel=mel_segment,\n                    num_frames=segment_size,\n                    prepend_punctuations=prepend_punctuations,\n                    append_punctuations=append_punctuations,\n                    last_speech_timestamp=last_speech_timestamp,\n                )\n\n                if not single_timestamp_ending:\n                    last_word_end = get_end(current_segments)\n                    if last_word_end is not None and last_word_end > time_offset:\n                        seek = round(last_word_end * FRAMES_PER_SECOND)\n\n                # skip silence before possible hallucinations\n                if hallucination_silence_threshold is not None:\n                    threshold = hallucination_silence_threshold\n                    if not single_timestamp_ending:\n                        last_word_end = get_end(current_segments)\n                        if last_word_end is not None and last_word_end > time_offset:\n                            remaining_duration = window_end_time - last_word_end\n                            if remaining_duration > threshold:\n                                seek = round(last_word_end * FRAMES_PER_SECOND)\n                            else:\n                                seek = previous_seek + segment_size\n\n                    # if first segment might be a hallucination, skip leading silence\n                    first_segment = next_words_segment(current_segments)\n                    if first_segment is not None and is_segment_anomaly(first_segment):\n                        gap = first_segment["start"] - time_offset\n                        if gap > threshold:\n                            seek = previous_seek + round(gap * FRAMES_PER_SECOND)\n                            continue\n\n                    # skip silence before any possible hallucination that is surrounded\n                    # by silence or more hallucinations\n                    hal_last_end = last_speech_timestamp\n                    for si in range(len(current_segments)):\n                        segment = current_segments[si]\n                        if not segment["words"]:\n                            continue\n                        if is_segment_anomaly(segment):\n                            next_segment = next_words_segment(\n                                current_segments[si + 1 :]\n                            )\n                            if next_segment is not None:\n                                hal_next_start = next_segment["words"][0]["start"]\n                            else:\n                                hal_next_start = time_offset + segment_duration\n                            silence_before = (\n                                segment["start"] - hal_last_end > threshold\n                                or segment["start"] < threshold\n                                or segment["start"] - time_offset < 2.0\n                            )\n                            silence_after = (\n                                hal_next_start - segment["end"] > threshold\n                                or is_segment_anomaly(next_segment)\n                                or window_end_time - segment["end"] < 2.0\n                            )\n                            if silence_before and silence_after:\n                                seek = round(\n                                    max(time_offset + 1, segment["start"])\n                                    * FRAMES_PER_SECOND\n                                )\n                                if content_duration - segment["end"] < threshold:\n                                    seek = content_frames\n                                current_segments[si:] = []\n                                break\n                        hal_last_end = segment["end"]\n\n                last_word_end = get_end(current_segments)\n                if last_word_end is not None:\n                    last_speech_timestamp = last_word_end\n\n            if verbose:\n                for segment in current_segments:\n                    start, end, text = segment["start"], segment["end"], segment["text"]\n                    line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"\n                    print(make_safe(line))\n\n            # if a segment is instantaneous or does not contain text, clear it\n            for i, segment in enumerate(current_segments):\n                if segment["start"] == segment["end"] or segment["text"].strip() == "":\n                    segment["text"] = ""\n                    segment["tokens"] = []\n                    segment["words"] = []\n\n            all_segments.extend(\n                [\n                    {"id": i, **segment}\n                    for i, segment in enumerate(\n                        current_segments, start=len(all_segments)\n                    )\n                ]\n            )\n            all_tokens.extend(\n                [token for segment in current_segments for token in segment["tokens"]]\n            )\n\n            if not condition_on_previous_text or result.temperature > 0.5:\n                # do not feed the prompt tokens if a high temperature was used\n                prompt_reset_since = len(all_tokens)\n\n            # update progress bar\n            pbar.update(min(content_frames, seek) - previous_seek)\n\n    return dict(\n        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),\n        segments=all_segments,\n        language=language,\n    )'
LOCKED_DECODE_FILE_SHA256 = '94084fedc5af74cabf8e1883a85552700057081f00ea669ad49435ecb29039db'
LOCKED_DECODE_FUNCTION = 'def decode(\n    model: "Whisper",\n    mel: Tensor,\n    options: DecodingOptions = DecodingOptions(),\n    **kwargs,\n) -> Union[DecodingResult, List[DecodingResult]]:\n    """\n    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).\n\n    Parameters\n    ----------\n    model: Whisper\n        the Whisper model instance\n\n    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)\n        A tensor containing the Mel spectrogram(s)\n\n    options: DecodingOptions\n        A dataclass that contains all necessary options for decoding 30-second segments\n\n    Returns\n    -------\n    result: Union[DecodingResult, List[DecodingResult]]\n        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)\n    """\n    if single := mel.ndim == 2:\n        mel = mel.unsqueeze(0)\n\n    if kwargs:\n        options = replace(options, **kwargs)\n\n    result = DecodingTask(model, options).run(mel)\n\n    return result[0] if single else result'

# Expected exact extracted-source bytes; independently measured from the locked files.
LOCKED_SEGMENT_SHA256 = {'LOCKED_TOKENIZER_ACCESSORS': '4d88869f94b08ac6e3f635a473dce23404c4535ed49944e30f5ffa6c642813a4', 'LOCKED_TRANSCRIBE_FUNCTION': '3b69ff98793efb9913a23b332dec10f9cda8f6fa18c945ac7160e2f2bf87e35f', 'LOCKED_DECODE_FUNCTION': 'd600a718760e5a5e45f1ff841859c85b377622b6c17abef74b9dfa1746eb53b0'}

class PrepareContracts(unittest.TestCase):
    def enabled(self):
        return {**c.FIXED, **c.BINDING_PINS, 'enabled': True, 'independent_source_review': True,
            'source_closure_ready': True, 'prior_prepare_history_sha256': c.digest(b'[]')}

    @contextmanager
    def disabled_release_file(self):
        value = c.decode(c.regular(HERE / 'prepare-release.json'))
        for key in ('enabled','independent_source_review','source_closure_ready'):value[key]=False
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'prepare-release.json').write_bytes(c.canonical(value))
            with patch.object(c,'HERE',root):
                yield

    @contextmanager
    def enabled_release_fixture(self):
        # Positive constructor sentinels use an explicit local fixture, independent
        # of checked-in approval state. The real release validator still executes.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'prepare-release.json').write_bytes(c.canonical(self.enabled()))
            with patch.object(c, 'HERE', root):
                yield

    def test_checked_in_release_scope_and_explicit_disabled_worker_runtime(self):
        value = c.decode(c.regular(HERE / 'prepare-release.json'))
        flags=tuple(value[k] for k in ('enabled','independent_source_review','source_closure_ready'))
        self.assertTrue(all(type(flag) is bool for flag in flags))
        self.assertIn(flags,((False,)*3,(True,)*3))
        self.assertEqual(value['max_new_asr_calls'],0)
        self.assertIs(value['audio_forward_allowed'],False)
        with self.disabled_release_file(),patch.object(c,'fixed_module',side_effect=AssertionError('runtime reached')):
            with self.assertRaisesRegex(ValueError,'prepare_execution_disabled'):
                worker.main(['qwen06'])

    def test_disabled_precedes_runtime_import_for_worker(self):
        with self.disabled_release_file(), patch.object(c,'fixed_module',side_effect=AssertionError('runtime reached')):
            with self.assertRaisesRegex(ValueError, 'prepare_execution_disabled'):
                worker.main(['qwen06'])

    def test_disabled_direct_constructor(self):
        with self.disabled_release_file(), patch.object(c,'fixed_module',side_effect=AssertionError('runtime reached')):
            with self.assertRaisesRegex(ValueError, 'prepare_execution_disabled'):
                worker.construct_family('qwen06', {}, {}, '1', 'a' * 40)

    def test_disabled_host_launcher(self):
        with self.disabled_release_file(), patch.object(c,'fixed_module',side_effect=AssertionError('runtime reached')):
            with self.assertRaisesRegex(ValueError, 'prepare_execution_disabled'):
                host.worker_arguments(name='x',image='x',code='x',runtime='x',output='x',metadata='x',
                    authority='x',family='qwen06',run_identity={},freeze_sha='x',claim_sha='x')

    def test_enabled_without_review_still_fails(self):
        value = self.enabled(); value['source_closure_ready'] = False
        with self.assertRaises(ValueError): c.validate_release(value)

    def test_no_asr_authority_or_history_alias(self):
        value = self.enabled(); value['max_new_asr_calls'] = 18
        with self.assertRaises(ValueError): c.validate_release(value)
        value = self.enabled(); value['experiment'] = 'k1-punctuation6-asr18-r1'
        with self.assertRaises(ValueError): c.validate_release(value)
        value = self.enabled(); value['run_attempt'] = 2
        with self.assertRaises(ValueError): c.validate_release(value)
        value = self.enabled(); value['max_new_asr_calls'] = False
        with self.assertRaises(ValueError): c.validate_release(value)

    def test_decode_and_extra_options_rejected_before_admission(self):
        with self.enabled_release_fixture(), patch.object(c, 'worker_admission', side_effect=AssertionError('admission reached')):
            for args in ([], ['decode'], ['qwen06', 'decode'], ['qwen06', 'prepare'],
                         ['whisper', '--input', '/input'], ['qwen06', '--release', 'other']):
                with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'only_one_fixed'):
                    worker.main(args)

    def test_canonical_and_duplicate_json(self):
        for raw in (b'{"x":1,"x":1}', b'{"x":NaN}', b'{"x": 1}'):
            with self.assertRaises(ValueError): c.decode(raw)

    def test_metadata_roster_rejects_audio_and_empty(self):
        for packet in ({}, {'sample.wav': b'RIFF'}, {n: b'{}' for n in c.PACKET_NAMES | {'extra.json'}}):
            with self.assertRaises(ValueError): c.validate_metadata_packet(packet, self.enabled())

    def test_symlink_fifo_and_oversize_rejected(self):
        import os
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); (root/'file').write_bytes(b'123')
            (root/'alias').symlink_to(root/'file')
            os.mkfifo(root/'fifo')
            for p in (root/'alias',root/'fifo'):
                with self.assertRaises(ValueError): c.regular(p)
            with self.assertRaises(ValueError): c.regular(root/'file',2)

    def test_no_constructor_decode_dispatch_in_application_ast(self):
        tree = ast.parse((HERE/'prepare_worker.py').read_text())
        constructor = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='construct_family')
        names = {n.func.attr for n in ast.walk(constructor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        for prohibited in ('bind_inputs','infer','emit_start','mark_native_enter','run_family','_native_call'):
            self.assertNotIn(prohibited,names)
        self.assertIn('load_qwen_after_approval', (HERE/'prepare_worker.py').read_text())
        params = [a.arg for a in constructor.args.args]
        self.assertNotIn('mode',params)
        self.assertNotIn('callback',params)
        for filename in ('prepare_worker.py','prepare_host.py'):
            for node in ast.walk(ast.parse((HERE/filename).read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                    self.assertNotIn(node.func.id,('eval','exec','recognition_stages'))

    def test_original_constructor_route_never_binds_or_infers(self):
        forbidden = Mock(side_effect=AssertionError('native entry reached'))
        retained = types.SimpleNamespace(verify_scope=Mock(), THREAD_ENV={}, OFFLINE_ENV={},
            prepare_caches=Mock(),verify_sources=Mock(),import_helpers=Mock(),validate_decoder=Mock(),
            RETAINED=Path('/retained'), read_regular=Mock(return_value=b'locks'),
            decode=Mock(return_value={'qwen06': {'asset_lock':{},'asset_lock_sha256':'a',
                'source_lock':[],'source_lock_sha256':'b'}}),
            execution_contract=Mock(return_value={'batch_size':1}),
            load_qwen_after_approval=Mock(return_value=types.SimpleNamespace(receipt={})),
            bind_inputs=forbidden, infer=forbidden, run=forbidden)
        contract = types.SimpleNamespace(read_json=Mock(return_value={}),digest=c.digest,
            write_once=Mock(return_value='d'),read_bytes=Mock(return_value=b'{}'))
        locks = types.SimpleNamespace(verify_installed_extension_sources=Mock(return_value={}),
            build_native_locks=Mock(return_value=(b'{}',b'{}')))
        native = types.SimpleNamespace(NativeCaptureAdapter=Mock(), infer=forbidden)
        shared = {'campaign_contract':contract,'native_locks':locks,
            'setup_adapter':types.SimpleNamespace(verify_runtime=Mock(return_value={}))}
        def fixed(relative,name):
            return retained if relative.endswith('/asr_worker.py') else native
        with self.enabled_release_fixture(), patch.object(c,'fixed_module',side_effect=fixed), \
             patch.object(c,'worker_admission',return_value=(self.enabled(),{}, {'identity':{'run_id':1,'executed_head_sha':'a'*40}}, {'decoder-inputs.json':b'{}'},{})), \
             patch.object(worker,'_shared',side_effect=lambda n:shared[n]), patch.object(worker,'_MODEL_STARTED',False), \
             patch.object(Path,'is_dir',return_value=True), patch.object(Path,'exists',return_value=False):
            worker.construct_family('qwen06',{'decoder-inputs.json':b'{}'}, {}, '1','a'*40)
        retained.load_qwen_after_approval.assert_called_once()
        native.NativeCaptureAdapter.assert_called_once()
        forbidden.assert_not_called()

    def test_sense_constructor_never_binds_or_infers(self):
        forbidden = Mock(side_effect=AssertionError('native entry reached'))
        retained = types.SimpleNamespace(verify_scope=Mock(), THREAD_ENV={}, OFFLINE_ENV={},
            prepare_caches=Mock(),verify_sources=Mock(),import_helpers=Mock(),validate_decoder=Mock(),
            RETAINED=Path('/retained'), read_regular=Mock(return_value=b'locks'),
            decode=Mock(return_value={'sensevoice': {'asset_lock':{},'asset_lock_sha256':'a',
                'source_lock':[],'source_lock_sha256':'b'}}),
            execution_contract=Mock(return_value={'batch_size':1}),
            load_sense_after_approval=Mock(return_value=types.SimpleNamespace(receipt={})),
            bind_inputs=forbidden, infer=forbidden, run=forbidden)
        contract = types.SimpleNamespace(read_json=Mock(return_value={}),digest=c.digest,
            write_once=Mock(return_value='d'),canonical=c.canonical,read_bytes=Mock(return_value=b'{}'))
        locks = types.SimpleNamespace(verify_installed_extension_sources=Mock(return_value={}),
            build_native_locks=Mock(return_value=(b'{}',b'{}')))
        native = types.SimpleNamespace(NativeCaptureAdapter=Mock(), infer=forbidden, sense_effective=Mock(return_value=({},{})))
        shared = {'campaign_contract':contract,'native_locks':locks,
            'setup_adapter':types.SimpleNamespace(verify_runtime=Mock(return_value={})),
            'optional_tools':types.SimpleNamespace(unavailable_ffmpeg=Mock(return_value=nullcontext()),policy_config=Mock(return_value={}))}
        def fixed(relative,name):
            return retained if relative.endswith('/asr_worker.py') else native
        with self.enabled_release_fixture(), patch.object(c,'fixed_module',side_effect=fixed), \
             patch.object(c,'worker_admission',return_value=(self.enabled(),{}, {'identity':{'run_id':1,'executed_head_sha':'a'*40}}, {'decoder-inputs.json':b'{}'},{})), \
             patch.object(worker,'_shared',side_effect=lambda n:shared[n]), patch.object(worker,'_MODEL_STARTED',False), \
             patch.object(Path,'is_dir',return_value=True), patch.object(Path,'exists',return_value=False):
            worker.construct_family('sensevoice',{'decoder-inputs.json':b'{}'}, {}, '1','a'*40)
        retained.load_sense_after_approval.assert_called_once()
        native.NativeCaptureAdapter.assert_called_once()
        forbidden.assert_not_called()

    def guard_attack_subprocess(self, attack, module='transformers.locked_test'):
        body = "def forward(label):\n    marker.write_text(marker.read_text() + label)\n"
        script = """import sys, threading, _thread
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import prepare_worker as worker
marker=Path(sys.argv[2])
namespace={'__name__':'transformers.locked_test','marker':marker}
saved_setprofile = sys.setprofile
saved_thread_start = _thread.start_new_thread
""" + "exec(compile(" + repr(body) + ", '/locked/test.py', 'exec'), namespace)\n" + attack
        script=script.replace("'transformers.locked_test'",repr(module),1)
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'entered.txt'
            marker.write_text('')
            flags=['-'+('O'*sys.flags.optimize)] if sys.flags.optimize else []
            result=subprocess.run([sys.executable,'-I','-S','-B',*flags,'-c',script,str(HERE),str(marker)],
                capture_output=True,timeout=10)
            self.assertEqual(result.returncode,91,result.stderr.decode(errors='replace'))
            self.assertEqual(marker.read_text(),'')
            self.assertEqual(result.stderr,b'')
            self.assertEqual(json.loads(result.stdout),{'schema':'k1-prepare-native-entry-blocked-v1',
                'status':'stopped_before_forbidden_native_body','exit_code':91})

    def test_forward_guard_cannot_be_caught_then_reentered(self):
        # A profiling callback exception used to clear the hook; then a caught
        # first call could permit the second body. Process exit is uncatchable.
        self.guard_attack_subprocess("""try:
    with worker.NoForward():
        try:
            namespace['forward']('first')
        except BaseException:
            pass
        namespace['forward']('second')
except BaseException:
    pass
""")

    def test_guard_cannot_temporarily_disable_and_restore_profile(self):
        for modify in ('saved_setprofile(None)', 'sys.setprofile(None)', 'threading.setprofile(None)'):
            with self.subTest(api=modify):
                self.guard_attack_subprocess("with worker.NoForward():\n    saved = sys.getprofile()\n    " +
                    modify + "\n    namespace['forward']('unguarded')\n    saved_setprofile(saved)\n")

    def test_guard_propagates_to_standard_thread_forward(self):
        # Thread.start itself is legal; its standard bootstrap installs the
        # inherited guard before the target invokes a known model entrypoint.
        self.guard_attack_subprocess("""with worker.NoForward():
    thread = threading.Thread(target=lambda: namespace['forward']('thread_forward'))
    thread.start()
    thread.join()
""")

    def test_guard_rejects_direct_unprofiled_lowlevel_thread(self):
        self.guard_attack_subprocess("""with worker.NoForward():
    saved_thread_start(lambda: (marker.write_text('raw_thread_started'), namespace['forward']('raw_thread')), ())
    __import__('time').sleep(.05)
""")

    def test_guard_allows_legitimate_monitor_thread(self):
        script = """import sys, threading, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import prepare_worker as worker
marker=Path(sys.argv[2])
with worker.NoForward() as guard:
    thread=threading.Thread(target=lambda: marker.write_text('monitor_completed'))
    thread.start()
    thread.join()
print(json.dumps(guard.receipt(), sort_keys=True))
"""
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'monitor.txt'
            flags=['-'+('O'*sys.flags.optimize)] if sys.flags.optimize else []
            result=subprocess.run([sys.executable,'-I','-S','-B',*flags,'-c',script,str(HERE),str(marker)],
                capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
            self.assertEqual(marker.read_text(),'monitor_completed')
            self.assertEqual(result.stderr,b'')
            receipt=json.loads(result.stdout)
            self.assertEqual(receipt['schema'],'k1-prepare-forward-guard-v1')
            self.assertEqual(receipt['forbidden_attempts'],0)
            self.assertIs(receipt['universal_native_extension_proof'],False)

    def test_locked_whisper_fixture_source_bytes(self):
        for name, expected in LOCKED_SEGMENT_SHA256.items():
            self.assertEqual(hashlib.sha256(globals()[name].encode()).hexdigest(),expected)

    def test_direct_whisper_model_forward_remains_blocked(self):
        self.guard_attack_subprocess("with worker.NoForward():\n    namespace['forward']('direct_model_forward')\n",
            module='whisper.model')

    def test_locked_whisper_tokenizer_accessors_are_not_neural_calls(self):
        # Retained accessor AST, not a fake class overriding the rule. Only the
        # text encoding backend is a test double; no model/audio object exists.
        script = """import sys, json, ast, __future__
from pathlib import Path
from functools import cached_property
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import prepare_worker as worker
source=""" + repr(LOCKED_TOKENIZER_ACCESSORS) + """
namespace={'__name__':'whisper.tokenizer','cached_property':cached_property}
code=ast.parse(source)
exec(compile(code,'/runtime/source/whisper/tokenizer.py','exec',flags=__future__.annotations.compiler_flag),namespace)
tokenizer=namespace['Tokenizer']()
tokenizer.special_tokens={'<|transcribe|>':50359,'<|0.00|>':50364}
tokenizer.encoding=SimpleNamespace(decode=lambda tokens, **kwargs:tokens)
with worker.NoForward() as guard:
    actual=tokenizer.transcribe
    decoded=tokenizer.decode([1,50364,2])
print(json.dumps({'transcribe_token':actual,'text_tokens':decoded,'guard':guard.receipt()},sort_keys=True))
"""
        flags=['-'+('O'*sys.flags.optimize)] if sys.flags.optimize else []
        result=subprocess.run([sys.executable,'-I','-S','-B',*flags,'-c',script,str(HERE)],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        self.assertEqual(result.stderr,b'')
        record=json.loads(result.stdout)
        self.assertEqual(record['transcribe_token'],50359)
        self.assertEqual(record['text_tokens'],[1,2])
        self.assertEqual(record['guard']['forbidden_attempts'],0)

    def test_locked_whisper_neural_entry_asts_still_stop_before_body(self):
        for module, source, call in (
            ('whisper.transcribe',LOCKED_TRANSCRIBE_FUNCTION,"namespace['transcribe'](None,None)"),
            ('whisper.decoding',LOCKED_DECODE_FUNCTION,"namespace['decode'](None,trap)")):
            script="""import sys, ast, __future__
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import prepare_worker as worker
marker=Path(sys.argv[2])
class BodyTrap:
    def __getattr__(self, name):
        marker.write_text('neural_body_entered')
        raise RuntimeError('model operations prohibited in this AST test')
trap=BodyTrap()
namespace={'__name__':"""+repr(module)+""",'torch':trap,'DecodingOptions':lambda:None}
source="""+repr(source)+"""
exec(compile(ast.parse(source),'/runtime/source/"""+module.replace('.','/')+""".py','exec',flags=__future__.annotations.compiler_flag),namespace)
with worker.NoForward():
    """+call+"\n"
            with self.subTest(module=module),tempfile.TemporaryDirectory() as directory:
                marker=Path(directory)/'neural-body.txt';marker.write_text('')
                flags=['-'+('O'*sys.flags.optimize)] if sys.flags.optimize else []
                result=subprocess.run([sys.executable,'-I','-S','-B',*flags,'-c',script,str(HERE),str(marker)],
                    capture_output=True,timeout=10)
                self.assertEqual(result.returncode,91,result.stderr.decode(errors='replace'))
                self.assertEqual(marker.read_text(),'')
                self.assertEqual(result.stderr,b'')
                self.assertEqual(json.loads(result.stdout)['exit_code'],91)

    def test_guard_allows_config_observation(self):
        namespace={'__name__':'transformers.locked_test'}
        exec(compile('def _prepare_generation_config():\n    return 1\n','/locked/test.py','exec'),namespace)
        with worker.NoForward() as guard:
            self.assertEqual(namespace['_prepare_generation_config'](),1)
        self.assertEqual(guard.receipt()['forbidden_attempts'],0)
        self.assertIs(guard.receipt()['universal_native_extension_proof'],False)

    def test_whisper_constructor_never_decodes(self):
        forbidden = Mock(side_effect=AssertionError('decode reached'))
        adapter = types.SimpleNamespace(FixedWhisper=Mock(), decode=forbidden)
        binding = types.SimpleNamespace(construct_whisper=Mock())
        contract = types.SimpleNamespace(read_json=Mock(return_value={}))
        locks = types.SimpleNamespace(build_whisper_locks=Mock(return_value=(b'{}',b'{}')))
        shared = {'campaign_contract':contract,'native_locks':locks,'whisper_binding':binding}
        with self.enabled_release_fixture(), patch.object(c,'fixed_module',return_value=adapter), \
             patch.object(c,'worker_admission',return_value=(self.enabled(),{}, {'identity':{'run_id':1,'executed_head_sha':'a'*40}}, {},{})), \
             patch.object(worker,'_shared',side_effect=lambda n:shared[n]), patch.object(worker,'_MODEL_STARTED',False), \
             patch.object(Path,'is_dir',return_value=True), patch.object(Path,'exists',return_value=False):
            worker.construct_family('whisper',{}, {}, '1','a'*40)
        binding.construct_whisper.assert_called_once()
        forbidden.assert_not_called()

    def launch_fixture(self):
        image='locked-image'
        args=['create','--name','test','--user','1000:1000']
        mounts=[]
        for dest in ('/code','/runtime','/output','/metadata','/authority','/gate'):
            writable=dest in ('/output','/gate')
            src='/host'+dest
            args+=['--mount','type=bind,src='+src+',dst='+dest+('' if writable else ',readonly')]
            mounts.append({'Type':'bind','Source':src,'Destination':dest,'RW':writable,'Propagation':'rprivate'})
        command=['python','-I','-B','/code/'+c.OLD+'cpu_gate.py','/runtime/venv/bin/python','-I','-B',
            '/code/'+c.REL+'prepare_worker.py','qwen06']
        args += [image]+command
        config={'Image':image,'Entrypoint':None,'Cmd':command,'Env':[],'User':'1000:1000'}
        limits={'Memory':12*1024**3,'MemorySwap':12*1024**3,'NanoCpus':4_000_000_000,'PidsLimit':256,
            'NetworkMode':'none','ReadonlyRootfs':True,'Privileged':False,'CgroupnsMode':'private','PidMode':'',
            'IpcMode':'private','CapAdd':[],'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges'],
            'LogConfig':{'Type':'local','Config':{'max-size':'1m','max-file':'1','compress':'false'}}}
        limits['Tmpfs'] = {p:'rw,noexec,nosuid,nodev,size='+str(size)+',uid=1000,gid=1000,mode=0700'
            for p,size in (('/scratch',268435456),('/tmp',134217728))}
        at=args.index(image)
        args[at:at]=[item for p,options in limits['Tmpfs'].items() for item in ('--tmpfs',p+':'+options)]
        return args,{'Config':config,'HostConfig':limits,'Mounts':mounts}

    def test_launch_fixed_metadata_only(self):
        args,record=self.launch_fixture()
        self.assertTrue(host.validate_launch(record,args))
        for key,value in (('NetworkMode','bridge'),('Privileged',True),('MemorySwap',0)):
            modified=deepcopy(record);modified['HostConfig'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):host.validate_launch(modified,args)
        modified=deepcopy(record);modified['Config']['Cmd'].append('decode')
        with self.assertRaises(ValueError):host.validate_launch(modified,args)
        modified=deepcopy(record);modified['Mounts'][3]['Destination']='/input'
        with self.assertRaises(ValueError):host.validate_launch(modified,args)
        modified=deepcopy(record);modified['Config']['Env'].append('GITHUB_TOKEN=secret')
        with self.assertRaises(ValueError):host.validate_launch(modified,args)
        modified=deepcopy(record);modified['Config']['User']='0:0'
        with self.assertRaises(ValueError):host.validate_launch(modified,args)

    def test_completed_guard_required_for_receipt(self):
        guard=worker.NoForward()
        with self.assertRaises(ValueError):guard.receipt()
        with guard:
            with self.assertRaises(ValueError):guard.receipt()
        self.assertEqual(guard.receipt()['forbidden_attempts'],0)

    def test_original_module_imports_are_model_free(self):
        source=c.ROOT
        if not (source/c.OLD/'native_worker.py').is_file():
            source=HERE.parents[1]/'asr18-r1-preparation/checkout'
        self.assertTrue((source/c.OLD/'native_worker.py').is_file())
        with patch.object(c,'ROOT',source):
            retained=c.fixed_module('research/source-screen32-v1/asr_worker.py','prepare_retained_worker')
            native=c.fixed_module('research/source-screen32-v1/native_v2_adapter.py','prepare_native_adapter')
            whisper=c.fixed_module('research/source-screen32-whisper-v1/adapter.py','prepare_whisper_adapter')
        self.assertTrue(callable(retained.load_qwen_after_approval))
        self.assertTrue(callable(retained.load_sense_after_approval))
        self.assertTrue(callable(native.NativeCaptureAdapter))
        self.assertTrue(callable(whisper.FixedWhisper))
        self.test_no_ml_imports()

    def test_no_ml_imports(self):
        self.assertFalse(any(n=='torch' or n.startswith(('torch.','qwen_asr','funasr','whisper.')) for n in sys.modules))


if __name__=='__main__': unittest.main()
