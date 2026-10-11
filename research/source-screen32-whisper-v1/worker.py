"""One fixed sixteen-clip, blind, CPU Whisper process. No retries or other models."""
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'source-screen32-v1'))
from common import BOUNDARY, require, read_json, write_once, atomic, bind_input, wav_bytes, file_hash
from runtime_scope import verify_runtime_scope


def alarm_handler(signum, frame):
    raise TimeoutError('bounded model stage')


def main():
    require(HERE == Path('/code/research/source-screen32-whisper-v1') and len(sys.argv) == 1,
            'fixed hosted worker entry')
    scope = verify_runtime_scope()
    root, output, runtime = Path('/input'), Path('/output'), Path('/runtime')
    require(not list(output.iterdir()), 'fresh blind output only')
    job = bind_input(root)
    write_once(output / 'blind-job.json', job)
    write_once(output / 'kernel-scope.json', scope)
    # No producer mapping, intended text, previous transcript, or roster exists in these mounts.
    atomic(output / 'progress.json', {'stage': 'load', 'index': 0, 'started_monotonic': time.monotonic()})
    signal.signal(signal.SIGALRM, alarm_handler); signal.alarm(1200)
    from adapter import FixedWhisper
    recognizer = FixedWhisper(runtime, output)
    signal.alarm(0)
    for index, clip in enumerate(job['clips'], 1):
        scope = verify_runtime_scope()
        pcm, frames = wav_bytes(root / clip['audio_path'], clip['wav_sha256'])
        directory = output / clip['audio_id']; directory.mkdir()
        write_once(directory / 'started.json', {'schema': 'whisper-screen-attempt-started-v1',
            'audio_id': clip['audio_id'], 'wav_sha256': clip['wav_sha256'], 'attempt': 1,
            'input_job_sha256': file_hash(root / 'job.json'),
            'native_config_sha256': file_hash(output / 'native-config.json'),
            'kernel_scope': scope, **BOUNDARY})
        atomic(output / 'progress.json', {'stage': 'cell', 'index': index, 'started_monotonic': time.monotonic()})
        signal.alarm(300)
        raw = recognizer.decode(pcm, frames, directory)
        signal.alarm(0)
        require(raw['decoder_completion_validated'] is True, 'native decode did not finish naturally; preserve raw and stop')
        write_once(directory / 'completed.json', {'schema': 'whisper-screen-attempt-completed-v1',
            'audio_id': clip['audio_id'], 'wav_sha256': clip['wav_sha256'], 'attempt': 1,
            'raw_output_sha256': file_hash(directory / 'raw-output.json'),
            'started_sha256': file_hash(directory / 'started.json'),
            'input_capture_sha256': file_hash(directory / 'input-capture.json'),
            'pre_finalize_sha256': file_hash(directory / 'pre-finalize.json'), **BOUNDARY})
    write_once(output / 'worker-complete.json', {'status': 'all_sixteen_returned_with_natural_eot',
               'calls': 16, **BOUNDARY})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never publish arbitrary exception text, paths, URLs, or environment values.
        signal.alarm(0)
        if Path('/output').is_dir():
            atomic(Path('/output/worker-failure.json'), {'status': 'failed_no_retry',
                   'error_type': type(error).__name__, **BOUNDARY})
        raise SystemExit(1)
