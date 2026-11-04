"""Two-phase isolated worker. Running it requires separate model-run authority.

Import and --help do not import ML packages. prepare is load-only; infer performs
one new preparation, verifies the already frozen policy, then consumes at most
16 audio-only entries once. Outputs and failures remain private.
"""
import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path
import signal
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from whisper_capture import prepare, require, read, parse, sha, canonical, persist, relative


@contextmanager
def deadline(seconds):
    require(type(seconds) is int and 1 <= seconds <= 1200, 'bounded worker deadline')
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'unexpected existing process timer')
    def expired(_signum, _frame):
        raise TimeoutError('single-attempt worker deadline exceeded')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def inputs(input_root, job_sha256, identity_map):
    root = Path(input_root).absolute()
    require(root.resolve() == root and root.is_dir(), 'unaliased input root')
    require({path.name for path in root.iterdir()} == {'job.json', 'audio'}, 'blind input root membership')
    job_raw = read(root/'job.json', 65536)
    require(sha(job_raw) == job_sha256, 'frozen original job byte hash')
    job = parse(job_raw)
    require(type(job) is dict and set(job) == {'schema', 'clips'} and
            job['schema'] == 'blind-source-screen32-job-v1', 'exact blind job schema')
    clips = job['clips']
    require(type(clips) is list and 1 <= len(clips) <= 16, 'bounded blind sixteen subset')
    identity_raw = read(identity_map, 65536)
    mapping = parse(identity_raw)
    require(type(mapping) is dict and set(mapping) == {'schema', 'clips'} and
            mapping['schema'] == 'blind-asr-source-request-map-v1', 'audio-only identity map schema')
    require(type(mapping['clips']) is list and len(mapping['clips']) == len(clips), 'exact identity denominator')
    rows = {}
    for row in mapping['clips']:
        require(type(row) is dict and set(row) == {'audio_id', 'source_id', 'request_id', 'wav_sha256'}, 'identity map cannot carry target text')
        require(type(row['audio_id']) is str and row['audio_id'] not in rows, 'unique mapped audio identity')
        rows[row['audio_id']] = row
    seen, requests, members = set(), set(), set()
    result = []
    for clip in clips:
        require(type(clip) is dict and set(clip) == {'audio_id', 'audio_path', 'wav_sha256'}, 'blind clip fields')
        opaque = clip['audio_id']
        require(type(opaque) is str and re.fullmatch('clip-[0-9]{6}', opaque) and
                opaque not in seen and opaque in rows, 'unique exact mapped clip')
        require(type(clip['wav_sha256']) is str and re.fullmatch('[a-f0-9]{64}', clip['wav_sha256']), 'WAV SHA256')
        seen.add(opaque)
        member = relative(clip['audio_path'])
        require(member == 'audio/' + clip['wav_sha256'] + '.wav', 'content-addressed audio member')
        require(member not in members, 'one attempt per source waveform')
        members.add(member)
        row = rows[opaque]
        require(row['wav_sha256'] == clip['wav_sha256'] and
                type(row['request_id']) is str and row['request_id'] not in requests,
                'identity byte binding and unique request')
        for key in ('source_id', 'request_id'):
            require(type(row[key]) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', row[key]), 'opaque source/request binding')
        requests.add(row['request_id'])
        result.append(dict(audio_path=root/member, opaque_id=opaque,
            expected_wav_sha256=clip['wav_sha256'], source_id=row['source_id'], request_id=row['request_id']))
    audio = root/'audio'
    require(audio.resolve() == audio and audio.is_dir() and
            {path.name for path in audio.iterdir()} == {Path(member).name for member in members}, 'exact audio membership')
    return result, dict(job_sha256=sha(job_raw), identity_map_sha256=sha(identity_raw))


def progress(args, stage, index):
    require(stage in ('load', 'cell') and type(index) is int and 0 <= index <= 16, 'retained progress contract')
    target = Path(args.output).parent/'progress.json'
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(dict(stage=stage, index=index, started_monotonic=time.monotonic())))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(target))


def run(args):
    prepared = None
    index = 0
    try:
        if args.phase == 'infer':
            require(all((args.input, args.job_sha256, args.identity_map, args.policy, args.policy_sha256)),
                    'infer requires frozen job, identity map and policy')
            clips, binding = inputs(args.input, args.job_sha256, args.identity_map)
            policy_raw = read(args.policy, 8*1024**2)
        else:
            require(not any((args.input, args.job_sha256, args.identity_map, args.policy, args.policy_sha256)),
                    'prepare must not receive audio, policy or source identities')
            clips, binding = [], {}
        progress(args, 'load', 0)
        with deadline(1200):
            prepared = prepare(args.retained_repo, args.runtime, args.output, args.executed_head,
                args.sink_source, args.consensus_tools, recognizer_id=args.recognizer_id, maximum=16)
        if args.phase == 'prepare':
            terminal = dict(schema='prospective-whisper-worker-terminal-v1', phase='prepare',
                status='loaded_without_recognition', actual_decode_calls=0,
                identity_sha256=sha(canonical(prepared.identity)), training_admitted=False)
            persist(Path(args.output)/'worker-terminal.json', canonical(terminal))
            return terminal
        progress(args, 'load', 0)
        with deadline(1200):
            prepared.bind_policy(policy_raw, args.policy_sha256)
        outcomes = []
        for index, clip in enumerate(clips, 1):
            progress(args, 'cell', index)
            with deadline(300):
                receipt = prepared.infer(**clip)
            outcomes.append(dict(opaque_id=clip['opaque_id'], status='success',
                completion_sha256=sha(canonical(receipt))))
        terminal = dict(schema='prospective-whisper-worker-terminal-v1', phase='infer', status='success',
            completed_count=len(outcomes), planned_count=len(clips), outcomes=outcomes,
            input_binding=binding, training_admitted=False)
        persist(Path(args.output)/'worker-terminal.json', canonical(terminal))
        return terminal
    except BaseException as exc:
        if prepared is not None:
            prepared.failed = True
            if prepared.sink is not None:
                prepared.sink.fail()
        # A separate exclusive terminal does not replace or repair evidence.
        output = Path(args.output)
        failure = dict(schema='prospective-whisper-worker-failure-v1', phase=args.phase,
            status='failed_no_retry', active_index=index, error_type=type(exc).__name__,
            error=str(exc), training_admitted=False)
        try:
            persist(output.parent/(output.name+'-failed.json'), canonical(failure))
        except (OSError, ValueError):
            pass
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=('prepare', 'infer'))
    for name in ('retained-repo', 'runtime', 'output', 'executed-head', 'sink-source', 'consensus-tools'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--recognizer-id', default='whisper-small')
    for name in ('input', 'job-sha256', 'identity-map', 'policy', 'policy-sha256'):
        p.add_argument('--'+name)
    return p


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
