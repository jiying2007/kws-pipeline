"""Bounded local lifecycle v2. Stdlib, Linux; no model or network operations.

Adapters must write all data through Writer and declare tool-internal peak copies.
This is cooperative I/O admission, not a filesystem quota or remote durable store.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

CATEGORIES = ('output', 'archive', 'upload', 'readback', 'temp')
DISK_FLOOR = 2 * 1024**3
CONTROL_RESERVE = 128 * 1024  # Includes old + new state, lock and directory metadata allowance.
STATE_LIMIT = 48 * 1024


class Rejected(ValueError):
    pass


def require(value, message):
    if not value:
        raise Rejected(message)


def integer(value):
    return type(value) is int and value >= 0


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def bounds(value):
    require(type(value) is dict and set(value) == set(CATEGORIES), 'all five category bounds required')
    require(all(integer(v) for v in value.values()), 'bounds must be nonnegative integer bytes')
    return dict(value)


def zero_bounds(**updates):
    return bounds({**dict.fromkeys(CATEGORIES, 0), **updates})


class Writer:
    """Exclusive-create writes; partial failures remain charged and are never deleted.

    Data is immutable after write. A callback may not write outside this API,
    spawn unmanaged writers, or retain this Writer after returning.
    """
    def __init__(self, controller, prefix, allowance):
        self.controller, self.prefix = controller, prefix
        self.remaining = bounds(allowance)
        self.closed = False
        self.files = []

    def write(self, category, name, data):
        require(type(data) is bytes, 'bytes required; use write_stream for large artifacts')
        return self.write_stream(category, name, io.BytesIO(data), expected_bytes=len(data))

    def write_stream(self, category, name, source, *, expected_bytes):
        """Read at most 64 KiB at a time; require an exact known bounded size.

        Source must honor read(n). A malicious adapter can allocate/write outside
        this API; this is not a sandbox. A failed partial file stays retained.
        """
        require(not self.closed, 'writer is closed')
        require(category in CATEGORIES, 'unknown storage category')
        require(type(name) is str and name not in ('', '.', '..') and '/' not in name and '\\' not in name,
                'one filename component required')
        require(integer(expected_bytes) and expected_bytes <= self.remaining[category],
                'declared category bound exceeded')
        self.controller._admit(zero_bounds(**{category: expected_bytes}))
        folder = self.controller.root / category / self.prefix
        folder.mkdir(exist_ok=True)
        require(not folder.is_symlink(), 'linked storage forbidden')
        sync_dir(folder.parent)  # Persist the new prefix entry before artifacts/terminal state.
        path = folder / name
        self.remaining[category] -= expected_bytes
        digest, written = hashlib.sha256(), 0
        with path.open('xb') as stream:
            while written < expected_bytes:
                requested = min(65536, expected_bytes - written)
                data = source.read(requested)
                require(type(data) is bytes and 0 < len(data) <= requested, 'source short or oversized read')
                self.controller._admit(zero_bounds(**{category: len(data)}))
                stream.write(data)
                stream.flush()  # Make already-written bytes visible to the next accounting scan.
                digest.update(data)
                written += len(data)
            require(source.read(1) == b'', 'source exceeds declared size')
            os.fsync(stream.fileno())
        sync_dir(folder)
        record = {'path': str(path.relative_to(self.controller.root)), 'bytes': written,
                  'sha256': digest.hexdigest()}
        self.files.append(record)
        return record


class Controller:
    def __init__(self, root, *, disk_free=None):
        self.root = Path(root).absolute()
        # Injection exists for deterministic tests; production CLI always uses disk_usage.
        self.disk_free = disk_free or (lambda: shutil.disk_usage(self.root).free)
        require(self.root.is_dir() and not self.root.is_symlink(), 'existing unlinked state directory required')

    @classmethod
    def create(cls, root, item_ids, *, total_bytes, disk_floor_bytes=DISK_FLOOR, disk_free=None):
        require(type(item_ids) is list and 0 < len(item_ids) <= 256 and
                all(type(x) is str and 0 < len(x) <= 128 for x in item_ids) and
                len(set(item_ids)) == len(item_ids), 'unique nonempty frozen roster required (max 256)')
        require(integer(total_bytes) and total_bytes >= CONTROL_RESERVE, 'total budget too small')
        require(integer(disk_floor_bytes) and disk_floor_bytes >= DISK_FLOOR, 'floor cannot be below 2GiB')
        root = Path(root).absolute()
        require(not root.exists(), 'state root already exists; never overwrite or reset')
        free = disk_free() if disk_free else shutil.disk_usage(root.parent).free
        require(free >= disk_floor_bytes + CONTROL_RESERVE, 'insufficient initialization headroom')
        root.mkdir()
        for name in CATEGORIES:
            (root / name).mkdir()
        (root / 'lock').touch(exist_ok=False)
        c = cls(root, disk_free=disk_free)
        c._save({'schema': 'bounded-lifecycle-v2', 'policy': {'total_bytes': total_bytes,
                    'disk_floor_bytes': disk_floor_bytes}, 'roster': item_ids,
                 'execution': {'status': 'pending', 'items': {x: 'pending' for x in item_ids}},
                 'delivery': {'status': 'pending', 'attempts': []}})
        sync_dir(root.parent)
        return c

    @contextmanager
    def _locked(self, *, blocking=True):
        path = self.root / 'lock'
        require(not path.is_symlink(), 'linked lock forbidden')
        with path.open('rb') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise Rejected('controller is active; cannot seal execution') from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _load(self):
        path = self.root / 'state.json'
        require(not path.is_symlink() and path.stat().st_size <= STATE_LIMIT, 'invalid state file')
        value = json.loads(path.read_text())
        require(value['schema'] == 'bounded-lifecycle-v2', 'wrong state schema')
        require(value['policy']['disk_floor_bytes'] >= DISK_FLOOR, 'invalid floor')
        return value

    def _save(self, value):
        raw = encoded(value)
        require(len(raw) <= STATE_LIMIT, 'state capacity exhausted; stop, retain evidence')
        require(self.disk_free() >= value['policy']['disk_floor_bytes'] + len(raw), 'state write risks disk floor')
        # Exclusive pending file: a crash leaves it for investigation, never overwrites it.
        tmp = self.root / 'state.pending'
        with tmp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.root / 'state.json')
        sync_dir(self.root)

    def _usage(self):
        totals = dict.fromkeys(CATEGORIES, 0)
        for category in CATEGORIES:
            folder = self.root / category
            require(not folder.is_symlink(), 'linked category forbidden')
            for path in folder.rglob('*'):
                require(not path.is_symlink(), 'linked artifact forbidden')
                if path.is_file():
                    totals[category] += path.stat().st_size
        return totals

    def _admit(self, future):
        future = bounds(future)
        state = self._load()
        used = self._usage()
        increment = sum(future.values())
        require(CONTROL_RESERVE + sum(used.values()) + increment <= state['policy']['total_bytes'],
                'total retained/peak budget exceeded')
        require(self.disk_free() >= state['policy']['disk_floor_bytes'] + increment + CONTROL_RESERVE,
                'projected write risks 2GiB disk floor')
        return used

    def status(self):
        with self._locked():
            state = self._load()
            values = list(state['execution']['items'].values())
            return {**state, 'counts': {'planned': len(state['roster']),
                    'consumed_or_unknown': sum(x != 'pending' for x in values),
                    'completed': values.count('completed'),
                    'unknown': sum(x in ('started', 'unknown') for x in values),
                    'not_started': values.count('pending')}, 'storage_bytes': self._usage()}

    def execute(self, callback, *, per_item_bounds):
        """Invoke callback(item_id, writer) once per roster item, serially.

        A started/failed/completed run is never resumed automatically, even after
        process death. Delivery errors cannot call this entry point again.
        """
        allowance = bounds(per_item_bounds)
        with self._locked():
            state = self._load()
            require(state['execution']['status'] == 'pending', 'execution already claimed; no retry')
            self._admit({k: v * len(state['roster']) for k, v in allowance.items()})
            state['execution']['status'] = 'started'
            self._save(state)
            for index, item in enumerate(state['roster']):
                state['execution']['items'][item] = 'started'
                self._save(state)  # Consume before invoking any model/adaptor callback.
                writer = Writer(self, f'exec-{index:04d}', allowance)
                try:
                    callback(item, writer)
                    self._admit(zero_bounds())
                except BaseException:
                    state['execution']['items'][item] = 'unknown'
                    state['execution']['status'] = 'failed'
                    self._save(state)
                    raise
                finally:
                    writer.closed = True
                state['execution']['items'][item] = 'completed'
                self._save(state)
            state['execution']['status'] = 'completed'
            self._save(state)

    def seal_failed_execution(self, *, processes_stopped_verified, evidence):
        """Explicit crash recovery, never retry or successful completion.

        Caller must independently verify model/child process cleanup. An unlocked
        file alone does not establish that descendants have stopped. Active local
        controller locks are rejected immediately, without waiting for completion.
        """
        require(processes_stopped_verified is True and type(evidence) is str and 0 < len(evidence) <= 512,
                'verified stopped process scope and bounded evidence reference required')
        with self._locked(blocking=False):
            state = self._load()
            require(state['execution']['status'] == 'started', 'only interrupted execution can be sealed')
            for item, status in state['execution']['items'].items():
                if status == 'started':
                    state['execution']['items'][item] = 'unknown'
            state['execution'].update(status='failed', recovery={
                'reason': 'interrupted_execution_sealed_failed',
                'processes_stopped_evidence': evidence,
                'consumption_preserved': True, 'retry_allowed': False})
            self._save(state)

    def deliver(self, callback, *, peak_bounds):
        """Invoke an explicitly requested delivery attempt without model execution.

        callback(writer) returns a small verified delivery receipt. Real adapters
        must independently verify remote bytes before returning. Unknown remote
        sends must be reconciled before retry; this class cannot deduplicate a
        remote service or authorize uploads. Failed local artifacts stay charged.
        """
        allowance = bounds(peak_bounds)
        with self._locked():
            state = self._load()
            require(state['execution']['status'] in ('completed', 'failed'), 'execution is not terminal')
            require(state['delivery']['status'] in ('pending', 'failed'),
                    'delivery completed or uncertain; reconcile before retry')
            require(len(state['delivery']['attempts']) < 64, 'delivery attempt cap reached')
            self._admit(allowance)
            attempt = {'status': 'started'}
            state['delivery']['attempts'].append(attempt)
            state['delivery']['status'] = 'started'
            self._save(state)
            writer = Writer(self, f'delivery-{len(state["delivery"]["attempts"]):04d}', allowance)
            try:
                receipt = callback(writer)
                require(type(receipt) is dict and receipt.get('verified') is True and
                        len(encoded(receipt)) <= 2048, 'bounded verified delivery receipt required')
                self._admit(zero_bounds())
            except BaseException:
                # A callback may have sent remotely: preserve uncertainty; do not retry.
                attempt['status'] = 'unknown'
                state['delivery']['status'] = 'unknown'
                self._save(state)
                raise
            finally:
                writer.closed = True
            attempt.update(status='completed', receipt=receipt)
            state['delivery']['status'] = 'completed'
            self._save(state)

    def reconcile_delivery(self, *, remote_absence_verified, evidence):
        """Operator-provided reconciliation; no automatic retry after uncertain send.

        The caller must actually establish remote absence; a boolean is not an
        authentication mechanism. Never use to reset execution consumption.
        """
        require(remote_absence_verified is True and type(evidence) is str and 0 < len(evidence) <= 512,
                'verified remote absence and bounded evidence reference required')
        with self._locked():
            state = self._load()
            require(state['delivery']['status'] in ('started', 'unknown'), 'no uncertain delivery')
            state['delivery']['attempts'][-1].update(status='failed', reconciliation=evidence)
            state['delivery']['status'] = 'failed'
            self._save(state)


def fixture(root):
    """Real entry-point integration fixture: 16 calls, archive/copy/readback/temp I/O.

    Synthetic bytes only. No ZIP, upload, inference, training or external action.
    """
    c = Controller.create(root, [f'item-{i:02d}' for i in range(16)], total_bytes=CONTROL_RESERVE + 65536)
    c.execute(lambda item, writer: writer.write('output', 'sample.bin', item.encode()),
              per_item_bounds=zero_bounds(output=32))
    def delivery(writer):
        payload = b'bounded synthetic delivery fixture\n'
        for category in ('archive', 'upload', 'readback', 'temp'):
            record = writer.write(category, 'fixture.bin', payload)
            require((c.root / record['path']).read_bytes() == payload, 'fixture copy mismatch')
        return {'verified': True, 'scope': 'local artificial byte equality only',
                'sha256': hashlib.sha256(payload).hexdigest()}
    c.deliver(delivery, peak_bounds=zero_bounds(archive=64, upload=64, readback=64, temp=64))
    return c.status()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('fixture', 'status', 'seal-failed'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--processes-stopped-verified', action='store_true')
    p.add_argument('--evidence')
    args = p.parse_args()
    if args.action == 'seal-failed':
        Controller(args.root).seal_failed_execution(
            processes_stopped_verified=args.processes_stopped_verified, evidence=args.evidence)
    print(json.dumps(fixture(args.root) if args.action == 'fixture' else Controller(args.root).status(), indent=2))
