"""Dormant CONNECT transport core; not wired to any execution entry point.

No TLS termination or HTTP response parsing. The budget counts bytes returned by
socket.recv in BOTH tunnel directions, before forwarding. CONNECT request/reply
headers, kernel buffering, TCP/IP overhead and retransmission are excluded.
Ciphertext accounting is not the historical exact response-body accounting.

A trusted supervisor owns this object, its immutable plan and asset transitions.
The data connection has no control-plane protocol. One broker lifetime covers the
configured asset plan. Serial one-shot brokers may inherit the trusted host
ledger only after a clean final receipt. A crash or poisoned session MUST abort
the entire attempt, never retry the failed asset with a fresh ledger. The eventual launcher must enforce this and
network-none containment; this module alone does not enforce client egress.
The launcher must also kill the entire broker at the absolute deadline: libc DNS
resolution can block beyond a Python socket timeout.
"""
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import signal
import stat
import argparse
import errno
import math
import select
import socket
import threading
import time

AGGREGATE_CAP = 1536 * 1024**2
HEADER_CAP = 8192
CHUNK = 65536
MAX_CONNECTIONS = 4
HOSTS = frozenset(('raw.githubusercontent.com', 'files.pythonhosted.org',
                  'download.pytorch.org', 'download-r2.pytorch.org',
                  'openaipublic.azureedge.net'))
CODES = frozenset(('plan', 'state', 'request', 'destination', 'dns', 'connection',
                   'connection_limit', 'deadline', 'byte_cap', 'io', 'poisoned', 'sealed'))


class Rejected(Exception):
    """Fixed public category only; never interpolate remote input or exceptions."""
    def __init__(self, code):
        self.code = code if code in CODES else 'io'
        super().__init__(self.code)


def require(ok, code):
    if not ok:
        raise Rejected(code)


@dataclass(frozen=True)
class Asset:
    asset_id: str
    kind: str
    expected_size: int

    def __post_init__(self):
        require(type(self.asset_id) is str and 1 <= len(self.asset_id) <= 64 and
                all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in self.asset_id), 'plan')
        require(self.kind in ('wheel', 'source', 'model') and
                type(self.expected_size) is int and 0 < self.expected_size <= 1024**3, 'plan')

    @property
    def cap(self):
        # Exact integer ceil(size * 1.02), no floating-point rounding.
        return min(AGGREGATE_CAP, (self.expected_size * 102 + 99) // 100 + 1024**2) if self.kind == 'wheel' else AGGREGATE_CAP


def parse_connect(header):
    require(type(header) is bytes and len(header) <= HEADER_CAP and
            header.endswith(b'\r\n\r\n'), 'request')
    try:
        lines = header[:-4].decode('ascii').split('\r\n')
    except UnicodeDecodeError:
        raise Rejected('request') from None
    require(lines and all('\r' not in x and '\n' not in x for x in lines), 'request')
    fields = lines[0].split(' ')
    require(len(fields) == 3 and fields[0] == 'CONNECT' and
            fields[2] in ('HTTP/1.0', 'HTTP/1.1'), 'request')
    authority = fields[1]
    require(authority in {host + ':443' for host in HOSTS}, 'destination')
    # No request body, duplicate Host, folding, auth, transfer framing or arbitrary headers.
    require(len(lines) <= 2, 'request')
    if len(lines) == 2:
        require(lines[1].lower() == 'host: ' + authority, 'request')
    else:
        require(fields[2] == 'HTTP/1.0', 'request')
    return authority[:-4]


def resolve_public(host):
    """Resolve once; every answer must be public. Connect by numeric IP afterward."""
    rows = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    require(0 < len(rows) <= 32, 'dns')
    for family, socktype, proto, _, address in rows:
        require(family in (socket.AF_INET, socket.AF_INET6) and
                socktype == socket.SOCK_STREAM and proto == socket.IPPROTO_TCP and
                address[1] == 443, 'dns')
        try:
            ip = ipaddress.ip_address(address[0])
            public = ip.is_global and not ip.is_multicast and not ip.is_unspecified
        except ValueError:
            public = False
        require(public, 'dns')
        require(family != socket.AF_INET6 or address[2:] == (0, 0), 'dns')
    return rows[0]  # No address fallback/retry.


def connect_numeric(row, timeout):
    family, socktype, proto, _, address = row
    sock = socket.socket(family, socktype, proto)
    try:
        sock.settimeout(timeout)
        sock.connect(address)
        return sock
    except BaseException:
        sock.close()
        raise


class Broker:
    """Thread-safe, one-shot plan ledger and CONNECT relay.

    begin_asset/finish_asset are trusted-parent methods, NEVER data-socket commands.
    Receipts are snapshots, not restart tokens. No caller-selected quota exists.
    Source/model traffic is charged once to aggregate and once to its asset meter;
    asset meters are views, not a second aggregate debit. Their exact plaintext
    size/hash checks remain the caller's responsibility.
    """
    def __init__(self, plan, deadline, *, clock=time.monotonic,
                 resolver=resolve_public, connector=connect_numeric, prior_charged_bytes=0):
        require(type(plan) in (tuple, list) and 0 < len(plan) <= 128 and
                all(type(a) is Asset for a in plan), 'plan')
        self.plan = tuple(plan)
        require(len({a.asset_id for a in self.plan}) == len(self.plan), 'plan')
        require(type(deadline) in (int, float) and math.isfinite(deadline) and
                clock() < deadline <= clock() + 1200, 'deadline')
        self.deadline, self.clock = deadline, clock
        self.resolver, self.connector = resolver, connector
        self.lock = threading.RLock()
        self.index, self.active, self.connections = 0, None, 0
        require(type(prior_charged_bytes) is int and 0 <= prior_charged_bytes < AGGREGATE_CAP, 'plan')
        self.prior_charged_bytes = prior_charged_bytes
        self.failed, self.total = None, prior_charged_bytes
        self.sockets = set()
        self.sealed = False
        self.assets = {}

    def _check(self, *, allow_sealed=False):
        require(self.failed is None, 'poisoned')
        require(allow_sealed or not self.sealed, 'sealed')
        if self.clock() >= self.deadline:
            self._poison('deadline')
            raise Rejected('deadline')

    def _poison(self, code):
        self.failed = self.failed or (code if code in CODES else 'io')
        for sock in tuple(self.sockets):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def abort(self, code='io'):
        with self.lock:
            self._poison(code)

    def begin_asset(self, asset_id):
        with self.lock:
            self._check()
            require(self.active is None and self.connections == 0 and
                    self.index < len(self.plan) and
                    self.plan[self.index].asset_id == asset_id, 'state')
            self.active = self.plan[self.index]
            self.assets[asset_id] = {'upstream_read_bytes': 0, 'downstream_read_bytes': 0,
                                     'total_read_bytes': 0, 'connections_started': 0,
                                     'transport_closed': 0,
                                     'complete': False}

    def accepting(self):
        # Signal may seal between the listener loop condition and this check.
        # Intentional seal ends acceptance, but never suppresses prior failure
        # or an expired absolute deadline.
        with self.lock:
            self._check(allow_sealed=True)
            return not self.sealed

    def seal(self):
        with self.lock:
            self._check()
            self.sealed = True

    def finish_asset(self):
        with self.lock:
            self._check(allow_sealed=True)
            require(self.active is not None and self.connections == 0, 'state')
            self.assets[self.active.asset_id]['complete'] = True
            self.active = None
            self.index += 1

    def receipt(self):
        with self.lock:
            return {'schema': 'bounded-connect-ledger-v1',
                    'byte_scope': 'bidirectional_tunnel_socket_recv_before_forwarding',
                    'aggregate_cap': AGGREGATE_CAP, 'total_read_bytes': self.total,
                    'prior_charged_bytes': self.prior_charged_bytes,
                    'charged_bytes': self.total - self.prior_charged_bytes,
                    'failure_code': self.failed, 'plan_complete': self.index == len(self.plan),
                    'assets': {k: dict(v) for k, v in self.assets.items()}}

    def _recv(self, source, direction):
        # Source is nonblocking. Reserving and charging under one lock avoids quota
        # races without holding a global lock during a blocking network operation.
        with self.lock:
            self._check()
            require(self.active is not None, 'state')
            meter = self.assets[self.active.asset_id]
            remaining = min(AGGREGATE_CAP - self.total,
                            self.active.cap - meter['total_read_bytes'])
            if remaining <= 0:
                self._poison('byte_cap')
                raise Rejected('byte_cap')
            data = source.recv(min(CHUNK, remaining))
            self.total += len(data)
            meter['total_read_bytes'] += len(data)
            meter[direction + '_read_bytes'] += len(data)
            return data

    def _ordinary_transport_close(self, error, established):
        # Only established tunnel teardown. A reset during DNS/connect/CONNECT
        # negotiation, arbitrary I/O error, or earlier policy failure stays fatal.
        if not (established and isinstance(error, OSError) and error.errno in
                (errno.ECONNRESET, errno.EPIPE, errno.ENOTCONN)):
            return False
        with self.lock:
            if self.failed is not None:
                return False
            if self.clock() >= self.deadline:
                self._poison('deadline')
                return False
            require(self.active is not None, 'state')
            self.assets[self.active.asset_id]['transport_closed'] += 1
            return True

    def handle(self, client):
        """Fixed failure code/None, or transport_closed (never artifact success)."""
        origin, admitted, established = None, False, False
        try:
            with self.lock:
                self._check()
                require(self.active is not None, 'state')
                require(self.connections < MAX_CONNECTIONS and
                        self.assets[self.active.asset_id]['connections_started'] <
                        (128 if self.active.kind == 'source' else MAX_CONNECTIONS),
                        'connection_limit')
                self.connections += 1
                admitted = True
                self.sockets.add(client)
                self.assets[self.active.asset_id]['connections_started'] += 1
            client.settimeout(min(10, self.deadline - self.clock()))
            header = bytearray()
            while not header.endswith(b'\r\n\r\n'):
                self._check()
                require(len(header) < HEADER_CAP, 'request')
                # Read exactly through the CONNECT delimiter: no unaccounted
                # coalesced TLS bytes can enter this control-header buffer.
                client.settimeout(min(10, max(0.001, self.deadline - self.clock())))
                part = client.recv(1)
                require(bool(part), 'request')
                header.extend(part)
            host = parse_connect(bytes(header))
            self._check()
            row = self.resolver(host)
            self._check()
            origin = self.connector(row, min(45, self.deadline - self.clock()))
            with self.lock:
                self._check()
                self.sockets.add(origin)
            client.settimeout(min(10, max(0.001, self.deadline - self.clock())))
            client.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            client.setblocking(False)
            origin.setblocking(False)
            pending = {client: bytearray(), origin: bytearray()}
            live = {client, origin}
            peers = {client: origin, origin: client}
            established = True
            while live or any(pending.values()):
                with self.lock:
                    self._check()
                readers = [s for s in live if not pending[peers[s]]]
                writers = [s for s in pending if pending[s]]
                readable, writable, _ = select.select(readers, writers, [],
                    min(0.25, max(0, self.deadline - self.clock())))
                for target in writable:
                    try:
                        sent = target.send(pending[target])
                    except BlockingIOError:
                        continue
                    require(sent > 0, 'io')
                    del pending[target][:sent]
                    if not pending[target] and peers[target] not in live:
                        target.shutdown(socket.SHUT_WR)
                for source in readable:
                    try:
                        data = self._recv(source, 'upstream' if source is client else 'downstream')
                    except BlockingIOError:
                        continue
                    if data:
                        pending[peers[source]].extend(data)
                    else:
                        live.remove(source)
                        if not pending[peers[source]]:
                            peers[source].shutdown(socket.SHUT_WR)
            with self.lock:
                self.assets[self.active.asset_id]['transport_closed'] += 1
            return None
        except BaseException as error:
            if self._ordinary_transport_close(error, established):
                return 'transport_closed'
            code = error.code if isinstance(error, Rejected) else 'deadline' if isinstance(error, TimeoutError) else 'io'
            if code == 'sealed':
                return None
            self.abort(code)
            return code
        finally:
            with self.lock:
                for sock in (client, origin):
                    if sock is not None:
                        self.sockets.discard(sock)
                        sock.close()
                if admitted:
                    self.connections -= 1


PLAN_KEYS = frozenset(('schema', 'asset_id', 'kind', 'expected_size',
                       'prior_charged_bytes', 'deadline_seconds'))


def load_plan(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= HEADER_CAP, 'plan')
    with path.open('rb') as stream:
        raw = stream.read(HEADER_CAP + 1)
    require(len(raw) <= HEADER_CAP, 'plan')
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'plan')
            result[key] = value
        return result
    plan = json.loads(raw, object_pairs_hook=unique_pairs)
    require(type(plan) is dict and set(plan) == PLAN_KEYS and
            plan['schema'] == 'bounded-connect-plan-v1', 'plan')
    require(type(plan['deadline_seconds']) is int and
            1 <= plan['deadline_seconds'] <= 1200, 'plan')
    require(type(plan['prior_charged_bytes']) is int and
            0 <= plan['prior_charged_bytes'] < AGGREGATE_CAP, 'plan')
    asset = Asset(plan['asset_id'], plan['kind'], plan['expected_size'])
    return plan, asset


def private_directory(path):
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and
            stat.S_IMODE(info.st_mode) == 0o700, 'state')


def write_once(path, value):
    # Every evidence directory is new. Refuse overwrite/symlink destinations.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def serve_one(plan_path, socket_path, evidence):
    """Fixed one-asset process; trusted host SIGUSR1 seals after artifact verification.

    A clean broker receipt does not prove artifact validity or worker success.
    The trusted host must verify both before incrementing its ledger/advancing.
    SIGTERM/SIGINT always fail. SIGUSR1 cannot clear earlier poison.
    An outer process watchdog is required for deadline-stuck libc DNS threads.
    """
    plan, asset = load_plan(plan_path)
    private_directory(socket_path.parent)
    private_directory(evidence)
    require(not socket_path.exists() and not socket_path.is_symlink(), 'state')
    deadline_path = socket_path.parent / 'deadline.json'
    deadline_info = deadline_path.lstat()
    require(stat.S_ISREG(deadline_info.st_mode) and deadline_info.st_size <= 1024, 'deadline')
    with deadline_path.open('rb') as stream:
        deadline_raw = stream.read(1025)
    require(len(deadline_raw) <= 1024, 'deadline')
    absolute = json.loads(deadline_raw)
    require(type(absolute) is dict and set(absolute) == {'deadline_monotonic'} and
            type(absolute['deadline_monotonic']) in (int, float) and
            math.isfinite(absolute['deadline_monotonic']), 'deadline')
    deadline = min(time.monotonic() + plan['deadline_seconds'], absolute['deadline_monotonic'])
    broker = Broker((asset,), deadline,
                    prior_charged_bytes=plan['prior_charged_bytes'])
    broker.begin_asset(asset.asset_id)
    stopped = threading.Event()
    handlers = []
    slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_handlers = {}
    def stop(signum, frame):
        if signum == signal.SIGUSR1:
            try:
                broker.seal()
            except Rejected as error:
                broker.abort(error.code)
        else:
            broker.abort('state')
        stopped.set()
    def run_connection(client):
        try:
            broker.handle(client)
        finally:
            slots.release()
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            old_handlers[sig] = signal.signal(sig, stop)
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        listener.listen(MAX_CONNECTIONS)
        listener.settimeout(0.1)
        write_once(evidence / 'ready.json', {'schema': 'bounded-connect-ready-v1',
                   'status': 'ready', 'asset_id': asset.asset_id})
        while not stopped.is_set():
            if not broker.accepting():
                break
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            if not slots.acquire(blocking=False):
                client.close()
                raise Rejected('connection_limit')
            thread = threading.Thread(target=run_connection, args=(client,), daemon=True)
            handlers = [t for t in handlers if t.is_alive()]
            handlers.append(thread)
            thread.start()
        listener.close()
        # Sealing prevents new recv calls atomically. Relay loops notice within
        # one select interval; do not wait for origin keepalive EOF or drain bytes.
        # A blocked header/DNS/connection operation fails closed if not finished.
        drain_end = min(broker.deadline, time.monotonic() + 1)
        for thread in handlers:
            thread.join(max(0, drain_end - time.monotonic()))
        require(not any(t.is_alive() for t in handlers), 'state')
        require(broker.total > broker.prior_charged_bytes, 'state')
        broker.finish_asset()
    except BaseException as error:
        broker.abort(error.code if isinstance(error, Rejected) else 'io')
    finally:
        listener.close()
        broker.abort(broker.failed) if broker.failed else None
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        receipt = broker.receipt()
        receipt.update(status='complete' if receipt['plan_complete'] and not broker.failed else 'failed',
                       asset_id=asset.asset_id, kind=asset.kind, expected_size=asset.expected_size,
                       asset_cap=asset.cap)
        write_once(evidence / 'receipt.json', receipt)
    return 0 if receipt['status'] == 'complete' else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--socket', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        # Isolated Python excludes script-directory imports; load the adjacent
        # reviewed proof module explicitly, before opening any socket.
        import importlib.util
        proof_path = Path(__file__).with_name('bounded_host.py')
        spec = importlib.util.spec_from_file_location('bounded_host', proof_path)
        proof = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proof)
        scope = proof.verify_effective_scope('broker')
        private_directory(args.evidence)
        write_once(args.evidence / 'kernel-scope.json', scope)
        return serve_one(args.plan, args.socket, args.evidence)
    except BaseException as error:
        code = error.code if isinstance(error, Rejected) else 'io'
        print(json.dumps({'status': 'failed', 'failure_code': code}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
