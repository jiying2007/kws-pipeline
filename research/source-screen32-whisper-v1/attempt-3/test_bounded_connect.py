"""Offline-only tests: socketpair fixtures and injected/instrumented transports."""
import importlib.util
from pathlib import Path
import socket
import json
import os
import signal
import tempfile
import multiprocessing
import errno
import sys
import threading
import time
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('bounded_connect', Path(__file__).with_name('bounded_connect.py'))
bc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bc
spec.loader.exec_module(bc)


class Instrumented:
    def __init__(self, size, block=bc.CHUNK):
        self.remaining, self.block, self.calls, self.bytes = size, block, [], 0
    def recv(self, amount):
        self.calls.append(amount)
        n = min(amount, self.block, self.remaining)
        self.remaining -= n
        self.bytes += n
        return b'x' * n


class Contracts(unittest.TestCase):
    def broker(self, size=100, **kwargs):
        return bc.Broker((bc.Asset('wheel-01', 'wheel', size),), time.monotonic()+30, **kwargs)

    def test_exact_cap_formula(self):
        for size in (1, 49, 50, 51, 192268841):
            a = bc.Asset('wheel', 'wheel', size)
            self.assertEqual(a.cap, (size*102+99)//100+1024**2)

    def test_valid_requests(self):
        for host in bc.HOSTS:
            for version in ('1.0', '1.1'):
                raw = f'CONNECT {host}:443 HTTP/{version}\r\nHost: {host}:443\r\n\r\n'.encode()
                self.assertEqual(bc.parse_connect(raw), host)
        self.assertEqual(bc.parse_connect(b'CONNECT files.pythonhosted.org:443 HTTP/1.0\r\n\r\n'), 'files.pythonhosted.org')

    def test_authority_rejection(self):
        for authority in ('evil.test:443','files.pythonhosted.org:80','FILES.PYTHONHOSTED.ORG:443',
                          'files.pythonhosted.org.:443','files.pythonhosted.org:0443',
                          '127.0.0.1:443','[::1]:443','u@files.pythonhosted.org:443',
                          'files.pythonhosted.org:443/path','files.pythonhosted.org:443?x',
                          'files.pythonhosted.org:443#x','files.pythonhosted.org%00:443'):
            with self.subTest(authority=authority), self.assertRaises(bc.Rejected):
                bc.parse_connect(f'CONNECT {authority} HTTP/1.0\r\n\r\n'.encode())

    def test_header_smuggling_rejection(self):
        good = b'CONNECT files.pythonhosted.org:443 HTTP/1.1\r\n'
        for tail in (b'\r\n', b'Host: evil.test:443\r\n\r\n',
                     b'Host: files.pythonhosted.org:443\r\nHost: files.pythonhosted.org:443\r\n\r\n',
                     b'Host: files.pythonhosted.org:443\r\nContent-Length: 1\r\n\r\n',
                     b'Host: files.pythonhosted.org:443\r\nTransfer-Encoding: chunked\r\n\r\n',
                     b' Host: files.pythonhosted.org:443\r\n\r\n',
                     b'Host: files.pythonhosted.org:443\n\r\n',
                     b'Host: files.pythonhosted.org:443\r\nProxy-Authorization: secret\r\n\r\n'):
            with self.subTest(tail=tail), self.assertRaises(bc.Rejected):
                bc.parse_connect(good+tail)
        for raw in (b'GET https://files.pythonhosted.org/ HTTP/1.0\r\n\r\n',
                    b'CONNECT\tfiles.pythonhosted.org:443 HTTP/1.0\r\n\r\n',
                    b'x'*8193, b'\xff\r\n\r\n'):
            with self.assertRaises(bc.Rejected): bc.parse_connect(raw)

    def test_dns_rejects_all_nonpublic_answers(self):
        for ip in ('127.0.0.1','10.0.0.1','169.254.169.254','192.0.2.1','224.0.0.1'):
            rows = [(socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP,'',(ip,443))]
            with mock.patch.object(bc.socket,'getaddrinfo',return_value=rows):
                with self.subTest(ip=ip), self.assertRaises(bc.Rejected): bc.resolve_public('files.pythonhosted.org')

    def test_dns_chooses_one_numeric_address_without_fallback(self):
        row = (socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP,'',('8.8.8.8',443))
        with mock.patch.object(bc.socket,'getaddrinfo',return_value=[row]):
            self.assertEqual(bc.resolve_public('files.pythonhosted.org'), row)
        with mock.patch.object(bc.socket,'getaddrinfo',return_value=[row,(socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP,'',('127.0.0.1',443))]):
            with self.assertRaises(bc.Rejected): bc.resolve_public('files.pythonhosted.org')

    def test_asset_quota_atomic_no_extra_reads(self):
        b=self.broker(); b.begin_asset('wheel-01')
        sources=[Instrumented(2*1024**2) for _ in range(8)]
        def consume(s):
            try:
                while True: b._recv(s,'downstream')
            except bc.Rejected: pass
        ts=[threading.Thread(target=consume,args=(s,)) for s in sources]
        for t in ts:t.start()
        for t in ts:t.join()
        cap=b.plan[0].cap
        self.assertEqual(sum(s.bytes for s in sources),cap)
        self.assertEqual(b.total,cap)
        self.assertEqual(b.failed,'byte_cap')
        calls=sum(len(s.calls) for s in sources)
        with self.assertRaises(bc.Rejected):b._recv(sources[0],'downstream')
        self.assertEqual(sum(len(s.calls) for s in sources),calls)

    def test_bidirectional_single_debit(self):
        b=self.broker();b.begin_asset('wheel-01')
        b._recv(Instrumented(10),'upstream');b._recv(Instrumented(20),'downstream')
        r=b.receipt();self.assertEqual(r['total_read_bytes'],30)
        self.assertEqual(r['assets']['wheel-01']['total_read_bytes'],30)
        self.assertEqual(r['assets']['wheel-01']['upstream_read_bytes'],10)
        self.assertEqual(r['assets']['wheel-01']['downstream_read_bytes'],20)

    def test_aggregate_overflow_across_assets_no_reset(self):
        # Seed a trusted test ledger near the real immutable global ceiling.
        b=bc.Broker((bc.Asset('a','source',1),bc.Asset('b','source',1)),time.monotonic()+30)
        b.total=bc.AGGREGATE_CAP-30
        b.begin_asset('a');b._recv(Instrumented(10),'downstream');b.finish_asset()
        b.begin_asset('b');s=Instrumented(100);self.assertEqual(len(b._recv(s,'upstream')),20)
        with self.assertRaises(bc.Rejected):b._recv(s,'downstream')
        self.assertEqual(s.bytes,20);self.assertEqual(len(s.calls),1)
        self.assertEqual(b.total,bc.AGGREGATE_CAP)

    def test_plan_monotonic_and_one_use(self):
        b=self.broker()
        with self.assertRaises(bc.Rejected):b.begin_asset('unknown')
        b.begin_asset('wheel-01')
        with self.assertRaises(bc.Rejected):b.begin_asset('wheel-01')
        b.finish_asset()
        with self.assertRaises(bc.Rejected):b.begin_asset('wheel-01')
        self.assertTrue(b.receipt()['plan_complete'])

    def test_invalid_plan_and_deadline(self):
        for size in (0,-1,True,1024**3+1):
            with self.assertRaises(bc.Rejected):bc.Asset('wheel','wheel',size)
        a=bc.Asset('a','wheel',1)
        for plan in ((),(a,a)):
            with self.assertRaises(bc.Rejected):bc.Broker(plan,time.monotonic()+10)
        for deadline in (0,float('inf'),float('nan'),time.monotonic()+1201):
            with self.assertRaises(bc.Rejected):bc.Broker((a,),deadline)

    def test_deadline_prevents_receive(self):
        now=[1.0];b=bc.Broker((bc.Asset('a','wheel',1),),2.0,clock=lambda:now[0]);b.begin_asset('a')
        now[0]=2.0;s=Instrumented(1)
        with self.assertRaises(bc.Rejected):b._recv(s,'downstream')
        self.assertEqual(s.calls,[]);self.assertEqual(b.failed,'deadline')

    def test_socketpair_blackbox_relay(self):
        client, accepted=socket.socketpair();origin,remote=socket.socketpair()
        b=self.broker(resolver=lambda host:host,connector=lambda row,timeout:origin);b.begin_asset('wheel-01')
        result=[];t=threading.Thread(target=lambda:result.append(b.handle(accepted)));t.start()
        try:
            client.settimeout(2);remote.settimeout(2)
            client.sendall(b'CONNECT files.pythonhosted.org:443 HTTP/1.1\r\nHost: files.pythonhosted.org:443\r\n\r\n')
            self.assertEqual(client.recv(4096),b'HTTP/1.1 200 Connection Established\r\n\r\n')
            client.sendall(b'opaque-client');self.assertEqual(remote.recv(100),b'opaque-client')
            remote.sendall(b'opaque-origin');self.assertEqual(client.recv(100),b'opaque-origin')
            client.shutdown(socket.SHUT_WR);remote.shutdown(socket.SHUT_WR)
            t.join(3);self.assertFalse(t.is_alive());self.assertEqual(result,[None])
            self.assertEqual(b.total,len(b'opaque-clientopaque-origin'))
            b.finish_asset();self.assertTrue(b.receipt()['plan_complete'])
        finally:
            client.close();remote.close();b.abort();t.join(3)

    def test_invalid_request_never_resolves(self):
        client,accepted=socket.socketpair();resolver=mock.Mock()
        b=self.broker(resolver=resolver);b.begin_asset('wheel-01')
        client.sendall(b'CONNECT evil.test:443 HTTP/1.0\r\n\r\n')
        try:self.assertEqual(b.handle(accepted),'destination')
        finally:client.close()
        resolver.assert_not_called();self.assertEqual(b.total,0)

    def test_total_connection_limit_is_not_reset_by_disconnect(self):
        b=self.broker();b.begin_asset('wheel-01')
        b.assets['wheel-01']['connections_started']=4
        client,accepted=socket.socketpair()
        try:self.assertEqual(b.handle(accepted),'connection_limit')
        finally:client.close()
        self.assertEqual(b.total,0)

    def test_sanitized_exception(self):
        def bad(host):raise RuntimeError('secret URL and private text')
        b=self.broker(resolver=bad);b.begin_asset('wheel-01')
        client,accepted=socket.socketpair();client.sendall(b'CONNECT files.pythonhosted.org:443 HTTP/1.0\r\n\r\n')
        try:self.assertEqual(b.handle(accepted),'io')
        finally:client.close()
        self.assertNotIn('secret',str(b.receipt()))


class TeardownContracts(unittest.TestCase):
    class ResetSocket:
        def __init__(self, sock, after_reads=0, send_errno=None):
            self.sock=sock;self.after_reads=after_reads;self.send_errno=send_errno
        def __getattr__(self,key):return getattr(self.sock,key)
        def recv(self,amount):
            if self.after_reads==0:raise ConnectionResetError(errno.ECONNRESET,'remote text not retained')
            self.after_reads-=1;return self.sock.recv(amount)
        def send(self,data):
            if self.send_errno:raise OSError(self.send_errno,'remote text not retained')
            return self.sock.send(data)

    def reset_case(self, after_bytes):
        client,accepted=socket.socketpair();origin,remote=socket.socketpair()
        wrapped=self.ResetSocket(origin,after_reads=1 if after_bytes else 0)
        b=bc.Broker((bc.Asset('a','wheel',100),),time.monotonic()+30,
                    resolver=lambda h:h,connector=lambda r,t:wrapped)
        b.begin_asset('a');result=[]
        thread=threading.Thread(target=lambda:result.append(b.handle(accepted)));thread.start()
        try:
            client.settimeout(2);remote.settimeout(2)
            client.sendall(b'CONNECT files.pythonhosted.org:443 HTTP/1.0\r\n\r\n')
            self.assertIn(b'200',client.recv(100))
            if after_bytes:
                remote.sendall(b'already-read');self.assertEqual(client.recv(100),b'already-read')
            remote.sendall(b'reset-trigger');thread.join(2)
            self.assertFalse(thread.is_alive());self.assertEqual(result,['transport_closed'])
            self.assertIsNone(b.failed);self.assertEqual(b.total,len(b'already-read') if after_bytes else 0)
            meter=b.receipt()['assets']['a'];self.assertEqual(meter['transport_closed'],1)
            # Ordinary close is not completion or permission to advance.
            self.assertFalse(b.receipt()['plan_complete']);self.assertFalse(b.sealed)
        finally:
            client.close();remote.close();b.abort();thread.join(2)

    def test_reset_before_tunnel_bytes_is_not_success(self):self.reset_case(False)
    def test_reset_after_tunnel_bytes_preserves_counters(self):self.reset_case(True)

    def test_brokenpipe_counts_bytes_read_before_failed_forward(self):
        client,accepted=socket.socketpair();origin,remote=socket.socketpair()
        wrapped=self.ResetSocket(origin,after_reads=100,send_errno=errno.EPIPE)
        b=bc.Broker((bc.Asset('a','wheel',100),),time.monotonic()+30,
                    resolver=lambda h:h,connector=lambda r,t:wrapped)
        b.begin_asset('a');result=[]
        thread=threading.Thread(target=lambda:result.append(b.handle(accepted)));thread.start()
        try:
            client.settimeout(2)
            client.sendall(b'CONNECT files.pythonhosted.org:443 HTTP/1.0\r\n\r\n')
            self.assertIn(b'200',client.recv(100));client.sendall(b'count-before-forward')
            thread.join(2);self.assertEqual(result,['transport_closed'])
            self.assertEqual(b.total,len(b'count-before-forward'));self.assertIsNone(b.failed)
        finally:
            client.close();remote.close();b.abort();thread.join(2)

    def test_preconnect_reset_remains_fatal(self):
        def connect(row,timeout):raise ConnectionResetError(errno.ECONNRESET,'hidden')
        b=bc.Broker((bc.Asset('a','wheel',100),),time.monotonic()+30,
                    resolver=lambda h:h,connector=connect)
        b.begin_asset('a');client,accepted=socket.socketpair()
        client.sendall(b'CONNECT files.pythonhosted.org:443 HTTP/1.0\r\n\r\n')
        try:self.assertEqual(b.handle(accepted),'io');self.assertEqual(b.failed,'io')
        finally:client.close()

    def test_close_classification_cannot_hide_policy_failure_or_deadline(self):
        for reason in ('byte_cap','destination','request','dns','plan','poisoned'):
            b=bc.Broker((bc.Asset('a','wheel',100),),time.monotonic()+30);b.begin_asset('a');b.abort(reason)
            self.assertFalse(b._ordinary_transport_close(ConnectionResetError(errno.ECONNRESET,'hidden'),True))
            self.assertEqual(b.failed,reason)
        now=[1.0];b=bc.Broker((bc.Asset('a','wheel',100),),2.0,clock=lambda:now[0]);b.begin_asset('a');now[0]=2.0
        self.assertFalse(b._ordinary_transport_close(BrokenPipeError(errno.EPIPE,'hidden'),True))
        self.assertEqual(b.failed,'deadline')
        b=bc.Broker((bc.Asset('a','wheel',100),),time.monotonic()+30);b.begin_asset('a')
        self.assertFalse(b._ordinary_transport_close(OSError(errno.EIO,'hidden'),True))
        self.assertFalse(b._ordinary_transport_close(bc.Rejected('byte_cap'),True))



class ProcessContracts(unittest.TestCase):
    def test_clean_seal_without_upstream_eof(self):
        try:
            probe=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        except PermissionError:
            if os.environ.get('GITHUB_ACTIONS') == 'true':
                raise
            self.skipTest('Executor denies Unix listener socket creation; process fixture requires permitted local sockets')
        else:
            probe.close()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); sockdir=root/'socket'; evidence=root/'evidence'
            sockdir.mkdir(mode=0o700);evidence.mkdir(mode=0o700)
            plan={'schema':'bounded-connect-plan-v1','asset_id':'wheel-01','kind':'wheel',
                  'expected_size':100,'prior_charged_bytes':1000,'deadline_seconds':10}
            (root/'plan.json').write_text(json.dumps(plan))
            (sockdir/'deadline.json').write_text(json.dumps({'deadline_monotonic':time.monotonic()+10}))
            origin,remote=socket.socketpair()
            def child():
                remote.close()
                original=bc.Broker
                bc.Broker=lambda *a,**kw:original(*a,resolver=lambda h:h,
                    connector=lambda row,timeout:origin,**kw)
                raise SystemExit(bc.serve_one(root/'plan.json',sockdir/'broker.sock',evidence))
            process=multiprocessing.get_context('fork').Process(target=child);process.start();origin.close()
            client=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            try:
                deadline=time.monotonic()+3
                while not (evidence/'ready.json').exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue((evidence/'ready.json').exists())
                self.assertEqual((sockdir/'broker.sock').stat().st_mode & 0o777,0o600)
                client.settimeout(2);remote.settimeout(2);client.connect(str(sockdir/'broker.sock'))
                client.sendall(b'CONNECT files.pythonhosted.org:443 HTTP/1.0\r\n\r\n')
                self.assertIn(b'200',client.recv(100))
                client.sendall(b'tls-client');self.assertEqual(remote.recv(100),b'tls-client')
                remote.sendall(b'tls-origin');self.assertEqual(client.recv(100),b'tls-origin')
                # Both sockets deliberately stay open: sealing must not wait for EOF.
                os.kill(process.pid,signal.SIGUSR1);process.join(3)
                self.assertEqual(process.exitcode,0)
                receipt=json.loads((evidence/'receipt.json').read_text())
                self.assertEqual(receipt['status'],'complete')
                self.assertEqual(receipt['charged_bytes'],20)
                self.assertEqual(receipt['total_read_bytes'],1020)
                self.assertFalse((sockdir/'broker.sock').exists())
            finally:
                client.close();remote.close()
                if process.is_alive():process.kill();process.join()

    def test_socketpair_seal_closes_keepalive_without_eof(self):
        client,accepted=socket.socketpair();origin,remote=socket.socketpair()
        b=bc.Broker((bc.Asset('a','wheel',100),),time.monotonic()+30,
                    resolver=lambda h:h,connector=lambda r,t:origin)
        b.begin_asset('a');result=[]
        thread=threading.Thread(target=lambda:result.append(b.handle(accepted)));thread.start()
        try:
            client.settimeout(2);remote.settimeout(2)
            client.sendall(b'CONNECT files.pythonhosted.org:443 HTTP/1.0\r\n\r\n')
            self.assertIn(b'200',client.recv(100))
            client.sendall(b'abc');self.assertEqual(remote.recv(10),b'abc')
            b.seal();thread.join(2)
            self.assertFalse(thread.is_alive());self.assertEqual(result,[None])
            self.assertEqual(b.total,3);self.assertIsNone(b.failed)
            b.finish_asset();self.assertTrue(b.receipt()['plan_complete'])
        finally:
            client.close();remote.close();b.abort();thread.join(2)

    def test_seal_between_listener_condition_and_scope_check(self):
        b=bc.Broker((bc.Asset('a','wheel',10),),time.monotonic()+30)
        b.begin_asset('a')
        condition_already_evaluated=True
        # Deterministic SIGUSR1 interleaving: loop condition passed, then seal.
        b.seal()
        self.assertTrue(condition_already_evaluated)
        self.assertFalse(b.accepting());self.assertIsNone(b.failed)
        b.abort('byte_cap')
        with self.assertRaises(bc.Rejected):b.accepting()
        self.assertEqual(b.failed,'byte_cap')

    def test_seal_does_not_clear_failure(self):
        b=bc.Broker((bc.Asset('a','wheel',10),),time.monotonic()+30)
        b.begin_asset('a');b.abort('byte_cap')
        with self.assertRaises(bc.Rejected):b.seal()
        self.assertEqual(b.receipt()['failure_code'],'byte_cap')

    def test_sealed_ledger_never_reads_more(self):
        b=bc.Broker((bc.Asset('a','wheel',10),),time.monotonic()+30)
        b.begin_asset('a');s=Instrumented(100);b.seal()
        with self.assertRaises(bc.Rejected):b._recv(s,'downstream')
        self.assertEqual(s.calls,[])

    def test_plan_rejects_duplicates_and_untrusted_caps(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'plan.json'
            path.write_text('{"schema":"bounded-connect-plan-v1","schema":"evil"}')
            with self.assertRaises(bc.Rejected):bc.load_plan(path)
            plan={'schema':'bounded-connect-plan-v1','asset_id':'a','kind':'wheel',
                  'expected_size':100,'prior_charged_bytes':0,'deadline_seconds':10,
                  'asset_cap':9999999999}
            path.write_text(json.dumps(plan))
            with self.assertRaises(bc.Rejected):bc.load_plan(path)

    def test_private_socket_directory(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);path.chmod(0o755)
            with self.assertRaises(bc.Rejected):bc.private_directory(path)

    def test_inherited_aggregate_limit(self):
        b=bc.Broker((bc.Asset('a','wheel',10),),time.monotonic()+30,
                    prior_charged_bytes=bc.AGGREGATE_CAP-7)
        b.begin_asset('a');s=Instrumented(100)
        self.assertEqual(len(b._recv(s,'downstream')),7)
        with self.assertRaises(bc.Rejected):b._recv(s,'downstream')
        self.assertEqual(s.calls,[7]);self.assertEqual(b.receipt()['charged_bytes'],7)



if __name__=='__main__':unittest.main()
