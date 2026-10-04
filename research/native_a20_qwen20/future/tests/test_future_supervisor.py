"""Pure mock tests. No subprocess, /proc read, model, FFT, or audio execution."""
import importlib.util
import ast
import json
import pathlib
import signal
import unittest
from unittest import mock

PATH=pathlib.Path(__file__).resolve().parents[1]/'src/launch_once.py'
SOURCE=PATH.read_text()
SPEC=importlib.util.spec_from_file_location('future_launcher',PATH)
launch=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)
PID=424242  # Invented test identifier; never inspected on the host.
STATUS='VmRSS:\t123 kB\nVmHWM:\t456 kB\nThreads:\t1\nVmSize:\t789 kB\n'
IO=''.join(f'{key}: {number}\n' for number,key in enumerate(launch.IO_COUNTERS))


class FakeProcess:
    pid=PID
    def __init__(self, polls, waitcode=0):
        self.polls=list(polls)
        self.last=None
        self.returncode=None
        self.waitcode=waitcode
        self.wait_calls=0
        self.kills=[]
    def poll(self):
        if self.polls:self.last=self.polls.pop(0)
        if self.last is not None:self.returncode=self.last
        return self.last
    def wait(self):
        self.wait_calls+=1
        self.returncode=self.waitcode
        return self.returncode
    def kill(self,pid,sig):
        self.kills.append((pid,sig))
        self.waitcode=-sig


def reader(status=STATUS,children='',io=IO):
    values={f'/proc/{PID}/status':status,
            f'/proc/{PID}/task/{PID}/children':children,
            f'/proc/{PID}/io':io}
    calls=[]
    def read(path):
        calls.append(str(path))
        value=values[str(path)]
        if isinstance(value,BaseException):raise value
        return value
    read.calls=calls
    return read


def denied(suffix):
    return PermissionError(13,'Permission denied',f'/proc/{PID}/{suffix}')


def missing(suffix):
    return FileNotFoundError(2,'No such file or directory',f'/proc/{PID}/{suffix}')


class FutureSupervisorTests(unittest.TestCase):
    def setUp(self):
        # Defensive test harness: an accidental production read/launch/kill fails.
        self.no_read=mock.patch.object(pathlib.Path,'read_text',side_effect=AssertionError('real read forbidden')).start()
        self.no_popen=mock.patch.object(launch.subprocess,'Popen',side_effect=AssertionError('subprocess forbidden')).start()
        self.no_kill=mock.patch.object(launch.os,'killpg',side_effect=AssertionError('real kill forbidden')).start()
        self.addCleanup(mock.patch.stopall)

    def tearDown(self):
        self.no_read.assert_not_called()
        self.no_popen.assert_not_called()
        self.no_kill.assert_not_called()

    def test_normal_observation_has_all_critical_fields(self):
        state=launch.observe_process(FakeProcess([None,None]),reader())
        self.assertEqual(state['critical_monitoring'],'OBSERVED')
        self.assertEqual(state['critical_errors'],{})
        self.assertEqual((state['VmRSS'],state['Threads'],state['children']),(123,1,''))
        self.assertIsNone(launch.guard_reason(state,1,0))
        for index,key in enumerate(launch.IO_COUNTERS):
            self.assertEqual(state['io'][key],{'status':'AVAILABLE','value':index})

    def test_optional_io_denied_live_keeps_exact_error_without_fabricated_zero(self):
        error=denied('io')
        read=reader(io=error)
        state=launch.observe_process(FakeProcess([None,None]),read)
        self.assertEqual(state['critical_monitoring'],'OBSERVED')
        self.assertIsNone(launch.guard_reason(state,1,0))
        self.assertIn(f'/proc/{PID}/task/{PID}/children',read.calls)
        for counter in state['io'].values():
            self.assertEqual(counter['status'],'UNAVAILABLE')
            self.assertIsNone(counter['value'])
            self.assertEqual(counter['error']['type'],'PermissionError')
            self.assertEqual(counter['error']['message'],str(error))
            self.assertEqual(counter['error']['errno'],13)
            self.assertEqual(counter['error']['filename'],error.filename)

    def test_optional_final_io_denied_after_exit_is_not_guard_failure(self):
        state=launch.observe_process(FakeProcess([None,0]),reader(io=denied('io')))
        self.assertEqual(state['observation'],'EXITED_DURING_READ')
        self.assertEqual(state['returncode'],0)
        self.assertIsNone(launch.guard_reason(state,1,0))
        self.assertTrue(all(c['status']=='UNAVAILABLE' for c in state['io'].values()))

    def test_optional_io_missing_does_not_mask_critical_children_missing(self):
        state=launch.observe_process(FakeProcess([None,None]),reader(io=missing('io'),children=missing(f'task/{PID}/children')))
        self.assertNotIn('children',state)
        self.assertEqual(state['critical_errors']['children']['type'],'FileNotFoundError')
        self.assertEqual(launch.guard_reason(state,1,0),'CRITICAL_MONITORING_UNAVAILABLE')

    def test_individual_missing_or_malformed_io_counters_are_not_zero(self):
        state=launch.read_proc(PID,reader(io='rchar: 0\nwchar: bad\nsyscr: -1\n'))
        self.assertEqual(state['io']['rchar'],{'status':'AVAILABLE','value':0})
        for key in launch.IO_COUNTERS[1:]:
            self.assertEqual(state['io'][key]['status'],'UNAVAILABLE')
            self.assertIsNone(state['io'][key]['value'])
            self.assertIn('error',state['io'][key])
        self.assertEqual(state['io']['wchar']['error']['message'],"invalid wchar counter: 'bad'")
        self.assertEqual(state['io']['read_bytes']['error']['type'],'MissingProcField')

    def test_poll_first_already_exited_never_reads_proc(self):
        read=mock.Mock(side_effect=AssertionError('post-exit read'))
        state=launch.observe_process(FakeProcess([0]),read)
        read.assert_not_called()
        self.assertEqual(state['critical_monitoring'],'UNKNOWN_NOT_SAMPLED_AFTER_EXIT')
        self.assertTrue(all(c['status']=='UNKNOWN' and c['value'] is None for c in state['io'].values()))
        self.assertNotIn('VmRSS',state)
        self.assertNotIn('Threads',state)
        self.assertNotIn('children',state)

    def test_confirmed_disappearance_race_is_explicit_unknown(self):
        state=launch.observe_process(FakeProcess([None,0]),reader(
            status=missing('status'),children=missing(f'task/{PID}/children'),io=missing('io')))
        self.assertEqual(state['critical_monitoring'],'UNKNOWN_CONFIRMED_EXIT_RACE')
        self.assertIsNone(launch.guard_reason(state,1,0))
        self.assertEqual(set(state['critical_errors']),{'VmRSS','Threads','children'})
        self.assertNotIn('children',state)

    def test_missing_children_while_live_fails_closed(self):
        error=missing(f'task/{PID}/children')
        state=launch.observe_process(FakeProcess([None,None]),reader(children=error))
        self.assertNotIn('children',state)
        self.assertEqual(state['critical_errors']['children']['message'],str(error))
        self.assertEqual(launch.guard_reason(state,1,0),'CRITICAL_MONITORING_UNAVAILABLE')

    def test_missing_children_after_confirmed_exit_is_unknown_not_empty(self):
        state=launch.observe_process(FakeProcess([None,0]),reader(children=missing(f'task/{PID}/children')))
        self.assertEqual(state['critical_monitoring'],'UNKNOWN_CONFIRMED_EXIT_RACE')
        self.assertNotIn('children',state)
        self.assertIsNone(launch.guard_reason(state,1,0))

    def test_critical_permission_denied_fails_live_and_after_exit(self):
        for field in ('status','children'):
            for rc in (None,0):
                with self.subTest(field=field,returncode=rc):
                    error=denied('status' if field=='status' else f'task/{PID}/children')
                    state=launch.observe_process(FakeProcess([None,rc]),reader(**{field:error}))
                    self.assertEqual(launch.guard_reason(state,1,0),'CRITICAL_MONITORING_UNAVAILABLE')
                    self.assertTrue(all(e['message']==str(error) for e in state['critical_errors'].values()))

    def test_missing_rss_or_threads_while_live_fails_closed(self):
        for key in ('VmRSS','Threads'):
            with self.subTest(key=key):
                status='\n'.join(line for line in STATUS.splitlines() if not line.startswith(key+':'))
                state=launch.observe_process(FakeProcess([None,None]),reader(status=status))
                self.assertNotIn(key,state)
                self.assertEqual(launch.guard_reason(state,1,0),'CRITICAL_MONITORING_UNAVAILABLE')

    def test_missing_status_while_live_fails_closed(self):
        state=launch.observe_process(FakeProcess([None,None]),reader(status=missing('status')))
        self.assertEqual(set(state['critical_errors']),{'VmRSS','Threads'})
        self.assertEqual(launch.guard_reason(state,1,0),'CRITICAL_MONITORING_UNAVAILABLE')

    def test_malformed_critical_data_never_becomes_exit_race(self):
        for read in (reader(status=STATUS.replace('Threads:\t1','Threads:\tbad')),
                     reader(status=STATUS.replace('123 kB','123 MB')),
                     reader(children='not-a-pid')):
            for rc in (None,0):
                with self.subTest(returncode=rc,reader=read):
                    state=launch.observe_process(FakeProcess([None,rc]),read)
                    self.assertEqual(launch.guard_reason(state,1,0),'CRITICAL_MONITORING_UNAVAILABLE')

    def test_guard_boundary_and_limit_crossings(self):
        state=launch.observe_process(FakeProcess([None,None]),reader(status=STATUS.replace('123 kB','262144 kB')))
        self.assertIsNone(launch.guard_reason(state,120,14680064))
        cases=[({'VmRSS':262145},1,0,'RSS_LIMIT'),
               ({'Threads':2},1,0,'THREAD_LIMIT'),
               ({'children':'77'},1,0,'CHILD_PROCESS_FORBIDDEN'),
               ({},120.0001,0,'WALL_LIMIT'),
               ({},1,14680065,'OUTPUT_LIMIT')]
        for changes,elapsed,total,expected in cases:
            for rc in (None,0):
                with self.subTest(expected=expected,returncode=rc):
                    self.assertEqual(launch.guard_reason({**state,**changes,'returncode':rc},elapsed,total),expected)

    def test_normal_monitor_uses_fake_process_only(self):
        proc=FakeProcess([None,None,0])
        sleep=mock.Mock()
        clock=mock.Mock(side_effect=[0.1,0.2])
        result=launch.monitor_process(proc,0,lambda:0,read_text=reader(),clock=clock,sleep=sleep,killpg=proc.kill)
        self.assertEqual(result['returncode'],0)
        self.assertIsNone(result['guard_stop_reason'])
        self.assertEqual(proc.kills,[])
        self.assertEqual(proc.wait_calls,1)
        self.assertEqual(result['sampled_peak_rss_bytes'],456*1024)
        self.assertEqual(result['io_last_successful_observation']['rchar']['value'],0)
        self.assertEqual(result['io_last_successful_observation']['rchar']['wall_s'],0.1)
        self.assertEqual(result['io_final_observation']['rchar']['status'],'UNKNOWN')
        sleep.assert_called_once_with(0.1)

    def test_optional_denial_does_not_stop_fake_live_process(self):
        proc=FakeProcess([None,None,0])
        result=launch.monitor_process(proc,0,lambda:0,read_text=reader(io=denied('io')),clock=lambda:1,sleep=mock.Mock(),killpg=proc.kill)
        self.assertEqual(result['returncode'],0)
        self.assertIsNone(result['guard_stop_reason'])
        self.assertEqual(proc.kills,[])
        self.assertEqual(result['proc_samples'][0]['io']['read_bytes']['status'],'UNAVAILABLE')
        self.assertEqual(result['io_last_successful_observation']['read_bytes']['status'],'UNKNOWN')

    def test_monitor_no_sample_has_unknown_peak_not_zero(self):
        result=launch.monitor_process(FakeProcess([0]),0,lambda:0,read_text=mock.Mock(),clock=lambda:0.1,sleep=mock.Mock(),killpg=mock.Mock())
        self.assertIsNone(result['sampled_peak_rss_bytes'])
        self.assertTrue(all(c['value'] is None for c in result['io_last_successful_observation'].values()))

    def test_live_critical_failure_kills_fake_process_and_records_evidence(self):
        proc=FakeProcess([None,None,None],waitcode=-signal.SIGKILL)
        result=launch.monitor_process(proc,0,lambda:0,read_text=reader(children=missing(f'task/{PID}/children')),clock=lambda:0.1,sleep=mock.Mock(),killpg=proc.kill)
        self.assertEqual(result['guard_stop_reason'],'CRITICAL_MONITORING_UNAVAILABLE')
        self.assertEqual(result['returncode'],-signal.SIGKILL)
        self.assertEqual(proc.kills,[(PID,signal.SIGKILL)])
        self.assertEqual(result['proc_samples'][0]['critical_errors']['children']['type'],'FileNotFoundError')

    def test_each_live_guard_crossing_stops_fake_process(self):
        cases=[(reader(status=STATUS.replace('123 kB','262145 kB')),0.1,0,'RSS_LIMIT'),
               (reader(status=STATUS.replace('Threads:\t1','Threads:\t2')),0.1,0,'THREAD_LIMIT'),
               (reader(children='99'),0.1,0,'CHILD_PROCESS_FORBIDDEN'),
               (reader(),120.01,0,'WALL_LIMIT'),
               (reader(),0.1,14680065,'OUTPUT_LIMIT')]
        for read,elapsed,size,reason in cases:
            with self.subTest(reason=reason):
                proc=FakeProcess([None,None,None])
                result=launch.monitor_process(proc,0,lambda:size,read_text=read,clock=lambda:elapsed,sleep=mock.Mock(),killpg=proc.kill)
                self.assertEqual(result['guard_stop_reason'],reason)
                self.assertEqual(proc.kills,[(PID,signal.SIGKILL)])

    def test_exit_between_guard_and_kill_does_not_erase_failure(self):
        proc=FakeProcess([None,None,None])
        kill=mock.Mock(side_effect=ProcessLookupError(3,'No such process'))
        result=launch.monitor_process(proc,0,lambda:0,read_text=reader(children='99'),clock=lambda:1,sleep=mock.Mock(),killpg=kill)
        self.assertEqual(result['guard_stop_reason'],'CHILD_PROCESS_FORBIDDEN')
        self.assertEqual(result['returncode'],0)
        kill.assert_called_once_with(PID,signal.SIGKILL)

    def test_partial_observations_survive_later_supervisor_exception(self):
        record={}
        proc=FakeProcess([None,None])
        with self.assertRaisesRegex(RuntimeError,'invented output error'):
            launch.monitor_process(proc,0,mock.Mock(side_effect=RuntimeError('invented output error')),read_text=reader(),clock=lambda:1,sleep=mock.Mock(),killpg=proc.kill,record=record)
        self.assertEqual(len(record['proc_samples']),1)
        self.assertEqual(record['proc_samples'][0]['critical_monitoring'],'OBSERVED')

    def test_exact_hard_limits_affinity_and_alarm_are_mocked(self):
        self.assertEqual(launch.LIMITS,dict(rss_bytes=268435456,cpu_seconds=60,wall_seconds=120,output_bytes=16777216,collector_threads=1,collector_processes=1))
        with mock.patch.object(launch.os,'sched_setaffinity') as affinity, \
             mock.patch.object(launch.resource,'setrlimit') as limits, \
             mock.patch.object(launch.signal,'alarm') as alarm:
            launch.apply_hard_limits(3)
        affinity.assert_called_once_with(0,{3})
        self.assertEqual(limits.call_args_list,[
            mock.call(launch.resource.RLIMIT_AS,(268435456,268435456)),
            mock.call(launch.resource.RLIMIT_CPU,(60,60)),
            mock.call(launch.resource.RLIMIT_FSIZE,(6291456,6291456)),
            mock.call(launch.resource.RLIMIT_CORE,(0,0))])
        alarm.assert_called_once_with(120)

    def test_old_release_never_reaches_io_or_process_launch(self):
        with mock.patch.object(launch,'load') as load, \
             mock.patch.object(launch.os,'open') as open_ledger, \
             mock.patch.object(launch.fcntl,'flock') as lock:
            with self.assertRaisesRegex(RuntimeError,'FUTURE_DRAFT_NOT_RELEASED'):
                launch.run(PATH.parents[2]/'EXECUTION-RELEASE.json')
        load.assert_not_called()
        open_ledger.assert_not_called()
        lock.assert_not_called()
        self.assertEqual(launch.ROOT,PATH.parents[1])

    def test_once_only_lock_and_pinning_controls_retained_in_disabled_candidate(self):
        # Static evidence only: the disabled production body is not invoked.
        tree=ast.parse(SOURCE)
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        self.assertIsInstance(run.body[0],ast.Raise)
        for control in (
                'fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)',
                'os.O_WRONLY|os.O_CREAT|os.O_EXCL',
                'out.mkdir(exist_ok=False)',
                "ROOT.parent.parent/'kws-native-a20-recovery-v1/.heavy.lock'",
                "cpu=protocol['cpu_affinity'];require(cpu in os.sched_getaffinity(0)",
                "OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'",
                'start_new_session=True,preexec_fn=guard',
                'identities_before=verify_freeze(ROOT,freeze)',
                'after=verify_freeze(ROOT,freeze)'):
            self.assertIn(control,SOURCE)
        self.assertNotIn('unlink(',SOURCE)
        self.assertNotIn('rmtree(',SOURCE)

    def test_final_record_is_byte_bounded_and_fails_on_truncation(self):
        base={'status':'RAW_COMPLETE_PENDING_AUDIT','guard_stop_reason':None,
              'proc_samples':[{'sample':i,'message':'x'*1000} for i in range(100)]}
        good,data=launch.encode_resource_record(base,0)
        self.assertEqual(good['status'],'RAW_COMPLETE_PENDING_AUDIT')
        self.assertEqual(json.loads(data),base)
        reduced,data=launch.encode_resource_record(base,launch.LIMITS['output_bytes']-10000)
        self.assertEqual(reduced['status'],'FAILED_NO_RETRY')
        self.assertEqual(reduced['guard_stop_reason'],'RESOURCE_RECORD_OUTPUT_LIMIT')
        self.assertEqual(reduced['proc_samples_omitted'],98)
        self.assertLessEqual(len(data),10000)
        self.assertEqual([s['sample'] for s in reduced['proc_samples']],[0,99])
        self.assertEqual(len(base['proc_samples']),100)

    def test_record_failure_preserves_prior_guard_reason_and_never_writes_over_cap(self):
        base={'status':'FAILED_NO_RETRY','guard_stop_reason':'CRITICAL_MONITORING_UNAVAILABLE',
              'proc_samples':[{'message':'x'*100}]}
        record,data=launch.encode_resource_record(base,launch.LIMITS['output_bytes'])
        self.assertEqual(record['guard_stop_reason'],'CRITICAL_MONITORING_UNAVAILABLE')
        self.assertTrue(record['resource_record_output_limit'])
        self.assertIsNone(data)


if __name__=='__main__':
    unittest.main(verbosity=2)
