# Exact definition-byte subset of the historical supervisor source.
# The paired supervisor AST-selects these eight definitions and supplies globals.
# This is not the original whole module. Unused launch/admission code is omitted.
# See SOURCE-PROVENANCE.json for original-file and per-definition hash bindings.

class MissingProcField(ValueError):
    """A required field was absent, including a possible terminal status race."""


def error_record(error, path):
    """Preserve the exact exception text, rather than replacing a read with zero."""
    return {'type':type(error).__name__, 'message':str(error), 'path':str(path),
            'errno':getattr(error,'errno',None), 'filename':getattr(error,'filename',None)}


def unknown_io(reason):
    return {key:{'status':'UNKNOWN','value':None,'reason':reason} for key in IO_COUNTERS}


def read_proc(pid, read_text=None):
    """Collect critical guards separately from optional, per-counter I/O telemetry."""
    read_text = read_text or (lambda path: path.read_text())
    base=pathlib.Path('/proc')/str(pid)
    state={'critical_errors':{}, 'optional_status_errors':{},
           'lifetime_child_absence':'NOT_PROVEN'}
    status_path=base/'status'
    try:
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip()
                for line in read_text(status_path).splitlines() if ':' in line}
    except (OSError,UnicodeError) as error:
        fields=None
        for key in ('VmRSS','Threads'):
            state['critical_errors'][key]=error_record(error,status_path)
    if fields is not None:
        for key in ('VmRSS','Threads','VmHWM','VmSize'):
            try:
                if key not in fields:
                    raise MissingProcField('missing '+key+' in '+str(status_path))
                parts=fields[key].split()
                if key=='Threads':
                    if len(parts)!=1 or not parts[0].isdigit() or int(parts[0])<1:
                        raise ValueError('invalid '+key+' field: '+repr(fields[key]))
                elif len(parts)!=2 or not parts[0].isdigit() or parts[1]!='kB':
                    raise ValueError('invalid '+key+' field: '+repr(fields[key]))
                state[key]=int(parts[0])
            except (ValueError,IndexError) as error:
                target='critical_errors' if key in ('VmRSS','Threads') else 'optional_status_errors'
                state[target][key]=error_record(error,status_path)

    children_path=base/'task'/str(pid)/'children'
    try:
        children=read_text(children_path).strip()
        if any(not token.isdigit() or int(token)<1 for token in children.split()):
            raise ValueError('invalid children field: '+repr(children))
        state.update(children=children,children_count=len(children.split()),
                     children_observation='AVAILABLE_AT_SAMPLE')
    except (OSError,UnicodeError,ValueError) as error:
        evidence=error_record(error,children_path)
        state.update(children=None,children_count=None,children_error=evidence)
        # The approved exception applies only to ENOENT from this exact endpoint.
        # A different/missing filename or any other error remains a STOP condition.
        if isinstance(error,FileNotFoundError) and error.errno==errno.ENOENT and error.filename==str(children_path):
            state['children_observation']='NOT_AVAILABLE'
        else:
            state['children_observation']='FAILED_UNAVAILABLE'
            state['critical_errors']['children']=evidence

    io_path=base/'io'
    state['io']={}
    try:
        io_fields={line.split(':',1)[0]:line.split(':',1)[1].strip()
                   for line in read_text(io_path).splitlines() if ':' in line}
    except (OSError,UnicodeError) as error:
        for key in IO_COUNTERS:
            state['io'][key]={'status':'UNAVAILABLE','value':None,'error':error_record(error,io_path)}
    else:
        for key in IO_COUNTERS:
            try:
                if key not in io_fields:
                    raise MissingProcField('missing '+key+' in '+str(io_path))
                if not io_fields[key].isdigit():
                    raise ValueError('invalid '+key+' counter: '+repr(io_fields[key]))
                state['io'][key]={'status':'AVAILABLE','value':int(io_fields[key])}
            except ValueError as error:
                state['io'][key]={'status':'UNAVAILABLE','value':None,'error':error_record(error,io_path)}
    return state


def observe_process(proc, read_text=None):
    """Poll first. Only confirmed disappearance races relax missing terminal guards."""
    rc=proc.poll()
    if rc is not None:
        return {'returncode':rc,'observation':'EXITED_BEFORE_READ','critical_errors':{},
                'critical_monitoring':'UNKNOWN_NOT_SAMPLED_AFTER_EXIT',
                'children':None,'children_count':None,'children_observation':'UNKNOWN_NOT_SAMPLED_AFTER_EXIT',
                'lifetime_child_absence':'NOT_PROVEN',
                'io':unknown_io('process already exited; no post-exit read attempted')}
    state=read_proc(proc.pid,read_text)
    state['returncode']=proc.poll()
    state['observation']='LIVE' if state['returncode'] is None else 'EXITED_DURING_READ'
    errors=state['critical_errors']
    if not errors:
        state['critical_monitoring']='OBSERVED_RSS_THREADS_ONLY'
    elif state['returncode'] is not None and all(
            key in ('VmRSS','Threads') and error['type'] in ('FileNotFoundError','ProcessLookupError','MissingProcField')
            for key,error in errors.items()):
        state['critical_monitoring']='UNKNOWN_CONFIRMED_EXIT_RACE'
    else:
        # Permission denial and malformed critical data never become success at exit.
        state['critical_monitoring']='FAILED_UNAVAILABLE'
    return state


def guard_reason(state, elapsed, output_bytes, *, limits=None, reserve=RESERVED_RECORD_BYTES):
    """Enforce known limit crossings even if this observation also sees an exit."""
    limits=LIMITS if limits is None else limits
    if elapsed>limits['wall_seconds']:return 'WALL_LIMIT'
    if 'VmRSS' in state and state['VmRSS']*1024>limits['rss_bytes']:return 'RSS_LIMIT'
    if 'Threads' in state and state['Threads']>limits['collector_threads']:return 'THREAD_LIMIT'
    if state.get('children'):return 'CHILD_PROCESS_FORBIDDEN'
    if output_bytes>limits['output_bytes']-reserve:return 'OUTPUT_LIMIT'
    if state['critical_monitoring']=='FAILED_UNAVAILABLE':return 'CRITICAL_MONITORING_UNAVAILABLE'
    return None


def apply_hard_limits(cpu, *, limits=None, file_bytes=PER_FILE_BYTES):
    limits=LIMITS if limits is None else limits
    os.sched_setaffinity(0,{cpu})
    resource.setrlimit(resource.RLIMIT_AS,(limits['rss_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(limits['cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(file_bytes,)*2)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    signal.alarm(limits['wall_seconds'])


def cleanup_owned_process(proc, *, kill=True, killpg=None):
    """Never replace a primary failure with a cleanup exception or wait forever.

    The group belongs to the child created with start_new_session=True. Kill its
    group even if the leader just exited: an observed forbidden descendant may
    still be present. ESRCH is preserved as an exit race, never as success proof.
    """
    killpg=os.killpg if killpg is None else killpg
    result={'returncode':None,'kill_requested':kill,'kill_exit_race':False,
            'reaped':False,'errors':[]}
    if proc is None:return result
    if kill:
        try:killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError as error:
            result.update(kill_exit_race=True,kill_exit_race_evidence=error_record(error,'owned process group'))
        except BaseException as error:
            result['errors'].append(error_record(error,'kill owned process group'))
    try:
        result.update(returncode=proc.wait(timeout=1),reaped=True)
    except BaseException as error:
        result['errors'].append(error_record(error,'wait owned process'))
        result['returncode']=getattr(proc,'returncode',None)
    return result
