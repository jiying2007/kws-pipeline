"""Bounded no-audio preparation diagnostics. Standard library only."""
from pathlib import Path
import json
import os
import stat

SCHEMA='native-prepare-safe-diagnostic-v1'
FILENAME='prepare-diagnostic.json'
MAX_BYTES=4096
FAMILIES=('qwen06','sensevoice','whisper')
STAGES=('entry','verify_scope','verify_sources','verify_runtime','import_framework',
        'import_model','install_forward_guard','review_native_sources','load_model',
        'prepare_native_identity','verify_zero_forward','write_preparation_proof','complete')
REQUIRED=('verify_scope','verify_sources','verify_runtime','install_forward_guard','load_model',
          'prepare_native_identity','verify_zero_forward','write_preparation_proof')
ERROR_TYPES=(ValueError,TypeError,RuntimeError,KeyError,AttributeError,ImportError,ModuleNotFoundError,
             FileNotFoundError,PermissionError,OSError,MemoryError,AssertionError,TimeoutError,
             NotImplementedError,OverflowError,ZeroDivisionError,RecursionError,EOFError,
             InterruptedError,KeyboardInterrupt,SystemExit,Exception,BaseException)
ERROR_NAMES=tuple(t.__name__ for t in ERROR_TYPES)+('OtherException',)
FIELDS={'schema','family','status','stage','started_stages','exception_class'}


def require(ok):
    if not ok:raise ValueError('invalid safe diagnostic')


def exception_class(error):
    # Never inspect error.args, str/repr, traceback, custom attributes or names.
    kind=type(error)
    for allowed in ERROR_TYPES:
        if kind is allowed:return allowed.__name__
    return 'OtherException'


def validate(value,expected_family):
    require(type(value) is dict and set(value)==FIELDS)
    require(expected_family in FAMILIES and value['schema']==SCHEMA and value['family']==expected_family)
    require(value['status'] in ('running','failed','complete') and type(value['status']) is str)
    stages=value['started_stages']
    require(type(stages) is list and 1<=len(stages)<=len(STAGES))
    require(all(type(s) is str and s in STAGES for s in stages))
    require(stages[0]=='entry' and len(set(stages))==len(stages) and
            [STAGES.index(s) for s in stages]==sorted(STAGES.index(s) for s in stages))
    require(value['stage']==stages[-1])
    if value['status']=='failed':
        require(type(value['exception_class']) is str and value['exception_class'] in ERROR_NAMES)
        require(value['stage']!='complete')
    else:require(value['exception_class'] is None)
    if value['status']=='complete':
        require(value['stage']=='complete' and all(s in stages for s in REQUIRED))
    else:require(value['stage']!='complete')
    return value


def _pairs(items):
    result={}
    for key,value in items:
        require(key not in result);result[key]=value
    return result


def _constant(_):raise ValueError('invalid safe diagnostic')


def decode(raw):
    require(type(raw) is bytes and 0<len(raw)<=MAX_BYTES)
    # No numeric fields exist in this schema, so any number is rejected at parse.
    return json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,
                      parse_constant=_constant,parse_float=_constant,parse_int=_constant)


def _write(path,value):
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
    require(len(raw)<=MAX_BYTES)
    temporary=path.with_name(path.name+'.next')
    with temporary.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


class StageTracker:
    """Call enter immediately BEFORE each operation; never infer a stage later."""
    def __init__(self,output,family):
        require(family in FAMILIES)
        self.path=Path(output)/FILENAME;self.family=family;self.stages=[];self.terminal=False;self.active=False;self.ready=False
    def _value(self,status,error=None):
        return dict(schema=SCHEMA,family=self.family,status=status,stage=self.stages[-1],
                    started_stages=list(self.stages),exception_class=error)
    def __enter__(self):
        require(not self.active and not self.stages and self.path.parent.is_dir())
        # Reserve the final name exclusively. An attempted tracker cannot restart.
        with self.path.open('xb') as stream:stream.flush();os.fsync(stream.fileno())
        self.active=True;self.enter('entry');return self
    def enter(self,stage):
        require(self.active and not self.terminal and not self.ready and stage in STAGES and stage!='complete')
        require(not self.stages or STAGES.index(stage)>STAGES.index(self.stages[-1]))
        require(self.stages or stage=='entry')
        self.stages.append(stage);value=self._value('running');validate(value,self.family);_write(self.path,value)
    def finish(self):
        require(self.active and not self.terminal and not self.ready and
                self.stages[-1]=='write_preparation_proof' and all(s in self.stages for s in REQUIRED))
        self.ready=True  # Commit completion only on a clean context exit.
    def __exit__(self,kind,error,traceback):
        if error is not None:
            self.terminal=True;self.active=False
            try:_write(self.path,self._value('failed',exception_class(error)))
            except Exception:pass  # diagnostic I/O must not replace original failure
            return False
        self.active=False
        if not self.ready:
            error=ValueError('incomplete preparation tracker')
            self.terminal=True
            try:_write(self.path,self._value('failed','ValueError'))
            except Exception:pass
            raise error
        value=dict(self._value('running'),status='complete',stage='complete',started_stages=self.stages+['complete'])
        validate(value,self.family);_write(self.path,value)
        self.stages.append('complete');self.terminal=True
        return False


def _unavailable(family,reason):
    return dict(schema='native-prepare-safe-summary-v1',family=family,status='unavailable',reason=reason)


def read_summary(path,expected_family):
    """Host-side validation. Returns enums only; never echoes supplied bytes."""
    require(expected_family in FAMILIES)
    path=Path(path)
    try:
        if path.is_symlink():return _unavailable(expected_family,'unsafe_file')
        flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0)
        fd=os.open(str(path),flags)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or not 0<info.st_size<=MAX_BYTES:
                return _unavailable(expected_family,'unsafe_file')
            raw=stream.read(MAX_BYTES+1)
            if len(raw)!=info.st_size:return _unavailable(expected_family,'unsafe_file')
    except FileNotFoundError:return _unavailable(expected_family,'missing')
    except OSError:return _unavailable(expected_family,'unreadable')
    try:value=validate(decode(raw),expected_family)
    except (ValueError,TypeError,KeyError,UnicodeError,RecursionError):return _unavailable(expected_family,'invalid')
    if value['status']=='running':return _unavailable(expected_family,'not_terminal')
    # This is diagnostic completion only, NEVER authority for model success,
    # policy acceptance, training admission or permission to continue a failed job.
    return dict(schema='native-prepare-safe-summary-v1',family=expected_family,status=value['status'],
                stage=value['stage'],exception_class=value['exception_class'])
