"""Bounded predefined failure receipts. Never expose traceback, argv or messages."""
import ast
import errno
import re
from pathlib import Path
from common import read_json,write_json,replace_json

# The finite vocabulary is taken only from reviewed constant validation messages,
# never from HTTP bodies, package metadata, environment values or runtime paths.
def _vocabulary(root):
    result={}
    for relative in ('src/common.py','src/contracts.py','src/control.py','src/run_features_control.py',
                     'src/fetch_public_inputs.py','src/dependency_phase.py','src/launch_once.py','src/deps/exact_wheels.py'):
        for node in ast.walk(ast.parse((root/relative).read_text())):
            if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='require'and len(node.args)>1:
                arg=node.args[1]
                if isinstance(arg,ast.Constant)and isinstance(arg.value,str):
                    code=re.sub('[^A-Z0-9]+','_',arg.value.upper()).strip('_')[:90]
                    result[arg.value]=code
    return result


def describe(root,error,phase,stage):
    message=str(error);known=_vocabulary(root)
    code=known.get(message)
    if code is None and re.fullmatch(r'HTTP_REJECTED_[1-5][0-9]{2}',message):code=message
    if code is None:
        for prefix,label in [('tensor shape/type/finite:','CHECKPOINT_TENSOR_SCHEMA'),('tensor content:','CHECKPOINT_TENSOR_HASH'),('source drift:','SOURCE_HASH_DRIFT')]:
            if message.startswith(prefix):code=label
    if code is None:
        code={'MemoryError':'MEMORY_ALLOCATION_FAILURE','ModuleNotFoundError':'DEPENDENCY_IMPORT_FAILURE',
              'ImportError':'DEPENDENCY_IMPORT_FAILURE','TimeoutError':'SOCKET_OR_OPERATION_TIMEOUT',
              'HTTPError':'HTTP_RESPONSE_REJECTED','URLError':'NETWORK_TRANSPORT_FAILURE',
              'JSONDecodeError':'JSON_PARSE_FAILURE','PermissionError':'PERMISSION_DENIED',
              'FileNotFoundError':'REQUIRED_FILE_MISSING'}.get(type(error).__name__,'UNCLASSIFIED_RUNTIME_FAILURE')
    allowed_types={'ValueError','RuntimeError','Rejected','MemoryError','ModuleNotFoundError','ImportError','TimeoutError','HTTPError','URLError',
        'JSONDecodeError','PermissionError','FileNotFoundError','OSError','AssertionError','KeyboardInterrupt','SystemExit','SubprocessError','CalledProcessError','TypeError','KeyError'}
    result=dict(schema='a20-safe-phase-failure-v1',phase=phase,stage=stage,error_code=code,
                exception_type=type(error).__name__ if type(error).__name__ in allowed_types else 'OTHER_EXCEPTION',raw_message_captured=False)
    number=getattr(error,'errno',None)
    if type(number)is int and 0<=number<=4096:result['os_errno']=number
    if type(getattr(error,'code',None))is int and 100<=error.code<=599:result['http_status']=error.code
    return result


def progress(root,phase,stage,recording=None):
    obj=dict(phase=phase,stage=stage)
    if recording is not None:
        allowed={r['recording']for r in read_json(root/'metadata/TRAIN32.json')['rows']}
        if recording not in allowed:raise ValueError('unknown progress recording')
        obj['recording']=recording
    replace_json(root/'work'/(phase+'-progress.json'),obj,4096)


def save(root,error,phase,default_stage):
    p=root/'work'/(phase+'-progress.json')
    observed=read_json(p,4096)if p.exists()else {'stage':default_stage}
    result=describe(root,error,phase,observed['stage'])
    if 'recording'in observed:result['recording']=observed['recording']
    if (root/'work/EXECUTION-LEDGER.json').is_file():
        write_json(root/'work'/(phase+'-safe-error.json'),result,8192)
    print(phase+' failed: '+result['error_code'])
    return result
