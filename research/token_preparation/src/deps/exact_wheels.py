#!/usr/bin/env python3
"""Closed, stdlib-only CPU wheel acquisition/inspection. Never launches a child.

Default CLI is read-only. No extraction or import of wheel contents occurs here.
Release is an execution interlock, not a substitute for human authorization.
"""
from __future__ import annotations

import argparse
import collections
import email.policy
import email.parser
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
LOCK_SHA256 = '90a28180a51fae445941bd3c0c9fefb6fa569e2c622255d32c17d1021d6939fc'
REQ_SHA256 = '3c1f66518786842b6659da595ad08d651462d355dcca893fba831de4156b435f'
SOURCE_SHA256 = '85043c2c83f3ec8f9f8e334735546faf99f06245676a668d1020897887e508ca'
DOWNLOAD_CAP = 300 * 1024**2
DISK_FLOOR = 4 * 1024**3
UNCOMPRESSED_CAP = 2 * 1024**3
MEMBER_CAP = 1024**3
METADATA_CAP = 4 * 1024**2
LICENSE_CAP = 16 * 1024**2
MAX_ENTRIES = 100000
MAX_RATIO = 1000
CHUNK = 65536
HTTP_TIMEOUT = 30
DOWNLOAD_WALL_CAP = 600
HOSTS = frozenset({'files.pythonhosted.org', 'download-r2.pytorch.org'})
EXPECTED_NAMES = frozenset({'torch','numpy','filelock','typing-extensions','setuptools','sympy','networkx','jinja2','fsspec','mpmath','markupsafe'})


class Rejected(RuntimeError):
    """Public-safe code only: never include OS paths, exception text, or headers."""


def require(condition, code):
    if not condition:
        raise Rejected(code)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def safe_url(url):
    p = urllib.parse.urlsplit(url)
    require(p.scheme == 'https' and p.hostname in HOSTS and p.netloc == p.hostname,
            'URL_SCHEME_HOST_OR_AUTH_REJECTED')
    require(not p.query and not p.fragment and not p.username and not p.password,
            'URL_QUERY_FRAGMENT_OR_AUTH_REJECTED')
    require(p.path.endswith('.whl') and '\\' not in p.path and not any(ord(c)<33 for c in url),
            'URL_PATH_REJECTED')
    return p


def wheel_tags(filename):
    require(re.fullmatch(r'[A-Za-z0-9_.+\-]+\.whl', filename) is not None, 'WHEEL_FILENAME_REJECTED')
    parts = filename[:-4].split('-')
    require(len(parts) == 5, 'WHEEL_BUILD_TAG_NOT_IN_LOCK')
    return {'-'.join(p) for p in itertools.product(*(v.split('.') for v in parts[-3:]))}


def load_lock():
    raw = (HERE / 'exact-wheels.lock.json').read_bytes()
    require(sha256(raw) == LOCK_SHA256, 'LOCK_HASH_MISMATCH')
    lock = json.loads(raw)
    require(lock['source_proposal_sha256'] == SOURCE_SHA256, 'SOURCE_HASH_MISMATCH')
    require(lock['download_cap_bytes'] == DOWNLOAD_CAP and lock['free_disk_floor_bytes'] == DISK_FLOOR
            and lock['aggregate_uncompressed_cap_bytes'] == UNCOMPRESSED_CAP, 'LIMIT_MISMATCH')
    require(lock['redirects'] == [], 'UNREVIEWED_REDIRECT')
    packages = lock['packages']
    require(len(packages) == 11 and {canonical_name(p['project']) for p in packages} == EXPECTED_NAMES,
            'PACKAGE_SET_MISMATCH')
    require(len({p['filename'] for p in packages}) == 11, 'DUPLICATE_WHEEL_FILENAME')
    require(sum(p['bytes'] for p in packages) == lock['total_download_bytes'] == 220935082,
            'DOWNLOAD_TOTAL_MISMATCH')
    for p in packages:
        url = safe_url(p['url'])
        require(urllib.parse.unquote(url.path.rsplit('/',1)[-1]) == p['filename'], 'URL_FILENAME_MISMATCH')
        require(p['matched_tag'] in wheel_tags(p['filename']), 'FILENAME_TAG_MISMATCH')
        require(re.fullmatch(r'[0-9a-f]{64}',p['sha256']) is not None, 'WHEEL_HASH_INVALID')
        require(re.fullmatch(r'[0-9a-f]{64}',p.get('wheel_metadata_sha256','')) is not None,
                'WHEEL_METADATA_PIN_REQUIRED')
    raw_req = (HERE / 'requirements.proposed.lock').read_bytes()
    require(sha256(raw_req) == REQ_SHA256, 'REQUIREMENTS_HASH_MISMATCH')
    lines = {line.strip() for line in raw_req.decode('ascii').splitlines() if line and not line.startswith('#')}
    require(lines == {f"{p['project']}=={p['version']} --hash=sha256:{p['sha256']}" for p in packages},
            'REQUIREMENTS_PACKAGE_MISMATCH')
    return lock


def release_template():
    return {'schema':'a20-cpu-wheel-execution-release-v1', 'approved':False,
            'scope':'download-inspect-build-offline-venv-install-argv', 'lock_sha256':LOCK_SHA256,
            'download_cap_bytes':DOWNLOAD_CAP, 'free_disk_floor_bytes':DISK_FLOOR,
            'aggregate_uncompressed_cap_bytes':UNCOMPRESSED_CAP,
            'no_retry':True, 'no_redirect':True, 'no_auth':True, 'no_subprocess':True}


def validate_release(release):
    expected = release_template()
    expected['approved'] = True
    require(release == expected, 'EXECUTION_NOT_RELEASED')


def validate_runtime():
    require(platform.python_implementation() == 'CPython' and platform.python_version() == '3.12.3',
            'PYTHON_RUNTIME_MISMATCH')
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'PLATFORM_MISMATCH')
    libc, version = platform.libc_ver()
    require(libc == 'glibc' and re.fullmatch(r'\d+\.\d+',version) is not None
            and tuple(map(int,version.split('.'))) >= (2,28), 'LIBC_MISMATCH')


def no_symlink_path(path, *, leaf_must_exist=True):
    path = Path(os.path.abspath(path))
    for part in [*reversed(path.parents),path]:
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            require(part == path and not leaf_must_exist, 'PATH_COMPONENT_MISSING')
            continue
        require(not stat.S_ISLNK(mode), 'SYMLINK_PATH_REJECTED')
    return path


def workspace_root(workspace):
    root = no_symlink_path(workspace)
    st = root.stat()
    require(stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode)&0o077 == 0,
            'WORKSPACE_NOT_PRIVATE_OWNED_DIRECTORY')
    return root


def check_free(root, reserve=0):
    require(shutil.disk_usage(root).free >= DISK_FLOOR + reserve, 'FREE_DISK_FLOOR')


class Budget:
    def __init__(self, download_cap=DOWNLOAD_CAP, uncompressed_cap=UNCOMPRESSED_CAP):
        self.download = 0
        self.uncompressed = 0
        self.download_cap = download_cap
        self.uncompressed_cap = uncompressed_cap

    def add_download(self, amount):
        self.download += amount
        require(self.download <= self.download_cap, 'DOWNLOAD_CAP_EXCEEDED')

    def add_uncompressed(self, amount):
        self.uncompressed += amount
        require(self.uncompressed <= self.uncompressed_cap, 'UNCOMPRESSED_CAP_EXCEEDED')


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Rejected('REDIRECT_REJECTED')


def exact_https_open(url):
    safe_url(url)
    # Do not read proxy, netrc, cookies or authentication from the environment.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), RejectRedirects())
    req = urllib.request.Request(url, headers={'Accept-Encoding':'identity',
                                  'User-Agent':'a20-exact-wheel-reader/1'}, method='GET')
    return opener.open(req, timeout=HTTP_TIMEOUT)


def receive_wheel(response, package, target, budget, *, free_check=lambda:None, deadline=None):
    """Injectable read-only HTTP fixture interface. Caller owns exact target path."""
    require(response.status == 200, 'HTTP_STATUS_REJECTED')
    require(response.geturl() == package['url'], 'FINAL_URL_MISMATCH')
    lengths = response.headers.get_all('Content-Length', [])
    require(len(lengths) == 1 and re.fullmatch(r'[0-9]+',lengths[0]) is not None,
            'CONTENT_LENGTH_MISSING_OR_AMBIGUOUS')
    require(int(lengths[0]) == package['bytes'], 'CONTENT_LENGTH_MISMATCH')
    require(not response.headers.get_all('Transfer-Encoding',[]), 'TRANSFER_ENCODING_REJECTED')
    enc = response.headers.get_all('Content-Encoding',[])
    require(not enc or enc == ['identity'], 'CONTENT_ENCODING_REJECTED')
    require(budget.download + package['bytes'] <= budget.download_cap, 'DOWNLOAD_CAP_EXCEEDED')
    digest = hashlib.sha256()
    total = 0
    # Exclusive/no-follow creation: never overwrite a path or follow a link.
    fd = os.open(target, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as out:
            while True:
                require(deadline is None or time.monotonic() <= deadline, 'DOWNLOAD_WALL_CAP_EXCEEDED')
                free_check()
                block = response.read(min(CHUNK, package['bytes'] - total + 1))
                if not block:
                    break
                budget.add_download(len(block))
                total += len(block)
                require(total <= package['bytes'], 'BODY_LENGTH_EXCEEDED')
                out.write(block)
                digest.update(block)
            require(total == package['bytes'], 'BODY_TRUNCATED')
            require(digest.hexdigest() == package['sha256'], 'WHEEL_SHA256_MISMATCH')
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        Path(target).unlink(missing_ok=True)
        raise
    return {'filename':package['filename'],'bytes':total,'sha256':digest.hexdigest(),'status':200,
            'url':package['url'],'redirects':0}


def canonical_field(value):
    # Preserve syntax, case and quoted values. Equivalent but differently formatted
    # metadata can fail closed; this never evaluates attacker-controlled markers.
    return ' '.join(str(value).split())


def parse_headers(data):
    require(len(data) <= METADATA_CAP, 'METADATA_CAP_EXCEEDED')
    msg = email.parser.BytesParser(policy=email.policy.compat32).parsebytes(data)
    require(not msg.defects, 'METADATA_PARSE_DEFECT')
    return msg


def singleton(msg, key, default=None):
    vals = msg.get_all(key,[])
    require(len(vals) <= 1 and (vals or default is not None), 'METADATA_SINGLETON_MISMATCH')
    return canonical_field(vals[0]) if vals else default


def python_spec_matches(spec, version=(3,12,3)):
    for clause in spec.split(','):
        if not clause.strip():
            continue
        m = re.fullmatch(r'\s*(>=|<=|!=|==|>|<)\s*(\d+(?:\.\d+){0,2})\s*',clause)
        require(m is not None, 'PYTHON_SPEC_UNSUPPORTED')
        rhs = tuple(map(int,m[2].split('.')))
        rhs += (0,)*(3-len(rhs))
        require({'==':version==rhs,'!=':version!=rhs,'>=':version>=rhs,'<=':version<=rhs,
                 '>':version>rhs,'<':version<rhs}[m[1]], 'REQUIRES_PYTHON_MISMATCH')
    return True


def metadata_evidence(data, metadata, package):
    """Only approved hash-bound field text is public; unexpected text is hashed."""
    observed = [canonical_field(v) for v in metadata.get_all('Requires-Dist', [])]
    expected = [canonical_field(v) for v in package['requires_dist']]
    observed_hash = sha256(data)
    trusted = observed_hash == package['wheel_metadata_sha256']
    def fields(values, disclose):
        encoded = json.dumps(values, ensure_ascii=True, separators=(',', ':')).encode('ascii')
        return {'count':len(values), 'canonical_fields_sha256':sha256(encoded),
                'values':values if disclose else None,
                'text_status':'PINNED_PUBLIC_METADATA' if disclose else 'REDACTED_UNEXPECTED_METADATA'}
    return {'expected_metadata_sha256':package['wheel_metadata_sha256'],
            'observed_metadata_sha256':observed_hash, 'observed_metadata_bytes':len(data),
            'metadata_hash_matches':trusted,
            'expected_requires_dist':fields(expected, True),
            'observed_requires_dist':fields(observed, trusted)}


def validate_metadata(data, wheel_data, package, metadata_sink=None):
    metadata = parse_headers(data)
    if metadata_sink is not None:
        metadata_sink(metadata_evidence(data, metadata, package))
    require(canonical_name(singleton(metadata,'Name')) == canonical_name(package['project']), 'METADATA_NAME_MISMATCH')
    require(singleton(metadata,'Version') == package['version'], 'METADATA_VERSION_MISMATCH')
    rp = singleton(metadata,'Requires-Python','')
    require(rp == package['requires_python'], 'METADATA_REQUIRES_PYTHON_MISMATCH')
    python_spec_matches(rp)
    reqs = [canonical_field(v) for v in metadata.get_all('Requires-Dist',[])]
    require(collections.Counter(reqs) == collections.Counter(canonical_field(v) for v in package['requires_dist']),
            'METADATA_REQUIRES_DIST_MISMATCH')
    require(sha256(data) == package['wheel_metadata_sha256'], 'WHEEL_METADATA_HASH_MISMATCH')
    wheel = parse_headers(wheel_data)
    require(singleton(wheel,'Wheel-Version') == '1.0', 'WHEEL_VERSION_UNSUPPORTED')
    tags = wheel.get_all('Tag',[])
    require(tags and len(tags)==len(set(tags)) and set(tags) == wheel_tags(package['filename']),
            'WHEEL_TAG_MISMATCH')
    pure = singleton(wheel,'Root-Is-Purelib')
    require(pure == ('true' if package['matched_tag']=='py3-none-any' else 'false'), 'PURELIB_MISMATCH')
    return {'name':singleton(metadata,'Name'),'version':singleton(metadata,'Version'),
            'requires_python':rp,'requires_dist':reqs,'tags':sorted(tags),
            'metadata_sha256':sha256(data),'wheel_descriptor_sha256':sha256(wheel_data),
            'license_expression':singleton(metadata,'License-Expression',''),
            'license_header_sha256':sha256(singleton(metadata,'License','').encode()),
            'license_files_declared':[canonical_field(v) for v in metadata.get_all('License-File',[])]}


def safe_member(info):
    name = info.filename
    require(name == info.orig_filename and bool(name) and len(name) <= 512 and
            all(32 <= ord(c) < 127 for c in name), 'ZIP_MEMBER_NAME_REJECTED')
    require(not name.startswith('/') and '\\' not in name and ':' not in name, 'ZIP_TRAVERSAL_REJECTED')
    pieces = name.rstrip('/').split('/')
    require(all(p and p not in ('.','..') for p in pieces), 'ZIP_TRAVERSAL_REJECTED')
    mode = info.external_attr >> 16
    filetype = stat.S_IFMT(mode)
    require(filetype in (0,stat.S_IFREG,stat.S_IFDIR) and not mode&0o7000, 'ZIP_SPECIAL_FILE_REJECTED')
    require(filetype != stat.S_IFDIR or info.is_dir(), 'ZIP_DIRECTORY_MISMATCH')
    require(not info.flag_bits&1, 'ZIP_ENCRYPTED_REJECTED')
    require(info.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED), 'ZIP_COMPRESSION_REJECTED')
    require(0 <= info.file_size <= MEMBER_CAP and 0 <= info.compress_size, 'ZIP_MEMBER_SIZE_REJECTED')
    require(info.file_size <= max(info.compress_size,1)*MAX_RATIO, 'ZIP_RATIO_REJECTED')
    require(not info.is_dir() or info.file_size == 0, 'ZIP_DIRECTORY_SIZE_REJECTED')
    return '/'.join(pieces)


def is_license(name):
    base = PurePosixPath(name).name.lower()
    return ('license' in base or base.startswith('copying') or base.startswith('notice'))


def inspect_wheel(path, package, budget, metadata_sink=None):
    path = no_symlink_path(path)
    require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size == package['bytes'], 'WHEEL_FILE_SIZE_MISMATCH')
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(CHUNK):
            h.update(b)
    require(h.hexdigest()==package['sha256'], 'WHEEL_SHA256_MISMATCH')
    # Reject appended/concatenated ZIPs, comments, unbounded central directories,
    # and zip64 central directories. These exact wheels fit ordinary ZIP limits.
    with path.open('rb') as f:
        require(f.read(4)==b'PK\x03\x04', 'ZIP_PREFIX_REJECTED')
        require(package['bytes']>=22, 'ZIP_EOCD_MISSING')
        f.seek(-22,2)
        eocd=f.read(22)
    require(eocd[:4]==b'PK\x05\x06', 'ZIP_EOCD_OR_COMMENT_REJECTED')
    _, disk, start_disk, n_disk, n_total, cd_size, cd_offset, comment_len=struct.unpack('<4s4H2LH',eocd)
    require(disk==start_disk==comment_len==0 and n_disk==n_total and 0<n_total<=MAX_ENTRIES
            and n_total<65535 and cd_size<=32*1024**2 and cd_offset+cd_size+22==package['bytes'],
            'ZIP_CENTRAL_DIRECTORY_REJECTED')
    dist = package['filename'].split('-')[0]+'-'+package['version']+'.dist-info'
    metadata_name, wheel_name = dist+'/METADATA', dist+'/WHEEL'
    license_reports=[]
    buffers={}
    actual=0
    with zipfile.ZipFile(path) as z:
        infos=z.infolist()
        require(len(infos)==n_total, 'ZIP_ENTRY_COUNT_MISMATCH')
        names={}
        offsets=set()
        for info in infos:
            logical=safe_member(info)
            require(logical not in names, 'ZIP_DUPLICATE_MEMBER')
            names[logical]=info
            require(info.header_offset not in offsets and 0<=info.header_offset<cd_offset,
                    'ZIP_DUPLICATE_OR_INVALID_OFFSET')
            offsets.add(info.header_offset)
            require(info.compress_size<=package['bytes'], 'ZIP_COMPRESSED_SIZE_REJECTED')
            # Vendored distributions may retain nested .dist-info as ordinary
            # package data. Only a top-level distribution identity is admitted.
            top=info.filename.split('/',1)[0]
            if top.endswith('.dist-info'):
                require(top==dist, 'ZIP_FOREIGN_DIST_INFO')
        for logical in names:
            for ancestor in PurePosixPath(logical).parents:
                a=str(ancestor)
                require(a not in names or names[a].is_dir(), 'ZIP_FILE_DIRECTORY_COLLISION')
        # Reuse the established two-component top-level METADATA selector.
        top_metas=[n for n in names if n.endswith('.dist-info/METADATA') and len(PurePosixPath(n).parts)==2]
        require(top_metas==[metadata_name], 'WHEEL_TOP_LEVEL_METADATA_UNIQUE')
        declared=sum(i.file_size for i in infos)
        require(budget.uncompressed+declared<=budget.uncompressed_cap, 'UNCOMPRESSED_CAP_EXCEEDED')
        require(metadata_name in names and wheel_name in names and dist+'/RECORD' in names,
                'WHEEL_REQUIRED_FILES_MISSING')
        for info in infos:
            if info.is_dir():
                continue
            wanted=info.filename in (metadata_name,wheel_name)
            license_file=is_license(info.filename)
            if wanted:
                require(info.file_size<=METADATA_CAP,'METADATA_CAP_EXCEEDED')
            if license_file:
                require(info.file_size<=LICENSE_CAP,'LICENSE_CAP_EXCEEDED')
            content=bytearray()
            digest=hashlib.sha256()
            n=0
            with z.open(info,'r') as stream:
                while b := stream.read(CHUNK):
                    n+=len(b)
                    require(n<=info.file_size and n<=MEMBER_CAP,'ZIP_ACTUAL_MEMBER_SIZE_EXCEEDED')
                    budget.add_uncompressed(len(b))
                    actual+=len(b)
                    digest.update(b)
                    if wanted:
                        content.extend(b)
            require(n==info.file_size,'ZIP_MEMBER_TRUNCATED')
            if wanted:
                buffers[info.filename]=bytes(content)
            if license_file:
                license_reports.append({'member':info.filename,'bytes':n,'sha256':digest.hexdigest()})
    metadata=validate_metadata(buffers[metadata_name],buffers[wheel_name],package,metadata_sink)
    for license_name in metadata['license_files_declared']:
        # PEP 639 uses licenses/; historical wheels may use dist-info directly.
        info=zipfile.ZipInfo(license_name)
        safe_member(info)
        require(any(candidate in names and not names[candidate].is_dir() for candidate in
                    (dist+'/licenses/'+license_name, dist+'/'+license_name)),
                'DECLARED_LICENSE_FILE_MISSING')
    require(actual==declared,'ZIP_ACTUAL_TOTAL_MISMATCH')
    return {'project':package['project'],'version':package['version'],'filename':package['filename'],
            'bytes':package['bytes'],'sha256':h.hexdigest(),'source_url':package['url'],
            'uncompressed_bytes':actual,'member_count':n_total,'metadata':metadata,
            'license_members':license_reports,'lock_license_metadata_sha256':package['license_metadata_sha256'],
            'lock_metadata_url':package['metadata_url'],'lock_metadata_sha256':package['metadata_sha256']}


def inspect_wheelhouse(workspace, observer=None):
    lock=load_lock()
    root=workspace_root(workspace)
    wheels=no_symlink_path(root/'wheels')
    require(wheels.is_dir(),'WHEELHOUSE_MISSING')
    expected={p['filename'] for p in lock['packages']}
    require({p.name for p in wheels.iterdir()}==expected,'WHEELHOUSE_NOT_CLOSED')
    budget=Budget()
    packages=[]
    for p in lock['packages']:
        if observer is not None:observer('INSPECTION_STARTED',p,None)
        sink=(lambda evidence,p=p:observer('METADATA_OBSERVED',p,evidence)) if observer is not None else None
        result=inspect_wheel(wheels/p['filename'],p,budget,sink)
        packages.append(result)
        if observer is not None:observer('INSPECTION_PASSED',p,result['metadata']['metadata_sha256'])
    return {'schema':'a20-cpu-wheel-inspection-v1','status':'PASS_INSPECTION_ONLY',
            'lock_sha256':LOCK_SHA256,'package_count':len(packages),
            'compressed_bytes':lock['total_download_bytes'],'uncompressed_bytes':budget.uncompressed,
            'install_executed':False,'packages':packages}


def download_all(workspace, release, evidence_sink=None):
    validate_release(release)
    validate_runtime()
    lock=load_lock()
    root=workspace_root(workspace)
    check_free(root,lock['total_download_bytes'])
    require(not (root/'wheels').exists() and not (root/'wheels').is_symlink()
            and not (root/'requirements.offline.lock').exists(), 'DOWNLOAD_OUTPUT_ALREADY_EXISTS')
    wheels=root/'wheels'
    wheels.mkdir(mode=0o700)
    budget=Budget()
    deadline=time.monotonic()+DOWNLOAD_WALL_CAP
    receipts=[]
    state={'schema':'a20-cpu-wheel-progress-v1','status':'IN_PROGRESS',
           'lock_sha256':LOCK_SHA256,'request_attempts_reserved':0,'successful_downloads':[],
           'successful_download_bytes':0,'inspections_completed':[],
           'current':None,'wire_bytes':'UNKNOWN_NOT_MEASURED',
           'reservation_semantics':'Before-call durable upper bound; a crash may occur before the reserved request enters the HTTP opener. Verified completed bodies are exact.'}
    def emit():
        if evidence_sink is not None:evidence_sink(state)
    def identity(p):
        return {k:p[k] for k in ('project','version','filename','bytes','sha256')}
    def observe(stage,p,evidence):
        if stage=='INSPECTION_STARTED':
            state['current']={'stage':stage,'package':identity(p),
                'expected_metadata_sha256':p['wheel_metadata_sha256']}
        elif stage=='METADATA_OBSERVED':
            state['current']['stage']=stage
            state['current']['metadata']=evidence
        else:
            state['current']['stage']=stage
            state['inspections_completed'].append(identity(p)|{'metadata_sha256':evidence})
        emit()
    emit()
    try:
        # One sequential request per exact URL. No HEAD, fallback, retry, or resume.
        # Checkpoints are persisted before starting and after validating each body.
        for p in lock['packages']:
            check_free(root,p['bytes'])
            require(time.monotonic()<=deadline,'DOWNLOAD_WALL_CAP_EXCEEDED')
            state['current']={'stage':'DOWNLOAD_STARTED','package':identity(p)}
            state['request_attempts_reserved']+=1
            emit()
            try:
                with exact_https_open(p['url']) as response:
                    receipt=receive_wheel(response,p,wheels/p['filename'],budget,
                                    free_check=lambda:check_free(root,CHUNK),deadline=deadline)
            except urllib.error.HTTPError as e:
                raise Rejected('HTTP_REJECTED_'+str(e.code)) from None
            except urllib.error.URLError:
                raise Rejected('HTTP_TRANSPORT_REJECTED') from None
            receipts.append(receipt)
            state['successful_downloads'].append(identity(p))
            state['successful_download_bytes']+=receipt['bytes']
            state['current']['stage']='DOWNLOAD_HASH_VERIFIED'
            emit()
        report=inspect_wheelhouse(root,observe)
        req=(HERE/'requirements.proposed.lock').read_bytes()
        fd=os.open(root/'requirements.offline.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:
            f.write(req)
        report.update(status='PASS_DOWNLOADED_AND_INSPECTED_NO_INSTALL',download_receipts=receipts,
                      actual_download_bytes=budget.download)
        state['status']='PASS_DOWNLOADED_AND_INSPECTED_NO_INSTALL'
        emit()
        return report
    except BaseException:
        state['status']='FAILED_NO_RETRY'
        # receive_wheel accounts application bytes, including at most one overflow sentinel.
        state['application_body_bytes_read']=budget.download
        emit()
        raise


def build_venv_argv(workspace, release):
    """Build only; parent runs once as its owned child. No activation or shell."""
    validate_release(release)
    validate_runtime()
    load_lock()
    root=workspace_root(workspace)
    check_free(root,UNCOMPRESSED_CAP)
    target=root/'venv'
    require(not target.exists() and not target.is_symlink(),'VENV_ALREADY_EXISTS')
    return [sys.executable,'-I','-m','venv','--copies',str(target)]


def build_install_argv(workspace, release):
    """Reinspect immediately before supervisor-owned pip spawn; never spawn here.

    Supervisor must use a new private venv, a sanitized environment with
    PIP_CONFIG_FILE=/dev/null, an owned process group, and independent resource
    supervision. Do not publish this argv: it contains runtime-local paths.
    """
    validate_release(release)
    validate_runtime()
    root=workspace_root(workspace)
    inspect_wheelhouse(root)
    check_free(root,UNCOMPRESSED_CAP)
    req=no_symlink_path(root/'requirements.offline.lock')
    require(sha256(req.read_bytes())==REQ_SHA256,'OFFLINE_REQUIREMENTS_HASH_MISMATCH')
    venv=no_symlink_path(root/'venv')
    cfg=no_symlink_path(venv/'pyvenv.cfg')
    require(re.search(r'^include-system-site-packages\s*=\s*false\s*$',cfg.read_text(),re.M),
            'VENV_NOT_ISOLATED')
    bindir=no_symlink_path(venv/'bin')
    python=bindir/'python'
    # Standard venv Python may be a symlink; the parent validates interpreter identity.
    require(python.is_file(),'VENV_PYTHON_MISSING')
    return [str(python),'-I','-m','pip','--isolated','--disable-pip-version-check',
            '--no-input','--require-virtualenv','install','--no-index','--no-deps',
            '--require-hashes','--only-binary=:all:','--no-cache-dir','--no-compile',
            '--find-links',str(root/'wheels'),'--requirement',str(req)]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=['plan','inspect','download'],default='plan')
    parser.add_argument('--workspace')
    parser.add_argument('--release')
    args=parser.parse_args(argv)
    try:
        lock=load_lock()
        if args.action=='plan':
            result={'status':'PREPARATION_ONLY_NO_NETWORK_NO_INSTALL','lock_sha256':LOCK_SHA256,
                    'packages':len(lock['packages']),'bytes':lock['total_download_bytes'],
                    'release_template':release_template()}
        else:
            require(args.workspace is not None,'WORKSPACE_REQUIRED')
            if args.action=='inspect':
                result=inspect_wheelhouse(args.workspace)
            else:
                require(args.release is not None,'EXECUTION_NOT_RELEASED')
                result=download_all(args.workspace,json.loads(Path(args.release).read_bytes()))
        print(json.dumps(result,sort_keys=True,indent=2))
        return 0
    except Rejected as e:
        print(json.dumps({'status':'BLOCKED','code':str(e)}))
        return 2
    except Exception:
        # Do not leak local paths, environment, response headers or secrets.
        print(json.dumps({'status':'BLOCKED','code':'LOCAL_IO_OR_FORMAT_REJECTED'}))
        return 2


if __name__=='__main__':
    sys.exit(main())
