"""Reused reviewed wheel member checks and shared distribution selection."""
from pathlib import PurePosixPath
import re
import stat
import zipfile
MEMBER_CAP = 1024**3
MAX_RATIO = 1000

def require(condition, message):
    if not condition: raise ValueError(message)

def canonical_name(name):
    return re.sub(r'[-_.]+','-',name).lower()

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


def select_metadata(names, expected_name=None, expected_version=None):
    top=[n for n in names if n.endswith('.dist-info/METADATA') and len(PurePosixPath(n).parts)==2]
    require(len(top)==1,'WHEEL_TOP_LEVEL_METADATA_UNIQUE')
    name=top[0]
    require('\\' not in name and not name.startswith('/') and all(p not in ('','..','.') for p in name.split('/')),'METADATA_UNSAFE_PATH')
    if expected_name is not None:
        dist=name.split('/')[0][:-len('.dist-info')]
        require('-' in dist,'METADATA_DIST_INFO_NAME_VERSION')
        project,version=dist.rsplit('-',1)
        require(canonical_name(project)==canonical_name(expected_name),'METADATA_DIST_INFO_NAME')
        require(version==expected_version,'METADATA_DIST_INFO_VERSION')
    return name

def wheel_members(archive):
    infos=archive.infolist();names={}
    for info in infos:
        logical=safe_member(info)
        require(logical not in names,'ZIP_DUPLICATE_MEMBER')
        names[logical]=info
    for logical in names:
        for ancestor in PurePosixPath(logical).parents:
            a=str(ancestor)
            require(a not in names or names[a].is_dir(),'ZIP_FILE_DIRECTORY_COLLISION')
    require(sum(i.file_size for i in infos)<=3*1024**3,'WHEEL_EXPANDED_CAP')
    return names
