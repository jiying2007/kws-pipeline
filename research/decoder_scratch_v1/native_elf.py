"""Native ELF guard reused from public native_a20/build.py; no target execution.

The historical module is unchanged. These parsing and ABI comparison functions
are copied verbatim; native_expected/require_native are the small local adapter.
"""
from pathlib import Path
import ctypes
import struct
import sys

MACHINES = {'i386': 3, 'arm': 40, 'x86_64': 62, 'aarch64': 183}


def read_elf(path):
    """Inspect ELF identity and load-segment bounds without target execution."""
    with Path(path).open('rb') as handle:
        header = handle.read(64)
    if len(header) < 16 or header[:4] != b'\x7fELF':
        raise ValueError(f'{path}: expected ELF')
    elf_class, data, version = header[4:7]
    if elf_class not in (1, 2) or data not in (1, 2) or version != 1:
        raise ValueError(f'{path}: unsupported ELF identification')
    size = 52 if elf_class == 1 else 64
    if len(header) < size:
        raise ValueError(f'{path}: truncated ELF header')
    endian = '<' if data == 1 else '>'
    kind, machine, elf_version = struct.unpack_from(endian+'HHI', header, 16)
    flags_offset, size_offset = (36, 40) if elf_class == 1 else (48, 52)
    if elf_version != 1 or struct.unpack_from(endian+'H', header, size_offset)[0] != size:
        raise ValueError(f'{path}: invalid ELF version/header size')
    flags = struct.unpack_from(endian+'I', header, flags_offset)[0]
    if machine == MACHINES['arm'] and flags & 0x600 == 0x600:
        raise ValueError(f'{path}: conflicting ARM float ABI flags')
    if kind in (2, 3):
        phoff_offset, phsize_offset, phnum_offset = (28, 42, 44) if elf_class == 1 else (32, 54, 56)
        phoff = struct.unpack_from(endian+('I' if elf_class == 1 else 'Q'), header, phoff_offset)[0]
        phsize = struct.unpack_from(endian+'H', header, phsize_offset)[0]
        phnum = struct.unpack_from(endian+'H', header, phnum_offset)[0]
        file_size = Path(path).stat().st_size
        # Extended program-header numbering is deliberately unsupported.
        if (not 0 < phnum < 65535 or phsize != (32 if elf_class == 1 else 56) or
                phoff < size or phoff+phsize*phnum > file_size):
            raise ValueError(f'{path}: invalid/truncated ELF program headers')
        executable_load = False
        with Path(path).open('rb') as handle:
            handle.seek(phoff)
            for _ in range(phnum):
                program = handle.read(phsize)
                if elf_class == 1:
                    ptype, offset, vaddr, _, filesz, memsz, pflags, align = struct.unpack(endian+'IIIIIIII', program)
                else:
                    ptype, pflags, offset, vaddr, _, filesz, memsz, align = struct.unpack(endian+'IIQQQQQQ', program)
                if offset+filesz > file_size:
                    raise ValueError(f'{path}: ELF segment exceeds file bounds')
                if ptype == 1:
                    if (filesz > memsz or (align > 1 and
                            (align & (align-1) or vaddr % align != offset % align))):
                        raise ValueError(f'{path}: invalid ELF load segment')
                    executable_load |= bool(pflags & 1 and filesz)
        if not executable_load:
            raise ValueError(f'{path}: no executable ELF load segment')
    return {'class': 32 if elf_class == 1 else 64, 'machine': machine,
            'endian': 'little' if data == 1 else 'big', 'type': kind,
            'osabi': header[7], 'abi_version': header[8], 'flags': flags}


def check_elf(path, expected, allowed_types):
    header = read_elf(path)
    for key in ('class', 'machine', 'endian'):
        if header[key] != expected[key]:
            raise ValueError(f'{path}: ELF {key}={header[key]}, expected {expected[key]}; '
                             'cross compilers require --mode compile-only and explicit target identity')
    if header['type'] not in allowed_types:
        raise ValueError(f'{path}: unexpected ELF type {header["type"]}')
    # Native mode supplies the running process ABI; compile-only records these
    # fields without pretending to know a vendor's ABI or loader contract.
    # Linux tools mark GNU ELF extensions (e.g. SHF_GNU_RETAIN) as GNU=3,
    # while otherwise compatible products may be SYSV=0. glibc's
    # sysdeps/gnu/ldsodefs.h VALID_ELF_OSABI accepts both for linked products.
    # Apply only that known Linux pair, never arbitrary OSABI equivalence.
    linux_gnu_abi = (sys.platform == 'linux' and header['type'] in (1, 2, 3) and
                     expected.get('osabi') in (0, 3) and header['osabi'] in (0, 3))
    for key in ('osabi', 'abi_version'):
        if key in expected and header[key] != expected[key]:
            if key == 'osabi' and linux_gnu_abi:
                continue
            raise ValueError(f'{path}: incompatible native ELF {key}={header[key]}, '
                             f'expected {expected[key]} (type={header["type"]})')
    if header['machine'] == MACHINES['arm'] and 'flags' in expected:
        mask = 0xff000000 | (0x600 if header['type'] in (2, 3) else 0)
        if header['flags'] & mask != expected['flags'] & mask:
            raise ValueError(f'{path}: incompatible native ARM EABI/float flags')
    return header


def native_expected():
    process = Path('/proc/self/exe')
    if not process.exists():
        process = Path(sys.executable).resolve()
    expected = read_elf(process)
    if expected['class'] != ctypes.sizeof(ctypes.c_void_p) * 8:
        raise ValueError('Python ELF class disagrees with process pointer size')
    return expected


def require_native(path, allowed_types=(2, 3)):
    return check_elf(path, native_expected(), allowed_types)
