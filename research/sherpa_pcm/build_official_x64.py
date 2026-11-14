#!/usr/bin/env python3
"""Opt-in build against exact official Sherpa v1.13.8 x64 shared archive.

This DOES NOT replace the older host runtime lock in dependencies.lock.json.
Source-pinned research build only, no model data, no target-board claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ARCHIVE_SHA="c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4"
ARCHIVE_BYTES=28156791
ARCHIVE_ROOT="sherpa-onnx-v1.13.8-linux-x64-shared"
LIBRARIES={
    "libsherpa-onnx-c-api.so":(
        5124192,
        "b8351ca1632571ac108adbb317bcc4bf7cfe84b72690e3017316b0da3e1e344f"
    ),
    "libonnxruntime.so":(
        27026609,
        "4b3607aebd1784b26b6f9b20e4bd974c7ab8287043e4d095cb7d2cb40b5e566e"
    )
}
SOURCE_FILES=(
    "pcm_kws.cc","pcm_kws.h","dependencies.lock.json",
    "vendor/sherpa-onnx/c-api/c-api.h"
)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def hash_file(path):
    obj=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            obj.update(chunk)
    return obj.hexdigest()


def verify_original_archive(path):
    require(path.is_absolute() and path.is_file() and
            not path.is_symlink() and path.stat().st_size==ARCHIVE_BYTES and
            hash_file(path)==ARCHIVE_SHA,
            "EN_C_OFFICIAL_HOST_ARCHIVE_CHANGED")
    return True


def extract_only_verified_libraries(archive, dest):
    verify_original_archive(archive)
    require(dest.is_dir() and not dest.is_symlink(),"EN_C_LIB_DEST")
    wanted={ARCHIVE_ROOT+"/lib/"+name:name for name in LIBRARIES}
    done={}
    with tarfile.open(archive,"r:bz2") as tar:
        for item in tar:
            if item.name not in wanted:
                continue
            key=wanted[item.name]
            require(key not in done and item.isfile() and not item.issym() and
                    item.size==LIBRARIES[key][0],
                    "EN_C_OFFICIAL_LIBRARY_MEMBER_CHANGED")
            source=tar.extractfile(item)
            require(source is not None,"EN_C_OFFICIAL_LIBRARY_SHORT")
            path=dest/key
            with source, path.open("xb") as out:
                shutil.copyfileobj(source,out,1024*1024)
            require(path.stat().st_size==LIBRARIES[key][0] and
                    hash_file(path)==LIBRARIES[key][1],
                    "EN_C_OFFICIAL_LIBRARY_HASH_DRIFT")
            done[key]=path
    require(set(done)==set(LIBRARIES),
            "EN_C_OFFICIAL_LIBRARY_MISSING")
    return done


def build(archive,output):
    root=Path(__file__).resolve().parent
    require(output.is_absolute() and output.parent.is_dir() and
            not output.parent.is_symlink() and not output.exists(),
            "EN_C_OFFICIAL_OUTPUT_CREATE_ONLY")
    verify_original_archive(archive)
    subprocess.run([sys.executable,str(root/"verify_dependencies.py")],check=True)
    output.mkdir(mode=0o700)
    libdir=output/"runtime"
    libdir.mkdir(mode=0o700)
    members=extract_only_verified_libraries(archive,libdir)
    target=output/"libkws_sherpa_pcm.so"
    command=[
        "g++","-O2","-std=c++17","-Wall","-Wextra","-Werror",
        "-fPIC","-shared",str(root/"pcm_kws.cc"),
        "-I"+str(root/"vendor"),"-L"+str(libdir),
        "-Wl,-rpath,$ORIGIN/runtime","-lsherpa-onnx-c-api",
        "-o",str(target),
    ]
    subprocess.run(command,check=True,timeout=120)
    require(target.is_file() and target.stat().st_size>1000,
            "EN_C_OFFICIAL_NATIVE_BUILD_MISSING")
    result={
        "schema":"english63-official-host-c-runtime-source-locked-v1",
        "runtime_profile":"original-sherpa-v1.13.8-x64-release",
        "runtime_archive_sha256":ARCHIVE_SHA,
        "runtime_archive_bytes":ARCHIVE_BYTES,
        "runtime_archive_path":str(archive),
        "files":{f:hash_file(root/f) for f in SOURCE_FILES},
        "runtime_libraries":[{
            "filename":name,"sha256":hash_file(members[name]),
            "bytes":members[name].stat().st_size,
            "path":str(members[name]),
        } for name in LIBRARIES],
        "library_sha256":hash_file(target),
        "compiler":subprocess.check_output(["g++","--version"],text=True).splitlines()[0],
        "commands":[command],
        "stub_only":False,"speech_inference_tested":False,
        "model_promotion_allowed":False,
        "SSC305_qualification":"NOT_RUN",
    }
    path=output/"build-receipt.json"
    with path.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True)
        f.write("\n")
    path.chmod(0o600)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:
        result=build(args.runtime_archive,args.output)
        print(json.dumps({
            "status":"OFFICIAL_RUNTIME_SOURCE_PINNED_NATIVE_RESEARCH_BUILD",
            "library_sha256":result["library_sha256"],
            "runtime_archive_sha256":result["runtime_archive_sha256"],
            "model_promotion_allowed":False},sort_keys=True))
        return 0
    except (ValueError,OSError,RuntimeError,tarfile.TarError,subprocess.CalledProcessError) as e:
        print("EN_C_OFFICIAL_BUILD_STOP:"+str(e),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
