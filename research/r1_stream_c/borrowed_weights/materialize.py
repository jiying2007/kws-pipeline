#!/usr/bin/env python3
"""Add explicit immutable weight-view APIs to the frozen R1 qhead8/par4 SDK.

Source preparation only unless the existing owner model is explicitly supplied.
The copied constructor, model bytes, arithmetic and streaming policy stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
QROOT = HERE.parent / "quant_head8"
DEPENDENCIES = {
    "materialize.py": "c44aa8c0521cf1b6b7f2789eeb5d9983a2ebdcb354a9c06de82e03d06a4e3ac4",
    "pack_model.py": "ad564f0a60f2ab49e05e884302cf131669a42951a435b6efaed6f422a7e0603b",
    "owner_arm_trace.c": "7e947f7c1534ce191886b3506abb235d9342192409122525c552435c81c4891f",
}
SOURCE_PINS = {
    "core/r1dfsmn.c": "a88cd2e148eda19e50f11eb505362411ac04b44a4e2ade0e67e03641248306ed",
    "core/r1dfsmn.h": "97daebbc1dbf8b56edaf2bb36c880b70f67a5ca843411d01144919150fb88860",
    "core/r1d_stream.c": "5ca8bcc2b40ca312c778c5ce508ca5cde76b5aea802392f7c3fea9396142be1d",
    "core/r1d_stream.h": "3cab4cff776499fe7c5cb613d82358d3db7ff89cc6de419405829f6b3c52e00b",
    "include/fsmn_sdk.h": "ba115e3d56369ad655f6a7e16996c177d3cb243ce950eced7437600c0149d0b3",
    "src/fsmn_sdk.c": "69a7653bf726f44367d97bef7e77d16f9d927b10e7dacc5b7d78c7626b4f76c5",
    "src/fsmn_run.c": "f0e4fa53f9987e514cdd0123f592a206fe35de91d3310cbae13263af2478b6a9",
    "CMakeLists.txt": "7ccc59f487abfda191f3cf164eacfe8fd5666dfe07d35caf97acc128232ec877",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("missing or ambiguous pinned source transform: " + old[:70])
    return text.replace(old, new, 1)


def section(text, start, end):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("ambiguous function boundary")
    return text[text.index(start):text.index(end)]


def add_view_loader(source):
    original = section(source, "r1dfsmn_status r1dfsmn_load(",
                       "r1dfsmn_status r1dfsmn_reset(")
    view = replace_once(original,
        "r1dfsmn_status r1dfsmn_load(r1dfsmn_model *m, float *w, size_t nw,",
        "r1dfsmn_status r1dfsmn_load_view(r1dfsmn_model *m,")
    view = replace_once(view,
        "if (!m || !w || !payload || !hash) return R1DFSMN_ARGUMENT;",
        "if (!m || !payload || !hash) return R1DFSMN_ARGUMENT;\n"
        "    if ((uintptr_t)payload % R1DFSMN_VIEW_ALIGNMENT) return R1DFSMN_ARGUMENT;")
    view = replace_once(view,
        "if (nw != R1DFSMN_Q8_PACKAGE_BYTES / 4u || bytes != R1DFSMN_Q8_PACKAGE_BYTES)",
        "if (bytes != R1DFSMN_Q8_PACKAGE_BYTES)")
    view = replace_once(view,
        "if (overlaps(m, sizeof(*m), w, R1DFSMN_Q8_PACKAGE_BYTES)\n"
        "        || overlaps(m, sizeof(*m), payload, bytes)\n"
        "        || overlaps(w, R1DFSMN_Q8_PACKAGE_BYTES, payload, bytes)\n"
        "        || overlaps(m, sizeof(*m), hash, 65)\n"
        "        || overlaps(w, R1DFSMN_Q8_PACKAGE_BYTES, hash, 65)) return R1DFSMN_OVERLAP;",
        "if (overlaps(m, sizeof(*m), payload, bytes)\n"
        "        || overlaps(m, sizeof(*m), hash, 65)) return R1DFSMN_OVERLAP;")
    view = replace_once(view,
        "/* Allocate once outside the stream data path; all views are aligned. */\n"
        "    memcpy(w,payload,bytes);\n"
        "    unsigned char *raw=(unsigned char *)w;",
        "/* All checks completed before publishing immutable borrowed views. */\n"
        "    const unsigned char *raw=payload;")
    return replace_once(source, original, original + view)


def add_stream_view(source):
    original = section(source, "r1d_stream *r1d_create(", "int r1d_feed(")
    view = replace_once(original, "r1d_stream *r1d_create(", "r1d_stream *r1d_create_view(")
    view = replace_once(view,
        "    s->weights=(float *)malloc(R1DFSMN_PAYLOAD_BYTES);\n"
        "    if(!s->weights){free(s);return NULL;}\n",
        "    /* No owned model allocation; destroy never frees the caller view. */\n"
        "    s->weights=NULL;\n")
    view = replace_once(view,
        "r1dfsmn_load(&s->net,s->weights,R1DFSMN_Q8_PACKAGE_BYTES/4u,payload,bytes,1,R1D_MODEL_SHA)",
        "r1dfsmn_load_view(&s->net,payload,bytes,1,R1D_MODEL_SHA)")
    return replace_once(source, original, original + view)


def add_sdk_view(source):
    original = section(source, "fsmn_sdk *fsmn_sdk_create(", "int fsmn_sdk_feed(")
    view = replace_once(original, "fsmn_sdk *fsmn_sdk_create(", "fsmn_sdk *fsmn_sdk_create_view(")
    view = replace_once(view, "s->core=r1d_create(", "s->core=r1d_create_view(")
    return replace_once(source, original, original + view)


def view_cli(source):
    source = replace_once(source, "fsmn_sdk *s=fsmn_sdk_create(", "fsmn_sdk *s=fsmn_sdk_create_view(")
    source = replace_once(source, ");free(w);free(t);", ");free(t);")
    source = replace_once(source, "if (!s) {fprintf(stderr,", "if (!s) {free(w);fprintf(stderr,")
    old = section(source, "    struct rusage usage;int usage_ok=", "    return rc?2:0;")
    new = r'''    struct rusage usage;long peak=-1;
    if(getrusage(RUSAGE_SELF,&usage) || usage.ru_maxrss<0)rc=1;else peak=usage.ru_maxrss;
    uint64_t frames=fsmn_sdk_frame_count(s),events=fsmn_sdk_event_count(s);
    size_t state_bytes=fsmn_sdk_state_bytes(),model_bytes=fsmn_sdk_weight_bytes();
    if(fsmn_sdk_destroy(s))rc=1;
    free(w);
    if(fflush(stdout) || ferror(stdout))rc=1;
    if(fprintf(stderr,"{\"status\":\"%s\",\"pcm_bytes\":%" PRIu64 ",\"model_frames\":%" PRIu64 ",\"proposals\":%" PRIu64 ",\"compute_and_io_seconds\":%.9f,\"eof_padding_samples\":0,\"model_state_bytes\":%zu,\"decoded_weight_bytes\":%zu,\"max_rss_kib\":%ld,\"weight_storage\":\"borrowed_immutable\",\"owned_model_copy_bytes\":0,\"physical_ssc305\":false,\"calibrated_far\":null}\n",
        rc?"failed":"complete",bytes,frames,events,elapsed,state_bytes,model_bytes,peak)<0)rc=1;
    if(fflush(stderr) || ferror(stderr))rc=1;
'''
    return replace_once(source, old, new)


def view_trace(source):
    source = replace_once(source, "#include <errno.h>", "#include <errno.h>\n#include <time.h>\n#include <sys/resource.h>")
    source = replace_once(source,
        "    fsmn_sdk *sdk=fsmn_sdk_create(w,FSMN_MODEL_BYTES,t,FSMN_TABLE_BYTES,0,on_event,on_frame,NULL);\n"
        "    free(w);free(t);\n"
        "    if(!sdk){fputs(\"model identity/host rejected\\n\",stderr);return 2;}",
        "    struct timespec begin,end;\n"
        "    if(clock_gettime(CLOCK_MONOTONIC,&begin)){free(w);free(t);return 2;}\n"
        "    fsmn_sdk *sdk=fsmn_sdk_create_view(w,FSMN_MODEL_BYTES,t,FSMN_TABLE_BYTES,0,on_event,on_frame,NULL);\n"
        "    int bad_clock=clock_gettime(CLOCK_MONOTONIC,&end);free(t);\n"
        "    double init_ms=bad_clock?-1.0:((double)(end.tv_sec-begin.tv_sec)*1000.0+(double)(end.tv_nsec-begin.tv_nsec)/1e6);\n"
        "    if(!sdk || init_ms<0.0){if(sdk)(void)fsmn_sdk_destroy(sdk);free(w);fputs(\"model identity/host/clock rejected\\n\",stderr);return 2;}")
    source = replace_once(source,
        "FILE *pcm=fopen(argv[3],\"rb\");if(!pcm){fsmn_sdk_destroy(sdk);return 2;}",
        "FILE *pcm=fopen(argv[3],\"rb\");if(!pcm){fsmn_sdk_destroy(sdk);free(w);return 2;}")
    source = replace_once(source,
        "    fprintf(stderr,\"{\\\"ok\\\":%s,\\\"pcm_bytes\\\":%\" PRIu64 \",\\\"frames\\\":%lu,\\\"events\\\":%lu,\\\"weights\\\":%zu}\\n\",\n"
        "            failure?\"false\":\"true\",total,frames,events,fsmn_sdk_weight_bytes());\n"
        "    if(fsmn_sdk_destroy(sdk))failure=1;",
        "    struct rusage usage;long peak=-1;\n"
        "    if(getrusage(RUSAGE_SELF,&usage) || usage.ru_maxrss<0)failure=1;else peak=usage.ru_maxrss;\n"
        "    size_t weight_bytes=fsmn_sdk_weight_bytes();\n"
        "    if(fsmn_sdk_destroy(sdk))failure=1;\n    free(w);\n"
        "    if(fprintf(stderr,\"{\\\"ok\\\":%s,\\\"pcm_bytes\\\":%\" PRIu64 \",\\\"frames\\\":%lu,\\\"events\\\":%lu,\\\"weights\\\":%zu,\\\"weight_storage\\\":\\\"borrowed_immutable\\\",\\\"model_init_wall_ms\\\":%.6f,\\\"process_peak_rss_kib\\\":%ld,\\\"physical_ssc305\\\":false}\\n\",\n"
        "            failure?\"false\":\"true\",total,frames,events,weight_bytes,init_ms,peak)<0)failure=1;\n"
        "    if(fflush(stderr) || ferror(stderr))failure=1;")
    return source


def materialize(repo, output, model=None, frontend=None):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("fresh non-symlink output required")
    for name, expected in DEPENDENCIES.items():
        path = QROOT / name
        if path.is_symlink() or digest(path.read_bytes()) != expected:
            raise ValueError("quantized source dependency drift: " + name)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("r1_view_qhead8_source", QROOT / "materialize.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".r1-view-source-", dir=output.parent) as temp:
        generated = Path(temp) / "sdk"
        inherited = q.materialize_from_repository(Path(repo), generated, model, frontend, "par4")
        sources = {}
        for name, expected in SOURCE_PINS.items():
            raw = (generated / name).read_bytes()
            if digest(raw) != expected:
                raise ValueError("generated par4 source drift: " + name)
            sources[name] = raw.decode("utf-8")
        changed = {
            "core/r1dfsmn.c": add_view_loader(sources["core/r1dfsmn.c"]),
            "core/r1d_stream.c": add_stream_view(sources["core/r1d_stream.c"]),
            "src/fsmn_sdk.c": add_sdk_view(sources["src/fsmn_sdk.c"]),
            "src/fsmn_run_view.c": view_cli(sources["src/fsmn_run.c"]),
            "src/fsmn_trace_view.c": view_trace((QROOT / "owner_arm_trace.c").read_text()),
            "src/fsmn_init_bench.c": (HERE / "init_bench.c").read_text(),
        }
        changed["core/r1dfsmn.h"] = replace_once(sources["core/r1dfsmn.h"],
            "r1dfsmn_status r1dfsmn_reset(",
            "/* Same storage contract as fsmn_sdk_create_view: immutable, 8-byte\n"
            " * aligned malloc/read or read-only mmap storage outlives the model.\n"
            " * Do not populate the view by memcpy from a declared byte array. */\n"
            "#define R1DFSMN_VIEW_ALIGNMENT 8u\n"
            "r1dfsmn_status r1dfsmn_load_view(r1dfsmn_model *model,\n"
            "    const unsigned char *payload, size_t bytes, unsigned byte_order,\n"
            "    const char expected_sha256[65]);\n\n"
            "r1dfsmn_status r1dfsmn_reset(")
        changed["core/r1d_stream.h"] = replace_once(sources["core/r1d_stream.h"],
            "int r1d_feed(",
            "/* Borrowed model has the public SDK view ownership/alignment contract. */\n"
            "r1d_stream *r1d_create_view(const unsigned char *model,size_t model_bytes,\n"
            "    const unsigned char *tables,size_t table_bytes,r1d_sink sink,void *user);\n"
            "int r1d_feed(")
        changed["include/fsmn_sdk.h"] = replace_once(sources["include/fsmn_sdk.h"],
            "/* At most FSMN_MAX_INPUT_BYTES per call.",
            "/* Opt-in single-copy model storage. The original create still copies.\n"
            " * model must be 8-byte aligned, immutable, and remain alive until destroy.\n"
            " * Populate fresh malloc/aligned allocation directly with fread/read,\n"
            " * or use read-only mmap storage. Declared byte arrays, including memcpy\n"
            " * from them into heap storage, violate the typed-view aliasing contract.\n"
            " * For aligned_alloc round allocation capacity to 8 bytes; model_bytes\n"
            " * remains the exact FSMN_MODEL_BYTES, not the rounded capacity.\n"
            " * No ownership transfer on success or failure; caller frees model only\n"
            " * after destroy. Other threads/processes must not modify a shared view.\n"
            " * Tables are still copied and may be released after this call.\n"
            " * All original size, hash, structure and numeric validation is retained. */\n"
            "#define FSMN_MODEL_VIEW_ALIGNMENT 8u\n"
            "fsmn_sdk *fsmn_sdk_create_view(const unsigned char *model,size_t model_bytes,\n"
            "    const unsigned char *tables,size_t table_bytes,uint64_t origin_sample,\n"
            "    fsmn_event_fn event_fn,fsmn_frame_fn frame_fn,void *user);\n"
            "/* At most FSMN_MAX_INPUT_BYTES per call.")
        changed["CMakeLists.txt"] = sources["CMakeLists.txt"] + '''
# Explicit immutable-view executables; no automatic model execution.
foreach(target fsmn_run_view fsmn_trace_view fsmn_init_bench)
  add_executable(${target} src/${target}.c)
  target_link_libraries(${target} PRIVATE fsmn_sdk_static m)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
  if(FSMN_SANITIZE)
    target_compile_options(${target} PRIVATE -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
  install(TARGETS ${target} RUNTIME DESTINATION bin)
endforeach()
'''
        for name, body in changed.items():
            (generated / name).write_text(body)
        report = {
            "schema": "r1-qhead8-par4-borrowed-source-v1",
            "source_model_id": inherited["source_model_id"],
            "q8_model_sha256": inherited["q8_model_sha256"],
            "q8_model_bytes": inherited["q8_model_bytes"],
            "private_model_packed": inherited["private_model_packed"],
            "kernel": "par4_unchanged",
            "model_forwards": 0,
            "borrowed_alignment_bytes": 8,
            "borrowed_model_copy_bytes": 0,
            "copied_api_preserved": True,
            "measured_resource_reduction": "NOT_RUN_FOR_THIS_BUILD",
            "frame_event_clock_parity": "NOT_RUN_FOR_THIS_BUILD",
            "physical_ssc305": "NOT_RUN",
            "independent_acoustic_validation": "NOT_RUN",
            "original_fp32_numeric_gate": "FAIL_UNCHANGED",
            "source_archive_sha256": inherited["source_archive_sha256"],
            "input_source_sha256": SOURCE_PINS,
            "changed_source_sha256": {name: digest(body.encode()) for name, body in changed.items()},
        }
        (generated / "BORROWED_PREPARATION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (generated / "README-BORROWED.md").write_bytes((HERE / "README.md").read_bytes())
        if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError("output appeared during preparation")
        generated.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frontend", type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.repo, args.output, args.model, args.frontend), sort_keys=True))


if __name__ == "__main__":
    main()
