"""Retain actual exceptions around one unchanged, pinned native worker entry.

This shim never imports a model, changes a decoder, repairs a return, or retries.
The host includes its exact bytes in the executed control-source inventory.
"""
import json
from pathlib import Path
import runpy
import sys
import traceback

ALLOWED = {
    "/code/research/native-asr-integration-20261104/prepare_worker.py",
    "/code/research/native-asr-integration-20261104/qwen_worker.py",
    "/code/research/sense-native-capture-20261104/worker.py",
    "/code/research/whisper-native-capture-20261104/worker.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ALLOWED:
        raise ValueError("one fixed native entry required")
    original = sys.argv[1:]
    sys.argv = original
    try:
        runpy.run_path(original[0], run_name="__main__")
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            return
        chain = []
        current = error
        for _ in range(3):
            if current is None:
                break
            chain.append({"error_type": type(current).__name__, "message": str(current)[:2048],
                "frames": [{"file": row.filename, "line": row.lineno, "function": row.name}
                           for row in traceback.extract_tb(current.__traceback__)[-12:]]})
            current = current.__cause__ or current.__context__
        with Path("/output/worker-failure.json").open("x") as stream:
            json.dump({"schema": "public-zh64-actual-worker-failure-v1", "entry": original[0],
                       "status": "failed_no_retry", "exceptions": chain,
                       "training_admitted": False}, stream, ensure_ascii=False, allow_nan=False)
        raise


if __name__ == "__main__":
    main()
