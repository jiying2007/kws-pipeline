#!/usr/bin/env python3
"""Stage only the historically verified host bytes with source and notices.

Local copying and checksum generation only. No ELF or model execution. A rebuilt
binary with a different hash needs its own review; this script does not bless it.
Apache-2.0; see the repository LICENSE.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from materialize import ROOT, checked_bytes, digest, load_json, verify_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="verified materialized source tree")
    parser.add_argument("--runtime", required=True, type=Path, help="local bin/, lib/ and models/ layout")
    parser.add_argument("--model-notices", required=True, type=Path, help="original README, Apache text and ORIGIN.json")
    parser.add_argument("--output", required=True, type=Path, help="new local kit directory")
    args = parser.parse_args()
    source, output = args.source.absolute(), args.output.absolute()
    verify_tree(source, load_json(ROOT / "sources.lock.json"))
    if output.exists() or output.is_symlink() or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("output must be new, with canonical existing parent")
    for root in (ROOT.parents[1], source, args.runtime.resolve(), args.model_notices.resolve()):
        if output.is_relative_to(root):
            raise ValueError("output must be outside repository and input trees")
    lock = load_json(ROOT / "dependencies.lock.json")
    temp = Path(tempfile.mkdtemp(prefix=".kws-stage-", dir=output.parent))
    try:
        payloads = {}
        for name, record in lock["verified_runtime"].items():
            payloads[name] = checked_bytes(args.runtime, name, record["sha256"], record["bytes"])
        for name, record in lock["model_notices"].items():
            payloads["licenses/model/" + name] = checked_bytes(args.model_notices, name, record["sha256"], record["bytes"])
        for name in ("run-kws", "config/profile.json", "config/keywords.txt"):
            payloads[name] = (source / "cli" / name).read_bytes()
        payloads["LICENSE"] = (ROOT.parents[1] / "LICENSE").read_bytes()
        for name in ("README.md", "README.zh-CN.md", "BUILDING.md", "PROVENANCE.md", "RESULTS.md"):
            payloads[name] = (ROOT / name).read_bytes()
        for name, data in payloads.items():
            target = temp / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data); target.chmod(0o755 if name in ("run-kws", "bin/kws") else 0o644)
        shutil.copytree(source, temp / "source")
        verify_tree(temp / "source", load_json(ROOT / "sources.lock.json"))
        manifest = "".join(digest(f.read_bytes()) + "  " + f.relative_to(temp).as_posix() + "\n"
                           for f in sorted(temp.rglob("*")) if f.is_file())
        (temp / "MANIFEST.sha256").write_text(manifest)
        output.mkdir()
        for child in temp.iterdir():
            child.rename(output / child.name)
        temp.rmdir()
    except BaseException:
        shutil.rmtree(temp)
        raise
    print(json.dumps({"manifest_sha256": digest(manifest.encode()), "executed": False, "uploaded": False}))


if __name__ == "__main__":
    main()
