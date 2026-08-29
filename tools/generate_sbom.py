#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spdx_id(index: int) -> str:
    return f"SPDXRef-File-{index:06d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError("SBOM root must be a directory")
    if re.fullmatch(r"[0-9a-f]{40,64}", args.source_sha) is None:
        raise ValueError("source-sha must be lowercase git/SHA256 hex")

    files = []
    relationships = []
    for index, path in enumerate(
        sorted(item for item in root.rglob("*") if item.is_file()), 1
    ):
        identifier = spdx_id(index)
        files.append(
            {
                "SPDXID": identifier,
                "fileName": path.relative_to(root).as_posix(),
                "checksums": [
                    {"algorithm": "SHA256", "checksumValue": sha256_file(path)}
                ],
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Package",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": identifier,
            }
        )
    if not files:
        raise ValueError("SBOM root contains no files")

    namespace_seed = hashlib.sha256(
        f"{args.name}\0{args.version}\0{args.source_sha}".encode("utf-8")
    ).hexdigest()
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{args.name}-{args.version}",
        "documentNamespace": f"https://github.com/jiying2007/kws-pipeline/sbom/{namespace_seed}",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: kws-pipeline/tools/generate_sbom.py"],
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package",
                "name": args.name,
                "versionInfo": args.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": True,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "OTHER",
                        "referenceType": "gitCommit",
                        "referenceLocator": args.source_sha,
                    }
                ],
            }
        ],
        "files": files,
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package",
            },
            *relationships,
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
