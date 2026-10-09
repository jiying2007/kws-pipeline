#!/usr/bin/env python3
"""Validate the release SDK's SPDX 2.3 profile, optionally against SDK bytes.

This is intentionally not a general-purpose SPDX validator: unknown fields and
other SPDX profiles are rejected, rather than silently left unchecked. Rules
come from the v2.3 tag (not v2.3.1): chapters/document-creation-information.md,
package-information.md and file-information.md in https://github.com/spdx/spdx-spec.
In particular 8.4 requires one SHA1 per file. Section 7.9 says Required: Yes but
cardinality 0..1 for analyzed packages; this release profile always supplies and
verifies the code. SHA256 remains required by our stronger artifact contract.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re

NAMESPACE_PREFIX = "https://github.com/jiying2007/kws-pipeline/sbom/"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fields(value: object, expected: set[str], label: str) -> None:
    require(isinstance(value, dict) and set(value) == expected,
            f"{label}: missing or unsupported fields")


def line(value: object, label: str) -> None:
    require(isinstance(value, str) and bool(value.strip()) and
            not any(ord(char) < 32 or ord(char) == 127 for char in value),
            f"{label}: expected nonempty single-line text")


def hex_digest(value: object, length: int, label: str) -> None:
    require(isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
            f"{label}: expected {length} lowercase hex digits")


def file_checksums(path: pathlib.Path) -> dict[str, str]:
    # SHA1 is a required SPDX identifier, not a security primitive here.
    digests = {"SHA1": hashlib.sha1(usedforsecurity=False), "SHA256": hashlib.sha256()}
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for digest in digests.values():
                digest.update(chunk)
    return {algorithm: digest.hexdigest() for algorithm, digest in digests.items()}


def sdk_files(root: pathlib.Path) -> list[pathlib.Path]:
    require(root.is_dir(), "SBOM root must be a directory")
    paths = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"SDK symlinks are unsupported: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"SDK entry is not a regular file: {path}")
        paths.append(path)
    require(bool(paths), "SBOM root contains no files")
    return paths


def verification_code(sha1_values: list[str]) -> str:
    return hashlib.sha1("".join(sorted(sha1_values)).encode("ascii"),
                        usedforsecurity=False).hexdigest()


def document_namespace(document: dict) -> str:
    # Hash the entire document except its own namespace, including paths, both
    # digests, source revision and metadata. Relocation does not change identity.
    content = {key: value for key, value in document.items() if key != "documentNamespace"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return NAMESPACE_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate(document: object, root: pathlib.Path | None = None) -> None:
    fields(document, {"spdxVersion", "dataLicense", "SPDXID", "name", "documentNamespace",
                      "creationInfo", "packages", "files", "relationships"}, "document")
    require(document["spdxVersion"] == "SPDX-2.3", "expected SPDX-2.3")
    require(document["dataLicense"] == "CC0-1.0", "expected CC0-1.0 data license")
    require(document["SPDXID"] == "SPDXRef-DOCUMENT", "invalid document SPDXID")
    line(document["name"], "document name")
    creation = document["creationInfo"]
    fields(creation, {"created", "creators"}, "creationInfo")
    created = creation["created"]
    require(isinstance(created, str) and
            re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created) is not None,
            "invalid creation timestamp")
    datetime.datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
    creators = creation["creators"]
    require(isinstance(creators, list) and bool(creators), "creators must be nonempty")
    for creator in creators:
        line(creator, "creator")
        require(re.fullmatch(r"(?:Person|Organization|Tool): \S.*", creator) is not None,
                "invalid SPDX creator")

    packages = document["packages"]
    require(isinstance(packages, list) and len(packages) == 1,
            "SDK profile requires exactly one package")
    package = packages[0]
    fields(package, {"SPDXID", "name", "versionInfo", "downloadLocation", "filesAnalyzed",
                     "packageVerificationCode", "licenseConcluded", "licenseDeclared",
                     "copyrightText", "externalRefs"}, "package")
    require(package["SPDXID"] == "SPDXRef-Package", "invalid package SPDXID")
    for key in ("name", "versionInfo"):
        line(package[key], f"package {key}")
    require(document["name"] == f"{package['name']}-{package['versionInfo']}",
            "document/package names disagree")
    require(package["filesAnalyzed"] is True, "SDK package must analyze its files")
    for key in ("downloadLocation", "licenseConcluded", "licenseDeclared", "copyrightText"):
        require(package[key] == "NOASSERTION", f"unsupported SDK profile {key}")
    refs = package["externalRefs"]
    require(isinstance(refs, list) and len(refs) == 1, "expected one source revision")
    fields(refs[0], {"referenceCategory", "referenceType", "referenceLocator"}, "source reference")
    require(refs[0]["referenceCategory"] == "OTHER" and refs[0]["referenceType"] == "gitCommit",
            "invalid source reference type")
    revision = refs[0]["referenceLocator"]
    require(isinstance(revision, str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision)
            is not None, "source-sha must be 40 or 64 lowercase hex digits")

    files = document["files"]
    require(isinstance(files, list) and bool(files), "files must be nonempty")
    ids = {document["SPDXID"], package["SPDXID"]}
    checksums_by_name = {}
    for file in files:
        fields(file, {"SPDXID", "fileName", "checksums"}, "file")
        identifier = file["SPDXID"]
        require(isinstance(identifier, str) and
                re.fullmatch(r"SPDXRef-[A-Za-z0-9.-]+", identifier) is not None,
                "invalid file SPDXID")
        require(identifier not in ids, "duplicate SPDXID")
        ids.add(identifier)
        name = file["fileName"]
        line(name, "fileName")
        require(not name.startswith("/") and "\\" not in name and
                all(part not in ("", ".", "..") for part in name.split("/")),
                "fileName must be a normalized package-relative path")
        require(name not in checksums_by_name, "duplicate fileName")
        checksums = file["checksums"]
        require(isinstance(checksums, list), "checksums must be a list")
        values = {}
        for checksum in checksums:
            fields(checksum, {"algorithm", "checksumValue"}, "checksum")
            algorithm = checksum["algorithm"]
            require(isinstance(algorithm, str) and algorithm in ("SHA1", "SHA256"),
                    "unsupported SDK checksum algorithm")
            require(algorithm not in values, "duplicate checksum algorithm")
            hex_digest(checksum["checksumValue"], 40 if algorithm == "SHA1" else 64, algorithm)
            values[algorithm] = checksum["checksumValue"]
        require(set(values) == {"SHA1", "SHA256"}, "each file requires SHA1 and SHA256")
        checksums_by_name[name] = values

    code = package["packageVerificationCode"]
    fields(code, {"packageVerificationCodeValue"}, "packageVerificationCode")
    hex_digest(code["packageVerificationCodeValue"], 40, "packageVerificationCode")
    require(code["packageVerificationCodeValue"] ==
            verification_code([value["SHA1"] for value in checksums_by_name.values()]),
            "package verification code mismatch")
    expected = {("SPDXRef-DOCUMENT", "DESCRIBES", "SPDXRef-Package")}
    expected.update(("SPDXRef-Package", "CONTAINS", file["SPDXID"]) for file in files)
    relationships = document["relationships"]
    require(isinstance(relationships, list), "relationships must be a list")
    actual = set()
    for relationship in relationships:
        fields(relationship, {"spdxElementId", "relationshipType", "relatedSpdxElement"},
               "relationship")
        for value in relationship.values():
            line(value, "relationship value")
        value = (relationship["spdxElementId"], relationship["relationshipType"],
                 relationship["relatedSpdxElement"])
        require(value not in actual, "duplicate relationship")
        require(value in expected, "unsupported or dangling relationship")
        actual.add(value)
    require(actual == expected, "incomplete document/package/file relationships")
    require(document["documentNamespace"] == document_namespace(document),
            "document namespace does not identify its content")
    if root is not None:
        root = root.resolve()
        paths = {path.relative_to(root).as_posix(): path for path in sdk_files(root)}
        require(set(paths) == set(checksums_by_name), "SBOM/SDK file inventory mismatch")
        for name, path in paths.items():
            require(file_checksums(path) == checksums_by_name[name],
                    f"SBOM/SDK checksum mismatch: {name}")


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=pathlib.Path)
    parser.add_argument("--root", type=pathlib.Path, required=True,
                        help="installed SDK directory to verify against the SBOM")
    args = parser.parse_args()
    try:
        document = json.loads(args.document.read_text(encoding="utf-8"),
                              object_pairs_hook=unique_object)
        validate(document, args.root)
    except (ValueError, OSError) as error:
        parser.exit(1, f"invalid SDK SBOM: {error}\n")
    print("SDK SPDX-2.3 semantics and file bytes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
