from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMAKE = os.environ.get("CMAKE_COMMAND", "cmake")


def run(*args: str | pathlib.Path) -> str:
    result = subprocess.run([str(arg) for arg in args], check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise AssertionError(f"command failed ({result.returncode}): {args}\n{result.stdout}")
    return result.stdout


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(build: pathlib.Path, *options: str, generator: str = "Unix Makefiles",
              source: pathlib.Path = ROOT) -> None:
    run(CMAKE, "-S", source, "-B", build, "-G", generator,
        "-DKWS_BUILD_TESTS=OFF", "-DKWS_BUILD_BENCH=OFF", "-DKWS_BUILD_TOOLS=OFF",
        "-DKWS_SOURCE_REVISION=identity-regression", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        *options)


def build_identity(build: pathlib.Path, config: str = "Release") -> dict[str, str]:
    run(CMAKE, "--build", build, "--config", config or "Release", "--parallel", "2")
    header = build / "generated" / config / "kws_build_config.h"
    if not header.exists():
        header = build / "generated" / "kws_build_config.h"
    identity = dict(re.findall(r'^#define KWS_BUILD_(\w+) "([^"]*)"$',
                               header.read_text(), flags=re.MULTILINE))
    assert re.fullmatch(r"[0-9a-f]{64}", identity["CONFIG_DIGEST"]), identity
    assert identity["TYPE"] == (config or "unspecified"), identity
    # Check the selected identity in the built archive, not only an unused header.
    library = build / config / "libkws_pipeline.a"
    if not library.exists():
        library = build / "libkws_pipeline.a"
    assert identity["CONFIG_DIGEST"].encode() in library.read_bytes(), library
    return identity


def test_inventory(root: pathlib.Path) -> None:
    first = root / "a"
    second = root / "b"
    first.mkdir()
    second.mkdir()
    (first / "lib.a").write_bytes(b"same")
    (second / "lib.a").write_bytes(b"same")
    command = [sys.executable, str(ROOT / "tools" / "check_reproducible_sdk.py"),
               str(first), str(second)]
    assert subprocess.run(command, check=False).returncode == 0
    (second / "lib.a").write_bytes(b"different")
    assert subprocess.run(command, check=False).returncode == 1


def test_single_config_identity(root: pathlib.Path) -> None:
    slow, fast, repeat = (root / name for name in ("o0", "o3", "o3-again"))
    for build, flag in ((slow, "-O0"), (fast, "-O3"), (repeat, "-O3")):
        configure(build, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_FLAGS_RELEASE={flag}")
    slow_id, fast_id, repeat_id = (build_identity(build) for build in (slow, fast, repeat))
    assert slow_id["CONFIG_DIGEST"] != fast_id["CONFIG_DIGEST"], "Release flags were ignored"
    assert fast_id == repeat_id, "build-directory paths contaminated the identity"
    for build, flag in ((slow, "-O0"), (fast, "-O3")):
        commands = json.loads((build / "compile_commands.json").read_text())
        frontend = next(row for row in commands if row["file"].endswith("/src/frontend.c"))
        assert flag in frontend["command"].split(), frontend
    object_path = pathlib.Path("CMakeFiles/kws_pipeline.dir/src/frontend.c.o")
    assert sha256(slow / object_path) != sha256(fast / object_path), "optimization was not applied"
    for build in (fast, repeat):
        run(CMAKE, "--install", build, "--prefix", build / "sdk")
    run(sys.executable, ROOT / "tools/check_reproducible_sdk.py", fast / "sdk", repeat / "sdk")

    # An existing build directory must not keep the old configuration header.
    configure(slow, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_FLAGS_RELEASE=-O3")
    assert build_identity(slow) == fast_id, "reconfigure retained a stale fingerprint"
    # Empty build type has no configuration-specific flags and remains supported.
    unspecified = root / "unspecified"
    configure(unspecified, "-DCMAKE_BUILD_TYPE=")
    build_identity(unspecified, "")


def test_relocated_debug_sdk(root: pathlib.Path) -> None:
    identities, sdks = [], []
    for name in ("relocated-a", "relocated-b"):
        source, build = root / name / "source", root / name / "build"
        source.mkdir(parents=True)
        shutil.copyfile(ROOT / "CMakeLists.txt", source / "CMakeLists.txt")
        for directory in ("cmake", "src", "include", "configs", "tools"):
            shutil.copytree(ROOT / directory, source / directory)
        # Debug paths in both the binary and flags are deliberately relocatable.
        # The fingerprint must not undo compiler prefix-map reproducibility.
        flags = (f"-g -ffile-prefix-map={source}=/kws/source "
                 f"-ffile-prefix-map={build}=/kws/build")
        configure(build, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_FLAGS={flags}", source=source)
        identities.append(build_identity(build))
        sdks.append(build / "sdk")
        run(CMAKE, "--install", build, "--prefix", sdks[-1])
    assert identities[0] == identities[1], "normalized source/build paths changed the fingerprint"
    run(sys.executable, ROOT / "tools/check_reproducible_sdk.py", *sdks)


def test_toolchain_identity(root: pathlib.Path) -> None:
    first, copied, changed = (root / name for name in ("toolchain-a", "toolchain-b", "toolchain-c"))
    for path in (first, copied, changed):
        path.mkdir()
        (path / "toolchain.cmake").write_text("set(CMAKE_SYSTEM_NAME Linux)\n")
    (changed / "toolchain.cmake").write_text("set(CMAKE_SYSTEM_NAME Linux)\nset(CMAKE_SYSTEM_PROCESSOR identity-test)\n")
    identities = []
    for path in (first, copied, changed):
        configure(path / "build", "-DCMAKE_BUILD_TYPE=Release",
                  f"-DCMAKE_TOOLCHAIN_FILE={path / 'toolchain.cmake'}")
        identities.append(build_identity(path / "build"))
    assert identities[0] == identities[1], "toolchain location contaminated the identity"
    assert identities[0]["CONFIG_DIGEST"] != identities[2]["CONFIG_DIGEST"], "toolchain changes were ignored"


def test_multi_config_identity(root: pathlib.Path) -> None:
    capabilities = json.loads(run(CMAKE, "-E", "capabilities"))
    generators = {row["name"] for row in capabilities["generators"]}
    if "Ninja Multi-Config" not in generators or not shutil.which("ninja"):
        print("Ninja Multi-Config identity: skipped (requires CMake >= 3.17 and ninja)")
        return
    first, repeat = root / "multi", root / "multi-again"
    identities = []
    for build in (first, repeat):
        # Build type is deliberately misleading: a multi-config generator must
        # choose --config, never CMAKE_BUILD_TYPE, for flags and embedded identity.
        configure(build, "-DCMAKE_BUILD_TYPE=Ignored", "-DCMAKE_CONFIGURATION_TYPES=Debug;Release",
                  "-DCMAKE_C_FLAGS_DEBUG=-O0", "-DCMAKE_C_FLAGS_RELEASE=-O3",
                  generator="Ninja Multi-Config")
        identities.append({config: build_identity(build, config) for config in ("Debug", "Release")})
    assert identities[0] == identities[1], "multi-config identity is not repeatable"
    assert identities[0]["Debug"]["CONFIG_DIGEST"] != identities[0]["Release"]["CONFIG_DIGEST"]
    # Vary the flags of the same configuration; differing names alone are not a test.
    configure(first, "-DCMAKE_BUILD_TYPE=Ignored", "-DCMAKE_CONFIGURATION_TYPES=Debug;Release",
              "-DCMAKE_C_FLAGS_DEBUG=-O0", "-DCMAKE_C_FLAGS_RELEASE=-O0",
              generator="Ninja Multi-Config")
    assert build_identity(first, "Release")["CONFIG_DIGEST"] != identities[0]["Release"]["CONFIG_DIGEST"]
    assert build_identity(first, "Debug") == identities[0]["Debug"], "inactive Release flags affected Debug"
    print("Ninja Multi-Config identity: ok")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        test_inventory(root)
        test_single_config_identity(root)
        test_relocated_debug_sdk(root)
        test_toolchain_identity(root)
        test_multi_config_identity(root)
    print("test_reproducible_sdk: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
