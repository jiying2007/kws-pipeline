from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMAKE = os.environ.get("CMAKE_COMMAND", "cmake")


def run(*args: str | pathlib.Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run([str(arg) for arg in args], check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
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
    definitions = re.findall(r'^#define KWS_BUILD_(\w+) ("(?:\\.|[^"\\])*")$',
                             header.read_text() + (build / "generated/kws_source_revision.h").read_text(),
                             flags=re.MULTILINE)
    identity = {key: json.loads(value) for key, value in definitions}
    assert re.fullmatch(r"[0-9a-f]{64}", identity["CONFIG_DIGEST"]), identity
    assert identity["TYPE"] == (config or "unspecified"), identity
    # Check the selected identity in the built archive, not only an unused header.
    library = build / config / "libkws_pipeline.a"
    if not library.exists():
        library = build / "libkws_pipeline.a"
    assert identity["CONFIG_DIGEST"].encode() in library.read_bytes(), library
    assert identity["SOURCE_REVISION"].encode() in library.read_bytes(), library
    return identity


def copy_source(source: pathlib.Path) -> None:
    source.mkdir(parents=True)
    shutil.copyfile(ROOT / "CMakeLists.txt", source / "CMakeLists.txt")
    for directory in ("cmake", "src", "include", "configs", "tools"):
        shutil.copytree(ROOT / directory, source / directory)


def compiler(build: pathlib.Path) -> str:
    commands = json.loads((build / "compile_commands.json").read_text())
    return shlex.split(commands[0]["command"])[0]


def archive(build: pathlib.Path, config: str = "Release") -> pathlib.Path:
    path = build / config / "libkws_pipeline.a"
    return path if path.exists() else build / "libkws_pipeline.a"


def runtime_identity(source: pathlib.Path, build: pathlib.Path,
                     config: str = "Release") -> dict[str, str]:
    probe = build / "identity-probe.c"
    probe.write_text('#include <kws_pipeline/kws.h>\n#include <stdio.h>\n'
                     'int main(void) { const kws_build_info_t *i = kws_build_info(); '
                     'puts(i->source_revision); puts(i->config_digest); puts(i->build_type); return 0; }\n')
    executable = build / f"identity-probe-{config}"
    run(compiler(build), "-I", source / "include", probe, archive(build, config),
        "-lm", "-o", executable)
    revision, digest, build_type = run(executable).splitlines()
    return {"SOURCE_REVISION": revision, "CONFIG_DIGEST": digest, "TYPE": build_type}


def assert_revision(source: pathlib.Path, build: pathlib.Path, expected: str,
                    config: str = "Release") -> dict[str, str]:
    identity = build_identity(build, config)
    observed = runtime_identity(source, build, config)
    for field, value in observed.items():
        assert identity[field] == value, (field, identity, observed)
    assert observed["SOURCE_REVISION"] == expected, (expected, observed)
    return identity


def assert_no_rebuild(build: pathlib.Path, config: str = "Release") -> None:
    paths = [archive(build, config), build / "generated/kws_source_revision.h",
             *build.glob("CMakeFiles/kws_pipeline.dir/**/*.o")]
    before = {str(path): path.stat().st_mtime_ns for path in paths}
    build_identity(build, config)
    after = {str(path): path.stat().st_mtime_ns for path in paths}
    assert before == after, "unchanged identity needlessly compiled or relinked the library"


def init_git(source: pathlib.Path) -> str:
    run("git", "init", "-q", source)
    run("git", "-C", source, "config", "user.name", "Local SDK regression")
    run("git", "-C", source, "config", "user.email", "sdk-test@example.invalid")
    run("git", "-C", source, "config", "commit.gpgsign", "false")
    run("git", "-C", source, "add", ".")
    run("git", "-C", source, "commit", "-qm", "Synthetic source identity fixture")
    return run("git", "-C", source, "rev-parse", "HEAD").strip()


def change_runtime(source: pathlib.Path) -> None:
    path = source / "src/frontend.c"
    original = path.read_text()
    changed = original.replace("1.4142135623730950488f", "1.5142135623730950488f")
    assert changed != original, "fixture constant changed; choose another observable source edit"
    path.write_text(changed)


def test_incremental_source_identity(root: pathlib.Path) -> None:
    source, build = root / "git-source", root / "git-build"
    copy_source(source)
    initial = init_git(source)
    configure(build, "-DCMAKE_BUILD_TYPE=Release", "-DKWS_SOURCE_REVISION=", source=source)
    first = assert_revision(source, build, initial)
    first_archive = sha256(archive(build))
    assert_no_rebuild(build)
    change_runtime(source)
    dirty = assert_revision(source, build, initial + "-dirty")
    assert dirty["CONFIG_DIGEST"] == first["CONFIG_DIGEST"], "Git state changed configuration identity"
    assert sha256(archive(build)) != first_archive
    assert_no_rebuild(build)
    run("git", "-C", source, "checkout", "--", "src/frontend.c")
    assert_revision(source, build, initial)
    assert sha256(archive(build)) == first_archive, "dirty-to-clean rebuild did not restore the SDK"
    change_runtime(source)
    run("git", "-C", source, "add", "src/frontend.c")
    assert_revision(source, build, initial + "-dirty")
    run("git", "-C", source, "commit", "-qm", "Synthetic runtime edit")
    second = run("git", "-C", source, "rev-parse", "HEAD").strip()
    assert_revision(source, build, second)
    # No source timestamp changes: only HEAD changed. This must still refresh.
    run("git", "-C", source, "commit", "--allow-empty", "-qm", "Synthetic identity-only edit")
    third = run("git", "-C", source, "rev-parse", "HEAD").strip()
    assert_revision(source, build, third)
    run("git", "-C", source, "pack-refs", "--all")
    assert_no_rebuild(build)
    assert run("git", "-C", source, "status", "--porcelain", "--untracked-files=all") == "", \
        "build identity generation polluted the source tree"

    worktree, work_build = root / "linked-worktree", root / "linked-build"
    run("git", "-C", source, "worktree", "add", "--detach", worktree, third)
    assert (worktree / ".git").is_file()
    configure(work_build, "-DCMAKE_BUILD_TYPE=Release", "-DKWS_SOURCE_REVISION=", source=worktree)
    assert_revision(worktree, work_build, third)
    run("git", "-C", worktree, "checkout", "--detach", initial)
    assert_revision(worktree, work_build, initial)
    change_runtime(worktree)
    assert_revision(worktree, work_build, initial + "-dirty")
    run("git", "-C", worktree, "checkout", "--", "src/frontend.c")
    assert_revision(worktree, work_build, initial)
    assert_no_rebuild(work_build)
    run(CMAKE, "--build", work_build, "--parallel", "2",
        env={**os.environ, "GIT_DIR": str(source / ".git"), "GIT_WORK_TREE": str(source)})
    assert runtime_identity(worktree, work_build)["SOURCE_REVISION"] == initial, \
        "inherited Git plumbing redirected worktree source identity"

    override_build = root / "override-build"
    configure(override_build, "-DCMAKE_BUILD_TYPE=Release", source=source)
    assert_revision(source, override_build, "identity-regression")
    run("git", "-C", source, "commit", "--allow-empty", "-qm", "Override stays authoritative")
    assert_revision(source, override_build, "identity-regression")
    assert_no_rebuild(override_build)
    literal_override = 'release;$<CONFIG>;quote";slash\\'
    configure(override_build, "-DCMAKE_BUILD_TYPE=Release",
              f"-DKWS_SOURCE_REVISION={literal_override}", source=source)
    assert_revision(source, override_build, literal_override)
    assert_no_rebuild(override_build)
    # Submodules use a gitfile too, but have their own independent HEAD.
    superproject, sub_build = root / "superproject", root / "submodule-build"
    superproject.mkdir()
    (superproject / "README").write_text("Synthetic submodule parent\n")
    init_git(superproject)
    run("git", "-c", "protocol.file.allow=always", "-C", superproject,
        "submodule", "add", "-q", source, "sdk")
    submodule = superproject / "sdk"
    assert (submodule / ".git").is_file()
    configure(sub_build, "-DCMAKE_BUILD_TYPE=Release", "-DKWS_SOURCE_REVISION=", source=submodule)
    assert_revision(submodule, sub_build, run("git", "-C", source, "rev-parse", "HEAD").strip())
    run("git", "-C", submodule, "checkout", "--detach", initial)
    assert_revision(submodule, sub_build, initial)
    assert_no_rebuild(sub_build)
    print("Incremental source identity: clean/dirty/staged/clean/new HEAD/packed refs/worktree/submodule/detached/override ok")


def test_exported_source_identity(root: pathlib.Path) -> None:
    parent = root / "unrelated-enclosing-repo"
    parent.mkdir()
    (parent / "unrelated.txt").write_text("Not the SDK source\n")
    init_git(parent)
    source, build = parent / "exported-sdk", root / "exported-build"
    copy_source(source)
    configure(build, "-DCMAKE_BUILD_TYPE=Release", "-DKWS_SOURCE_REVISION=", source=source)
    assert_revision(source, build, "unknown")
    change_runtime(source)
    assert_revision(source, build, "unknown")
    assert_no_rebuild(build)
    # A tracked vendor copy still has no independent project Git identity.
    run("git", "-C", parent, "add", "exported-sdk")
    run("git", "-C", parent, "commit", "-qm", "Track the exported vendor copy")
    assert_revision(source, build, "unknown")
    configure(build, "-DCMAKE_BUILD_TYPE=Release", "-DKWS_SOURCE_REVISION=export-release-label", source=source)
    assert_revision(source, build, "export-release-label")
    print("Exported source identity: unknown without local Git; explicit override ok")


def test_pkgconfig_consumers(root: pathlib.Path) -> None:
    if not shutil.which("pkg-config"):
        raise AssertionError("pkg-config is required for real installed SDK consumer regressions")
    # Explicit multilevel layout is portable; /usr also exercises the platform's
    # actual GNUInstallDirs default (e.g. lib/x86_64-linux-gnu on Debian).
    cases = (("lib", "lib", "include", "/opt/kws-original"),
             ("lib64", "lib64", "include", "/opt/kws-original"),
             ("multiarch", "lib/test-multiarch", "include/kws-sdk", "/opt/kws-original"),
             ("usr-default", None, None, "/usr"))
    for name, libdir, includedir, prefix in cases:
        build = root / f"pc-build-{name}"
        options = ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}"]
        if libdir is not None:
            options.extend((f"-DCMAKE_INSTALL_LIBDIR={libdir}", f"-DCMAKE_INSTALL_INCLUDEDIR={includedir}"))
        configure(build, *options)
        build_identity(build)
        staged = root / f"pc-stage-{name}"
        if name == "usr-default":
            run(CMAKE, "--install", build, env={**os.environ, "DESTDIR": str(staged)})
            installed = staged / "usr"
        else:
            installed = staged
            run(CMAKE, "--install", build, "--prefix", installed)
        moved = root / f"pc-relocated-{name}"
        for index, sdk in enumerate((installed, moved)):
            if index:
                shutil.move(installed, moved)
                assert not installed.exists(), "original install remains and may mask relocation failures"
            pc_files = list(sdk.rglob("kws-pipeline.pc"))
            assert len(pc_files) == 1, pc_files
            pc = pc_files[0]
            env = {**os.environ, "PKG_CONFIG_PATH": "", "PKG_CONFIG_LIBDIR": str(pc.parent),
                   "PKG_CONFIG_SYSROOT_DIR": ""}
            assert run("pkg-config", "--modversion", "kws-pipeline", env=env).strip() == "0.4.0"
            observed_prefix = pathlib.Path(run("pkg-config", "--variable=prefix", "kws-pipeline", env=env).strip())
            assert observed_prefix.resolve() == sdk.resolve(), (pc.read_text(), observed_prefix, sdk)
            flags = shlex.split(run("pkg-config", "--cflags", "--libs", "kws-pipeline", env=env))
            executable = root / f"pc-consumer-{name}-{index}"
            run(compiler(build), ROOT / "tests/consumer/main.c", *flags, "-o", executable)
            assert "KwsPipeline consumer:" in run(executable)
            # CMake exports must retain relocation too, even for custom layouts.
            consumer = root / f"cmake-consumer-{name}-{index}"
            config = next(sdk.rglob("KwsPipelineConfig.cmake"))
            run(CMAKE, "-S", ROOT / "tests/consumer", "-B", consumer,
                f"-DKwsPipeline_DIR={config.parent}")
            run(CMAKE, "--build", consumer, "--parallel", "2")
            assert "KwsPipeline consumer:" in run(consumer / "kws_pipeline_consumer")
        print(f"pkg-config/CMake consumer {name} ({pc.parent.relative_to(moved)}): compile/link/run before and after relocation ok")

    for index, option in enumerate(("-DCMAKE_INSTALL_LIBDIR=/tmp/absolute-sdk-lib",
                                    "-DCMAKE_INSTALL_INCLUDEDIR=/tmp/absolute-sdk-include",
                                    "-DCMAKE_INSTALL_LIBDIR=../outside-lib",
                                    "-DCMAKE_INSTALL_INCLUDEDIR=include/../../outside-include")):
        try:
            configure(root / f"invalid-layout-{index}", option)
        except AssertionError as exc:
            assert "prefix-relative directory" in str(exc), exc
        else:
            raise AssertionError(f"non-relocatable layout was silently accepted: {option}")
    print("SDK absolute/parent-traversing install directories: rejected at configure time")

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
        copy_source(source)
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
    source, build = root / "multi-git-source", root / "multi-git-build"
    copy_source(source)
    revision = init_git(source)
    configure(build, "-DKWS_SOURCE_REVISION=", "-DCMAKE_CONFIGURATION_TYPES=Debug;Release",
              generator="Ninja Multi-Config", source=source)
    clean = {config: assert_revision(source, build, revision, config) for config in ("Debug", "Release")}
    change_runtime(source)
    for config in ("Debug", "Release"):
        dirty = assert_revision(source, build, revision + "-dirty", config)
        assert dirty["CONFIG_DIGEST"] == clean[config]["CONFIG_DIGEST"]
        assert_no_rebuild(build, config)
    run("git", "-C", source, "checkout", "--", "src/frontend.c")
    for config in ("Release", "Debug"):
        assert_revision(source, build, revision, config)
        assert_no_rebuild(build, config)
    print("Ninja Multi-Config identity: configuration fingerprints and incremental source refresh ok")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        test_inventory(root)
        test_single_config_identity(root)
        test_relocated_debug_sdk(root)
        test_toolchain_identity(root)
        test_multi_config_identity(root)
        test_incremental_source_identity(root)
        test_exported_source_identity(root)
        test_pkgconfig_consumers(root)
    print("test_reproducible_sdk: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
