#!/usr/bin/env python3
"""Inventory executable test declarations; never claim they prove execution.

Run blocks are inspected, rather than comments/path filters/arbitrary mentions.
Actual case IDs, Python modes, discovery counts and skip outcomes are validated
separately by checked_unittest.py for its explicitly contracted suite subset.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import fnmatch
import json
import pathlib
import re
import shlex
import sys
import textwrap

OFFICIAL_WORKFLOWS = (
    '.github/workflows/ci.yml',
    '.github/workflows/training-integration.yml',
    '.github/workflows/training-audit-closure.yml',
    '.github/workflows/speech-like-external-base-bundle-contract.yml',
    '.github/workflows/product-training-data-contract.yml',
    '.github/workflows/release.yml',
)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_blocks(text):
    """Read YAML run scalars without treating unrelated YAML text as commands."""
    lines = text.splitlines()
    index = 0
    job = ''
    job_gated = step_gated = False
    while index < len(lines):
        line = lines[index]
        match = re.fullmatch(r'  ([a-zA-Z0-9_-]+):', line)
        if match:
            job = match[1]
            job_gated = step_gated = False
        if re.match(r'^      - ', line):
            step_gated = False
        if re.match(r'^    (?:if|continue-on-error):', line):
            job_gated = True
        if re.match(r'^        (?:if|continue-on-error):', line):
            step_gated = True
        match = re.match(r'^(\s*)(?:- )?run:\s*(.*)$', line)
        index += 1
        if not match:
            continue
        indent, value = len(match[1]), match[2]
        if value in ('|', '|-', '>', '>-'):
            body = []
            while index < len(lines) and (not lines[index].strip() or
                    len(lines[index]) - len(lines[index].lstrip()) > indent):
                body.append(lines[index])
                index += 1
            block = textwrap.dedent('\n'.join(body))
            if value.startswith('>'):
                block = ' '.join(block.splitlines())
        else:
            block = value
        yield job, block, job_gated or step_gated


def python_commands(block):
    # Join only explicit shell continuations. Refuse comments, echo, heredoc
    # content, and arbitrary shell syntax rather than counting text mentions.
    block = block.replace('\\\n', ' ')
    heredoc = None
    for line in block.splitlines():
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            continue
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if not words:
            continue
        marker = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)", line)
        if marker:
            heredoc = marker[1]
        if re.fullmatch(r'(?:[^\s]+/)?python(?:3(?:\.\d+)?)?', words[0]):
            yield words


def declared_paths(block):
    paths = []
    specs = json.loads((ROOT / 'tools/test_execution_contracts.json').read_text())['suites']
    for words in python_commands(block):
        if '-c' in words:
            # Only recognize the explicit loadTestsFromNames program actually
            # invoked via its shell variable in this same run block.
            value = words[words.index('-c') + 1]
            if not re.fullmatch(r'\$[A-Za-z_][A-Za-z0-9_]*', value):
                continue
            assignment = re.search(r'^' + re.escape(value[1:]) + r'=(.+)$', block, re.M)
            if not assignment:
                continue
            try:
                tree = ast.parse(ast.literal_eval(assignment[1]))
            except (ValueError, SyntaxError):
                continue
            roots = [node.args[0].value for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'Path' and node.args
                     and isinstance(node.args[0], ast.Constant)
                     and isinstance(node.args[0].value, str) and node.args[0].value != 'tools']
            loaded = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and node.func.attr == 'loadTestsFromNames' for node in ast.walk(tree))
            names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                     and isinstance(node.value, str) and re.fullmatch(r'test_[a-z0-9_]+(?:\.py)?', node.value)}
            if len(roots) == 1 and loaded:
                paths.extend((str(pathlib.PurePosixPath(roots[0]) / (name if name.endswith('.py') else name + '.py')), 'unittest-modules') for name in names)
            continue
        if 'tools/checked_unittest.py' in words and '--suite' in words and '--verify-reports' not in words:
            name = words[words.index('--suite') + 1]
            if name in specs:
                spec = specs[name]
                paths.extend((str(pathlib.PurePosixPath(spec['root']) / (module + '.py')), 'checked-unittest') for module in spec['modules'])
            continue
        if 'unittest' in words and 'discover' in words and '-s' in words:
            root = words[words.index('-s') + 1]
            pattern = words[words.index('-p') + 1] if '-p' in words else 'test*.py'
            paths.append((str(pathlib.PurePosixPath(root) / pattern), 'unittest-discovery'))
            continue
        # A test filename passed to another program is data, not an invocation.
        scripts = [(index, word) for index, word in enumerate(words[1:], 1)
                   if not word.startswith('-')]
        if scripts:
            index, path = scripts[0]
            if re.fullmatch(r'(?:[^\s]+/)?test_[A-Za-z0-9_]+\.py', path):
                paths.append((path, 'script-selected' if len(words) > index + 1 else 'script'))
    return paths


def scope(workflow, job, text, gated=False):
    if workflow.name == 'ci.yml':
        return 'conditional-mainline' if gated else 'mainline-required'
    if workflow.name == 'release.yml':
        return 'release-only'
    if 'self-hosted' in text:
        return 'hardware-or-self-hosted'
    if re.search(r'^\s+paths(?:-ignore)?:', text, re.M):
        return 'path-filtered'
    return 'manual-or-scheduled'


def inventory(root, workflows, excluded=()):
    tests = sorted(path.name for path in root.glob('test_*.py'))
    if not tests:
        raise ValueError('no Python tests discovered')
    excluded = set(excluded)
    if excluded - set(tests):
        raise ValueError('inventory exclusions do not exist: ' + ', '.join(sorted(excluded - set(tests))))
    entries = {name: [] for name in tests}
    for workflow in workflows:
        source = workflow.read_text(encoding='utf-8')
        for job, block, gated in run_blocks(source):
            for declared, kind in declared_paths(block):
                for name in tests:
                    # Compare the complete test-root suffix, not only basename.
                    target = root / name
                    if root.is_absolute():
                        matches = fnmatch.fnmatch(str(target), declared) or fnmatch.fnmatch(str(pathlib.PurePosixPath(root.name) / name), declared)
                    else:
                        matches = fnmatch.fnmatch(target.as_posix(), declared)
                    if matches:
                        entries[name].append(dict(workflow=str(workflow), job=job,
                                                  scope=scope(workflow, job, source, gated), kind=kind))
    missing = [name for name, routes in entries.items() if not routes and name not in excluded]
    return dict(schema=1, evidence='executable-declarations-only', root=str(root),
                execution_coverage_proven=False, files=entries,
                externally_executed=sorted(excluded), missing=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, default=pathlib.Path('tests'))
    workflows = parser.add_mutually_exclusive_group(required=True)
    workflows.add_argument('--workflow', action='append', type=pathlib.Path)
    workflows.add_argument('--official-workflows', action='store_true')
    parser.add_argument('--exclude', action='append', default=[])
    parser.add_argument('--output', type=pathlib.Path, help='write declaration scopes; not execution evidence')
    args = parser.parse_args()
    paths = [pathlib.Path(path) for path in OFFICIAL_WORKFLOWS] if args.official_workflows else args.workflow
    report = inventory(args.root, paths, args.exclude)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    if report['missing']:
        print('Python tests missing executable declarations: ' + ', '.join(report['missing']), file=sys.stderr)
        return 1
    counts = Counter('mainline-required' if any(r['scope'] == 'mainline-required' for r in routes)
                     else '|'.join(sorted({r['scope'] for r in routes})) or 'external'
                     for routes in report['files'].values())
    print(f"declared {len(report['files'])} Python test file(s): " + json.dumps(counts, sort_keys=True))
    for name, routes in report['files'].items():
        if not any(route['scope'] == 'mainline-required' for route in routes):
            print('  non-mainline: ' + name + ' [' + ', '.join(sorted({r['scope'] for r in routes})) + ']')
    print('Declarations are not proof of execution; checked_unittest reports cover only their contracted suites.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
