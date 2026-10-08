"""Offline-only standard-Git fixtures. Never addresses a network remote."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import tempfile

ZERO = '0' * 40
ANCHOR = 'refs/tags/archive/branches-2026-10-07-prune-anchor'
ORIGINAL = 'refs/tags/archive/branches-2026-10-07'
SOURCE = 'refs/transfer/archive-target'


def run_git(*args, cwd=None, input=None, check=True, env=None):
    clean = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent',
             'LANG': 'C', 'LC_ALL': 'C', 'GIT_CONFIG_NOSYSTEM': '1',
             'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_TERMINAL_PROMPT': '0',
             'GIT_AUTHOR_NAME': 'Offline Test', 'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
             'GIT_COMMITTER_NAME': 'Offline Test', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid',
             'GIT_AUTHOR_DATE': '2000-01-01T00:00:00+0000',
             'GIT_COMMITTER_DATE': '2000-01-01T00:00:00+0000'}
    if env:
        clean.update(env)
    return subprocess.run(['git', *map(str, args)], cwd=cwd, env=clean, input=input,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


class GitFixture:
    """Disposable bare receiver/sender with independently inspectable reference state."""
    def __init__(self, count=57):
        self.tmp = tempfile.TemporaryDirectory(prefix='git-atomic-offline-')
        self.root = Path(self.tmp.name)
        self.remote = self.root / 'receiver.git'
        self.local = self.root / 'sender.git'
        self.hooks = self.root / 'hooks'
        self.hooks.mkdir()
        self.count = count
        for repo in (self.remote, self.local):
            run_git('init', '--bare', '--initial-branch=main', repo)
        tree = run_git('-C', self.remote, 'mktree', input=b'').stdout.strip().decode()
        self.old = self.commit(tree, 'frozen original head')
        self.archive = self.commit(tree, 'archive target', self.old)
        self.other = self.commit(tree, 'concurrent change', self.old)
        self.candidates = {f'refs/heads/frozen/topic-{n:03}': self.old for n in range(count)}
        self.preserved = {
            'refs/heads/main': self.other,
            'refs/heads/feature/held-one': self.old,
            'refs/heads/feature/held-two': self.other,
            'refs/heads/feature/held-three': self.archive,
            'refs/heads/unrelated-extra': self.other,
            'refs/tags/unrelated-tag': self.old,
            ORIGINAL: self.archive,
        }
        for ref, oid in {**self.candidates, **self.preserved}.items():
            self.set_ref(ref, oid)
        run_git('-C', self.local, '-c', 'protocol.file.allow=always', 'fetch', '--no-tags',
                self.remote, f'{ORIGINAL}:{SOURCE}')
        self.initial = self.refs()

    def commit(self, tree, message, parent=None):
        args = ['-C', self.remote, 'commit-tree', tree, '-m', message]
        if parent:
            args += ['-p', parent]
        return run_git(*args).stdout.strip().decode()

    def set_ref(self, ref, oid):
        if oid is None:
            run_git('-C', self.remote, 'update-ref', '-d', ref)
        else:
            run_git('-C', self.remote, 'update-ref', ref, oid)

    def refs(self):
        raw = run_git('-C', self.remote, 'for-each-ref', '--format=%(refname) %(objectname)').stdout
        return dict(line.decode().split(' ') for line in raw.splitlines())

    def set_receiver_hook(self, name, body):
        path = self.remote / 'hooks' / name
        path.write_text('#!/bin/sh\nset -eu\n' + body + '\n', encoding='ascii')
        path.chmod(0o700)
        return path

    def race_after_advertisement(self, ref, oid):
        # Native receive-pack runs pre-receive after reading the command batch and
        # before its atomic ref transaction. The update below models another writer.
        import shlex
        body = ('unset GIT_QUARANTINE_PATH GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES\n'
                'git --git-dir=' + shlex.quote(str(self.remote)) + ' update-ref '
                + shlex.quote(ref) + ' ' + shlex.quote(oid) + '\ncat >/dev/null')
        return self.set_receiver_hook('pre-receive', body)

    def close(self):
        self.tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
