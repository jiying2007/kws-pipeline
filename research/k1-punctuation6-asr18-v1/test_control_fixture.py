"""Test-only enabled source copies. No execution authority or model acquisition.

Copy the explicit public source closure and adjacent test/source files, change
only approval flags in the temporary controls, and remeasure that copy's freeze.
The checked-in source is never edited. Callers still exercise production gates.
"""
from contextlib import contextmanager
from pathlib import Path
import tempfile

import campaign_contract as c
import prepare_freeze

APPROVALS = {
    'execution-release.json': tuple(c.RELEASE_APPROVALS),
    'publication-release.json': ('enabled','independent_source_review','root_publication_release'),
}

DISABLED_APPROVALS = {name:tuple(key for key in keys if key!='historical_screening_reviewed')
    for name,keys in APPROVALS.items()}


@contextmanager
def synthetic_enabled_source(source):
    source=Path(source).resolve(); root=source.parents[1]
    rosters=prepare_freeze.declared_rosters(root)
    names=set().union(*rosters.values())
    names.add('.github/workflows/k1-punctuation6-asr18-v1.yml')
    # Only the source package's direct public files, never a recursive data walk.
    names.update(str(path.relative_to(root)) for path in source.iterdir()
        if path.is_file() and path.suffix in ('.py','.json','.md','.yml','.yaml','.txt'))
    original={name:c.read_bytes(root/name,cap=8*c.MIB) for name in sorted(names)}
    with tempfile.TemporaryDirectory(prefix='k1-synthetic-enabled-source-') as name:
        target=Path(name)/'checkout'
        for relative,body in original.items():
            path=target/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
        copied=target/prepare_freeze.NEW
        for filename,keys in APPROVALS.items():
            before=c.decode_json((copied/filename).read_bytes())
            value=dict(before)
            for key in keys:value[key]=True
            c.require(all(value[key]==item for key,item in before.items() if key not in keys),
                'synthetic copy changes only approval flags')
            (copied/filename).write_bytes(c.canonical(value))
        (copied/'execution-freeze.json').write_bytes(c.canonical(prepare_freeze.build(target)))
        try:
            yield copied
        finally:
            c.require(all(c.read_bytes(root/relative,cap=8*c.MIB)==body for relative,body in original.items()),
                'checked-in source bytes must remain unchanged by test fixture')
