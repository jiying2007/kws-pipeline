"""Test-only enabled source copies. No execution authority or model acquisition.

Copy the explicit public source closure and adjacent test/source files, change
approval flags and invented prerequisite metadata in temporary controls, and remeasure that copy's freeze.
The checked-in source is never edited. Callers still exercise production gates.
"""
from contextlib import contextmanager
from pathlib import Path
import tempfile

import campaign_contract as c
import prepare_freeze
import prepare_prerequisite


def synthetic_prerequisite(release, *, root=None, freeze=None):
    """Invented success-shaped bytes only for temporary zero-model tests; never actual proof."""
    p = prepare_prerequisite
    value = p.disabled_receipt()
    value.update(evidence_kind="actual_read_only", identity={"repository":p.REPOSITORY,
        "run_id":123456789, "run_attempt":1, "job_key":"prepare_diagnostic", "executed_head_sha":"e"*40},
        numeric_job_id=987654321, result_commit="f"*40, archive_sha256="a"*64,
        prepare_freeze_sha256="b"*64, posterior_policy_sha256="c"*64,
        outcome="complete_prepare_verified", run_conclusion="success", job_conclusion="success",
        step_conclusions={"contracts":"success","host":"success","publisher":"success"},
        host_receipt_status="all3_prepared_no_recognition", verified_prepared_families=list(c.FAMILIES),
        verified_completed_model_constructions=3, diagnostic_forward_calls=0,
        zero_forward_scope="fixed_reviewed_prepare_entrypoints_and_no_audio_mount_only",
        input_authorities={key:release[key] for key in p.INPUT_KEYS}, acceptance=dict(p.ACCEPTANCE),
        scientific_source_sha256=p.scientific_identity(root, freeze) if root is not None else "d"*64)
    raw=c.canonical(value);release["prepare_prerequisite_sha256"]=c.digest(raw)
    return raw

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
        release=c.read_json(copied/'execution-release.json')
        import hosted_run
        release['prior_public_claim_history_sha256']=c.digest(c.canonical(hosted_run.prior_history_identity()))
        raw=synthetic_prerequisite(release,root=target,freeze=prepare_freeze.build(target))
        (copied/'prepare-prerequisite.json').write_bytes(raw)
        (copied/'execution-release.json').write_bytes(c.canonical(release))
        (copied/'execution-freeze.json').write_bytes(c.canonical(prepare_freeze.build(target)))
        try:
            yield copied
        finally:
            c.require(all(c.read_bytes(root/relative,cap=8*c.MIB)==body for relative,body in original.items()),
                'checked-in source bytes must remain unchanged by test fixture')
