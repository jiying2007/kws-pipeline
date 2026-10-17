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


# Test-only preservation of exact source bytes expected by historical contracts.
# Live admission/staging continues to check the maintained paths and reject drift.
HISTORICAL_RETAINED = frozenset('research/qwen6_asr/' + name for name in (
    'core/asr_stage/adapters.py', 'core/asr_stage/execution.py', 'core/sense_binding.py'))
HISTORICAL_ARCHIVE = 'research/consolidation/archive/source-revisions/sense-runtime-binding-20261017/'


def historical_source_bytes(root, names):
    """Verified test source projection; never a production source resolver."""
    root = Path(root)
    pins = {'research/qwen6_asr/' + name: sha for name, sha in
            prepare_freeze.retained_pins(root / prepare_freeze.OLD / 'asr_worker.py').items()}
    result = {}
    for name in sorted(names):
        raw = c.read_bytes(root / name, cap=8*c.MIB)
        # Nested temporary fixtures may already hold the exact historical bytes.
        if name in HISTORICAL_RETAINED and c.digest(raw) != pins[name]:
            raw = c.read_bytes(root / HISTORICAL_ARCHIVE / name, cap=8*c.MIB)
        if name in pins:
            c.require(c.digest(raw) == pins[name], 'historical fixture retained source pin: ' + name)
        result[name] = raw
    return result


@contextmanager
def historical_source_tree(root, *, extra=()):
    """Materialize the explicit public closure for zero-model fixture validation."""
    names = set(prepare_freeze.declared_rosters(root)['host_code']) | set(extra)
    original = historical_source_bytes(root, names)
    with tempfile.TemporaryDirectory(prefix='k1-historical-source-') as directory:
        target = Path(directory) / 'checkout'
        for name, raw in original.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        prepare_freeze.build(target)  # The real unchanged live gate verifies the copy.
        yield target


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
    'claim-admission-control.json': tuple(c.CLAIM_APPROVALS),
}

DISABLED_APPROVALS = {name:tuple(key for key in keys if key!='historical_screening_reviewed')
    for name,keys in APPROVALS.items()}


@contextmanager
def synthetic_enabled_source(source):
    source=Path(source).resolve(); root=source.parents[1]
    rosters=prepare_freeze.declared_rosters(root)
    names=set().union(*rosters.values())
    names.add(c.current_scope()['workflow'])
    # Only the source package's direct public files, never a recursive data walk.
    names.update(str(path.relative_to(root)) for path in source.iterdir()
        if path.is_file() and path.suffix in ('.py','.json','.md','.yml','.yaml','.txt'))
    original={name:c.read_bytes(root/name,cap=8*c.MIB) for name in sorted(names)}
    projected=historical_source_bytes(root,names)
    with tempfile.TemporaryDirectory(prefix='k1-synthetic-enabled-source-') as name:
        target=Path(name)/'checkout'
        for relative,body in projected.items():
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
        control=c.read_json(copied/c.CLAIM_CONTROL_FILE)
        control.update(campaign=release['experiment'],source_ref=release['source_ref'],
            evidence_ref=release['evidence_ref'],workflow=release['workflow'],
            claim_ref='refs/heads/claims/'+release['experiment'])
        control_raw=c.canonical(control)
        (copied/c.CLAIM_CONTROL_FILE).write_bytes(control_raw)
        release['claim_admission_control_sha256']=c.digest(control_raw)
        import hosted_run
        release['prior_public_claim_history_sha256']=c.digest(c.canonical(hosted_run.bound_prior_history_identity(control)))
        raw=synthetic_prerequisite(release,root=target,freeze=prepare_freeze.build(target))
        (copied/'prepare-prerequisite.json').write_bytes(raw)
        (copied/'execution-release.json').write_bytes(c.canonical(release))
        (copied/'execution-freeze.json').write_bytes(c.canonical(prepare_freeze.build(target)))
        try:
            yield copied
        finally:
            c.require(all(c.read_bytes(root/relative,cap=8*c.MIB)==body for relative,body in original.items()),
                'checked-in source bytes must remain unchanged by test fixture')
