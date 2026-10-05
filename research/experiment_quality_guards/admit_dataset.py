"""Builder gate: exit 0=stated qualification allowed, 1=rejected, 2=invalid input.

Only coverage and declared identity evidence are checked. No training, model,
fresh-validation, rights or product authorization is granted.
"""
import argparse
import json
from pathlib import Path
import sys
from quality_gates import coverage_admission, identity_audit, require


def assess_admission(payload):
    """Return (JSON report, exit code); explicit regression mode is report-only."""
    scope = {"training_authorized": False, "product_qualified": False,
             "fresh_validation_qualified": False}
    try:
        require(type(payload) is dict, "input must be an object")
        json.dumps(payload, allow_nan=False)
        qualification = payload["requested_qualification"]
        require(qualification in ("balanced_source_groups", "exposed_regression"),
                "unknown requested qualification")
        require(type(payload.get("require_unseen_voice", False)) is bool,
                "require_unseen_voice must be boolean")
        rows, history = payload["rows"], payload["history"]
        for name, values in (("rows", rows), ("history", history),
                             ("declarations", payload["declarations"])):
            require(type(values) is list and all(type(x) is dict for x in values),
                    name + " must be a list of objects")
            require(all(x.get("split") in ("train", "dev", "heldout", "background", "regression")
                        for x in values), name + " requires canonical train/dev/heldout/background/regression split")
        require(bool(rows), "rows must not be empty")
        coverage = coverage_admission(rows, payload["declarations"], payload["policy"],
                                      payload["frozen_policy_sha256"])
        identity = identity_audit(rows, history)
        reasons = []
        if identity["status"] == "FAIL":
            reasons.append("IDENTITY_OR_EXPOSURE_CONFLICT")
        if payload.get("require_unseen_voice") and any(
                not row["unseen_voice"] or row["lineage_status"] != "verified"
                for row in identity["rows"]):
            reasons.append("UNSEEN_VOICE_NOT_ESTABLISHED")
        if qualification == "balanced_source_groups":
            if not coverage["balanced_admission"]:
                reasons.append("BALANCED_COVERAGE_INSUFFICIENT")
        elif any(row.get("exposure") != "EXPOSED" or row.get("use") != "regression"
                 for row in rows):
            reasons.append("EXPLICIT_EXPOSED_REGRESSION_USE_REQUIRED")
        allowed = not reasons
        return {"status": "QUALIFICATION_ALLOWED" if allowed else "QUALIFICATION_REJECTED",
                "requested_qualification": qualification, "eligible": allowed,
                "coverage_eligible": qualification == "balanced_source_groups" and coverage["balanced_admission"],
                "qualification_scope": "coverage and declared identity only" if qualification == "balanced_source_groups"
                else "exposed historical report/code regression only; no balanced-data claim",
                "reasons": reasons, "coverage": coverage, "identity": identity, **scope}, 0 if allowed else 1
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        return {"status": "INVALID_INPUT", "eligible": False,
                "error": str(error), **scope}, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON input file, or - for stdin")
    parser.add_argument("--output", type=Path, help="Otherwise report JSON on stdout")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
        report, code = assess_admission(json.loads(raw))
    except (OSError, ValueError) as error:
        report, code = {"status": "INVALID_INPUT", "eligible": False, "error": str(error)}, 2
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        if args.output:
            args.output.write_text(encoded)
        else:
            print(encoded, end="")
    except OSError as error:
        print(json.dumps({"status": "REPORT_WRITE_FAILED", "error": str(error)}), file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
