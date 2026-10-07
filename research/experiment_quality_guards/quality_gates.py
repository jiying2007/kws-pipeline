"""Pure research admission/reporting checks; no inference, process launch or network.

Rows are JSON objects, not generation requests. Missing evidence fails closed.
These checks validate records and claim boundaries, never acoustic performance.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import unicodedata

KEYWORDS = {"K1": "你好小窝", "K2": "小窝小窝"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_actual(text):
    """Same comparison-only whitespace/punctuation policy as actual_label_policy."""
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def human_truth(row):
    """Actual words only. ASR/intent/greedy fields cannot promote a weak label."""
    review = row.get("review", {})
    if (review.get("status") != "clean"
            or review.get("independent_human") is not True
            or review.get("complete") is not True
            or not isinstance(row.get("actual_text"), str)
            or not normalized_actual(row["actual_text"])):
        return None
    return {key for key, text in KEYWORDS.items() if text in normalized_actual(row["actual_text"])}


def label_preparation(row):
    """Check original-label evidence without replacing human actual words.

    Machine states follow the saved dual-ASR review: complete, nonempty
    agreement is weak evidence. A missing plan is never filled from actual text.
    Eligibility here is for the label check only, never dataset/training admission.
    """
    review = row.get("review", {})
    require(type(review) is dict, "review must be an object")
    actual, intended = row.get("actual_text"), row.get("intended_text")
    for value in (actual, intended):
        require(value is None or type(value) is str, "label text must be string or null")
    observations = row.get("asr_results", [None, None])
    require(type(observations) is list and len(observations) == 2,
            "asr_results must contain exactly two saved observations")
    texts, complete, flags = [], [], []
    for observation in observations:
        require(observation is None or type(observation) is dict,
                "ASR observation must be an object or null")
        observation = observation or {}
        raw = observation.get("raw_text")
        require(raw is None or type(raw) is str, "ASR raw_text must be string or null")
        text = None if raw is None else normalized_actual(raw)
        quality_flags = observation.get("quality_flags", [])
        require(type(quality_flags) is list and all(type(f) is str for f in quality_flags),
                "ASR quality_flags must be a list of strings")
        flags.extend(quality_flags)
        texts.append(text)
        complete.append(observation.get("status") == "complete" and bool(text))
    machine = ("unresolved" if not all(complete) else
               "disagree" if texts[0] != texts[1] else "machinesAgreeWeak")
    plan = normalized_actual(intended or "")
    actual_normalized = normalized_actual(actual or "")
    support = ("UNKNOWN" if not plan or machine == "unresolved" else
               "SUPPORTED" if all(t == plan for t in texts) else "REJECTED")
    human_match = (actual_normalized == plan if plan and actual_normalized
                   and review.get("independent_human") is True else None)
    truth = human_truth(row)
    reasons = []
    if not plan:
        reasons.append("INTENDED_TEXT_MISSING")
    if support != "SUPPORTED":
        reasons.append("DUAL_ASR_PLAN_SUPPORT_" + support)
    if human_match is False:
        reasons.append("HUMAN_ACTUAL_DIFFERS_FROM_PLAN")
    if truth is None:
        reasons.append("CLEAN_COMPLETE_HUMAN_ACTUAL_REQUIRED")
    if flags:
        reasons.append("ASR_QUALITY_FLAGS_RETAINED")
    eligible = not reasons
    return {"actual_text": actual, "intended_text": intended,
            "label_source": "human_actual_only", "machine_state": machine,
            "machine_normalized_texts": texts, "planned_lexical_support": support,
            "plan_matches_human_actual": human_match,
            "original_complete_label_eligible": eligible, "reasons": reasons,
            "actual_expected_keywords": None if truth is None else sorted(truth),
            "human_actual_review_complete": truth is not None,
            "acoustic_completeness": "UNKNOWN",
            "automatic_relabel": False, "ctc_target": None,
            "silence_or_blank_target_inferred": False, "training_admitted": False,
            "independent_accuracy_qualified": False}


def _unique_rows(rows):
    ids = [row["id"] for row in rows]
    require(len(set(ids)) == len(ids), "duplicate row ID")


def coverage_admission(rows, declarations, policy, frozen_policy_sha256):
    """Frozen absolute counts per declared source group/split, no percentages.

    Nonwake case names are policy-defined and must be independently assigned.
    Exact duplicate audio contributes at most once per group/category.
    """
    _unique_rows(rows)
    require(hashlib.sha256(json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() == frozen_policy_sha256,
            "frozen coverage policy digest mismatch")
    require(policy.get("frozen") is True and policy.get("policy_id"), "coverage policy not frozen")
    minima = policy["minima"]
    require(set(KEYWORDS) <= set(minima), "both wake minima required")
    require(all(type(minima[k]) is int and minima[k] > 0 for k in KEYWORDS), "positive wake minima required")
    nonwake = policy.get("nonwake_minima", {})
    require(nonwake and all(type(v) is int and v > 0 for v in nonwake.values()), "nonwake minima required")
    targets = dict(minima) | {"nonwake:" + k: v for k, v in nonwake.items()}
    require(set(minima) == set(KEYWORDS), "unknown wake minimum")
    declared_keys = [(d["source_group"], d["split"]) for d in declarations]
    require(len(set(declared_keys)) == len(declared_keys), "duplicate group declaration")
    require(all((r["source_group"], r["split"]) in declared_keys for r in rows), "undeclared group")
    wav_to_pcm = {}
    for row in rows:
        wav, pcm = row.get("wav_sha256"), row.get("pcm_sha256")
        if wav and pcm:
            require(wav not in wav_to_pcm or wav_to_pcm[wav] == pcm, "same WAV has conflicting PCM hashes")
            wav_to_pcm[wav] = pcm
    reports = []
    for declaration in declarations:
        require(declaration["role"] in ("balanced", "negative_only"), "unknown cohort role")
        key = declaration["source_group"], declaration["split"]
        members = [r for r in rows if (r["source_group"], r["split"]) == key]
        seen = defaultdict(set)
        excluded = []
        positive_rows = []
        for row in members:
            truth = human_truth(row)
            digest = row.get("pcm_sha256") or wav_to_pcm.get(row.get("wav_sha256")) or row.get("wav_sha256")
            if truth is None or not digest:
                excluded.append(row["id"])
                continue
            if truth:
                positive_rows.append(row["id"])
                for keyword in truth:
                    seen[keyword].add(digest)
            else:
                actual = normalized_actual(row["actual_text"])
                for category in nonwake:
                    actual_match = actual == category
                    if category == "out_of_vocabulary":
                        require(bool(policy.get("vocabulary")), "OOV coverage needs frozen vocabulary")
                        actual_match = bool(set(actual) - set(policy["vocabulary"]))
                    if actual_match:
                        seen["nonwake:" + category].add(digest)
        counts = {category: len(seen[category]) for category in targets}
        missing = {k: n - counts[k] for k, n in targets.items() if counts[k] < n}
        if declaration["role"] == "negative_only":
            status = "INVALID_NEGATIVE_ONLY" if positive_rows else "SEPARATE_NEGATIVE_ONLY"
        else:
            status = "ADMITTED_BALANCED" if not missing else "REJECTED_COVERAGE"
        reports.append(dict(declaration, status=status, counts=counts, missing=missing,
                            excluded_unclean_or_unbound=excluded, positive_rows=positive_rows))
    return {"policy_id": policy["policy_id"], "groups": reports,
            "balanced_admission": bool(reports) and all(r["status"] == "ADMITTED_BALANCED"
              for r in reports if r["role"] == "balanced") and any(r["role"] == "balanced" for r in reports)
              and not any(r["status"] == "INVALID_NEGATIVE_ONLY" for r in reports),
            "background_is_positive_coverage": False, "training_admitted": False,
            "scope": "coverage only; exposure, lineage, rights and authorized use are separate gates"}


def identity_audit(rows, history=()):
    """Union content/reference/voice/explicit lineage across all generators.

    Persist returned ledger with previous records; never drop history. Declared
    lineage is evidence supplied by a reviewer, not an identity inferred here.
    """
    _unique_rows(rows)
    all_rows = list(history) + list(rows)
    parent = list(range(len(all_rows)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def tokens(row):
        values = {"row:" + row["id"]}
        for field in ("wav_sha256", "pcm_sha256", "reference_sha256"):
            if row.get(field):
                values.add("audio:" + row[field])
        if row.get("voice_identity"):
            values.add("voice:" + row["voice_identity"])
        values.update("lineage:" + x for x in row.get("lineage_ids", []))
        return values

    seen = {}
    for i, row in enumerate(all_rows):
        require(row.get("exposure") in ("FRESH", "EXPOSED", "UNKNOWN"), "explicit exposure required")
        for token in tokens(row):
            if token in seen:
                parent[root(i)] = root(seen[token])
            seen[token] = i
    # Use restrictions follow the recording and derivatives, not every other
    # recording made with the same stock voice or a shared parent reference.
    restricted = set()
    for row in all_rows:
        if row.get("restriction") == "regression_only":
            restricted.update(tokens(row) - {"voice:" + row.get("voice_identity", ""), "audio:" + row.get("reference_sha256", "")})
    restricted_indices = set()
    for _ in all_rows:
        before = len(restricted_indices)
        for i, row in enumerate(all_rows):
            nonvoice = {t for t in tokens(row) if not t.startswith("voice:")}
            if nonvoice & restricted:
                restricted_indices.add(i)
                restricted.update(nonvoice - {"audio:" + row.get("reference_sha256", "")})
        if len(restricted_indices) == before:
            break
    groups = defaultdict(list)
    for i, row in enumerate(all_rows):
        groups[root(i)].append((i, row))
    audit = []
    conflicts = []
    ledger = [dict(row) for row in history]
    history_audio = {r.get(f) for r in history for f in ("wav_sha256", "pcm_sha256") if r.get(f)}
    current_content = defaultdict(set)
    for row in rows:
        for field in ("wav_sha256", "pcm_sha256"):
            if row.get(field):
                current_content[row[field]].add(row["id"])
    for members in groups.values():
        active = [r for i, r in members if i >= len(history)]
        if not active:
            continue
        exposed = any(r["exposure"] == "EXPOSED" or r.get("use") in
                      ("training", "development", "calibration", "regression") for _, r in members)
        effective = "EXPOSED" if exposed else ("UNKNOWN" if any(r["exposure"] == "UNKNOWN" for _, r in members) else "FRESH")
        splits = {r.get("split") for _, r in members} & {"train", "dev", "heldout"}
        if len(splits) > 1:
            conflicts.append({"kind": "SHARED_IDENTITY_ACROSS_SPLITS", "ids": sorted({r["id"] for _, r in members}), "splits": sorted(splits)})
        for row in active:
            restricted_recording = any(i in restricted_indices and r is row for i, r in members)
            verified = row.get("lineage_status") == "verified" and bool(row.get("voice_identity") or row.get("lineage_ids") or row.get("reference_sha256"))
            historical_identity = any(i < len(history) for i, _ in members)
            hashes = [row[f] for f in ("wav_sha256", "pcm_sha256") if row.get(f)]
            fresh_recording = bool(hashes) and not any(h in history_audio or len(current_content[h]) > 1 for h in hashes)
            unseen_voice = verified and not historical_identity
            if row["exposure"] == "FRESH" and effective == "EXPOSED":
                conflicts.append({"kind": "EXPOSURE_DOWNGRADE", "id": row["id"]})
            if effective != "FRESH" and row.get("split") == "heldout":
                conflicts.append({"kind": "EXPOSED_OR_UNKNOWN_HELDOUT", "id": row["id"]})
            if row.get("restriction") == "regression_only" and row.get("use") != "regression":
                conflicts.append({"kind": "REGRESSION_FIXTURE_REPURPOSED", "id": row["id"]})
            if restricted_recording and row.get("use") != "regression":
                conflicts.append({"kind": "REGRESSION_LINEAGE_REPURPOSED", "id": row["id"]})
            audit.append({"id": row["id"], "effective_exposure": effective,
                          "fresh_recording": fresh_recording, "unseen_voice": unseen_voice,
                          "strong_independence": verified and fresh_recording and unseen_voice and effective == "FRESH",
                          "lineage_status": "verified" if verified else "unknown"})
            persisted = dict(row, exposure=effective)
            if restricted_recording:
                persisted["restriction"] = "regression_only"
            ledger.append(persisted)
    return {"status": "FAIL" if conflicts else "CHECKED_DECLARED_LINEAGE", "conflicts": conflicts,
            "rows": sorted(audit, key=lambda r: r["id"]), "ledger": ledger,
            "repeated_content": [sorted(ids) for ids in current_content.values() if len(ids) > 1],
            "unknown_lineage_never_proves_independence": True}


def keyword_name(value):
    if value in (1, "1", "K1", KEYWORDS["K1"]):
        return "K1"
    if value in (2, "2", "K2", KEYWORDS["K2"]):
        return "K2"
    raise ValueError("unknown event keyword")


def event_comparison(rows, original, candidate, training_diagnostics=None):
    """Presence/false/wrong/repeat counts on independent saved event records.

    One expected detection per present keyword per clip. Multiple actual
    occurrences need independently annotated event instances before using this
    clip-level guard; they are reported as unknown, not silently collapsed.
    """
    _unique_rows(rows)
    known = {r["id"] for r in rows}
    require(set(original) <= known and set(candidate) <= known, "events without labels")
    findings = []
    unknown = []
    summaries = {arm: Counter() for arm in ("original", "candidate")}
    for row in rows:
        truth = human_truth(row)
        actual = normalized_actual(row["actual_text"]) if truth is not None else ""
        # An overlap (e.g. 小窝小窝小窝) also has ambiguous event instances.
        multiple = any(actual.find(text) != actual.rfind(text) for text in KEYWORDS.values())
        if truth is None or multiple or row["id"] not in original or row["id"] not in candidate:
            unknown.append(row["id"])
            continue
        arms = {}
        for name, events in (("original", original[row["id"]]), ("candidate", candidate[row["id"]])):
            require(isinstance(events, list), "missing independent event list")
            counts = Counter(keyword_name(e["keyword"]) for e in events)
            misses = sorted(truth - counts.keys())
            false = {k: n for k, n in counts.items() if k not in truth and not truth}
            wrong = {k: n for k, n in counts.items() if k not in truth and truth}
            repeats = {k: n - 1 for k, n in counts.items() if n > 1}
            arms[name] = dict(misses=misses, false_keywords=false, wrong_keywords=wrong, repeats=repeats,
                              event_total=sum(counts.values()))
            summaries[name].update(events=sum(counts.values()), misses=len(misses),
                                   false_events=sum(false.values()), wrong_events=sum(wrong.values()), repeats=sum(repeats.values()))
        old, new = arms["original"], arms["candidate"]
        deltas = {"new_misses": sorted(set(new["misses"]) - set(old["misses"]))}
        for category in ("false_keywords", "wrong_keywords", "repeats"):
            deltas["new_" + category] = dict(Counter(new[category]) - Counter(old[category]))
        findings.append({"id": row["id"], "expected": sorted(truth), **arms, **deltas})
    regressions = [r for r in findings if any(r[k] for k in ("new_misses", "new_false_keywords", "new_wrong_keywords", "new_repeats"))]
    return {"event_status": "EVENT_REGRESSION" if regressions else ("UNKNOWN" if unknown or not findings else "EVENT_PROTECTION_PASS"),
            "rows": findings, "regressions": [r["id"] for r in regressions], "unscored": unknown,
            "counts": {k: dict(v) for k, v in summaries.items()},
            "training_diagnostics": training_diagnostics or {}, "training_fit_implies_event_pass": False,
            "fresh_model_validation": False, "scientific_qualification": False}


def availability_report(original, candidate, sample_rates):
    """Unique retained event availability; no greedy alignment or nearest match."""
    matches, ambiguous, unmatched, unknown = [], [], [], []
    for recording in sorted(set(original) | set(candidate)):
        if recording not in original or recording not in candidate:
            unknown.append({"id": recording, "reason": "MISSING_EVENT_ARM"})
            continue
        arms = []
        for source in (original, candidate):
            grouped = defaultdict(list)
            require(isinstance(source[recording], list), "missing independent event list")
            for event in source[recording]:
                grouped[keyword_name(event["keyword"])].append(event)
            arms.append(grouped)
        for keyword in sorted(set(arms[0]) | set(arms[1])):
            old, new = (arm[keyword] for arm in arms)
            key = {"id": recording, "keyword": keyword}
            if len(old) > 1 or len(new) > 1:
                ambiguous.append(dict(key, original_count=len(old), candidate_count=len(new)))
                continue
            if not old or not new:
                unmatched.append(dict(key, change="added" if new else "removed"))
                continue
            rate = sample_rates.get(recording)
            values = old[0].get("available_samples"), new[0].get("available_samples")
            if not (type(rate) in (int, float) and math.isfinite(rate) and rate > 0 and all(type(v) is int and v >= 0 for v in values)):
                unknown.append(key)
                continue
            delta = values[1] - values[0]
            matches.append(dict(key, original_available_samples=values[0], candidate_available_samples=values[1],
                                delta_samples=delta, availability_delta_ms=delta * 1000 / rate))
    counts = {"later": sum(r["delta_samples"] > 0 for r in matches),
              "unchanged": sum(r["delta_samples"] == 0 for r in matches),
              "earlier": sum(r["delta_samples"] < 0 for r in matches)}
    return {"status": "AMBIGUOUS_OR_UNKNOWN" if ambiguous or unknown else "SAVED_AVAILABILITY_ACCOUNTED",
            "matched_events": len(matches), "counts": counts, "matches": matches,
            "ambiguous": ambiguous, "unknown": unknown, "unmatched": unmatched,
            "coordinate": "input samples available at event callback", "word_end_latency_ms": None,
            "word_end_latency_status": "NOT_MEASURED", "wall_service_latency_status": "NOT_MEASURED"}


def supervision_interpretation(resource):
    """Read the existing supervisor schema, without replacing its executor."""
    issues = []
    live = []

    def check_children(status, count, observation):
        if status == "NOT_AVAILABLE":
            if count is not None:
                issues.append("UNAVAILABLE_CHILDREN_REWRITTEN_AS_COUNT")
        elif status == "AVAILABLE_AT_SAMPLE":
            if type(count) is not int or count < 0:
                issues.append("MISSING_OR_FAILED_CHILDREN_OBSERVATION")
        elif status == "UNKNOWN_NOT_SAMPLED_AFTER_EXIT" and observation == "EXITED_BEFORE_READ":
            if count is not None:
                issues.append("UNAVAILABLE_CHILDREN_REWRITTEN_AS_COUNT")
        else:
            issues.append("CHILDREN_OBSERVATION_FAILED")
        if type(count) is int and count > 0:
            issues.append("CHILD_PROCESS_FORBIDDEN")

    pre = resource.get("pre_go_observation", {})
    if (type(pre.get("VmRSS")) is not int or pre["VmRSS"] < 0
            or type(pre.get("Threads")) is not int or pre["Threads"] < 1
            or pre.get("critical_monitoring") != "OBSERVED_RSS_THREADS_ONLY"
            or pre.get("critical_errors")):
        issues.append("MISSING_OR_FAILED_PRE_GO_CRITICAL_OBSERVATION")
    check_children(pre.get("children_observation"), pre.get("children_count"), pre.get("observation"))
    for sample in resource.get("proc_samples_compact", []):
        require(len(sample) >= 13, "truncated compact supervision sample")
        if sample[7] == "LIVE":
            live.append(sample)
            if type(sample[2]) is not int or sample[2] < 0 or type(sample[5]) is not int or sample[5] < 1 or sample[8] != "OBSERVED_RSS_THREADS_ONLY" or sample[10] is not None:
                issues.append("MISSING_OR_FAILED_CRITICAL_RSS_THREADS")
        check_children(sample[9], sample[6], sample[7])
        if sample[8] == "FAILED_UNAVAILABLE":
            issues.append("CRITICAL_OBSERVATION_FAILED")
    if not live or resource.get("critical_post_go_samples", 0) < 1:
        issues.append("NO_CRITICAL_LIVE_OBSERVATIONS")
    if resource.get("schema") != "a20-endpoint-arm-resource-v1":
        issues.append("UNKNOWN_SUPERVISION_SCHEMA")
    if resource.get("status") != "COMPLETE_ACQUISITION_PENDING_TRACE_AUDIT":
        issues.append("TERMINAL_STATUS_NOT_SUCCESS")
    if resource.get("guard_stop_reason") is not None or resource.get("returncode") != 0:
        issues.append("EXECUTION_FAILED_OR_UNKNOWN")
    return {"integrity_status": "FAIL" if issues else "PASS_SAVED_LIMITED_OBSERVATIONS",
            "issues": sorted(set(issues)), "original_execution_status": resource.get("status", "UNKNOWN"),
            "children_observation": sorted({s[9] for s in resource.get("proc_samples_compact", [])}),
            "lifetime_child_absence": "NOT_PROVEN", "continuous_lifetime_compliance_proven": False,
            "scientific_outcome": "NOT_ASSESSED"}


def recovery_interpretation(original_status, expected, recovered_bytes):
    """Only supplied bytes matching frozen size/hash recover an object."""
    files = []
    for item in expected:
        data = recovered_bytes.get(item["path"])
        if data is None:
            status = "METADATA_ONLY_NOT_BYTE_RECOVERED"
        elif not isinstance(data, bytes):
            status = "NOT_BYTES"
        else:
            status = "BYTE_VERIFIED" if len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"] else "CORRUPT"
        files.append({"path": item["path"], "status": status, "required_raw": bool(item.get("required_raw"))})
    recovered = bool(files) and all(f["status"] == "BYTE_VERIFIED" for f in files)
    missing_raw = any(f["required_raw"] and f["status"] != "BYTE_VERIFIED" for f in files)
    return {"byte_recovery": "BYTE_VERIFIED" if recovered else "INCOMPLETE", "files": files,
            "missing_raw": missing_raw, "original_execution_status": original_status,
            "execution_status": original_status, "scientific_outcome": "NOT_ASSESSED",
            "recovery_implies_execution_pass": False}


def rss_comparison(original_lifetime_kib, candidate_lifetime_kib, original_sampled_bytes, candidate_sampled_bytes):
    def difference(a, b):
        return None if a is None or b is None else b - a
    return {"lifetime_native": {"scope": "whole process lifetime high water", "unit": "KiB",
                               "original": original_lifetime_kib, "candidate": candidate_lifetime_kib,
                               "delta": difference(original_lifetime_kib, candidate_lifetime_kib)},
            "sampled": {"scope": "supervisor observation instants", "unit": "bytes",
                        "original": original_sampled_bytes, "candidate": candidate_sampled_bytes,
                        "delta": difference(original_sampled_bytes, candidate_sampled_bytes)},
            "model_memory_regression": "NOT_ESTABLISHED", "scopes_interchangeable": False}
