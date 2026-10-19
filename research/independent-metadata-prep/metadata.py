"""Offline metadata only. No acoustic inference or qualification decisions."""
import argparse
import hashlib
import json
import re
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from pathlib import Path

NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
UNKNOWN_DELIMITERS = re.compile(r'[\[\]()（）【】<>{}［］〈〉]')
UNKNOWN_TOKEN = re.compile(r'(?<![A-Za-z0-9_])(?:UNK|UNKNOWN)(?![A-Za-z0-9_])', re.I)


def uncertain_text(text):
    # Finite conservative rules, not an exhaustive annotation-language parser.
    return bool(UNKNOWN_DELIMITERS.search(text) or UNKNOWN_TOKEN.search(text)
                or text.strip() in {'未知', '不清', '听不清'})


QUOTE = r'"(?:[^\"]|\"\")*"'


class Reader:
    def __init__(self, text):
        self.text, self.pos = text, 0

    def take(self, pattern):
        match = re.compile(r'\s*(?:' + pattern + ')', re.S).match(self.text, self.pos)
        if not match:
            raise ValueError(f'unsupported TextGrid syntax at offset {self.pos}')
        self.pos = match.end()
        return match.group().strip()

    def number(self):
        value = Decimal(self.take(NUM))
        if not value.is_finite() or len(value.as_tuple().digits) > 40 or abs(value.adjusted()) > 40:
            raise ValueError('unsupported time precision/range')
        return value

    def quoted(self):
        return self.take(QUOTE)[1:-1].replace('""', '"')


def parse_textgrid(data):
    """Strict long IntervalTier parser; Decimal, no tolerance or text normalization."""
    encoding = 'utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    r = Reader(data.decode(encoding))
    r.take(r'File type\s*=\s*"ooTextFile"')
    r.take(r'Object class\s*=\s*"TextGrid"')
    r.take('xmin\\s*='); lo = r.number()
    r.take('xmax\\s*='); hi = r.number()
    if lo != 0 or hi <= 0:
        raise ValueError('invalid extent')
    r.take(r'tiers\?\s*<exists>\s*size\s*=')
    count = int(r.take(r'\d+'))
    r.take(r'item\s*\[\]:')
    intervals = []
    for tier in range(1, count + 1):
        r.take(r'item\s*\[' + str(tier) + r'\]:')
        r.take(r'class\s*=\s*"IntervalTier"\s*name\s*='); r.quoted()
        r.take('xmin\\s*='); tlo = r.number()
        r.take('xmax\\s*='); thi = r.number()
        if (tlo, thi) != (lo, hi):
            raise ValueError('tier extent mismatch')
        r.take(r'intervals:\s*size\s*='); n = int(r.take(r'\d+'))
        previous = lo
        for index in range(1, n + 1):
            r.take(r'intervals\s*\[' + str(index) + r'\]:')
            r.take('xmin\\s*='); start = r.number()
            r.take('xmax\\s*='); end = r.number()
            r.take('text\\s*='); text = r.quoted()
            if not lo <= previous <= start < end <= hi:
                raise ValueError('overlap, reversed, empty or out-of-range interval')
            previous = end
            intervals.append((tier, index, start, end, text))
    if count < 1 or r.text[r.pos:].strip():
        raise ValueError('empty tiers or trailing syntax')
    return hi, intervals


def sample(time, rate, rounding):
    # More precision than the parser's bounded Decimal input and integer rate.
    with localcontext() as context:
        context.prec = 120
        return int((time * rate).to_integral_value(rounding=rounding))


def mask(data, rate, samples, review_terms, historical_pending):
    if type(rate) is not int or type(samples) is not int or not 0 < rate <= 1000000 or samples <= 0:
        raise ValueError('positive integer rate/sample extent required')
    if not review_terms or any(not isinstance(t, str) or not t.strip() for t in review_terms):
        raise ValueError('explicit nonempty review terms required')
    hi, intervals = parse_textgrid(data)
    if sample(hi, rate, ROUND_CEILING) > samples:
        raise ValueError('transcript extends beyond declared audio sample count')
    support, review, uncertain = [], [], []
    for tier, index, start, end, text in intervals:
        # Review terms take precedence even inside annotation markers.
        if any(term in text for term in review_terms):
            review.append((sample(start, rate, ROUND_FLOOR), sample(end, rate, ROUND_CEILING)))
        elif uncertain_text(text):
            # Unknown annotation syntax blocks support even if mixed with speech.
            uncertain.append((sample(start, rate, ROUND_FLOOR), sample(end, rate, ROUND_CEILING)))
        elif any(c.isalnum() for c in text):
            # Blank and annotation-only markers do not establish speech support.
            support.append((sample(start, rate, ROUND_CEILING), sample(end, rate, ROUND_FLOOR)))
    points = sorted({0, samples} | {p for span in support + review + uncertain for p in span})
    result = []
    for start, end in zip(points, points[1:]):
        if start == end:
            continue
        label = 'unknown'
        if any(a <= start and end <= b for a, b in support):
            label = 'transcript_supported_candidate'
        if any(a <= start and end <= b for a, b in uncertain):
            label = 'unknown'
        if any(a <= start and end <= b for a, b in review):
            label = 'target_or_near_target_review'
        if result and result[-1]['category'] == label:
            result[-1]['end_sample'] = end
        else:
            result.append(dict(start_sample=start, end_sample=end, category=label))
    validate_partition(result, samples)
    return dict(schema='development-mask-draft-v1', split='development',
                new_independent_hours=0, human_gold=False, negative_admitted=False,
                accepted_negative_seconds=0, expected=None, audio_verified_this_run=False,
                speaker_overlap='unknown', human_review='pending',
                historical_pending=historical_pending, sample_rate=rate, num_samples=samples,
                textgrid_sha256=hashlib.sha256(data).hexdigest(), intervals=result,
                screening_complete=False, cross_segment_search='not_performed',
                all_sound_sources_annotated='unknown', review_terms=review_terms)



def validate_partition(rows, samples):
    """Production check stays active under python -O and -OO."""
    cursor = 0
    allowed = {'unknown', 'transcript_supported_candidate', 'target_or_near_target_review'}
    for row in rows:
        start, end = row['start_sample'], row['end_sample']
        if (type(start) is not int or type(end) is not int or start != cursor
                or not start < end <= samples or row['category'] not in allowed):
            raise ValueError('invalid sample partition')
        cursor = end
    if cursor != samples or not rows:
        raise ValueError('incomplete sample partition')


def checked_mask(path, contract):
    """Bind explicit caller-reviewed identity before generating private metadata."""
    sid = contract['session_id']
    digest = contract['textgrid_sha256']
    family = contract['source_family']
    if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_]+', sid):
        raise ValueError('invalid session identity')
    if Path(path).name != sid + '.TextGrid':
        raise ValueError('session/filename mismatch')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('expected TextGrid SHA-256 required')
    if not isinstance(family, str) or not family.strip():
        raise ValueError('explicit source family required')
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('TextGrid identity mismatch')
    result = mask(data, contract['sample_rate'], contract['num_samples'],
                  contract['review_terms'], contract['historical_pending'])
    result.update(session_id=sid, source_family=family,
                  identity_basis='caller-reviewed contract; audio not verified')
    return result

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def exposure(candidates, ledgers):
    """Exact token contact clues only; never upgrades an absence to independence."""
    if len(set(candidates)) != len(candidates) or any(not isinstance(c, str) or not c for c in candidates):
        raise ValueError('nonempty unique candidate IDs required')
    evidence = []
    for path in ledgers:
        data = Path(path).read_bytes()
        evidence.append((hashlib.sha256(data).hexdigest(), list(strings(json.loads(data)))))
    rows = []
    for candidate in candidates:
        token = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(candidate) + r'(?![A-Za-z0-9])')
        hits = [digest for digest, values in evidence if any(token.search(v) for v in values)]
        rows.append(dict(candidate_id=candidate, scanned_ledger_matches=hits,
                         contact_clue='found_requires_review' if hits else 'no_match_in_supplied_ledgers',
                         historical_exposure='unknown', speaker_overlap='unknown',
                         pretraining_exposure='unknown', sourcefamily_overlap='unknown',
                         official_membership='unknown', selection_independent='unknown',
                         negative_admitted=False, new_independent_hours=0))
    return dict(schema='exposure-clues-draft-v1', ledger_count=len(evidence),
                evidence_scope='supplied JSON only; no match is not independence', candidates=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('exposure')
    audit.add_argument('--candidates', required=True, help='private JSON array of candidate IDs')
    audit.add_argument('--ledger', action='append', default=[])
    masking = sub.add_parser('mask')
    masking.add_argument('--textgrid', required=True)
    masking.add_argument('--contract', required=True, help='adapted reviewed private contract')
    for child in (audit, masking):
        child.add_argument('--output', required=True, help='new private output, outside public repository')
    args = parser.parse_args()
    if args.command == 'exposure':
        result = exposure(json.loads(Path(args.candidates).read_text()), args.ledger)
    else:
        contract = json.loads(Path(args.contract).read_text())
        result = checked_mask(args.textgrid, contract)
    with open(args.output, 'x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
