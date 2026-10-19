"""Small stdlib boundary for existing clip metrics; no inference or private data.

The caller supplies an independently reviewed input manifest. SHA256 binding
checks identity, not label truth, acoustic independence, or provenance honesty.
"""
import copy
import hashlib
import json
import math
import re

SCHEMA = 'kws-bound-events-v1'
PROFILE = 'seed0_B-offline-16khz-stride3-tail300-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _index(rows):
    require(isinstance(rows, list) and bool(rows), 'nonempty roster list required')
    out = {}
    for row in rows:
        require(isinstance(row, dict), 'row object required')
        name = row.get('recording')
        require(isinstance(name, str) and bool(name), 'recording ID required')
        require(name not in out, 'duplicate recording ID: ' + name)
        require(_sha(row.get('pcm_sha256')), 'PCM SHA256 required: ' + name)
        out[name] = row
    return out


def roster_digest(rows):
    """Order-independent identity digest; labels are NOT authenticated by this."""
    indexed = _index(rows)
    identity = [[name, indexed[name]['pcm_sha256']] for name in sorted(indexed)]
    return digest(json.dumps(identity, separators=(',', ':'), ensure_ascii=False).encode())


def align(rows, document):
    """Require exact identity sets; safely reorder results by recording ID."""
    expected = _index(rows)
    require(isinstance(document, dict) and document.get('schema') == SCHEMA,
            'explicit bound-events-v1 schema required; raw lists are forbidden')
    actual = _index(document.get('results'))
    require(set(actual) == set(expected), 'missing or extra result IDs')
    aligned = []
    for name, row in expected.items():
        result = actual[name]
        require(result['pcm_sha256'] == row['pcm_sha256'], 'PCM identity mismatch: ' + name)
        events = result.get('events')
        require(isinstance(events, list), 'explicit events list required: ' + name)
        for event in events:
            require(isinstance(event, dict), 'event object required')
            require(type(event.get('keyword_id')) is int and event['keyword_id'] in (1, 2), 'keyword ID')
            require(type(event.get('decision_row')) is int and event['decision_row'] >= 0, 'decision row')
            require(type(event.get('score')) in (int, float) and math.isfinite(event['score']) and
                    0 <= event['score'] <= 1, 'score must be in [0,1]')
            if 'recording_id' in event:
                require(event['recording_id'] == name, 'nested event recording mismatch')
        aligned.append(copy.deepcopy(events))
    return aligned


def measure_bound(rows, document, *, metric, window):
    """Keep frozen arithmetic via an explicitly supplied trusted metric function.

    Only the audited seed0_B analytical offline profile is supported here.
    This boundary does not turn its timestamps into online availability.
    """
    events = align(rows, document)
    require(window in ('pre_eof', 'eof_plus300'), 'unsupported diagnostic window')
    for row, es in zip(rows, events):
        require(type(row.get('samples')) is int and row['samples'] > 0, 'positive sample count')
        require(type(row.get('frames')) is int and row['frames'] > 0, 'positive frame count')
        labels = row.get('labels')
        require(isinstance(labels, list) and len(labels) == 2 and
                all(y is None or type(y) is int and y in (0, 1) for y in labels), 'two explicit labels')
        require(all(e['decision_row'] < row['frames'] for e in es), 'event outside frame domain')
    return metric(copy.deepcopy(rows), events, 'seed0_B', window)

