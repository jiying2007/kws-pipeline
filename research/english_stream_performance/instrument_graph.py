#!/usr/bin/env python3
"""Create an explicitly diagnostic copy of the pinned frontend; never edit it."""
import argparse
import hashlib
from pathlib import Path

EXPECTED = "5a476954887b59032ea5d419e97d6fb913695e3ad99f5d09164513a4cbd87a89"


def instrument(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("unexpected frozen frontend source")
    text = raw.decode("utf-8")
    changes = [
        ("    if (!output || ef_graph_validate(g) != EF_OK || !input) return EF_INVALID;",
         "    if (!output) return EF_INVALID;\n"
         "    {\n"
         "        ekws_profile_stamp start = ekws_profile_now();\n"
         "        int valid = ef_graph_validate(g);\n"
         "        ekws_profile_add(g, -1, -1, start);\n"
         "        if (valid != EF_OK || !input) return EF_INVALID;\n"
         "    }"),
        ("        status = EF_OK;\n        if (n->op >= EF_ADD", 
         "        ekws_profile_stamp node_started = ekws_profile_now();\n"
         "        status = EF_OK;\n        if (n->op >= EF_ADD"),
        ("        if (trace && trace(trace_context, ni, y, out)) return EF_CALLBACK;",
         "        ekws_profile_add(g, (int)ni, (int)n->op, node_started);\n"
         "        if (trace && trace(trace_context, ni, y, out)) return EF_CALLBACK;")]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError("diagnostic insertion anchor is not unique")
        text = text.replace(before, after, 1)
    text = '#include "ekws_graph_profile.h"\n' + text
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    instrument(args.source, args.output)
