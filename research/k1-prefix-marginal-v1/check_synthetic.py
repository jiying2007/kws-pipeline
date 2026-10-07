#!/usr/bin/env python3
"""Run the independent bounded synthetic checks without replacing saved reports."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "review"))
from independent_review import run as oracle_run
from edge_review import run as edge_run
from underflow_mock_review import run as underflow_run


def main():
    module = HERE / "frozen/k1-prefix-marginal-v1/k1_prefix_marginal.py"
    oracle = oracle_run(module)
    assert not oracle["failures"], oracle["failures"]
    assert (oracle["cases"], oracle["raw_paths_visited"]) == (612, 5523157)
    edges = edge_run(module)
    assert len(edges["checks_passed"]) == 11
    underflow = underflow_run(
        HERE / "frozen/k1-prefix-saved-evaluation-v1/runner.py",
        HERE / "frozen/k1-prefix-saved-evaluation-recovery-v2/runner.py",
    )
    assert underflow["status"] == "PASS_ADAPTER_MOCKS_ONLY"
    assert len(underflow["checks"]) == 8
    print(json.dumps({"oracle": oracle, "edges": edges, "underflow": underflow},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
