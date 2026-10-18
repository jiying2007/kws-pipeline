"""Explicit, source-pinned numerical/node patch under review.

Only two near-zero branch tests and one node copy-on-write insertion change.
The retained source files and legacy decoder remain unchanged.
"""
import ast
import math
from collections import defaultdict
from pathlib import Path
from posterior_events import DecoderError, _retained


def patched_methods():
    methods = _retained()  # Validates both source hashes; retains score/suffix fixes.
    tree = ast.parse((Path(__file__).parent / 'vendor' / 'stream_kws_ctc.py').read_bytes())
    beam = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'ctc_prefix_beam_search')
    branches = []
    copies = 0
    for node in ast.walk(beam):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            call = test.operand
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == 'math' and call.func.attr == 'isclose'):
                if (len(call.args) != 2 or not isinstance(call.args[0], ast.Name)
                        or call.args[0].id not in ('pb', 'pnb')
                        or ast.literal_eval(call.args[1]) != 0.0
                        or len(call.keywords) != 1
                        or call.keywords[0].arg != 'abs_tol'
                        or ast.literal_eval(call.keywords[0].value) != 1e-6):
                    raise DecoderError('unexpected near-zero branch source shape')
                variable = call.args[0].id
                branches.append(variable)
                node.test = ast.Compare(left=ast.Name(id=variable, ctx=ast.Load()),
                                        ops=[ast.Gt()], comparators=[ast.Constant(0.0)])
        if len(node.body) == 2 and all(isinstance(n, ast.Assign) for n in node.body):
            targets = [ast.unparse(n.targets[0]) for n in node.body]
            if targets == ["nodes[-1]['prob']", "nodes[-1]['frame']"]:
                node.body.insert(0, ast.parse('nodes[-1] = nodes[-1].copy()').body[0])
                copies += 1
    if sorted(branches) != ['pb', 'pnb'] or copies != 1:
        raise DecoderError('minimal patch changed an unexpected number of sites')
    module = ast.fix_missing_locations(ast.Module(body=[beam], type_ignores=[]))
    namespace = dict(math=math, defaultdict=defaultdict)
    exec(compile(module, 'pinned-wekws[minimal-positive-mass-and-node-cow]', 'exec'), namespace)
    methods['ctc_prefix_beam_search'] = namespace['ctc_prefix_beam_search']
    return methods
