import copy
import itertools
import math
import unittest
from fractions import Fraction
from posterior_events import KEYWORDS, _retained, _Row
from three_window_events import ThreeWindowEvents
from test_posterior_events import row
from beam_minimal_patch_candidate import patched_methods

ALPHABET = (0, 1027, 1401)
PROBS = ((Fraction(1,2), Fraction(3,10), Fraction(1,5)),
         (Fraction(1,4), Fraction(1,4), Fraction(1,2)),
         (Fraction(1,5), Fraction(1,2), Fraction(3,10)),
         (Fraction(1,2), Fraction(1,5), Fraction(3,10)))


def enumeration_oracle(rows, scale):
    totals = {}
    for path in itertools.product(range(3), repeat=len(rows)):
        mass = scale
        collapsed, last = [], None
        for t, index in enumerate(path):
            token = ALPHABET[index]
            mass *= rows[t][index]
            if token != last and token != 0:
                collapsed.append(token)
            last = token
        key = tuple(collapsed)
        pb, pnb = totals.get(key, (Fraction(0), Fraction(0)))
        totals[key] = (pb + mass, pnb) if last == 0 else (pb, pnb + mass)
    return totals


def run_beam(beam, rows, scale):
    hypotheses = [((), (float(scale), 0., []))]
    for t, probabilities in enumerate(rows):
        values = [0.] * 2599
        for token, p in zip(ALPHABET, probabilities):
            values[token] = float(p)
        hypotheses = beam(3*t, _Row(values), hypotheses, set(ALPHABET), 3)
    return {prefix: (pb, pnb) for prefix, (pb, pnb, _) in hypotheses}


class MinimalPatchFixtures(unittest.TestCase):
    def test_01_independent_fraction_ctc_oracle(self):
        beam = patched_methods()['ctc_prefix_beam_search']
        for length in range(1, 5):
            for scale in (Fraction(1), Fraction(1, 10**9)):
                expected = enumeration_oracle(PROBS[:length], scale)
                actual = run_beam(beam, PROBS[:length], scale)
                self.assertEqual(actual.keys(), expected.keys())
                for prefix, expected_pair in expected.items():
                    for result, exact in zip(actual[prefix], expected_pair):
                        self.assertTrue(math.isclose(result, float(exact), rel_tol=2e-14,
                                                     abs_tol=1e-30), (prefix, result, exact))
                self.assertTrue(math.isclose(sum(sum(p) for p in actual.values()),
                                            float(scale), rel_tol=2e-14))

    def test_02_old_core_loses_positive_mass(self):
        old = _retained()['ctc_prefix_beam_search']
        expected = enumeration_oracle(PROBS[:2], Fraction(1, 10**9))
        actual = run_beam(old, PROBS[:2], Fraction(1, 10**9))
        self.assertLess(sum(sum(p) for p in actual.values()),
                        sum(float(sum(p)) for p in expected.values()))

    def test_03_repeated_target_after_blank_old_failure_new_pass(self):
        for silence in (20, 62):
            results = []
            for patched in (False, True):
                d = ThreeWindowEvents('fixture')
                d._methods = patched_methods() if patched else _retained()
                events = list(d.process(row(0, .5) for _ in range(silence)))
                for token in KEYWORDS[0][1]:
                    events.extend(d.process(row(token) for _ in range(2)))
                results.append((len(events), d.empty_beam_window_rows, d.zero_mass_window_rows))
            self.assertEqual(results[0][0], 0)
            self.assertGreater(results[0][1], 0)
            self.assertEqual(results[1], (1, 0, 0))

    def test_04_old_alias_pollution_and_patch_isolation(self):
        for patched in (False, True):
            shared = dict(token=1027, frame=0, prob=.2)
            second = dict(token=1401, frame=3, prob=.3)
            hypotheses = [((1027,), (.4,.2,[shared])),
                          ((1027,1401), (.1,.1,[shared,second]))]
            before = copy.deepcopy(hypotheses)
            values = [0.] * 2599
            values[1027], values[2] = .6, .4  # Other class is filtered after global top3.
            beam = (patched_methods() if patched else _retained())['ctc_prefix_beam_search']
            after = beam(6, _Row(values), hypotheses, {0,1027,1401,1462,2093}, 3)
            other = next(nodes for prefix, (_,_,nodes) in after if prefix == (1027,1401,1027))
            if patched:
                self.assertEqual(hypotheses, before)
                self.assertEqual([n['frame'] for n in other], [0,3,6])
                updated = next(nodes for prefix, (_,_,nodes) in after if prefix == (1027,))
                self.assertEqual(updated[-1]['frame'], 6)
                self.assertIsNot(updated[-1], shared)
            else:
                self.assertNotEqual(hypotheses, before)
                self.assertEqual([n['frame'] for n in other], [6,3,6])

    def test_05_scale_preserves_mass_ratios_and_rank(self):
        beam = patched_methods()['ctc_prefix_beam_search']
        normal = run_beam(beam, PROBS, Fraction(1))
        small = run_beam(beam, PROBS, Fraction(1, 10**9))
        for key in normal:
            for high, low in zip(normal[key], small[key]):
                self.assertTrue(math.isclose(high * 1e-9, low, rel_tol=2e-14, abs_tol=1e-30))
        self.assertEqual(sorted(normal, key=lambda k: sum(normal[k]), reverse=True),
                         sorted(small, key=lambda k: sum(small[k]), reverse=True))

    def test_06_blank_last_writer_is_retained_without_input_mutation(self):
        # COW deliberately does not replace the upstream metadata winner policy.
        shared = dict(token=1027, frame=0, prob=.2)
        hypotheses = [((1027,), (.4,.2,[shared]))]
        before = copy.deepcopy(hypotheses)
        values = [0.] * 2599
        values[1027], values[0] = .6, .4
        beam = patched_methods()['ctc_prefix_beam_search']
        after = beam(6, _Row(values), hypotheses, {0,1027,1401,1462,2093}, 3)
        self.assertEqual(hypotheses, before)
        selected = next(nodes for prefix, (_,_,nodes) in after if prefix == (1027,))
        self.assertEqual(selected[-1]['frame'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
