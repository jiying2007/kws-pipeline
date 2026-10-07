"""Synthetic-only regressions; no saved logits, audio, model, or decoder runs."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import repeat
import math
import random
import sys
import unittest

from k1_prefix_marginal import (
    CERTIFICATE_SCALE,
    MAX_FRAMES,
    STATE_NAMES,
    TOKENS,
    CapacityError,
    FinalizedError,
    InputError,
    K1PrefixMarginal,
)


def row(token=None, **weights):
    if token is not None:
        return tuple(int(value == token) for value in TOKENS)
    return tuple(weights.get(value, 0) for value in TOKENS)


def rows(*values):
    return [row(value) if isinstance(value, str) else row(**value) for value in values]


PREFIX = rows("你", "好", "小")
FIXTURES = (
    ("empty", [], (0, 0, 1)),
    ("silence", rows("blank", "blank", "blank"), (0, 0, 1)),
    ("incomplete", PREFIX, (0, 0, 1)),
    ("true_prefix", rows("你", "好", "小", "窝"), (1, 0, 0)),
    ("confuser_prefix", rows("你", "好", "小", "屋"), (0, 1, 0)),
    ("leading_blanks", rows("blank", "blank", "你", "好", "小", "窝", "blank"), (1, 0, 0)),
    ("adjacent_repeat", rows("你", "你", "好", "小", "窝"), (1, 0, 0)),
    ("blank_separated_repeat", rows("你", "blank", "你", "好", "小", "窝"), (0, 0, 1)),
    ("n1_semantic_suffix_protection_synthetic", rows("你", "好", "小", "窝", "blank", "屋"), (1, 0, 0)),
    ("initial_confuser_then_wake", rows("你", "好", "小", "屋", "blank", "你", "好", "小", "窝"), (0, 1, 0)),
    ("incompatible_start", rows("屋", "blank", "你", "好", "小", "窝"), (0, 0, 1)),
    ("alignment_marginal", PREFIX + rows({"窝": 2, "blank": 3}, {"窝": 2, "blank": 3}), (Fraction(16, 25), 0, Fraction(9, 25))),
    ("six_competitors", PREFIX + [(1,) * 6], (Fraction(1, 6), Fraction(1, 6), Fraction(2, 3))),
    ("binary_grouping", PREFIX + rows({"窝": 4, "屋": 3, "blank": 3}), (Fraction(2, 5), Fraction(3, 10), Fraction(3, 10))),
    ("exact_tie", PREFIX + rows({"窝": 1, "屋": 1}), (Fraction(1, 2), Fraction(1, 2), 0)),
    ("irreversible_false_wo_limitation", PREFIX + rows({"窝": 3, "blank": 2}, "屋"), (Fraction(3, 5), Fraction(2, 5), 0)),
    ("changing_conditional_odds", PREFIX + rows({"窝": 4, "屋": 3, "blank": 3}, "屋"), (Fraction(2, 5), Fraction(3, 5), 0)),
)


class SyntheticFixtureTests(unittest.TestCase):
    def assert_masses(self, snapshot, expected):
        for actual, exact, bounds in zip(snapshot.masses, expected, snapshot.mass_bounds):
            self.assertAlmostEqual(actual, float(exact), delta=2e-13)
            self.assertLessEqual(bounds[0], exact)
            self.assertGreaterEqual(bounds[1], exact)
        for value, log_value in zip(snapshot.masses, snapshot.log_masses):
            self.assertEqual(value, math.exp(log_value))

    def test_directed_fixtures_and_every_two_way_split(self):
        for name, observation, expected in FIXTURES:
            with self.subTest(fixture=name):
                verifier = K1PrefixMarginal()
                reference = verifier.update_chunk(observation)
                self.assert_masses(reference, expected)
                result = verifier.finalize()
                self.assertEqual(result.decision, "ACCEPT_K1" if expected[0] > Fraction(1, 2) else "REJECT_K1")
                for split in range(len(observation) + 1):
                    split_verifier = K1PrefixMarginal()
                    split_verifier.update_chunk([])
                    split_verifier.update_chunk(observation[:split])
                    split_verifier.update_chunk([])
                    split_verifier.update_chunk(observation[split:])
                    self.assertEqual(split_verifier.snapshot(), reference)
                    self.assertEqual(split_verifier.finalize(), result)
                individual = K1PrefixMarginal()
                for item in observation:
                    individual.update(item)
                self.assertEqual(individual.snapshot(), reference)

    def test_repeats_on_each_shared_label_across_chunk_boundary(self):
        for index in range(3):
            repeated = PREFIX[: index + 1] + [PREFIX[index]] + PREFIX[index + 1:] + rows("窝")
            separated = PREFIX[: index + 1] + rows("blank") + [PREFIX[index]] + PREFIX[index + 1:] + rows("窝")
            for observation, expected in ((repeated, (1, 0, 0)), (separated, (0, 0, 1))):
                verifier = K1PrefixMarginal()
                verifier.update_chunk(observation[: index + 1])
                verifier.update_chunk(observation[index + 1:])
                self.assert_masses(verifier.snapshot(), expected)

    def test_arbitrary_suffix_never_changes_absorbed_buckets(self):
        generator = random.Random(82391)
        suffix = [tuple(generator.randrange(1, 12) for _ in TOKENS) for _ in range(80)]
        for observation, expected in (
            (rows("你", "好", "小", "窝", "blank", "屋"), (1, 0, 0)),
            (PREFIX + rows({"窝": 3, "blank": 2}, "屋"), (Fraction(3, 5), Fraction(2, 5), 0)),
            (rows("屋"), (0, 0, 1)),
        ):
            verifier = K1PrefixMarginal()
            before = verifier.update_chunk(observation)
            after = verifier.update_chunk(suffix)
            self.assertEqual(before.state_log_masses, after.state_log_masses)
            self.assert_masses(after, expected)

    def test_acceptance_can_arrive_later_but_cannot_be_retracted(self):
        verifier = K1PrefixMarginal()
        first = verifier.update_chunk(PREFIX + rows({"窝": 2, "blank": 3}))
        self.assertLess(first.A, 0.5)
        second = verifier.update(row(**{"窝": 2, "blank": 3}))
        self.assertGreater(second.A, 0.5)
        third = verifier.update(row("屋"))
        self.assertEqual(second.A, third.A)
        self.assertEqual(verifier.finalize().decision, "ACCEPT_K1")

    def test_conditional_odds_can_fall_without_A_falling(self):
        verifier = K1PrefixMarginal()
        before = verifier.update_chunk(PREFIX + rows({"窝": 4, "屋": 3, "blank": 3}))
        after = verifier.update(row("屋"))
        self.assertAlmostEqual(before.A / (before.A + before.B), 4 / 7)
        self.assertAlmostEqual(after.A / (after.A + after.B), 2 / 5)
        self.assertEqual(before.A, after.A)
        self.assertEqual(verifier.finalize().decision, "REJECT_K1")


class NumericalTests(unittest.TestCase):
    def test_exact_multialignment_tie_rejects_despite_float_log_order(self):
        # A = 1 - (4/5)*(5/8) = 1/2 exactly. Strict floating log comparison
        # accepts incorrectly on common libm implementations (review regression).
        verifier = K1PrefixMarginal()
        verifier.update_chunk(PREFIX + rows({"blank": 4, "窝": 1}, {"blank": 5, "窝": 3}))
        result = verifier.finalize()
        self.assertEqual(result.decision, "REJECT_K1")
        self.assertEqual(result.decision_status, "NUMERICALLY_UNRESOLVED")
        lower, upper = result.snapshot.mass_bounds[0]
        self.assertLessEqual(lower, Fraction(1, 2))
        self.assertGreaterEqual(upper, Fraction(1, 2))

    def test_single_row_exact_tie_is_certified_reject(self):
        verifier = K1PrefixMarginal()
        verifier.update_chunk(PREFIX + rows({"窝": 1, "屋": 1}))
        result = verifier.finalize()
        self.assertEqual(result.decision, "REJECT_K1")
        self.assertEqual(result.decision_status, "CERTIFIED_REJECT")
        self.assertEqual(result.snapshot.mass_bounds[0], (Fraction(1, 2), Fraction(1, 2)))

    def test_nearest_binary64_weights_on_both_sides_of_tie(self):
        for weight, decision in (
            (math.nextafter(1.0, 0.0), "REJECT_K1"),
            (math.nextafter(1.0, math.inf), "ACCEPT_K1"),
        ):
            verifier = K1PrefixMarginal()
            verifier.update_chunk(PREFIX + rows({"窝": weight, "屋": 1.0}))
            result = verifier.finalize()
            self.assertEqual(result.decision, decision)
            self.assertNotEqual(result.decision_status, "NUMERICALLY_UNRESOLVED")
            exact_a = Fraction.from_float(weight) / (Fraction.from_float(weight) + 1)
            self.assertLessEqual(result.snapshot.mass_bounds[0][0], exact_a)
            self.assertGreaterEqual(result.snapshot.mass_bounds[0][1], exact_a)

    def test_subgrid_positive_margin_is_explicitly_unresolved(self):
        verifier = K1PrefixMarginal()
        verifier.update_chunk(PREFIX + rows({"窝": 1, "blank": 1}))
        verifier.update(row(**{"窝": math.ulp(0.0), "blank": 1.0}))
        result = verifier.finalize()
        # True A is strictly above 1/2, by less than 2^-256. This is not a
        # certified mathematical rejection, nor is a wider threshold fitted.
        self.assertEqual(result.decision, "REJECT_K1")
        self.assertEqual(result.decision_status, "NUMERICALLY_UNRESOLVED")
        self.assertGreater(result.snapshot.mass_bounds[0][1], Fraction(1, 2))

    def test_overflowing_raw_sum_is_normalized_without_overflow(self):
        verifier = K1PrefixMarginal()
        snapshot = verifier.update_chunk(PREFIX + [(sys.float_info.max,) * 6])
        for actual, expected in zip(snapshot.masses, (1 / 6, 1 / 6, 2 / 3)):
            self.assertAlmostEqual(actual, expected, delta=2e-13)

    def test_all_subnormal_equal_weights_remain_uniform(self):
        verifier = K1PrefixMarginal()
        snapshot = verifier.update_chunk(PREFIX + [(math.ulp(0.0),) * 6])
        for actual, expected in zip(snapshot.masses, (1 / 6, 1 / 6, 2 / 3)):
            self.assertAlmostEqual(actual, expected, delta=2e-13)

    def test_extreme_dynamic_range_preserves_nonzero_log_mass(self):
        tiny, large = math.ulp(0.0), sys.float_info.max
        verifier = K1PrefixMarginal()
        snapshot = verifier.update_chunk(rows({"你": tiny, "屋": large}, "好", "小", "窝"))
        self.assertEqual(snapshot.A, 0.0)
        self.assertTrue(math.isfinite(snapshot.log_A))
        self.assertAlmostEqual(snapshot.log_A, math.log(tiny) - math.log(large), delta=1e-12)
        self.assertEqual(snapshot.log_B, -math.inf)
        self.assertEqual(verifier.finalize().decision_status, "CERTIFIED_REJECT")

    def test_long_tiny_positive_path_does_not_underflow_log_state(self):
        repeat_count = MAX_FRAMES - 3
        verifier = K1PrefixMarginal()
        verifier.update_chunk(repeat(row(**{"你": 1e-300, "屋": 1}), repeat_count))
        snapshot = verifier.update_chunk(rows("好", "小", "窝"))
        self.assertEqual(snapshot.frames, MAX_FRAMES)
        self.assertEqual(snapshot.A, 0.0)
        self.assertTrue(math.isfinite(snapshot.log_A))
        self.assertAlmostEqual(snapshot.log_A, repeat_count * math.log(1e-300), delta=5e-7)
        self.assertEqual(snapshot.B, 0.0)
        self.assertEqual(snapshot.R, 1.0)
        self.assertEqual(verifier.finalize().decision_status, "CERTIFIED_REJECT")

    def test_exact_power_of_two_common_row_scaling_invariance(self):
        generator = random.Random(80913)
        observation = [tuple(generator.randrange(8) for _ in TOKENS) for _ in range(150)]
        base, scaled = K1PrefixMarginal(), K1PrefixMarginal()
        base.update_chunk(observation)
        # One common scale per complete row, not an independent token scale.
        scaled_rows = []
        for item in observation:
            exponent = generator.randrange(-700, 701)
            scaled_rows.append(tuple(math.ldexp(value, exponent) for value in item))
        scaled.update_chunk(scaled_rows)
        self.assertEqual(base.snapshot().state_lower_numerators, scaled.snapshot().state_lower_numerators)
        self.assertEqual(base.snapshot().state_upper_numerators, scaled.snapshot().state_upper_numerators)
        for left, right in zip(base.snapshot().masses, scaled.snapshot().masses):
            self.assertAlmostEqual(left, right, delta=2e-12)
        self.assertEqual(base.finalize().decision, scaled.finalize().decision)

    def test_conservation_monotonicity_and_fixed_grid_error_budget(self):
        generator = random.Random(739102)
        verifier = K1PrefixMarginal()
        previous = verifier.snapshot()
        for index in range(MAX_FRAMES):
            weights = tuple(generator.randrange(1, 257) for _ in TOKENS)
            current = verifier.update(weights)
            self.assertAlmostEqual(math.fsum(current.state_masses), 1.0, delta=2e-11)
            self.assertAlmostEqual(math.fsum(current.masses), 1.0, delta=2e-11)
            self.assertGreaterEqual(current.A, previous.A)
            self.assertGreaterEqual(current.B, previous.B)
            self.assertLessEqual(current.R, previous.R + 2e-15)
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in current.state_masses))
            lower, upper = current.state_lower_numerators, current.state_upper_numerators
            self.assertLessEqual(sum(lower), CERTIFICATE_SCALE)
            self.assertGreaterEqual(sum(upper), CERTIFICATE_SCALE)
            self.assertLessEqual(CERTIFICATE_SCALE - sum(lower), 10 * (index + 1))
            self.assertLessEqual(sum(upper) - CERTIFICATE_SCALE, 10 * (index + 1))
            self.assertTrue(all(0 <= low <= high <= CERTIFICATE_SCALE + 10 * MAX_FRAMES for low, high in zip(lower, upper)))
            previous = current


class LifecycleTests(unittest.TestCase):
    def test_snapshot_is_immutable_and_does_not_change_with_updates(self):
        verifier = K1PrefixMarginal()
        before = verifier.snapshot()
        with self.assertRaises(FrozenInstanceError):
            before.A = 1.0
        verifier.update(row("你"))
        self.assertEqual(before.frames, 0)
        self.assertEqual(before.masses, (0.0, 0.0, 1.0))
        self.assertFalse(before.finalized)
        self.assertEqual(len(before.state_masses), len(STATE_NAMES))

    def test_finalize_inserts_no_frame_and_emits_once_per_reset(self):
        verifier = K1PrefixMarginal()
        before = verifier.update_chunk(PREFIX)
        first, second, third = verifier.finalize(), verifier.finalize(), verifier.finalize()
        self.assertTrue(first.emitted)
        self.assertFalse(second.emitted)
        self.assertEqual(second, third)
        self.assertIs(first.snapshot, second.snapshot)
        self.assertEqual(first.snapshot.state_log_masses, before.state_log_masses)
        self.assertEqual(first.snapshot.frames, before.frames)
        self.assertTrue(first.snapshot.finalized)
        self.assertEqual(first.decision, "REJECT_K1")
        for update in (lambda: verifier.update(row("窝")), lambda: verifier.update_chunk([])):
            with self.assertRaises(FinalizedError):
                update()
        self.assertIs(verifier.snapshot(), first.snapshot)
        verifier.reset()
        verifier.update_chunk(PREFIX + rows("窝"))
        self.assertTrue(verifier.finalize().emitted)
        self.assertEqual(verifier.finalize().decision, "ACCEPT_K1")

    def test_reset_partial_and_finalized_runs_match_fresh(self):
        for finalized in (False, True):
            reused = K1PrefixMarginal()
            reused.update_chunk(rows("屋", "blank", "你"))
            if finalized:
                reused.finalize()
            self.assertEqual(reused.reset(), K1PrefixMarginal().snapshot())
            fresh = K1PrefixMarginal()
            for verifier in (reused, fresh):
                verifier.update_chunk(PREFIX + rows("窝"))
            self.assertEqual(reused.finalize(), fresh.finalize())

    def test_random_deterministic_ragged_chunk_partition_invariance(self):
        generator = random.Random(440193)
        for case in range(20):
            observation = [tuple(generator.randrange(1, 101) for _ in TOKENS) for _ in range(generator.randrange(0, 201))]
            whole = K1PrefixMarginal()
            whole.update_chunk(observation)
            split = K1PrefixMarginal()
            position = 0
            while position < len(observation):
                split.update_chunk([])
                size = generator.randrange(1, 24)
                split.update_chunk(iter(observation[position:position + size]))
                position += size
            self.assertEqual(split.snapshot(), whole.snapshot(), msg=f"case {case}")
            self.assertEqual(split.finalize(), whole.finalize(), msg=f"case {case}")

    def test_empty_chunks_are_noops_before_eof(self):
        verifier = K1PrefixMarginal()
        for observation in ([], PREFIX):
            verifier.update_chunk(observation)
            before = verifier.snapshot()
            self.assertEqual(verifier.update_chunk(iter(())), before)

    def test_capacity_is_observation_bound_and_never_an_automatic_window(self):
        verifier = K1PrefixMarginal()
        before = verifier.update_chunk(repeat(row("blank"), MAX_FRAMES))
        self.assertEqual(verifier.update_chunk([]), before)
        with self.assertRaises(CapacityError):
            verifier.update(row("你"))
        self.assertEqual(verifier.snapshot(), before)
        self.assertEqual(verifier.finalize().snapshot.frames, MAX_FRAMES)
        verifier.reset()
        self.assertEqual(verifier.update(row("你")).frames, 1)

    def test_overcapacity_chunk_is_atomic(self):
        verifier = K1PrefixMarginal()
        before = verifier.update_chunk(repeat(row("blank"), MAX_FRAMES - 1))
        with self.assertRaises(CapacityError):
            verifier.update_chunk(rows("你", "好"))
        self.assertEqual(verifier.snapshot(), before)

    def test_unbounded_chunk_stops_at_capacity_without_committing(self):
        verifier = K1PrefixMarginal()
        before = verifier.snapshot()
        with self.assertRaises(CapacityError):
            verifier.update_chunk(repeat(row("blank")))
        self.assertEqual(verifier.snapshot(), before)


class InputValidationTests(unittest.TestCase):
    def test_wrong_or_missing_token_mapping_fails(self):
        for mapping in (TOKENS[::-1], TOKENS[:-1], TOKENS + ("extra",), ("_",) + TOKENS[1:], "blank你好小窝屋", set(TOKENS), None):
            with self.subTest(mapping=mapping), self.assertRaises(InputError):
                K1PrefixMarginal(token_order=mapping)
        self.assertEqual(K1PrefixMarginal(token_order=list(TOKENS)).snapshot(), K1PrefixMarginal().snapshot())

    def test_invalid_rows_do_not_mutate_state(self):
        invalid = (
            (), [1] * 5, [1] * 7, None, "111111", {value: 1 for value in TOKENS}, set(range(6)),
            [0] * 6, [0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, math.nan],
            [0, 0, 0, 0, 0, math.inf], [0, 0, 0, 0, 0, -math.inf],
            [1, 0, 0, 0, 0, True], [1, 0, 0, 0, 0, "1"], [1, 0, 0, 0, 0, 1j],
            [1, 0, 0, 0, 0, 10**1000], [1, 0, 0, 0, 0, Fraction(1, 10**1000)],
        )
        verifier = K1PrefixMarginal()
        before = verifier.update_chunk(PREFIX)
        for value in invalid:
            with self.subTest(value=repr(value)[:100]), self.assertRaises(InputError):
                verifier.update(value)
            self.assertEqual(verifier.snapshot(), before)
        self.assertEqual(verifier.update(row("窝")).A, 1.0)

    def test_invalid_later_chunk_row_does_not_commit_earlier_rows(self):
        verifier = K1PrefixMarginal()
        before = verifier.update(row("你"))
        with self.assertRaises(InputError):
            verifier.update_chunk([row("好"), row("小"), [0] * 6])
        self.assertEqual(verifier.snapshot(), before)

    def test_iterator_failure_is_atomic(self):
        def failing_rows():
            yield row("好")
            raise RuntimeError("synthetic source failure")
        verifier = K1PrefixMarginal()
        before = verifier.update(row("你"))
        with self.assertRaisesRegex(RuntimeError, "synthetic source failure"):
            verifier.update_chunk(failing_rows())
        self.assertEqual(verifier.snapshot(), before)

    def test_invalid_chunk_container_fails_without_mutation(self):
        verifier = K1PrefixMarginal()
        before = verifier.snapshot()
        for value in (None, 1, "123456", {}, set(), frozenset({row("你")})):
            with self.subTest(value=value), self.assertRaises(InputError):
                verifier.update_chunk(value)
            self.assertEqual(verifier.snapshot(), before)

    def test_unbounded_row_is_rejected_after_seven_values(self):
        verifier = K1PrefixMarginal()
        with self.assertRaises(InputError):
            verifier.update(repeat(1))
        self.assertEqual(verifier.snapshot().frames, 0)

    def test_explicit_zero_weights_are_legal(self):
        verifier = K1PrefixMarginal()
        self.assertEqual(verifier.update([-0.0, 1, 0, 0, 0, 0]).frames, 1)
        self.assertEqual(verifier.snapshot().state_masses[2], 1.0)


if __name__ == "__main__":
    unittest.main()
