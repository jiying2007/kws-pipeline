"""Same Decimal80 softmax enclosure, reconstructed; no model/Torch/native calls."""
import decimal
D = decimal.Decimal
ZERO, ONE, TWO = D(0), D(1), D(2)


def contexts():
    result = []
    for mode in [decimal.ROUND_HALF_EVEN, decimal.ROUND_CEILING, decimal.ROUND_FLOOR]:
        context = decimal.Context(prec=80, rounding=mode, Emin=-999999, Emax=999999)
        for trap in [decimal.InvalidOperation, decimal.Overflow, decimal.Underflow,
                     decimal.Subnormal, decimal.DivisionByZero]:
            context.traps[trap] = True
        result.append(context)
    return result


def softmax_interval(values, lowers, uppers):
    near, up, down = contexts()
    centers, lower, upper = ([D(v) for v in sequence] for sequence in [values, lowers, uppers])
    assert len(centers) == len(lower) == len(upper) == 6
    assert all(v.is_finite() for sequence in [centers, lower, upper] for v in sequence)
    assert all(lo <= value <= hi for lo, value, hi in zip(lower, centers, upper))
    logit_radius = max(max(up.subtract(value, lo), up.subtract(hi, value))
                       for value, lo, hi in zip(centers, lower, upper))
    assert ZERO <= logit_radius <= D('1e-20')
    maximum, exponentials, enclosures = max(centers), [], []
    for value in centers:
        shifted = near.subtract(value, maximum)
        shift_radius = max(up.subtract(shifted, down.subtract(value, maximum)),
                           up.subtract(up.subtract(value, maximum), shifted))
        assert ZERO <= shift_radius <= D('0.5')
        rounded = near.exp(shifted)
        assert ZERO < rounded <= ONE
        exp_radius = max(up.subtract(near.next_plus(rounded), rounded),
                         up.subtract(rounded, near.next_minus(rounded)))
        upper_exp = up.add(rounded, exp_radius)
        radius = up.add(exp_radius, up.multiply(up.multiply(TWO, shift_radius), upper_exp))
        exponentials.append(rounded)
        enclosures.append([max(ZERO, down.subtract(rounded, radius)), up.add(rounded, radius)])
    near_sum = low_sum = high_sum = ZERO
    for value, (lo, hi) in zip(exponentials, enclosures):
        near_sum = near.add(near_sum, value)
        low_sum = down.add(low_sum, lo)
        high_sum = up.add(high_sum, hi)
    assert low_sum > ZERO
    probabilities = [near.divide(value, near_sum) for value in exponentials]
    radius = up.divide(logit_radius, TWO)
    intervals = [[max(ZERO, down.subtract(down.divide(lo, high_sum), radius)),
                  min(ONE, up.add(up.divide(hi, low_sum), radius))] for lo, hi in enclosures]
    assert all(lo <= value <= hi for value, (lo, hi) in zip(probabilities, intervals))
    return dict(probabilities=[str(v) for v in probabilities],
                probability_intervals=[[str(lo), str(hi)] for lo, hi in intervals],
                max_logit_uncertainty=str(logit_radius), decimal_exp_calls=6,
                source='unrounded propagated mathematical final logits')


def compare_absolute(actual, intervals, tolerance):
    _, up, _ = contexts()
    tolerance = D(tolerance)
    assert tolerance > 0 and len(actual) == len(intervals)
    failures, maximum = [], ZERO
    for index, (value, bounds) in enumerate(zip(actual, intervals)):
        observed = D.from_float(float(value))
        lo, hi = map(D, bounds)
        assert observed.is_finite() and lo.is_finite() and hi.is_finite() and lo <= hi
        distance = max(up.subtract(observed, lo), up.subtract(hi, observed))
        maximum = max(maximum, distance)
        if distance > tolerance:
            failures.append(dict(index=index, actual=str(observed), lower=str(lo), upper=str(hi),
                                 distance_upper=str(distance), tolerance=str(tolerance)))
    return dict(passed=not failures, elements=len(actual), failures=failures,
                maximum_absolute_error_upper=str(maximum), tolerance=str(tolerance))
