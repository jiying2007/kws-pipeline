"""Research-only, fixed six-token first-prefix CTC marginal at supplied EOF.

No model, audio, saved-data reader, endpoint detector, or decoder integration.
Inputs are literal full-head weights, never inferred logits. See README.md.
"""

from dataclasses import dataclass
from collections.abc import Mapping, Set
from fractions import Fraction
from itertools import islice
import math
from numbers import Real
from typing import Iterable, Literal


TOKENS = ("blank", "你", "好", "小", "窝", "屋")
STATE_NAMES = ("e", "b1", "r1", "b2", "r2", "b3", "r3", "A", "B", "D")
MAX_FRAMES = 4096  # An explicit research scope bound; never an automatic window.
CERTIFICATE_BITS = 256  # Arithmetic precision, not a decision threshold.
CERTIFICATE_SCALE = 1 << CERTIFICATE_BITS
Decision = Literal["ACCEPT_K1", "REJECT_K1"]
DecisionStatus = Literal["CERTIFIED_ACCEPT", "CERTIFIED_REJECT", "NUMERICALLY_UNRESOLVED"]
_NEG_INF = -math.inf
_INITIAL_STATE = (0.0,) + (_NEG_INF,) * 9
_INITIAL_BOUNDS = (CERTIFICATE_SCALE,) + (0,) * 9


class InputError(ValueError):
    """An observation row or declared token order violates the input contract."""


class CapacityError(ValueError):
    """The observation would exceed the explicit 4096-row research bound."""


class FinalizedError(RuntimeError):
    """An update was attempted after EOF; reset explicitly to start again."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable readout; logs retain positive masses below float exp range.

    A/B/R and their natural logs are approximate float64 readouts. Linear
    masses can underflow to zero while their logs remain finite; exact zero
    has log -inf. Exact integer enclosures, not these logs, certify decisions.
    State tuples follow STATE_NAMES. No pre-EOF decision is made here.
    """

    frames: int
    finalized: bool
    A: float
    B: float
    R: float
    log_A: float
    log_B: float
    log_R: float
    state_masses: tuple[float, ...]
    state_log_masses: tuple[float, ...]
    state_lower_numerators: tuple[int, ...]
    state_upper_numerators: tuple[int, ...]

    @property
    def masses(self) -> tuple[float, float, float]:
        return self.A, self.B, self.R

    @property
    def log_masses(self) -> tuple[float, float, float]:
        return self.log_A, self.log_B, self.log_R

    @property
    def mass_bounds(self) -> tuple[tuple[Fraction, Fraction], ...]:
        """Rigorous exact-rational A/B/R enclosures for the float64 inputs.

        Fractions avoid rounding a narrow interval to two identical floats.
        Bounds are diagnostics/certificates; they never renormalize masses.
        """
        lower, upper = self.state_lower_numerators, self.state_upper_numerators
        return tuple(
            (Fraction(low, CERTIFICATE_SCALE), Fraction(high, CERTIFICATE_SCALE))
            for low, high in (
                (lower[7], upper[7]),
                (lower[8], upper[8]),
                (sum(lower[:7]) + lower[9], sum(upper[:7]) + upper[9]),
            )
        )


@dataclass(frozen=True, slots=True)
class Finalization:
    """EOF result; only the first finalize after reset has emitted=True.

    Every result includes the full immutable mass/log snapshot. Repeated EOF
    calls preserve that snapshot and decision but do not emit another event.
    """

    snapshot: Snapshot
    decision: Decision
    emitted: bool
    decision_status: DecisionStatus

    @property
    def masses(self) -> tuple[float, float, float]:
        return self.snapshot.masses

    @property
    def log_masses(self) -> tuple[float, float, float]:
        return self.snapshot.log_masses


def _logadd(left: float, right: float) -> float:
    if left == _NEG_INF:
        return right
    if right == _NEG_INF:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def _logsum(values: Iterable[float]) -> float:
    values = tuple(values)
    high = max(values, default=_NEG_INF)
    if high == _NEG_INF:
        return _NEG_INF
    return high + math.log(math.fsum(math.exp(value - high) for value in values))


def _six_values(values: Iterable[object], name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise InputError(f"{name} must be an ordered iterable with exactly six values")
    try:
        # Read at most seven values even if a malformed iterable is unbounded.
        result = tuple(islice(iter(values), 7))
    except TypeError as exc:
        raise InputError(f"{name} must be an ordered iterable with exactly six values") from exc
    if len(result) != 6:
        raise InputError(f"{name} must contain exactly six values")
    return result


def _prepare_weights(weights: Iterable[Real]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    raw = _six_values(weights, "weights")
    logs = []
    ratios = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InputError(f"weight {index} must be a finite nonnegative real number")
        try:
            weight = float(value)
        except (OverflowError, ValueError) as exc:
            raise InputError(f"weight {index} must be representable as finite float64") from exc
        if not math.isfinite(weight) or weight < 0:
            raise InputError(f"weight {index} must be finite and nonnegative")
        if weight == 0 and value != 0:
            raise InputError(f"weight {index} underflows when converted to float64")
        logs.append(math.log(weight) if weight > 0 else _NEG_INF)
        ratios.append(weight.as_integer_ratio())
    high = max(logs)
    if high == _NEG_INF:
        raise InputError("weights must include at least one positive value")
    shifted = tuple(value - high for value in logs)
    # Normalize all six competing weights without overflowing their sum or
    # underflowing a small numerator. No floors or bucket renormalization.
    log_denominator = math.log(math.fsum(math.exp(value) for value in shifted))
    # Every binary64 denominator is a power of two. This common denominator
    # produces the exact six relative integer weights, without float division.
    common_denominator = max(denominator for _, denominator in ratios)
    integers = tuple(
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    )
    return tuple(value - log_denominator for value in shifted), integers


def _step(state: tuple[float, ...], p: tuple[float, ...]) -> tuple[float, ...]:
    """Ten-accumulator recurrence; every right-hand side uses the old state."""
    e, b1, r1, b2, r2, b3, r3, a, b, d = state
    blank, ni, hao, xiao, wo, wu = p
    h1, h2, h3 = _logadd(b1, r1), _logadd(b2, r2), _logadd(b3, r3)
    # Direct nonnegative mismatch flux, with blank-separated repeats included.
    next_d = _logsum((
        d,
        e + _logsum((hao, xiao, wo, wu)),
        b1 + _logsum((ni, xiao, wo, wu)),
        r1 + _logsum((xiao, wo, wu)),
        b2 + _logsum((ni, hao, wo, wu)),
        r2 + _logsum((ni, wo, wu)),
        b3 + _logsum((ni, hao, xiao)),
        r3 + _logsum((ni, hao)),
    ))
    return (
        blank + e,
        blank + h1,
        ni + _logadd(r1, e),
        blank + h2,
        hao + _logadd(r2, h1),
        blank + h3,
        xiao + _logadd(r3, h2),
        _logadd(a, wo + h3),
        _logadd(b, wu + h3),
        next_d,
    )


def _step_bounds(
    values: tuple[int, ...], weights: tuple[int, ...], *, upper: bool
) -> tuple[int, ...]:
    """Directed integer evaluation of the same ten-state nonnegative map.

    If values/2**256 bound the old probabilities, these bound the new ones.
    All intermediate sums/products are exact Python integers. Each output is
    rounded once, downward for lower bounds or upward for upper bounds.
    """
    e, b1, r1, b2, r2, b3, r3, a, b, d = values
    blank, ni, hao, xiao, wo, wu = weights
    denominator = sum(weights)
    h1, h2, h3 = b1 + r1, b2 + r2, b3 + r3
    numerators = (
        blank * e,
        blank * h1,
        ni * (r1 + e),
        blank * h2,
        hao * (r2 + h1),
        blank * h3,
        xiao * (r3 + h2),
        a * denominator + wo * h3,
        b * denominator + wu * h3,
        d * denominator
        + e * (hao + xiao + wo + wu)
        + b1 * (ni + xiao + wo + wu)
        + r1 * (xiao + wo + wu)
        + b2 * (ni + hao + wo + wu)
        + r2 * (ni + wo + wu)
        + b3 * (ni + hao + xiao)
        + r3 * (ni + hao),
    )
    adjustment = denominator - 1 if upper else 0
    return tuple((value + adjustment) // denominator for value in numerators)


def _snapshot(
    state: tuple[float, ...], lower: tuple[int, ...], upper: tuple[int, ...],
    frames: int, finalized: bool,
) -> Snapshot:
    log_a, log_b = state[7], state[8]
    log_r = _logsum((*state[:7], state[9]))
    return Snapshot(
        frames=frames,
        finalized=finalized,
        A=math.exp(log_a),
        B=math.exp(log_b),
        R=math.exp(log_r),
        log_A=log_a,
        log_B=log_b,
        log_R=log_r,
        state_masses=tuple(math.exp(value) for value in state),
        state_log_masses=state,
        state_lower_numerators=lower,
        state_upper_numerators=upper,
    )


class K1PrefixMarginal:
    """One finite first-prefix observation, at most MAX_FRAMES full-head rows.

    Alphabet and class prefixes are fixed, not generic configurable targets.
    update_chunk is transactional for this verifier: any invalid row, iterator
    failure, or overflow leaves its previous state intact. External iterator
    consumption cannot be rolled back. This object is not thread-safe.
    """

    __slots__ = ("_state", "_lower", "_upper", "_frames", "_final")

    def __init__(self, token_order: Iterable[str] = TOKENS) -> None:
        if _six_values(token_order, "token_order") != TOKENS:
            raise InputError(f"token_order must exactly equal {TOKENS!r}")
        self._state = _INITIAL_STATE
        self._lower = self._upper = _INITIAL_BOUNDS
        self._frames = 0
        self._final: Finalization | None = None

    def snapshot(self) -> Snapshot:
        """Read without changing state or making a decision."""
        if self._final is not None:
            return self._final.snapshot
        return _snapshot(self._state, self._lower, self._upper, self._frames, False)

    def update(self, weights: Iterable[Real]) -> Snapshot:
        """Consume one full six-weight row; failure leaves the state intact."""
        return self.update_chunk((weights,))

    def update_chunk(self, rows: Iterable[Iterable[Real]]) -> Snapshot:
        """Consume rows in order with no chunk-boundary blank or reset.

        Empty chunks are no-ops before EOF. Every update after EOF, including
        an empty chunk, raises FinalizedError until an explicit reset.
        """
        if self._final is not None:
            raise FinalizedError("observation is finalized; call reset before updating")
        if isinstance(rows, (str, bytes, bytearray, Mapping, Set)):
            raise InputError("rows must be an ordered iterable of six-weight rows")
        try:
            iterator = iter(rows)
        except TypeError as exc:
            raise InputError("rows must be an ordered iterable of six-weight rows") from exc
        state, frames = self._state, self._frames
        lower, upper = self._lower, self._upper
        for row in iterator:
            if frames >= MAX_FRAMES:
                raise CapacityError(f"observation exceeds the {MAX_FRAMES}-row research bound")
            logs, weights = _prepare_weights(row)
            state = _step(state, logs)
            lower = _step_bounds(lower, weights, upper=False)
            upper = _step_bounds(upper, weights, upper=True)
            frames += 1
        # Do not commit until the entire chunk has succeeded. No row history.
        self._state, self._frames = state, frames
        self._lower, self._upper = lower, upper
        return self.snapshot()

    def finalize(self) -> Finalization:
        """At EOF only: test fixed rule A > B + R; emit at most once.

        Exact integer bounds certify the comparison for normalized binary64
        inputs. An unresolved comparison fails closed and is explicitly marked;
        it is not claimed to be a mathematically proven negative. Exact ties
        can never accept. No fitted tolerance, blank, or terminal is inserted.
        """
        if self._final is not None:
            return Finalization(
                self._final.snapshot, self._final.decision, False,
                self._final.decision_status,
            )
        snapshot = _snapshot(self._state, self._lower, self._upper, self._frames, True)
        # Exhaustive exact partition gives A > B+R iff A > 1/2. Using that
        # identity tightens a certificate without dropping/reweighting R.
        half = CERTIFICATE_SCALE // 2
        if self._lower[7] > half:
            decision: Decision = "ACCEPT_K1"
            status: DecisionStatus = "CERTIFIED_ACCEPT"
        elif self._upper[7] <= half:
            decision, status = "REJECT_K1", "CERTIFIED_REJECT"
        else:
            decision, status = "REJECT_K1", "NUMERICALLY_UNRESOLVED"
        self._final = Finalization(snapshot, decision, True, status)
        return self._final

    def reset(self) -> Snapshot:
        """Start a new explicitly bounded observation, clearing the EOF event."""
        self._state = _INITIAL_STATE
        self._lower = self._upper = _INITIAL_BOUNDS
        self._frames = 0
        self._final = None
        return self.snapshot()
